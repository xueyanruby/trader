"""
盯盤調度器：定時信號計算 + 分市場實時價格監控

┌─────────────────────────────────────────────────────────────────┐
│  市場          │  價格獲取方式          │  數據延遲               │
├─────────────────────────────────────────────────────────────────┤
│  A 股（CN）    │  AKShare 分鐘線        │  ~1 分鐘（東方財富源）  │
│  港股（HK）    │  AKShare 實時快照輪詢  │  ~10-30 秒              │
│  美股（US）    │  yfinance 快照輪詢     │  ~15 秒                 │
└─────────────────────────────────────────────────────────────────┘

兩種觸發邏輯
-----------
1. 定時信號任務（Cron）
   每個交易日收盤後自動執行：拉日線 → 計算策略信號 → 比對上次倉位
   → 推送「買入 / 賣出 / 繼續持有」通知 + 每日持倉日報

2. 盤中價格監控（Interval 輪詢）
   以可配置的間隔持續拉取各市場最新價格：
   - A 股：akshare.stock_zh_a_hist_min_em  拉最新分鐘 K 線
   - 港股：akshare.stock_hk_spot_em        拉實時快照
   - 美股：yfinance Ticker.fast_info        拉快速報價
   價格一旦觸碰止損 / 止盈 / 熔斷閾值，立即推送高優先級警報

依賴
----
pip install apscheduler akshare yfinance

啟動方式（通過 main.py watch 命令）：
    scheduler = WatchScheduler(config, notifier, broker, risk_manager)
    scheduler.start()   # 阻塞，Ctrl-C 退出
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from loguru import logger

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False

from src.risk.metrics import RealTimeMetrics
from src.risk.regime_detector import RegimeDetector
from src.risk.black_swan_detector import BlackSwanDetector

from src.analysis.trade_reviewer import TradeReviewer
from src.analysis.attribution import ReturnAttributor
from src.analysis.strategy_optimizer import StrategyOptimizer
from src.analysis.report_generator import DailyReportGenerator


# ---------------------------------------------------------------------------
# 輔助：市場分類
# ---------------------------------------------------------------------------

def _classify_symbols(
    symbols: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    將持倉列表按市場分組。
    規則（粗略，可根據實際代碼格式調整）：
      A 股 : 6 位純數字，如 "000001" / "600519"
      港股 : 5 位純數字或帶 ".HK" 後綴
      美股 : 其餘（字母為主）
    """
    cn, hk, us = [], [], []
    for sym in symbols:
        s = sym.upper()
        if s.endswith(".HK") or (s.isdigit() and len(s) == 5):
            hk.append(sym)
        elif sym.isdigit() and len(sym) in (5, 6):
            cn.append(sym)
        else:
            us.append(sym)
    return cn, hk, us


def _select_column(columns: Sequence[Any], candidates: Sequence[str]) -> Optional[Any]:
    """Return the first matching column name, accepting simplified/traditional variants."""
    normalized_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        match = normalized_map.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# 價格抓取器（每個市場一個靜態方法）
# ---------------------------------------------------------------------------

class PriceFetcher:
    """各市場的最新價格快照抓取。"""

    @staticmethod
    def fetch_futu_snapshot(symbols: List[str]) -> Dict[str, float]:
        """
        HK/CN：通過 futu OpenAPI 快照拉取最新價（無需訂閱）。
        兼容輸入："00700" / "00700.HK" / "HK.00700" / "600519" / "SH.600519" ...
        """
        try:
            from src.futu.quote_client import get_global_quote_client
        except Exception as exc:
            logger.warning(f"futu 行情不可用：{exc}")
            return {}

        try:
            qc = get_global_quote_client()
            return qc.get_latest_prices(symbols)
        except Exception as exc:
            logger.warning(f"futu 快照獲取失敗：{exc}")
            return {}

    @staticmethod
    def fetch_cn_minute(symbols: List[str]) -> Dict[str, float]:
        """
        A 股：通過 AKShare 分鐘線取最新一根 K 線的收盤價。
        接口：ak.stock_zh_a_hist_min_em(symbol, period="1", ...)
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安裝，A 股分鐘線不可用")
            return {}

        prices: Dict[str, float] = {}
        now = datetime.now()
        # 往前取 10 分鐘窗口，確保拿到最新一根
        start = (now - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        end = now.strftime("%Y-%m-%d %H:%M:%S")

        for sym in symbols:
            try:
                df = ak.stock_zh_a_hist_min_em(
                    symbol=sym,
                    period="1",
                    start_datetime=start,
                    end_datetime=end,
                    adjust="",
                )
                if df is not None and not df.empty:
                    # 列名：时间 / 开盘 / 收盘 / 最高 / 最低 / 成交量 / 成交额 / 振幅 / 涨跌幅 / 涨跌额 / 换手率
                    close_col = "收盘" if "收盘" in df.columns else df.columns[2]
                    prices[sym] = float(df[close_col].iloc[-1])
            except Exception as exc:
                logger.debug(f"A 股分鐘線獲取失敗 {sym}: {exc}")

        logger.debug(f"A 股分鐘線：成功獲取 {len(prices)}/{len(symbols)} 只")
        return prices

    @staticmethod
    def fetch_hk_snapshot(symbols: List[str]) -> Dict[str, float]:
        """
        港股：通過 AKShare stock_hk_spot_em 拉取實時快照。
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安裝，港股快照不可用")
            return {}

        prices: Dict[str, float] = {}
        try:
            df = ak.stock_hk_spot_em()
            # 列名包含：代码 / 名称 / 最新价 / 涨跌幅 / ...
            if df is None or df.empty:
                return prices

            code_col = "代码" if "代码" in df.columns else df.columns[0]
            price_col = "最新价" if "最新价" in df.columns else df.columns[2]

            sym_set = set(symbols)
            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(5)
                if code in sym_set or row[code_col] in sym_set:
                    try:
                        prices[str(row[code_col])] = float(row[price_col])
                    except (ValueError, TypeError):
                        pass
        except Exception as exc:
            logger.warning(f"港股快照獲取失敗：{exc}")

        logger.debug(f"港股快照：成功獲取 {len(prices)}/{len(symbols)} 只")
        return prices

    @staticmethod
    def fetch_us_snapshot(symbols: List[str]) -> Dict[str, float]:
        """
        美股：通過 yfinance Ticker.fast_info 拉取最快報價。
        fast_info 不需要下載完整歷史，速度遠快於 download()。
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance 未安裝，美股快照不可用")
            return {}

        prices: Dict[str, float] = {}
        for sym in symbols:
            try:
                fi = yf.Ticker(sym).fast_info
                # fast_info 有 last_price 屬性（yfinance >= 0.2）
                price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
                if price and float(price) > 0:
                    prices[sym] = float(price)
            except Exception as exc:
                logger.debug(f"美股快照獲取失敗 {sym}: {exc}")

        logger.debug(f"美股快照：成功獲取 {len(prices)}/{len(symbols)} 只")
        return prices

    @staticmethod
    def fetch_us_futu_snapshot(symbols: List[str]) -> Dict[str, float]:
        """US：通過 futu OpenAPI 快照拉取最新價（無需訂閱）。"""
        return PriceFetcher.fetch_futu_snapshot(symbols)

    @staticmethod
    def fetch_all(
        symbols: List[str],
        market_mode: str = "us",
    ) -> Dict[str, float]:
        """
        根據市場模式路由到對應抓取方法。
        multi 模式：自動按代碼特徵分組，分別抓取後合並。
        """
        if market_mode == "cn":
            return PriceFetcher.fetch_cn_minute(symbols)

        if market_mode == "hk":
            return PriceFetcher.fetch_hk_snapshot(symbols)

        if market_mode == "us":
            return PriceFetcher.fetch_us_snapshot(symbols)

        if market_mode == "multi":
            cn_syms, hk_syms, us_syms = _classify_symbols(symbols)
            result: Dict[str, float] = {}
            if cn_syms:
                result.update(PriceFetcher.fetch_cn_minute(cn_syms))
            if hk_syms:
                result.update(PriceFetcher.fetch_hk_snapshot(hk_syms))
            if us_syms:
                result.update(PriceFetcher.fetch_us_snapshot(us_syms))
            return result

        logger.warning(f"未知市場模式 '{market_mode}'，跳過價格獲取")
        return {}


# ---------------------------------------------------------------------------
# 調度器主類
# ---------------------------------------------------------------------------

class WatchScheduler:
    """
    盯盤調度器。

    Parameters
    ----------
    config       : 完整 config.yaml 內容（dict）
    notifier     : Notifier 實例
    broker       : BaseBroker 實例（PaperBroker 或 AlpacaBroker）
    risk_manager : RiskManager 實例
    """

    def __init__(
        self,
        config: Dict[str, Any],
        notifier,
        broker,
        risk_manager,
    ):
        if not _HAS_APSCHEDULER:
            raise ImportError("APScheduler 未安裝，請運行：pip install apscheduler")

        self.config = config
        self.notifier = notifier
        self.broker = broker
        self.risk_manager = risk_manager

        self.market_mode: str = config.get("market", {}).get("mode", "us")
        self.strategy_name: str = config.get("strategy", {}).get("active", "momentum_12_1")

        watch_cfg = config.get("watch", {})
        self.signal_cron: str = watch_cfg.get("signal_cron", "0 16 * * 1-5")

        # 各市場輪詢間隔（秒）
        self.cn_interval: int = watch_cfg.get("cn_minute_interval_seconds", 60)
        self.hk_interval: int = watch_cfg.get("hk_snapshot_interval_seconds", 120)
        self.us_interval: int = watch_cfg.get("us_snapshot_interval_seconds", 300)
        self.enable_price_monitor: bool = watch_cfg.get("enable_price_monitor", True)

        # 自選股名單
        self.watchlist: List[str] = watch_cfg.get("watchlist", [])
        self.also_monitor_positions: bool = watch_cfg.get("also_monitor_positions", True)
        self.repeat_alert: bool = watch_cfg.get("repeat_alert", False)

        # 已推送過警報的股票集合（用於去重，每日重置）
        self._alerted_today: Set[str] = set()
        self._alert_reset_date: Optional[date] = None

        # 盤中技術信號掃描配置
        self.signal_monitor_interval: int = watch_cfg.get("signal_monitor_interval_seconds", 1800)
        self.short_signal_cfg: Dict[str, Any] = watch_cfg.get("short_term_signal", {})
        self.long_signal_cfg: Dict[str, Any] = watch_cfg.get("long_term_signal", {})
        self.signal_monitor_enabled: bool = (
            self.short_signal_cfg.get("enabled", True)
            or self.long_signal_cfg.get("enabled", True)
        )

        # 內部狀態
        self._last_signals: Dict[str, int] = {}
        self._scheduler = BlockingScheduler(timezone="Asia/Shanghai")

        # ── 智能風控模塊 ───────────────────────────────────────────────
        smart_cfg = config.get("risk", {}).get("smart_risk", {})
        self.smart_risk_enabled: bool = smart_cfg.get("enabled", True)

        self.rt_metrics = RealTimeMetrics(
            lookback_days=smart_cfg.get("metrics_lookback_days", 20),
        )
        self.regime_detector = RegimeDetector.from_config(smart_cfg)
        self.black_swan_detector = BlackSwanDetector.from_config(smart_cfg)

        # ── 收盤复盘/归因/策略升级 ───────────────────────────────────
        self.trade_reviewer = TradeReviewer()
        self.attributor = ReturnAttributor()
        self.strategy_optimizer = StrategyOptimizer()
        self.report_generator = DailyReportGenerator()

        # 交易開關
        self.trade_on_signal: bool = watch_cfg.get("trade_on_signal", False)
        self.weekly_report_cron: str = watch_cfg.get("weekly_report_cron", "0 18 * * 5")

        # 黑天鵝緊急倉位（觸發後目標敞口，0 = 清倉）
        self.emergency_exposure: float = smart_cfg.get("black_swan_emergency_exposure", 0.0)

        # 本週期間追蹤：用於週報計算
        self._week_start_equity: float = 0.0
        self._current_regime: str = "neutral"
        self._current_regime_scale: float = 0.20

        if self.trade_on_signal:
            logger.info("✅ trade_on_signal=true：信號觸發後將自動在模擬盤下單")
        if self.smart_risk_enabled:
            logger.info(
                f"🛡️ 智能風控已啟用：regime_fast={self.regime_detector.fast_period}d  "
                f"regime_slow={self.regime_detector.slow_period}d  "
                f"bs_zscore={self.black_swan_detector.zscore_threshold}σ"
            )

        logger.info(
            f"WatchScheduler 初始化：strategy={self.strategy_name}, "
            f"market={self.market_mode}, signal_cron='{self.signal_cron}'"
        )
        if self.enable_price_monitor:
            logger.info(
                f"價格監控間隔 — "
                f"A 股:{self.cn_interval}s  "
                f"港股:{self.hk_interval}s  "
                f"美股:{self.us_interval}s"
            )
        if self.watchlist:
            logger.info(f"自選股名單（{len(self.watchlist)} 只）：{self.watchlist}")
            logger.info(f"同時盯持倉：{self.also_monitor_positions}  重複警報：{self.repeat_alert}")
        else:
            logger.info("自選股名單為空，僅盯當前持倉")

    # ------------------------------------------------------------------
    # 啟動 / 停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        """啟動調度器（阻塞，直到 Ctrl-C）。"""
        # 定時信號任務
        self._scheduler.add_job(
            func=self._run_signal_job,
            trigger=CronTrigger.from_crontab(self.signal_cron),
            id="signal_job",
            name="每日收盤信號計算",
            misfire_grace_time=300,
        )
        logger.info(f"已添加定時信號任務：cron='{self.signal_cron}'")

        # 分市場盯盤任務
        if self.enable_price_monitor:
            self._add_price_monitor_jobs()

        # 技術信號掃描任務（短線 / 長線買入機會）
        if self.signal_monitor_enabled:
            self._scheduler.add_job(
                func=self._run_signal_monitor,
                trigger=IntervalTrigger(seconds=self.signal_monitor_interval),
                id="signal_monitor",
                name="技術信號掃描（短線/長線）",
                coalesce=True,
                next_run_time=datetime.now(),   # 啟動後立即掃描一次
            )
            logger.info(
                f"已添加技術信號掃描任務："
                f"間隔={self.signal_monitor_interval}s  "
                f"短線={self.short_signal_cfg.get('enabled', True)}  "
                f"長線={self.long_signal_cfg.get('enabled', True)}"
            )

        # 週報任務
        self._scheduler.add_job(
            func=self._run_weekly_report_job,
            trigger=CronTrigger.from_crontab(self.weekly_report_cron),
            id="weekly_report",
            name="週度持倉報告",
            misfire_grace_time=3600,
        )
        logger.info(f"已添加週報任務：cron='{self.weekly_report_cron}'")

        logger.info("盯盤調度器啟動，按 Ctrl-C 退出...")
        try:
            self._scheduler.start()
        except KeyboardInterrupt:
            logger.info("調度器已停止")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def _add_price_monitor_jobs(self) -> None:
        """根據市場模式，添加對應的價格監控任務。"""
        if self.market_mode in ("cn", "multi"):
            self._scheduler.add_job(
                func=lambda: self._run_price_monitor("cn"),
                trigger=IntervalTrigger(seconds=self.cn_interval),
                id="price_monitor_cn",
                name="A 股分鐘線止損監控",
                coalesce=True,
                next_run_time=datetime.now(),
            )
            logger.info(f"A 股分鐘線監控：每 {self.cn_interval}s")

        if self.market_mode in ("hk", "multi"):
            self._scheduler.add_job(
                func=lambda: self._run_price_monitor("hk"),
                trigger=IntervalTrigger(seconds=self.hk_interval),
                id="price_monitor_hk",
                name="港股快照止損監控",
                coalesce=True,
                next_run_time=datetime.now(),
            )
            logger.info(f"港股快照監控：每 {self.hk_interval}s")

        if self.market_mode in ("us", "multi"):
            self._scheduler.add_job(
                func=lambda: self._run_price_monitor("us"),
                trigger=IntervalTrigger(seconds=self.us_interval),
                id="price_monitor_us",
                name="美股快照止損監控",
                coalesce=True,
                next_run_time=datetime.now(),
            )
            logger.info(f"美股快照監控：每 {self.us_interval}s")

        # single-market 模式也保底添加
        if self.market_mode not in ("cn", "hk", "us", "multi"):
            self._scheduler.add_job(
                func=lambda: self._run_price_monitor(self.market_mode),
                trigger=IntervalTrigger(seconds=self.us_interval),
                id="price_monitor_default",
                name="默認止損監控",
                coalesce=True,
                next_run_time=datetime.now(),
            )

    # ------------------------------------------------------------------
    # 任務 1：每日收盤信號計算
    # ------------------------------------------------------------------

    def _run_signal_job(self) -> None:
        """
        每日收盤後執行：
          拉日線 → 計算信號 → 市況研判 → 黑天鵝檢測
          → (可選) Futu 模擬盤下單 → 推送通知 + 日報 + 風控儀表板
        """
        logger.info(f"[定時任務] 開始每日信號計算 — {datetime.now():%Y-%m-%d %H:%M}")
        try:
            prices = self._fetch_daily_data()
            if prices is None or prices.empty:
                logger.warning("[定時任務] 日線數據拉取失敗，跳過")
                return

            # ── 計算策略信號 ──────────────────────────────────────────
            import pandas as pd
            signals_raw = self._compute_signals(prices)
            signals = pd.Series(signals_raw, dtype=float)
            buy, sell, hold = self._diff_signals(signals_raw)

            # ── 獲取最新收盤價（用於下單和風控）────────────────────────
            from src.data.preprocessor import Preprocessor
            close_prices = Preprocessor.get_close(prices)
            latest_prices: Dict[str, float] = close_prices.iloc[-1].dropna().to_dict()

            # ── 智能風控：市況研判 ────────────────────────────────────
            regime = "neutral"
            regime_scale = 0.20
            if self.smart_risk_enabled:
                try:
                    benchmark_prices = self._fetch_benchmark_prices()
                    equity = self.broker.get_equity()
                    metrics = self.rt_metrics.update(equity)
                    vol = metrics.get("volatility", 0.0)
                    regime, regime_scale = self.regime_detector.detect(benchmark_prices, vol)
                    self._current_regime = regime
                    self._current_regime_scale = regime_scale
                    logger.info(
                        f"[風控] 市況={regime.upper()}  最大倉位={regime_scale:.0%}  "
                        f"波動率={vol:.2%}  {self.rt_metrics.summary_str()}"
                    )

                    # 黑天鵝盤前檢測：基於組合昨日收益率
                    portfolio_return = metrics.get("today_return", 0.0)
                    history = list(self.rt_metrics._return_history)[:-1]  # 去掉今日
                    if history:
                        bs_triggered, bs_reason = self.black_swan_detector.check(
                            today_return=portfolio_return,
                            return_history=history,
                            current_vol=vol,
                            long_term_vol=self._get_long_term_vol(),
                        )
                        if bs_triggered:
                            logger.warning(f"[黑天鵝] 每日信號任務觸發！原因：{bs_reason}")
                            self._execute_emergency_reduce(
                                latest_prices, bs_reason, equity
                            )
                            return   # 緊急縮倉後不繼續正常信號流程
                except Exception as exc:
                    logger.warning(f"[風控] 市況研判異常，使用默認 neutral：{exc}")

            # ── (可選) 根據信號下模擬盤訂單 ──────────────────────────
            if self.trade_on_signal and not self.risk_manager.is_halted:
                self._execute_signal_trades(signals, latest_prices, regime, regime_scale)

            # ── 推送每日信號通知 ──────────────────────────────────────
            self.notifier.send_daily_signals(
                strategy=self.strategy_name,
                buy=buy, sell=sell, hold=hold,
                date=str(date.today()),
            )
            self._last_signals = dict(signals_raw)

            # ── 持倉報告 ──────────────────────────────────────────────
            try:
                summary = self.broker.portfolio_summary()
                self.notifier.send_portfolio_report(summary)
            except Exception:
                pass

            # ── 風控儀表板 ────────────────────────────────────────────
            if self.smart_risk_enabled:
                try:
                    rt = self.rt_metrics.get_last()
                    dashboard = self.risk_manager.risk_dashboard(rt)
                    dashboard["regime"] = regime
                    dashboard["max_exposure"] = regime_scale
                    self.notifier.send_risk_dashboard(dashboard)
                except Exception as exc:
                    logger.warning(f"[風控] 儀表板推送異常：{exc}")

            # ── 收盤后自动复盘日报（结论先行）──────────────────────────
            try:
                # portfolio_summary 可能取不到，这里尽量复用上面 summary
                portfolio_summary = {}
                try:
                    portfolio_summary = summary  # type: ignore[name-defined]
                except Exception:
                    portfolio_summary = self.broker.portfolio_summary()

                rt = self.rt_metrics.get_last()
                dashboard = self.risk_manager.risk_dashboard(rt)
                dashboard["regime"] = regime
                dashboard["max_exposure"] = regime_scale

                self._run_daily_review_job(
                    prices=prices,
                    close_prices=close_prices,
                    latest_prices=latest_prices,
                    portfolio_summary=portfolio_summary,
                    risk_dashboard=dashboard,
                )
            except Exception as exc:
                logger.warning(f"[复盘] 日报生成/推送异常：{exc}")

            logger.info(
                f"[定時任務] 完成：買入 {len(buy)} | 賣出 {len(sell)} | 持有 {len(hold)}  "
                f"市況={regime}  最大倉位={regime_scale:.0%}"
            )
        except Exception as exc:
            logger.exception(f"[定時任務] 異常：{exc}")
            self.notifier.send_raw(
                subject="【系統錯誤】每日信號任務異常",
                body=f"錯誤：{exc}\n時間：{datetime.now()}",
            )

    # ------------------------------------------------------------------
    # 任務 2：分市場止損監控
    # ------------------------------------------------------------------

    def _run_price_monitor(self, market: str) -> None:
        """
        拉取指定市場的最新價格，執行止損 / 止盈 / 熔斷檢查。

        Parameters
        ----------
        market : "cn" | "hk" | "us"
        """
        try:
            positions = self.broker.get_positions()

            # 構建本次要盯的股票列表
            # 優先級：watchlist > 持倉 > 空
            if self.watchlist:
                target: Set[str] = set(self.watchlist)
                if self.also_monitor_positions:
                    target |= set(positions.keys())
            else:
                if not positions:
                    return
                target = set(positions.keys())

            # 按市場篩選
            all_symbols = sorted(target)
            cn_syms, hk_syms, us_syms = _classify_symbols(all_symbols)
            market_syms = {"cn": cn_syms, "hk": hk_syms, "us": us_syms}.get(market, all_symbols)

            if not market_syms:
                return

            # 按市場選用對應拉取方式
            if market == "cn":
                provider = self.config.get("data", {}).get("provider", {}).get("cn", "futu")
                if provider == "akshare":
                    latest_prices = PriceFetcher.fetch_cn_minute(market_syms)
                    source = "分鐘線"
                else:
                    latest_prices = PriceFetcher.fetch_futu_snapshot(market_syms)
                    source = "快照"
            elif market == "hk":
                provider = self.config.get("data", {}).get("provider", {}).get("hk", "futu")
                if provider == "akshare":
                    latest_prices = PriceFetcher.fetch_hk_snapshot(market_syms)
                    source = "快照"
                else:
                    latest_prices = PriceFetcher.fetch_futu_snapshot(market_syms)
                    source = "快照"
            else:
                provider = self.config.get("data", {}).get("provider", {}).get("us", "yfinance")
                if provider == "futu":
                    latest_prices = PriceFetcher.fetch_us_futu_snapshot(market_syms)
                else:
                    latest_prices = PriceFetcher.fetch_us_snapshot(market_syms)
                source = "快照"

            if not latest_prices:
                logger.debug(f"[{market.upper()} {source}] 未獲取到價格，跳過本輪")
                return

            logger.debug(
                f"[{market.upper()} {source}] "
                f"獲取 {len(latest_prices)}/{len(market_syms)} 只最新價格"
            )

            # 更新 Broker 價格 + 風控資產淨值
            self.broker.update_prices(latest_prices)
            equity = self.broker.get_equity()
            self.risk_manager.update_equity(equity)

            # ── 實時風控指標更新 + 盤中黑天鵝檢測 ───────────────────
            if self.smart_risk_enabled:
                try:
                    metrics = self.rt_metrics.update(equity)
                    self._check_black_swan_intraday(metrics, latest_prices, equity)
                except Exception as exc:
                    logger.debug(f"[風控] 實時指標更新異常：{exc}")

            # 熔斷優先檢查
            if self.risk_manager.is_halted:
                drawdown = self._calc_drawdown(equity)
                halt_thr = abs(self.config.get("risk", {}).get("max_drawdown_halt", 0.20))
                self.notifier.send_circuit_breaker_alert(drawdown, halt_thr, equity)
                return

            # 每日重置去重集合
            today = date.today()
            if self._alert_reset_date != today:
                self._alerted_today.clear()
                self._alert_reset_date = today

            # 逐股止損 / 止盈
            risk_cfg = self.config.get("risk", {})
            stop_thr = abs(risk_cfg.get("stop_loss_pct", 0.08))
            tp_thr = risk_cfg.get("take_profit_pct", 0.25)

            for sym in market_syms:
                price = latest_prices.get(sym)
                if not price:
                    continue

                # 自選股但未持倉：只做價格播報，不做止損（無建倉成本）
                if sym not in positions:
                    logger.debug(f"[自選股 {sym}] 當前價格 {price:.3f}（未持倉，跳過止損檢查）")
                    continue

                # 首次出現持倉：記錄建倉成本
                if sym not in self.risk_manager._entry_prices:
                    self.risk_manager.record_entry(sym, positions[sym].avg_cost)
                    continue

                entry = self.risk_manager._entry_prices[sym]
                if entry == 0:
                    continue
                pnl_pct = (price - entry) / entry

                # 去重 key：區分止損和止盈，避免同一天同一只股票重複刷屏
                sl_key = f"sl:{sym}"
                tp_key = f"tp:{sym}"

                if self.risk_manager.check_stop_loss(sym, price):
                    if self.repeat_alert or sl_key not in self._alerted_today:
                        self.notifier.send_stop_loss_alert(
                            symbol=sym,
                            entry_price=entry,
                            current_price=price,
                            loss_pct=pnl_pct,
                            threshold_pct=stop_thr,
                        )
                        self._alerted_today.add(sl_key)

                elif self.risk_manager.check_take_profit(sym, price):
                    if self.repeat_alert or tp_key not in self._alerted_today:
                        self.notifier.send_raw(
                            subject=f"【止盈提示】{sym} 已達目標收益",
                            body=(
                                f"股票代碼 ：{sym}\n"
                                f"建倉價格 ：{entry:.3f}\n"
                                f"當前價格 ：{price:.3f}\n"
                                f"盈利幅度 ：{pnl_pct*100:.2f}%\n"
                                f"止盈目標 ：{tp_thr*100:.2f}%\n\n"
                                f"數據來源 ：{market.upper()} {source}\n"
                                f"時間     ：{datetime.now():%H:%M:%S}\n"
                                "— Auto Trader"
                            ),
                        )
                        self._alerted_today.add(tp_key)

        except Exception as exc:
            logger.exception(f"[{market.upper()} 價格監控] 異常：{exc}")

    # ------------------------------------------------------------------
    # 日線數據（供每日信號任務使用）
    # ------------------------------------------------------------------

    def _fetch_daily_data(self):
        """拉取最新日線數據（用於信號計算，帶緩存禁用確保新鮮度）。"""
        from src.data.us_fetcher import USFetcher
        from src.data.cn_hk_fetcher import CNHKFetcher
        from src.data.futu_cnhk_fetcher import FutuCNHKFetcher
        from src.data.preprocessor import Preprocessor

        data_cfg = self.config.get("data", {})
        provider_cfg = data_cfg.get("provider", {}) or {}
        start = data_cfg.get("start_date", "2020-01-01")
        end = str(date.today())

        market_cfg = self.config.get("market", {})

        if self.market_mode in ("us", "multi"):
            fetcher = USFetcher(cache_dir=data_cfg.get("cache_dir", ".cache/data"), cache_enabled=False)
            universe = market_cfg.get("us", {}).get("universe") or fetcher.get_universe()
            prices = fetcher.get_prices(universe, start=start, end=end)

        elif self.market_mode == "cn":
            cn_provider = provider_cfg.get("cn", "futu")
            if cn_provider == "akshare":
                fetcher = CNHKFetcher(market="cn", cache_dir=data_cfg.get("cache_dir", ".cache/data"), cache_enabled=False)
            else:
                fetcher = FutuCNHKFetcher(market="cn", cache_dir=data_cfg.get("cache_dir", ".cache/data"), cache_enabled=False)
            universe = market_cfg.get("cn", {}).get("universe") or fetcher.get_universe()
            prices = fetcher.get_prices(universe, start=start, end=end)

        elif self.market_mode == "hk":
            hk_provider = provider_cfg.get("hk", "futu")
            if hk_provider == "akshare":
                fetcher = CNHKFetcher(market="hk", cache_dir=data_cfg.get("cache_dir", ".cache/data"), cache_enabled=False)
            else:
                fetcher = FutuCNHKFetcher(market="hk", cache_dir=data_cfg.get("cache_dir", ".cache/data"), cache_enabled=False)
            universe = market_cfg.get("hk", {}).get("universe") or fetcher.get_universe()
            prices = fetcher.get_prices(universe, start=start, end=end)
        else:
            return None

        return Preprocessor.clean(prices)

    def _compute_signals(self, prices) -> Dict[str, int]:
        """用配置策略計算信號，返回 {symbol: +1/-1}。"""
        from src.strategies import get_strategy
        from src.selection.screener import Screener

        screener = Screener(self.config.get("screener", {}))
        screened = screener.apply(prices)

        strategy_cls = get_strategy(self.strategy_name)
        strategy_cfg = self.config.get("strategy", {}).get(self.strategy_name, {})
        strategy = strategy_cls(config=strategy_cfg)

        filtered = prices[screened] if screened else prices
        raw = strategy.generate_signals(filtered)
        return {sym: int(v) for sym, v in raw.items()}

    def _diff_signals(self, new_signals: Dict[str, int]):
        """比對新舊信號，返回 (buy, sell, hold) 三個列表。"""
        prev_longs: Set[str] = {s for s, v in self._last_signals.items() if v == 1}
        new_longs: Set[str] = {s for s, v in new_signals.items() if v == 1}
        return sorted(new_longs - prev_longs), sorted(prev_longs - new_longs), sorted(prev_longs & new_longs)

    def _calc_drawdown(self, current_equity: float) -> float:
        peak = self.risk_manager._peak_equity
        return (current_equity - peak) / peak if peak > 0 else 0.0

    # ------------------------------------------------------------------
    # 任務 3：盤中技術信號掃描（短線 / 長線買入機會）
    # ------------------------------------------------------------------

    def _run_signal_monitor(self) -> None:
        """
        以配置的時間間隔執行技術信號掃描。
        對 watchlist + 持倉中的股票拉取日線歷史，分析短線/長線買入信號，
        發現信號後推送郵件（同一只股票同一天去重）。
        """
        from src.signal_analyzer import SignalAnalyzer

        logger.info(f"[信號掃描] 開始 — {datetime.now():%Y-%m-%d %H:%M}")

        # 構建掃描股票列表
        positions = {}
        try:
            positions = self.broker.get_positions()
        except Exception:
            pass

        target: Set[str] = set(self.watchlist)
        if self.also_monitor_positions or not self.watchlist:
            target |= set(positions.keys())

        if not target:
            logger.info("[信號掃描] 無目標股票，跳過")
            return

        symbols = sorted(target)
        logger.info(f"[信號掃描] 掃描 {len(symbols)} 只：{symbols}")

        # 決定需要的最大回溯天數
        short_days = int(self.short_signal_cfg.get("lookback_days", 60)) if self.short_signal_cfg.get("enabled", True) else 0
        long_days = int(self.long_signal_cfg.get("lookback_days", 250)) if self.long_signal_cfg.get("enabled", True) else 0
        lookback_days = max(short_days, long_days, 60)

        # 拉取日線歷史（各市場路由）
        price_data = self._fetch_short_history(symbols, lookback_days=lookback_days)
        if not price_data:
            logger.warning("[信號掃描] 未能獲取任何歷史數據，跳過")
            return

        # 分析信號
        analyzer = SignalAnalyzer(
            short_cfg=self.short_signal_cfg,
            long_cfg=self.long_signal_cfg,
        )
        fired_results = analyzer.analyze_batch(symbols, price_data)

        if not fired_results:
            logger.info("[信號掃描] 本輪無買入信號")
            return

        # 每日去重，避免重複推送同一只股票同一類型信號
        today = date.today()
        if self._alert_reset_date != today:
            self._alerted_today.clear()
            self._alert_reset_date = today

        pushed = 0
        for r in fired_results:
            alert_key = f"sig:{r.symbol}:{r.signal_type}"
            if not self.repeat_alert and alert_key in self._alerted_today:
                logger.debug(f"[信號掃描] {r.symbol} {r.signal_type} 今日已推送，跳過")
                continue

            self.notifier.send_buy_signal_alert(
                symbol=r.symbol,
                price=r.price,
                signal_type=r.signal_type,
                short_detail=r.short_detail if r.short_fired else {},
                long_detail=r.long_detail if r.long_fired else {},
            )
            self._alerted_today.add(alert_key)
            pushed += 1

        logger.info(f"[信號掃描] 完成：發現 {len(fired_results)} 個信號，推送 {pushed} 個")

    # ------------------------------------------------------------------
    # 週報任務
    # ------------------------------------------------------------------

    def _run_weekly_report_job(self) -> None:
        """每週五收盤後執行：查詢週度成交記錄，計算週 P&L，推送週報。"""
        logger.info(f"[週報] 開始生成週報 — {datetime.now():%Y-%m-%d %H:%M}")
        try:
            end_equity = self.broker.get_equity()
            start_equity = self._week_start_equity if self._week_start_equity > 0 else end_equity
            return_pct = (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0

            # 查詢週成交記錄（FutuTradeBroker 有 weekly_fill_log，PaperBroker 有 fill_log）
            fill_log: List[Dict[str, Any]] = []
            if hasattr(self.broker, "weekly_fill_log"):
                weekly = self.broker.weekly_fill_log()
                fill_log = weekly.get("fills", [])
                # 使用 Futu 返回的精確數值覆蓋（如可用）
                if weekly.get("start_equity", 0) > 0:
                    start_equity = weekly["start_equity"]
                    return_pct = (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0
            elif hasattr(self.broker, "fill_log"):
                fill_log = self.broker.fill_log()

            # 整理週成交摘要
            top_buys = list({f["symbol"] for f in fill_log if f.get("side") == "buy"})[:5]
            top_sells = list({f["symbol"] for f in fill_log if f.get("side") == "sell"})[:5]
            today = date.today()
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)

            summary = {
                "period": f"{week_start.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}",
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return_pct": return_pct,
                "total_trades": len(fill_log),
                "fills": fill_log,
                "top_buys": top_buys,
                "top_sells": top_sells,
            }

            self.notifier.send_weekly_report(summary)

            # ── 周度策略升级总结（归因 + 调参建议）────────────────────
            try:
                weekly_att = self.attributor.weekly_summary(days=5)
                optimizer = self.strategy_optimizer.propose(
                    period=summary["period"],
                    review_summary={
                        "total_trades": len(fill_log),
                        "win_rate": 0.0,
                        "signal_issues": 0,
                        "exec_issues": 0,
                        "market_issues": 0,
                    },
                    attribution_summary=weekly_att,
                    risk_dashboard={
                        "regime": self._current_regime,
                        "max_exposure": self._current_regime_scale,
                    },
                    current_config=self.config,
                )
                payload = self.report_generator.build_weekly_upgrade(
                    period=summary["period"],
                    weekly_return_pct=return_pct,
                    weekly_attribution=weekly_att,
                    optimizer_text=optimizer.to_text(),
                )
                self.notifier.send_weekly_strategy_upgrade(payload)
            except Exception as exc:
                logger.warning(f"[週報] 策略升级总结异常：{exc}")

            # 重置週初資產（用於下週計算）
            self._week_start_equity = end_equity
            logger.info(
                f"[週報] 完成：期間={summary['period']}  "
                f"週回報={return_pct*100:.2f}%  "
                f"成交={len(fill_log)} 筆"
            )
        except Exception as exc:
            logger.exception(f"[週報] 任務異常：{exc}")
            self.notifier.send_raw(
                subject="【系統錯誤】週報生成異常",
                body=f"錯誤：{exc}\n時間：{datetime.now()}",
            )

    # ------------------------------------------------------------------
    # 智能風控輔助方法
    # ------------------------------------------------------------------

    def _execute_signal_trades(
        self,
        signals: "pd.Series",
        latest_prices: Dict[str, float],
        regime: str,
        regime_scale: float,
    ) -> None:
        """根據策略信號在 Futu 模擬盤下單。"""
        try:
            target_weights = self.risk_manager.size_positions(
                signals=signals,
                prices=pd.Series(latest_prices),
                regime_scale=regime_scale,
            )
            if not target_weights:
                logger.info("[下單] 無有效信號，跳過")
                return

            orders = self.broker.rebalance(target_weights, latest_prices)
            if orders:
                order_dicts = [
                    {
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": o.qty,
                        "status": o.status,
                    }
                    for o in orders
                ]
                self.notifier.send_trade_executed(order_dicts, regime=regime)
                logger.info(f"[下單] 模擬盤下單完成：{len(orders)} 筆  市況={regime}")
            else:
                logger.info("[下單] rebalance 無需調整（倉位已達目標）")
        except Exception as exc:
            logger.error(f"[下單] 執行異常：{exc}")

    def _execute_emergency_reduce(
        self,
        latest_prices: Dict[str, float],
        reason: str,
        equity: float,
    ) -> None:
        """黑天鵝觸發：緊急縮倉至 emergency_exposure。"""
        try:
            if self.emergency_exposure <= 0:
                # 清倉：全部平倉
                logger.warning("[黑天鵝] 緊急清倉所有持倉")
                orders = self.broker.rebalance({}, latest_prices)
                action = f"已緊急清倉 {len(orders)} 個持倉"
            else:
                # 按緊急倉位縮倉
                import pandas as pd
                signals = pd.Series({sym: 1 for sym in self.broker.get_positions()})
                if not signals.empty:
                    target_weights = self.risk_manager.size_positions(
                        signals=signals,
                        prices=pd.Series(latest_prices),
                        regime_scale=self.emergency_exposure,
                    )
                    orders = self.broker.rebalance(target_weights, latest_prices)
                    action = f"已縮倉至 {self.emergency_exposure:.0%}，調整 {len(orders)} 個持倉"
                else:
                    action = "無持倉，無需操作"

            # 黑天鵝通知
            portfolio_return = 0.0
            if self.rt_metrics._return_history:
                portfolio_return = list(self.rt_metrics._return_history)[-1]

            self.notifier.send_black_swan_alert(
                return_pct=portfolio_return,
                reason=reason,
                action=action,
                equity=equity,
            )
            logger.warning(f"[黑天鵝] 緊急操作完成：{action}")
        except Exception as exc:
            logger.error(f"[黑天鵝] 緊急縮倉異常：{exc}")
            self.notifier.send_raw(
                subject="【緊急】黑天鵝縮倉操作失敗",
                body=f"原因：{reason}\n錯誤：{exc}\n時間：{datetime.now()}",
            )

    def _check_black_swan_intraday(
        self,
        metrics: Dict[str, Any],
        latest_prices: Dict[str, float],
        equity: float,
    ) -> None:
        """盤中黑天鵝檢測（由 _run_price_monitor 每輪調用）。"""
        try:
            history = list(self.rt_metrics._return_history)
            if len(history) < 2:
                return
            today_return = history[-1] if history else 0.0
            current_vol = metrics.get("volatility", 0.0)
            long_vol = self._get_long_term_vol()

            triggered, reason = self.black_swan_detector.check(
                today_return=today_return,
                return_history=history[:-1],
                current_vol=current_vol,
                long_term_vol=long_vol,
            )
            if triggered:
                logger.warning(f"[黑天鵝] 盤中觸發！{reason}")
                self._execute_emergency_reduce(latest_prices, reason, equity)
        except Exception as exc:
            logger.debug(f"[黑天鵝] 盤中檢測異常：{exc}")

    def _get_long_term_vol(self) -> float:
        """返回長期波動率估算（使用完整歷史的均值）。"""
        returns = list(self.rt_metrics._return_history)
        if len(returns) < 5:
            return 0.0
        import math
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
        return math.sqrt(var) * math.sqrt(252)

    def _fetch_benchmark_prices(self) -> "pd.Series":
        """
        拉取市場基準指數的日線收盤價（供 RegimeDetector 使用）。
        按 market.mode 自動選擇 benchmark：US=SPY，HK=^HSI，CN=000300.SH。
        """
        import pandas as pd
        data_cfg = self.config.get("data", {})
        market_cfg = self.config.get("market", {})
        start = (date.today() - timedelta(days=120)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")

        benchmark = {
            "us": market_cfg.get("us", {}).get("benchmark", "SPY"),
            "hk": market_cfg.get("hk", {}).get("benchmark", "^HSI"),
            "cn": market_cfg.get("cn", {}).get("benchmark", "000300.SH"),
        }.get(self.market_mode, "SPY")

        try:
            import yfinance as yf
            df = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                return close.dropna()
        except Exception as exc:
            logger.debug(f"[風控] 基準指數拉取失敗 ({benchmark})：{exc}")
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # 收盤复盘：交易为什么赚/亏 + 归因 + 策略升级建议
    # ------------------------------------------------------------------

    def _run_daily_review_job(
        self,
        *,
        prices,
        close_prices: "pd.DataFrame | pd.Series",
        latest_prices: Dict[str, float],
        portfolio_summary: Dict[str, Any],
        risk_dashboard: Dict[str, Any],
    ) -> None:
        """
        在每日收盘任务尾部调用：
        - 逐笔交易复盘：信号/执行/行情问题
        - 收益归因：趋势/震荡/alpha
        - 策略升级建议：输出可执行建议（不自动改策略代码）
        - 推送邮件：结论先行
        """
        date_str = str(date.today())

        # 1) 获取成交记录（优先 broker.fill_log）
        fills: List[Dict[str, Any]] = []
        if hasattr(self.broker, "fill_log"):
            try:
                fills = self.broker.fill_log()
            except Exception:
                fills = []

        # 2) 逐笔复盘：用成交价 -> 当日收盘价变化做简化 PnL
        price_changes: Dict[str, float] = {}
        for f in fills:
            sym = f.get("symbol")
            if not sym:
                continue
            try:
                entry = float(f.get("price", 0) or 0)
                cur = float(latest_prices.get(sym, 0) or 0)
                if entry > 0 and cur > 0:
                    price_changes[str(sym)] = (cur - entry) / entry
            except Exception:
                continue

        # 市场收益：用基准指数最后一日收益近似（如取不到则 0）
        market_return = 0.0
        try:
            bench = self._fetch_benchmark_prices()
            if bench is not None and len(bench) >= 2:
                market_return = float((bench.iloc[-1] - bench.iloc[-2]) / bench.iloc[-2])
        except Exception:
            pass

        review = self.trade_reviewer.review_day(
            fills=fills,
            price_changes=price_changes,
            market_return=market_return,
            day_open_prices=None,
            date=date_str,
        )
        review_summary = {
            "date": review.date,
            "total_trades": review.total_trades,
            "signal_issues": review.signal_issues,
            "exec_issues": review.exec_issues,
            "market_issues": review.market_issues,
            "clean_trades": review.clean_trades,
            "avg_pnl_pct": review.avg_pnl_pct,
            "win_rate": review.win_rate,
            "overall_verdict": review.overall_verdict,
            "top_lesson": review.top_lesson,
        }

        # 3) 收益归因：组合收益（用 RealTimeMetrics.today_return）+ 持仓收益
        rt = self.rt_metrics.get_last()
        portfolio_return = float(rt.get("today_return", 0.0) or 0.0)

        positions = {}
        try:
            positions = self.broker.get_positions()
        except Exception:
            positions = {}

        position_returns: Dict[str, float] = {}
        position_weights: Dict[str, float] = {}
        equity = float(portfolio_summary.get("equity", 0) or 0)
        if equity > 0 and positions:
            for sym, pos in positions.items():
                try:
                    avg_cost = float(getattr(pos, "avg_cost", 0) or 0)
                    cur = float(latest_prices.get(sym, 0) or 0)
                    mv = float(getattr(pos, "market_value", 0) or 0)
                    if avg_cost > 0 and cur > 0:
                        position_returns[str(sym)] = (cur - avg_cost) / avg_cost
                    if mv > 0:
                        position_weights[str(sym)] = mv / equity
                except Exception:
                    continue

        # 动量分数：用当前信号近似（若缺失则默认顺势）
        momentum_scores: Dict[str, float] = {}
        try:
            for sym, v in self._last_signals.items():
                momentum_scores[str(sym)] = 1.0 if int(v) > 0 else -1.0
        except Exception:
            pass

        att = self.attributor.decompose(
            portfolio_return=portfolio_return,
            market_return=market_return,
            position_returns=position_returns,
            position_weights=position_weights,
            momentum_scores=momentum_scores,
            date=date_str,
        )
        attribution = {
            "date": att.date,
            "total_return": att.total_return,
            "market_beta": att.market_beta,
            "trend": att.trend,
            "revert": att.revert,
            "alpha": att.alpha,
            "luck": att.luck,
            "estimated_beta": att.estimated_beta,
            "market_return": att.market_return,
            "dominant_source": att.dominant_source,
            "verdict": att.verdict,
        }

        # 4) 策略升级建议（仅建议）
        opt = self.strategy_optimizer.propose(
            period=date_str,
            review_summary=review_summary,
            attribution_summary=attribution,
            risk_dashboard=risk_dashboard,
            current_config=self.config,
        )

        # 5) 生成结论先行日报并推送
        daily_review_payload = self.report_generator.build_daily_review(
            date_str=date_str,
            portfolio_summary=portfolio_summary,
            risk_dashboard=risk_dashboard,
            review_summary=review_summary,
            attribution=attribution,
            optimizer_text=opt.to_text(),
        )
        self.notifier.send_daily_review(daily_review_payload)


    def _fetch_short_history(
        self,
        symbols: List[str],
        lookback_days: int = 250,
    ) -> Dict[str, "pd.Series"]:
        """
        按市場路由，批量拉取最近 lookback_days 根日線收盤價。
        返回 {symbol: pd.Series(close, index=date)}。
        """
        import pandas as pd
        from datetime import timedelta

        cn_syms, hk_syms, us_syms = _classify_symbols(symbols)
        result: Dict[str, "pd.Series"] = {}

        end_date = date.today()
        # 多取 50% 的日曆天，確保交易日夠用（跳過週末/節假日）
        start_date = end_date - timedelta(days=int(lookback_days * 1.6))
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # ── 美股：yfinance ──────────────────────────────────────────────
        if us_syms:
            try:
                import yfinance as yf
                raw = yf.download(
                    us_syms,
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                )
                if not raw.empty:
                    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
                    if isinstance(close, pd.DataFrame):
                        for sym in us_syms:
                            if sym in close.columns:
                                s = close[sym].dropna()
                                if not s.empty:
                                    result[sym] = s
                    else:
                        # 單只股票時返回 Series
                        sym = us_syms[0]
                        s = close.dropna()
                        if not s.empty:
                            result[sym] = s
                logger.debug(f"[信號掃描] 美股日線：{len([s for s in us_syms if s in result])}/{len(us_syms)} 只")
            except Exception as exc:
                logger.warning(f"[信號掃描] 美股日線拉取失敗：{exc}")

        # ── 港股：AKShare ───────────────────────────────────────────────
        if hk_syms:
            try:
                from src.data.futu_cnhk_fetcher import FutuCNHKFetcher
                from src.data.cn_hk_fetcher import CNHKFetcher

                provider = self.config.get("data", {}).get("provider", {}).get("hk", "futu")
                if provider == "akshare":
                    fetcher = CNHKFetcher(market="hk", cache_enabled=False)
                else:
                    fetcher = FutuCNHKFetcher(market="hk", cache_enabled=False)

                prices = fetcher.get_prices(hk_syms, start=start_str, end=end_str, interval="1d")
                if not prices.empty:
                    close = prices.xs("close", level="field", axis=1)
                    for sym in hk_syms:
                        if sym in close.columns:
                            s = close[sym].dropna()
                            if not s.empty:
                                result[sym] = s
                logger.debug(f"[信號掃描] 港股日線：{len([s for s in hk_syms if s in result])}/{len(hk_syms)} 只")
            except Exception as exc:
                logger.warning(f"[信號掃描] 港股日線拉取失敗：{exc}")

        # ── A 股：AKShare ───────────────────────────────────────────────
        if cn_syms:
            try:
                from src.data.futu_cnhk_fetcher import FutuCNHKFetcher
                from src.data.cn_hk_fetcher import CNHKFetcher

                provider = self.config.get("data", {}).get("provider", {}).get("cn", "futu")
                if provider == "akshare":
                    fetcher = CNHKFetcher(market="cn", cache_enabled=False)
                else:
                    fetcher = FutuCNHKFetcher(market="cn", cache_enabled=False)

                prices = fetcher.get_prices(cn_syms, start=start_str, end=end_str, interval="1d")
                if not prices.empty:
                    close = prices.xs("close", level="field", axis=1)
                    for sym in cn_syms:
                        if sym in close.columns:
                            s = close[sym].dropna()
                            if not s.empty:
                                result[sym] = s
                logger.debug(f"[信號掃描] A 股日線：{len([s for s in cn_syms if s in result])}/{len(cn_syms)} 只")
            except Exception as exc:
                logger.warning(f"[信號掃描] A 股日線拉取失敗：{exc}")

        return result
