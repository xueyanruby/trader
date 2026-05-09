"""
A 股（CN market）前置過濾器

在進入策略信號計算之前，先把「爛股票」全部排除：
1. ST / *ST / 退市 / 停牌
2. 股價：5 元 ～ 50 元
3. 流通市值：30 億 ～ 800 億 CNY
4. 換手率：2% ～ 15%
5. 近 1 個月（20 個交易日）無暴跌（單日 < -8%）且無連續跌停（連續 2 日 < -9.5%）

Usage
-----
snapshot = ak.stock_zh_a_spot_em()
screener = CNScreener(config.get("cn_screener", {}))
passing  = screener.apply_cn(prices, snapshot)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .screener import Screener


class CNScreener(Screener):
    """
    A 股專用前置過濾器，繼承通用 Screener 並疊加 CN 特有規則。

    Parameters
    ----------
    config : dict — 可覆蓋以下默認值：
        price_min               float  股價下限（元）         default 5.0
        price_max               float  股價上限（元）         default 50.0
        float_cap_min           float  流通市值下限（元）     default 30e8
        float_cap_max           float  流通市值上限（元）     default 800e8
        turnover_min            float  換手率下限（%）        default 2.0
        turnover_max            float  換手率上限（%）        default 15.0
        exclude_st              bool   排除 ST / *ST 股       default True
        crash_lookback_days     int    暴跌/跌停回望窗口      default 20
        crash_threshold         float  單日暴跌閾值           default -0.08
        limit_down_threshold    float  跌停判定閾值           default -0.095
    """

    _CN_DEFAULTS: Dict[str, Any] = {
        "price_min": 5.0,
        "price_max": 50.0,
        "float_cap_min": 30e8,
        "float_cap_max": 800e8,
        "turnover_min": 2.0,
        "turnover_max": 15.0,
        "exclude_st": True,
        "crash_lookback_days": 20,
        "crash_threshold": -0.08,
        "limit_down_threshold": -0.095,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**self._CN_DEFAULTS, **(config or {})}
        # 傳給父類的通用篩選用 CN 合理默認
        super().__init__(
            config={
                "min_price": merged["price_min"],
                "min_avg_dollar_volume": 0,   # CN 用換手率代替絕對成交額
                "min_listing_days": 60,
                "exclude_sectors": [],
            }
        )
        self.cn_config = merged

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def apply_cn(
        self,
        prices: pd.DataFrame,
        snapshot: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        執行完整的 CN 前置過濾。

        Parameters
        ----------
        prices   : MultiIndex (symbol, field) × date — 日線歷史數據
        snapshot : akshare stock_zh_a_spot_em() 返回的 DataFrame
                   需包含欄位：代码、名称、最新价、换手率、流通市值
                   為 None 時跳過快照依賴的過濾（僅做基於價格歷史的過濾）

        Returns
        -------
        list of symbol strings that pass all filters
        """
        from ..data.preprocessor import Preprocessor

        close = Preprocessor.get_close(prices)
        all_syms: List[str] = close.columns.tolist()
        logger.info(f"CNScreener: 起始股票池 {len(all_syms)} 只")

        # ── 步驟 1：基於 akshare 快照的即時過濾 ─────────────────────────
        snap_pass = all_syms
        if snapshot is not None and not snapshot.empty:
            snap_pass = self._filter_by_snapshot(all_syms, snapshot)
        else:
            logger.warning("CNScreener: 未提供快照，跳過 ST/股價/市值/換手率過濾")

        # ── 步驟 2：基於價格歷史的暴跌/跌停過濾 ────────────────────────
        clean_pass = self._filter_crash_and_limit_down(prices, snap_pass)

        logger.info(f"CNScreener: 最終通過 {len(clean_pass)} 只")
        return clean_pass

    # ------------------------------------------------------------------
    # 私有過濾方法
    # ------------------------------------------------------------------

    def _filter_by_snapshot(
        self,
        symbols: List[str],
        snapshot: pd.DataFrame,
    ) -> List[str]:
        """
        依賴 akshare 快照，過濾：
        - ST / *ST（名稱含 ST）
        - 停牌 / 退市（最新價為 0 或 NaN）
        - 股價超出 [price_min, price_max]
        - 流通市值超出 [float_cap_min, float_cap_max]
        - 換手率超出 [turnover_min, turnover_max]
        """
        cfg = self.cn_config

        # 標準化快照索引：嘗試用「代码」欄位，向後兼容「symbol」
        code_col = self._detect_col(snapshot, ["代码", "symbol", "代碼"])
        name_col = self._detect_col(snapshot, ["名称", "name", "名稱"])
        price_col = self._detect_col(snapshot, ["最新价", "latest_price", "最新價", "price"])
        cap_col = self._detect_col(snapshot, ["流通市值", "float_cap", "流通市值(元)"])
        turnover_col = self._detect_col(snapshot, ["换手率", "turnover_rate", "換手率"])

        if code_col is None:
            logger.warning("CNScreener: 快照缺少股票代码列，跳過快照過濾")
            return symbols

        snap = snapshot.copy()
        snap[code_col] = snap[code_col].astype(str).str.zfill(6)
        snap = snap.set_index(code_col)

        result = []
        st_excluded = price_excluded = cap_excluded = turnover_excluded = halted = 0

        for sym in symbols:
            code = sym.split(".")[0] if "." in sym else sym
            code = code.zfill(6)

            if code not in snap.index:
                result.append(sym)  # 快照中無此股 → 保留（避免遺漏）
                continue

            row = snap.loc[code]

            # ST / *ST 排除
            if cfg["exclude_st"] and name_col is not None:
                name = str(row.get(name_col, ""))
                if "ST" in name.upper():
                    st_excluded += 1
                    continue

            # 停牌 / 退市（最新價 = 0 / NaN）
            if price_col is not None:
                price = pd.to_numeric(row.get(price_col), errors="coerce")
                if pd.isna(price) or price == 0:
                    halted += 1
                    continue
                # 股價區間
                if price < cfg["price_min"] or price > cfg["price_max"]:
                    price_excluded += 1
                    continue

            # 流通市值
            if cap_col is not None:
                cap = pd.to_numeric(row.get(cap_col), errors="coerce")
                if pd.notna(cap):
                    if cap < cfg["float_cap_min"] or cap > cfg["float_cap_max"]:
                        cap_excluded += 1
                        continue

            # 換手率（單位：%，如 3.21 表示 3.21%）
            if turnover_col is not None:
                turnover = pd.to_numeric(row.get(turnover_col), errors="coerce")
                if pd.notna(turnover):
                    if turnover < cfg["turnover_min"] or turnover > cfg["turnover_max"]:
                        turnover_excluded += 1
                        continue

            result.append(sym)

        logger.debug(
            f"快照過濾: ST排除 {st_excluded}, 停牌/退市 {halted}, "
            f"股價排除 {price_excluded}, 市值排除 {cap_excluded}, "
            f"換手率排除 {turnover_excluded}, 通過 {len(result)}"
        )
        return result

    def _filter_crash_and_limit_down(
        self,
        prices: pd.DataFrame,
        symbols: List[str],
    ) -> List[str]:
        """
        排除近 N 個交易日內：
        - 任意單日跌幅 < crash_threshold（暴跌）
        - 連續 2 日以上跌幅 < limit_down_threshold（連續跌停）
        """
        from ..data.preprocessor import Preprocessor

        cfg = self.cn_config
        lookback = cfg["crash_lookback_days"]
        crash_thr = cfg["crash_threshold"]
        ld_thr = cfg["limit_down_threshold"]

        close = Preprocessor.get_close(prices)
        if len(close) < 2:
            return symbols

        window_close = close.iloc[-(lookback + 1):]
        daily_ret = window_close.pct_change().iloc[1:]  # shape: lookback × symbols

        result = []
        crash_excluded = ld_excluded = 0

        for sym in symbols:
            if sym not in daily_ret.columns:
                result.append(sym)
                continue

            ret = daily_ret[sym].dropna()
            if ret.empty:
                result.append(sym)
                continue

            # 單日暴跌
            if (ret < crash_thr).any():
                crash_excluded += 1
                continue

            # 連續跌停：連續 2 天 ret < ld_thr
            if self._has_consecutive_limit_down(ret, ld_thr, consecutive=2):
                ld_excluded += 1
                continue

            result.append(sym)

        logger.debug(
            f"歷史過濾: 暴跌排除 {crash_excluded}, 連續跌停排除 {ld_excluded}, "
            f"通過 {len(result)}"
        )
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _has_consecutive_limit_down(
        ret: pd.Series,
        threshold: float,
        consecutive: int = 2,
    ) -> bool:
        """判斷 ret 序列中是否存在連續 consecutive 天均低於 threshold。"""
        count = 0
        for r in ret:
            if r < threshold:
                count += 1
                if count >= consecutive:
                    return True
            else:
                count = 0
        return False

    @staticmethod
    def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """從候選欄位名列表中找到 df 實際存在的第一個，返回 None 若全部缺失。"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def summary(self) -> Dict[str, Any]:
        base = super().summary()
        base.update({"cn_config": self.cn_config})
        return base
