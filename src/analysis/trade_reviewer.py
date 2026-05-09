"""
逐筆交易復盤分析器（Trade Reviewer）

對每筆成交記錄進行「三問診斷」：
  1. 信號問題 (signal_issue)  : 策略方向判斷錯誤（買了跌 / 賣了漲）
  2. 執行問題 (exec_issue)    : 信號正確但成交價偏離理論最優值太多
  3. 行情問題 (market_issue)  : 市場系統性反轉，即使信號和執行都對，也難逃虧損

同時輸出每筆交易的「學習點」（lesson），供 ReportGenerator 直接插入日報。

設計原則
--------
- 僅使用已有數據（fill_log + 日線 close），不依賴 tick 或訂單簿
- 規則簡單可審計，避免「黑箱復盤」
- 允許部分數據缺失（graceful degradation）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 數據結構
# ---------------------------------------------------------------------------

@dataclass
class TradeAnalysis:
    """單筆交易復盤結果。"""
    # 基本信息
    symbol: str
    side: str               # "buy" | "sell"
    qty: float
    fill_price: float
    fill_date: str          # YYYY-MM-DD

    # PnL 結果（正 = 盈，負 = 虧）
    pnl_pct: float          # 相對成交價的收益率（持倉期間）

    # 三問診斷（True = 該維度存在問題）
    signal_issue: bool = False
    exec_issue: bool = False
    market_issue: bool = False

    # 各維度貢獻
    signal_contribution: float = 0.0    # 信號方向貢獻的 pnl
    exec_contribution: float = 0.0      # 執行偏差的 pnl 損耗
    market_contribution: float = 0.0    # 市場 beta 貢獻的 pnl

    # 主要問題標籤（供報告用）
    primary_issue: str = "正常"         # "信號問題" | "執行問題" | "行情問題" | "正常"

    # 可讀結論
    verdict: str = ""
    lesson: str = ""        # 給策略的改進提示

    # 輔助字段
    market_return: float = 0.0          # 同期大盤收益率
    entry_vs_open: float = 0.0          # 成交價相對開盤價的偏差（執行滑點估算）


@dataclass
class ReviewSummary:
    """一日所有交易的復盤彙總。"""
    date: str
    total_trades: int
    trades: List[TradeAnalysis] = field(default_factory=list)

    # 問題分類計數
    signal_issues: int = 0
    exec_issues: int = 0
    market_issues: int = 0
    clean_trades: int = 0

    # 整體指標
    avg_pnl_pct: float = 0.0
    win_count: int = 0
    lose_count: int = 0
    win_rate: float = 0.0

    # 彙總結論
    top_lesson: str = ""
    overall_verdict: str = ""


# ---------------------------------------------------------------------------
# 主分析器
# ---------------------------------------------------------------------------

class TradeReviewer:
    """
    對當日 fill_log 進行逐筆復盤，輸出 ReviewSummary。

    Parameters
    ----------
    exec_slippage_threshold : 認定「執行問題」的滑點閾值（相對成交價），默認 0.5%
    signal_threshold        : 認定「信號問題」的收益率邊界，默認 0（跌即是問題）
    market_beta_threshold   : 認定「行情問題」的大盤跌幅，默認 -1%
    """

    def __init__(
        self,
        exec_slippage_threshold: float = 0.005,
        signal_threshold: float = 0.0,
        market_beta_threshold: float = -0.01,
    ) -> None:
        self.exec_threshold = exec_slippage_threshold
        self.signal_threshold = signal_threshold
        self.market_beta_threshold = market_beta_threshold

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def review_day(
        self,
        fills: List[Dict[str, Any]],
        price_changes: Dict[str, float],   # {symbol: pnl_pct since fill}
        market_return: float = 0.0,        # 今日大盤收益率（如 SPY / 恒指）
        day_open_prices: Optional[Dict[str, float]] = None,  # 今日開盤價
        date: str = "",
    ) -> ReviewSummary:
        """
        Parameters
        ----------
        fills          : broker.fill_log() 或 weekly_fill_log()['fills'] 中的列表
        price_changes  : {symbol: 成交後至今的價格變化率}（正 = 漲，負 = 跌）
        market_return  : 大盤同期收益率
        day_open_prices: 各股今日開盤價（用於估算執行滑點）
        date           : YYYY-MM-DD（用於標記）

        Returns
        -------
        ReviewSummary
        """
        analyses: List[TradeAnalysis] = []

        for fill in fills:
            ta = self._analyze_fill(
                fill=fill,
                price_changes=price_changes,
                market_return=market_return,
                day_open_prices=day_open_prices or {},
            )
            analyses.append(ta)

        return self._summarize(date, analyses, market_return)

    def review_positions(
        self,
        positions: Dict[str, Any],         # broker.get_positions()
        current_prices: Dict[str, float],
        market_return: float = 0.0,
        date: str = "",
    ) -> ReviewSummary:
        """
        復盤「當前持倉」的盈虧狀況（不依賴 fill_log，僅需持倉快照）。
        適用於當天沒有新交易但需要評估持倉質量的場景。
        """
        fills = []
        price_changes = {}
        for sym, pos in positions.items():
            fills.append({
                "symbol": sym,
                "side": "buy",
                "qty": pos.qty if hasattr(pos, "qty") else pos.get("qty", 0),
                "price": pos.avg_cost if hasattr(pos, "avg_cost") else pos.get("avg_cost", 0),
                "create_time": date,
            })
            cost = pos.avg_cost if hasattr(pos, "avg_cost") else pos.get("avg_cost", 1)
            cur = current_prices.get(sym, cost)
            price_changes[sym] = (cur - cost) / cost if cost > 0 else 0.0

        return self.review_day(
            fills=fills,
            price_changes=price_changes,
            market_return=market_return,
            date=date,
        )

    # ------------------------------------------------------------------
    # 私有：單筆分析
    # ------------------------------------------------------------------

    def _analyze_fill(
        self,
        fill: Dict[str, Any],
        price_changes: Dict[str, float],
        market_return: float,
        day_open_prices: Dict[str, float],
    ) -> TradeAnalysis:
        symbol = fill.get("symbol", "?")
        side = str(fill.get("side", "buy")).lower()
        qty = float(fill.get("qty", 0))
        fill_price = float(fill.get("price", fill.get("fill_price", 0)))
        fill_date = str(fill.get("create_time", fill.get("fill_date", "")))[:10]

        # 持倉期收益率（買入持有，賣出則 flip sign）
        raw_change = price_changes.get(symbol, 0.0)
        pnl_pct = raw_change if side == "buy" else -raw_change

        # ── 執行滑點估算 ──────────────────────────────────────────────
        open_price = day_open_prices.get(symbol, fill_price)
        if open_price > 0 and fill_price > 0:
            exec_slip = (fill_price - open_price) / open_price
        else:
            exec_slip = 0.0
        # 買入時滑點為正（比開盤貴）是損耗，賣出時滑點為負是損耗
        exec_loss = exec_slip if side == "buy" else -exec_slip
        entry_vs_open = exec_slip

        # ── 市場 beta 貢獻（簡化：假設 beta ≈ 1，用大盤收益作為市場貢獻）───
        market_contribution = market_return if side == "buy" else -market_return

        # ── 信號貢獻（去除市場和執行部分的剩餘） ──────────────────────
        signal_contribution = pnl_pct - market_contribution - (-exec_loss)

        # ── 問題診斷 ──────────────────────────────────────────────────
        signal_issue = pnl_pct < self.signal_threshold
        exec_issue = abs(exec_loss) > self.exec_threshold
        market_issue = market_return < self.market_beta_threshold

        # ── 主要問題（優先級：行情 > 信號 > 執行 > 正常）────────────
        if signal_issue and market_issue:
            # 信號和行情雙輸，主要問題是行情
            primary_issue = "行情問題"
        elif signal_issue:
            primary_issue = "信號問題"
        elif exec_issue and pnl_pct < 0:
            primary_issue = "執行問題"
        elif market_issue:
            primary_issue = "行情問題"
        else:
            primary_issue = "正常"

        # ── 可讀結論與學習點 ─────────────────────────────────────────
        verdict, lesson = self._make_verdict(
            symbol=symbol, side=side, pnl_pct=pnl_pct,
            signal_issue=signal_issue, exec_issue=exec_issue,
            market_issue=market_issue, market_return=market_return,
            exec_loss=exec_loss, fill_price=fill_price,
        )

        return TradeAnalysis(
            symbol=symbol, side=side, qty=qty,
            fill_price=fill_price, fill_date=fill_date,
            pnl_pct=pnl_pct,
            signal_issue=signal_issue,
            exec_issue=exec_issue,
            market_issue=market_issue,
            signal_contribution=signal_contribution,
            exec_contribution=-exec_loss,
            market_contribution=market_contribution,
            primary_issue=primary_issue,
            verdict=verdict,
            lesson=lesson,
            market_return=market_return,
            entry_vs_open=entry_vs_open,
        )

    @staticmethod
    def _make_verdict(
        symbol: str, side: str, pnl_pct: float,
        signal_issue: bool, exec_issue: bool, market_issue: bool,
        market_return: float, exec_loss: float, fill_price: float,
    ) -> Tuple[str, str]:
        direction = "買入" if side == "buy" else "賣出"
        pnl_str = f"{pnl_pct*100:+.2f}%"
        market_str = f"{market_return*100:+.2f}%"

        if not signal_issue and not exec_issue:
            verdict = f"✅ {direction} {symbol} 正確｜收益 {pnl_str}，大盤 {market_str}"
            lesson = ""
        elif market_issue and not signal_issue:
            verdict = f"🌧 {direction} {symbol} 大盤拖累｜收益 {pnl_str}，大盤 {market_str}"
            lesson = f"{symbol}：信號方向正確但大盤走弱。建議在熊市市況下降低此類標的的權重。"
        elif signal_issue and market_issue:
            verdict = f"❌ {direction} {symbol} 信號+行情雙殺｜收益 {pnl_str}，大盤 {market_str}"
            lesson = f"{symbol}：信號判斷錯誤，且大盤加速虧損。需複盤信號觸發條件是否符合當前市況。"
        elif signal_issue:
            verdict = f"⚠️ {direction} {symbol} 信號錯誤｜收益 {pnl_str}（大盤 {market_str} 但個股反向）"
            lesson = f"{symbol}：大盤表現尚可但個股反向。可能存在基本面/消息面因素未被模型捕捉，建議加入事件過濾。"
        elif exec_issue:
            verdict = f"🔧 {direction} {symbol} 執行偏差｜成交價偏離開盤 {exec_loss*100:.2f}%"
            lesson = f"{symbol}：信號方向正確，但成交價偏貴/偏低影響收益。考慮調整限價單策略或縮短下單延遲。"
        else:
            verdict = f"✅ {direction} {symbol} 正常｜收益 {pnl_str}"
            lesson = ""

        return verdict, lesson

    # ------------------------------------------------------------------
    # 私有：彙總
    # ------------------------------------------------------------------

    def _summarize(
        self,
        date: str,
        analyses: List[TradeAnalysis],
        market_return: float,
    ) -> ReviewSummary:
        if not analyses:
            return ReviewSummary(
                date=date, total_trades=0,
                overall_verdict="無成交記錄，無需復盤。",
            )

        signal_n = sum(1 for a in analyses if a.signal_issue)
        exec_n = sum(1 for a in analyses if a.exec_issue)
        market_n = sum(1 for a in analyses if a.market_issue)
        clean_n = sum(1 for a in analyses if a.primary_issue == "正常")

        pnl_list = [a.pnl_pct for a in analyses]
        avg_pnl = sum(pnl_list) / len(pnl_list)
        win_n = sum(1 for p in pnl_list if p > 0)
        lose_n = sum(1 for p in pnl_list if p <= 0)
        win_rate = win_n / len(pnl_list)

        # 主要學習點（取三問中最多的問題）
        issue_counts = {
            "信號問題": signal_n,
            "執行問題": exec_n,
            "行情問題": market_n,
        }
        dominant_issue = max(issue_counts, key=lambda k: issue_counts[k])
        lessons = [a.lesson for a in analyses if a.lesson]

        if not any(issue_counts.values()):
            overall = f"今日 {len(analyses)} 筆交易全部正常，策略執行良好。"
        elif dominant_issue == "行情問題":
            overall = (
                f"今日大盤下跌 {market_return*100:.2f}%，行情系統性拖累 {market_n} 筆交易。"
                f"策略本身信號無誤，建議在熊市模式下縮減倉位。"
            )
        elif dominant_issue == "信號問題":
            overall = (
                f"今日 {signal_n}/{len(analyses)} 筆出現信號判斷偏差，"
                f"勝率 {win_rate:.0%}（低於預期）。"
                "建議複盤近期信號 IC 是否衰退。"
            )
        else:
            overall = (
                f"今日 {exec_n} 筆存在執行偏差，"
                f"成交價偏離理論最優值。建議優化下單時序與限價設定。"
            )

        top_lesson = lessons[0] if lessons else "繼續保持當前策略紀律。"

        return ReviewSummary(
            date=date,
            total_trades=len(analyses),
            trades=analyses,
            signal_issues=signal_n,
            exec_issues=exec_n,
            market_issues=market_n,
            clean_trades=clean_n,
            avg_pnl_pct=avg_pnl,
            win_count=win_n,
            lose_count=lose_n,
            win_rate=win_rate,
            top_lesson=top_lesson,
            overall_verdict=overall,
        )
