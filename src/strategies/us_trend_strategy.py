"""
美股趨勢跟蹤策略（US Trend Follower）

設計理念
--------
美股不適合 A 股短線均線套路：
- 流動性充裕，不存在 A 股「主力操盤」的短期博弈邏輯
- 大盤 ETF（QQQ/SPY）和龍頭科技股有清晰的趨勢結構
- 60 日均線是機構常用的趨勢分界線，支撐/壓力意義明確

策略邏輯
--------
只在固定「優質標的池」中選股（ETF + 龍頭科技），不做全市場掃描：

入場條件（全部滿足）：
1. 收盤價 > MA60（處於上升趨勢，60 日均線是多/空分水嶺）
2. 近支撐入場：收盤價在 MA60 上方，但未偏離超過 entry_zone_pct
   即：MA60 <= 收盤價 <= MA60 × (1 + entry_zone_pct)
   意義：在均線支撐附近買入，而非追高
3. RSI(14) < rsi_max（未到超買區域，默認 70）
4. 近期無大幅回撤：過去 lookback 天最大跌幅 < max_drawdown_pct

離場條件（由風控模塊處理，策略側不輸出 -1）：
- 收盤跌破 MA60 時觸發止損

技術補充：支撐/壓力識別
- 當前最接近支撐：MA60 本身
- 第二支撐：近 20 日最低點（close 最近低點）
- 壓力位：近 20 日最高點（不在此附近入場）

Config keys（config.yaml strategy.us_trend 節）
-----------------------------------------------
watchlist           list[str]  自選股池（必填，ETF + 龍頭股）
                               默認: [QQQ, SPY, AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, AVGO]
ma_trend            int        趨勢均線週期                   default 60
entry_zone_pct      float      MA60 上方入場區間（小數）       default 0.05  (5%)
rsi_period          int        RSI 計算週期                   default 14
rsi_max             float      RSI 超買上限                   default 70
lookback_drawdown   int        近期回撤回望天數               default 10
max_drawdown_pct    float      最大允許近期回撤               default 0.12  (-12%)
top_n               int        最多持有標的數（0 = 不限）      default 5
rebalance_freq      str        "D" | "W"                     default "W"
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy

# 默認優質標的池：大盤 ETF + 龍頭科技股
_DEFAULT_WATCHLIST: List[str] = [
    # 大盤 / 行業 ETF
    "QQQ",   # 納指 100 ETF
    "SPY",   # S&P 500 ETF
    "XLK",   # 科技行業 ETF
    # 龍頭科技股
    "AAPL",  # 蘋果
    "MSFT",  # 微軟
    "NVDA",  # 英偉達
    "GOOGL", # Alphabet
    "META",  # Meta
    "AMZN",  # 亞馬遜
    "TSLA",  # 特斯拉
    "AVGO",  # 博通
    "ORCL",  # 甲骨文
]


@register_strategy("us_trend")
class USTrendStrategy(AbstractStrategy):
    """
    美股趨勢跟蹤策略。

    核心邏輯：固定優質標的池 + 60 日均線趨勢判斷 + 近支撐入場。
    不做短線均線金叉死叉，不掃全市場。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        watchlist: List[str] = self._cfg("watchlist", _DEFAULT_WATCHLIST)
        ma_period: int = self._cfg("ma_trend", 60)
        entry_zone: float = self._cfg("entry_zone_pct", 0.05)
        rsi_period: int = self._cfg("rsi_period", 14)
        rsi_max: float = self._cfg("rsi_max", 70.0)
        lookback_dd: int = self._cfg("lookback_drawdown", 10)
        max_dd: float = self._cfg("max_drawdown_pct", 0.12)
        top_n: int = self._cfg("top_n", 5)

        if not isinstance(data.columns, pd.MultiIndex):
            logger.warning("USTrend: data 不是 MultiIndex，跳過")
            return pd.Series(dtype=float)

        close = Preprocessor.get_close(data)
        required = ma_period + rsi_period + 5
        if len(close) < required:
            logger.warning(
                f"USTrend: 歷史不足（需 {required} 天，實有 {len(close)} 天）"
            )
            return pd.Series(dtype=float)

        # 只看 watchlist 中存在於 data 的標的
        available = [s for s in watchlist if s in close.columns]
        if not available:
            logger.warning(
                "USTrend: watchlist 中沒有任何標的在 data 中，"
                f"watchlist={watchlist[:5]}..."
            )
            return pd.Series(dtype=float)

        signals: Dict[str, int] = {}
        # 記錄「近支撐程度」用於排序（越接近 MA60 越好）
        proximity: Dict[str, float] = {}

        for sym in available:
            c = close[sym].dropna()
            if len(c) < required:
                continue

            price_now = float(c.iloc[-1])

            # ── 條件 1：收盤 > MA60（確認上升趨勢）────────────────────
            ma60 = float(c.iloc[-ma_period:].mean())
            if price_now <= ma60:
                continue

            # ── 條件 2：近支撐入場（不追高）────────────────────────────
            # 入場區間：MA60 到 MA60 × (1 + entry_zone_pct)
            upper_bound = ma60 * (1.0 + entry_zone)
            if price_now > upper_bound:
                # 已偏離 MA60 超過 entry_zone，暫不入場（等回調）
                continue

            # ── 條件 3：RSI < rsi_max（未超買）──────────────────────────
            rsi_val = self._compute_rsi_last(c, rsi_period)
            if rsi_val is not None and rsi_val >= rsi_max:
                continue

            # ── 條件 4：近期無大幅回撤────────────────────────────────────
            recent = c.iloc[-(lookback_dd + 1):]
            if len(recent) >= 2:
                period_high = float(recent.iloc[:-1].max())
                period_low = float(recent.iloc[-1])
                if period_high > 0:
                    drawdown = (period_low - period_high) / period_high
                    if drawdown < -max_dd:
                        continue

            signals[sym] = 1
            # proximity：越小說明越接近 MA60 支撐（更優）
            proximity[sym] = (price_now - ma60) / ma60

        if not signals:
            return pd.Series(dtype=float)

        result = pd.Series(signals, dtype=int)

        # 按「最近 MA60 支撐」排序，優先選回調到均線附近的標的
        if top_n > 0 and len(result) > top_n:
            buy_syms = result[result == 1].index.tolist()
            prox = pd.Series({s: proximity[s] for s in buy_syms})
            top_syms = prox.nsmallest(top_n).index.tolist()
            result = result[result.index.isin(top_syms)]

        logger.info(
            f"USTrend: {len(result)} 個標的滿足趨勢+支撐條件 "
            f"（watchlist {len(available)} 只，entry_zone={entry_zone:.0%}）"
        )
        return result[result != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "us_trend",
            "watchlist": self._cfg("watchlist", _DEFAULT_WATCHLIST),
            "ma_trend": self._cfg("ma_trend", 60),
            "entry_zone_pct": self._cfg("entry_zone_pct", 0.05),
            "rsi_period": self._cfg("rsi_period", 14),
            "rsi_max": self._cfg("rsi_max", 70.0),
            "lookback_drawdown": self._cfg("lookback_drawdown", 10),
            "max_drawdown_pct": self._cfg("max_drawdown_pct", 0.12),
            "top_n": self._cfg("top_n", 5),
            "rebalance_freq": self._cfg("rebalance_freq", "W"),
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi_last(close: pd.Series, period: int) -> float | None:
        """計算最新一根 K 線的 RSI(period)。"""
        if len(close) < period + 1:
            return None
        delta = close.diff().iloc[-(period + 1):]
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        last_loss = float(loss.iloc[-1])
        if last_loss == 0:
            return 100.0
        rs = float(gain.iloc[-1]) / last_loss
        return 100.0 - (100.0 / (1.0 + rs))
