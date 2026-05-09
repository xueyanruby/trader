"""
MACD 趨勢追蹤策略
移植來源：QuantConnect Lean Algorithm.Framework/Alphas/MacdAlphaModel.py

原始邏輯（Lean）
-----------------
- 計算 MACD signal line（EMA(fast) - EMA(slow) 的 EMA(signal_period)）
- normalized_signal = signal_line / price
- BUY  when normalized_signal >  bounce_threshold（默認 1%）
- SELL when normalized_signal < -bounce_threshold
- 方向不變時不重複發出信號

本策略在 trader 框架中的適配
------------------------------
- 接受 MultiIndex DataFrame（symbol, field），提取 close 價格
- 對每只股票獨立計算 MACD 指標
- 返回 pd.Series（index=symbols, values=+1/-1）的批量信號
- 不做跨股票排名，每只股票方向獨立判斷

Config keys（config.yaml strategy.macd 節）
-------------------------------------------
fast_period        : int    EMA 快線週期        (默認 12)
slow_period        : int    EMA 慢線週期        (默認 26)
signal_period      : int    Signal line 週期    (默認 9)
bounce_threshold   : float  normalized 觸發閾值 (默認 0.01 = 1%)
top_n              : int    最多持有數量         (默認 20，0 = 不限)
rebalance_freq     : str    "D" | "W" | "M"     (默認 "D")
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _macd_series(close: pd.Series, fast: int, slow: int, signal: int):
    """
    返回 (macd_line, signal_line, histogram) 三條序列。
    完全移植 Lean 的 MovingAverageConvergenceDivergence 邏輯：
      macd_line   = EMA(fast) - EMA(slow)
      signal_line = EMA(macd_line, signal_period)
      histogram   = macd_line - signal_line
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@register_strategy("macd")
class MACDStrategy(AbstractStrategy):
    """
    MACD Signal Line 策略（移植自 Lean MacdAlphaModel）。

    每只股票獨立計算 MACD，根據 signal line 的正負（相對於股價
    的歸一化值）給出買入（+1）或賣出（-1）信號。

    Lean 原版使用「方向不變不重複」的去重邏輯；在批量回測模式下，
    我們直接返回最新時間截點的方向，適合每日 / 每週重新計算。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast: int = self._cfg("fast_period", 12)
        slow: int = self._cfg("slow_period", 26)
        signal_p: int = self._cfg("signal_period", 9)
        threshold: float = self._cfg("bounce_threshold", 0.01)
        top_n: int = self._cfg("top_n", 20)

        # 最少需要 slow + signal_period 根數據
        min_bars = slow + signal_p
        close = Preprocessor.get_close(data)

        if len(close) < min_bars:
            return pd.Series(dtype=float)

        results: Dict[str, int] = {}
        buy_scores: Dict[str, float] = {}

        for sym in close.columns:
            series = close[sym].dropna()
            if len(series) < min_bars:
                continue

            price = float(series.iloc[-1])
            if price == 0:
                continue

            _, signal_line, _ = _macd_series(series, fast, slow, signal_p)

            normalized = float(signal_line.iloc[-1]) / price

            if normalized > threshold:
                results[sym] = 1
                buy_scores[sym] = normalized
            elif normalized < -threshold:
                results[sym] = -1

        if not results:
            return pd.Series(dtype=float)

        signals = pd.Series(results, dtype=int)

        # 對買入信號按 MACD 強度排名，最多取 top_n
        if top_n > 0:
            buy_syms = [s for s in signals[signals == 1].index]
            if len(buy_syms) > top_n:
                top_buy = sorted(buy_syms, key=lambda s: buy_scores.get(s, 0), reverse=True)[:top_n]
                keep = set(top_buy) | set(signals[signals == -1].index)
                signals = signals[signals.index.isin(keep)]

        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "macd",
            "fast_period": self._cfg("fast_period", 12),
            "slow_period": self._cfg("slow_period", 26),
            "signal_period": self._cfg("signal_period", 9),
            "bounce_threshold": self._cfg("bounce_threshold", 0.01),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "D"),
        }
