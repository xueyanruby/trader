"""
RSI 超買/超賣反轉策略
移植來源：QuantConnect Lean Algorithm.Framework/Alphas/RsiAlphaModel.py

原始邏輯（Lean）
-----------------
- 使用 Wilder's RSI（EMA 平滑法，com = period-1）
- 狀態機：TRIPPED_LOW / MIDDLE / TRIPPED_HIGH
  • RSI < 30  → TRIPPED_LOW  → 生成 UP（Buy）insight
  • RSI > 70  → TRIPPED_HIGH → 生成 DOWN（Sell）insight
  • 離開超賣/超買區時（RSI > 35 / RSI < 65）→ 回到 MIDDLE
- 狀態沒有變化時不重複發出信號

本策略在 trader 框架中的適配
------------------------------
- 批量模式：直接根據最新 RSI 值判斷當前狀態
- 也引入「前一日狀態」的去重：僅在狀態轉換時輸出信號，
  避免連續多天 RSI 在超賣區都輸出重複買入信號
- 每只股票獨立計算，不做跨股票排名
- 返回 pd.Series（index=symbols, values=+1/-1）

Config keys（config.yaml strategy.rsi_contrarian 節）
------------------------------------------------------
period         : int    RSI 週期         (默認 14)
oversold       : float  超賣閾值         (默認 30)
overbought     : float  超買閾值         (默認 70)
exit_oversold  : float  離開超賣閾值     (默認 35)
exit_overbought: float  離開超買閾值     (默認 65)
top_n          : int    最多持有數量      (默認 20，0 = 不限)
rebalance_freq : str    "D" | "W" | "M"  (默認 "D")
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict

import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


class _State(IntEnum):
    TRIPPED_LOW = 0
    MIDDLE = 1
    TRIPPED_HIGH = 2


def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI — 完全移植自 Lean RelativeStrengthIndex(MovingAverageType.WILDERS)。
    使用指數移動平均（com = period - 1），等價於 alpha = 1/period 的 SMMA。
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # com = period - 1 對應 Wilder's 平滑因子 alpha = 1/period
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - 100 / (1 + rs)


def _get_state(
    rsi_val: float,
    prev_state: _State,
    oversold: float,
    overbought: float,
    exit_oversold: float,
    exit_overbought: float,
) -> _State:
    """
    移植自 Lean RsiAlphaModel.get_state()。
    包含「bounce 容忍」邏輯：超賣/超買後需 RSI 回到 exit 閾值才轉 MIDDLE。
    """
    if rsi_val > overbought:
        return _State.TRIPPED_HIGH
    if rsi_val < oversold:
        return _State.TRIPPED_LOW
    if prev_state == _State.TRIPPED_LOW:
        if rsi_val > exit_oversold:
            return _State.MIDDLE
    if prev_state == _State.TRIPPED_HIGH:
        if rsi_val < exit_overbought:
            return _State.MIDDLE
    return prev_state


@register_strategy("rsi_contrarian")
class RSIContrarianStrategy(AbstractStrategy):
    """
    RSI 超買/超賣反轉策略（移植自 Lean RsiAlphaModel）。

    在超賣區（RSI < oversold）給出 Buy 信號，
    在超買區（RSI > overbought）給出 Sell 信號。
    使用 Wilder's 平滑 RSI（與 Lean 原版一致）。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        period: int = self._cfg("period", 14)
        oversold: float = self._cfg("oversold", 30.0)
        overbought: float = self._cfg("overbought", 70.0)
        exit_os: float = self._cfg("exit_oversold", 35.0)
        exit_ob: float = self._cfg("exit_overbought", 65.0)
        top_n: int = self._cfg("top_n", 20)

        min_bars = period * 3    # Wilder 需要足夠的 warm-up 期
        close = Preprocessor.get_close(data)

        if len(close) < min_bars:
            return pd.Series(dtype=float)

        results: Dict[str, int] = {}
        buy_scores: Dict[str, float] = {}    # 越低 RSI 越強

        for sym in close.columns:
            series = close[sym].dropna()
            if len(series) < min_bars:
                continue

            rsi = _wilder_rsi(series, period)
            if len(rsi) < 2:
                continue

            prev_rsi = float(rsi.iloc[-2])
            curr_rsi = float(rsi.iloc[-1])

            # 計算前一根和當前的狀態（移植 Lean 的狀態轉換邏輯）
            prev_state = _get_state(prev_rsi, _State.MIDDLE, oversold, overbought, exit_os, exit_ob)
            curr_state = _get_state(curr_rsi, prev_state, oversold, overbought, exit_os, exit_ob)

            # 僅在狀態發生轉換時輸出信號（移植 Lean 的 state != previous_state 判斷）
            if curr_state != prev_state:
                if curr_state == _State.TRIPPED_LOW:
                    results[sym] = 1
                    buy_scores[sym] = curr_rsi   # RSI 越低越強
                elif curr_state == _State.TRIPPED_HIGH:
                    results[sym] = -1

        if not results:
            return pd.Series(dtype=float)

        signals = pd.Series(results, dtype=int)

        # 按 RSI 越低越優先（超賣越深越優先）選 top_n 個買入
        if top_n > 0:
            buy_syms = list(signals[signals == 1].index)
            if len(buy_syms) > top_n:
                top_buy = sorted(buy_syms, key=lambda s: buy_scores.get(s, 50))[:top_n]
                keep = set(top_buy) | set(signals[signals == -1].index)
                signals = signals[signals.index.isin(keep)]

        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "rsi_contrarian",
            "period": self._cfg("period", 14),
            "oversold": self._cfg("oversold", 30.0),
            "overbought": self._cfg("overbought", 70.0),
            "exit_oversold": self._cfg("exit_oversold", 35.0),
            "exit_overbought": self._cfg("exit_overbought", 65.0),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "D"),
        }
