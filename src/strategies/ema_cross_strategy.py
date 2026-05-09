"""
EMA 金叉/死叉策略
移植來源：QuantConnect Lean Algorithm.Framework/Alphas/EmaCrossAlphaModel.py

原始邏輯（Lean）
-----------------
- 維護 fast_is_over_slow 狀態（防止同方向重複信號）
- BUY  when slow_is_over_fast AND fast crosses above slow（fast 剛上穿 slow）
- SELL when fast_is_over_slow AND slow crosses above fast（fast 剛下穿 slow）

本策略在 trader 框架中的適配
------------------------------
- 在批量模式下用「最近 cross_tolerance 根 K 線內發生穿越」代替狀態機
- 每只股票獨立計算，不做跨股票排名
- 返回 pd.Series（index=symbols, values=+1/-1）

Config keys（config.yaml strategy.ema_cross 節）
------------------------------------------------
fast_period      : int    EMA 快線週期              (默認 12)
slow_period      : int    EMA 慢線週期              (默認 26)
cross_tolerance  : int    偵測穿越的容忍視窗（K 線數）(默認 3)
top_n            : int    最多持有數量               (默認 20，0 = 不限)
rebalance_freq   : str    "D" | "W" | "M"           (默認 "D")
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _cross_direction(
    fast: pd.Series,
    slow: pd.Series,
    tolerance: int = 3,
) -> int:
    """
    偵測最近 tolerance 根 K 線內的穿越方向。

    Returns
    -------
    +1  : fast 剛從下方上穿 slow（金叉，Buy）
    -1  : fast 剛從上方下穿 slow（死叉，Sell）
     0  : 無穿越
    """
    if len(fast) < 2:
        return 0

    # 最新值必須已確定方向
    latest_fast_over = float(fast.iloc[-1]) > float(slow.iloc[-1])

    window = min(tolerance + 1, len(fast))
    f = fast.iloc[-window:]
    s = slow.iloc[-window:]

    # 在視窗內（排除最後一根）是否存在反方向的關係
    had_opposite = any(
        (float(f.iloc[i]) > float(s.iloc[i])) != latest_fast_over
        for i in range(len(f) - 1)
    )

    if not had_opposite:
        return 0

    return 1 if latest_fast_over else -1


@register_strategy("ema_cross")
class EMACrossStrategy(AbstractStrategy):
    """
    EMA 金叉 / 死叉策略（移植自 Lean EmaCrossAlphaModel）。

    每只股票獨立計算 EMA(fast)/EMA(slow) 的穿越方向，
    金叉返回 +1（Buy），死叉返回 -1（Sell）。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast: int = self._cfg("fast_period", 12)
        slow: int = self._cfg("slow_period", 26)
        tol: int = self._cfg("cross_tolerance", 3)
        top_n: int = self._cfg("top_n", 20)

        close = Preprocessor.get_close(data)
        if len(close) < slow + tol + 1:
            return pd.Series(dtype=float)

        results: Dict[str, int] = {}

        for sym in close.columns:
            series = close[sym].dropna()
            if len(series) < slow + tol + 1:
                continue

            ema_f = _ema(series, fast)
            ema_s = _ema(series, slow)
            direction = _cross_direction(ema_f, ema_s, tol)
            if direction != 0:
                results[sym] = direction

        if not results:
            return pd.Series(dtype=float)

        signals = pd.Series(results, dtype=int)

        # 對買入信號按 EMA 快線超出慢線的幅度排名
        if top_n > 0:
            buy_syms = list(signals[signals == 1].index)
            if len(buy_syms) > top_n:
                close_last = close.iloc[-1]
                ema_fast_last = close.apply(lambda s: float(_ema(s.dropna(), fast).iloc[-1]))
                ema_slow_last = close.apply(lambda s: float(_ema(s.dropna(), slow).iloc[-1]))
                spread = (ema_fast_last - ema_slow_last).abs()
                top_buy = spread[buy_syms].nlargest(top_n).index.tolist()
                keep = set(top_buy) | set(signals[signals == -1].index)
                signals = signals[signals.index.isin(keep)]

        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "ema_cross",
            "fast_period": self._cfg("fast_period", 12),
            "slow_period": self._cfg("slow_period", 26),
            "cross_tolerance": self._cfg("cross_tolerance", 3),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "D"),
        }
