"""
Short-term mean-reversion strategy.

Theoretical basis
-----------------
De Bondt & Thaler (1985) "Does the Stock Market Overreact?"
Journal of Finance, 40(3), 793-805.

Lehmann (1990) "Fads, Martingales, and Market Efficiency"
Quarterly Journal of Economics.

Strategy logic
--------------
At each weekly rebalance:
1. Compute the z-score of each stock's close relative to its 20-day rolling mean.
2. Buy the N most oversold stocks (lowest z-score, i.e. most negative).
3. Sell positions that have mean-reverted (z-score crosses back toward zero).

This is a contrarian strategy — the opposite of momentum — and works best
at weekly or shorter frequencies.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


@register_strategy("mean_reversion")
class MeanReversionStrategy(AbstractStrategy):
    """
    Short-term mean-reversion (oversold z-score).

    Config keys
    -----------
    lookback_days     : int   rolling window for z-score    (default 20)
    z_score_threshold : float buy if z < -threshold         (default 1.5)
    top_n             : int   max number of positions       (default 20)
    rebalance_freq    : str   "W" | "D"                     (default "W")
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        lookback: int = self._cfg("lookback_days", 20)
        threshold: float = self._cfg("z_score_threshold", 1.5)
        top_n: int = self._cfg("top_n", 20)

        z_scores = Preprocessor.compute_z_score(data, window=lookback)

        if z_scores.empty:
            return pd.Series(dtype=float)

        # Oversold candidates: z-score below -threshold
        oversold = z_scores[z_scores < -threshold]

        if oversold.empty:
            return pd.Series(dtype=float)

        # Pick the most extreme oversold stocks (most negative z)
        buy_candidates = oversold.nsmallest(top_n).index
        signals = pd.Series(0, index=z_scores.index, dtype=int)
        signals[buy_candidates] = 1
        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "mean_reversion",
            "lookback_days": self._cfg("lookback_days", 20),
            "z_score_threshold": self._cfg("z_score_threshold", 1.5),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "W"),
        }
