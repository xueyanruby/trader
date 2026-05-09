"""
Momentum strategy — Jegadeesh & Titman (1993)
"Returns to Buying Winners and Selling Losers"
Journal of Finance, 48(1), 65-91.

Strategy logic
--------------
At each rebalance date:
1. Compute the cumulative return for each stock over the past J months,
   excluding the most recent K months (skip period to avoid short-term reversal).
   Classic parameter: J=12, K=1 ("12-1 momentum").
2. Rank all stocks by this momentum score.
3. Go long the top N stocks (equal-weighted).
4. Exit positions not in the top N.

This file is also a reference implementation showing how to translate a
paper strategy into the system's plugin interface.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


@register_strategy("momentum_12_1")
class MomentumStrategy(AbstractStrategy):
    """
    Cross-sectional price momentum.

    Config keys
    -----------
    lookback_months : int   total lookback window in months  (default 12)
    skip_months     : int   months to skip at the end        (default 1)
    top_n           : int   number of stocks to hold         (default 20)
    rebalance_freq  : str   "M" | "W" | "Q"                 (default "M")
    """

    # approximate trading-days-per-month conversion
    _DAYS_PER_MONTH = 21

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Parameters
        ----------
        data : MultiIndex price DataFrame up to the current rebalance date.

        Returns
        -------
        pd.Series  index=symbol, value=+1 for the top_n momentum stocks.
        """
        lookback_months: int = self._cfg("lookback_months", 12)
        skip_months: int = self._cfg("skip_months", 1)
        top_n: int = self._cfg("top_n", 20)

        lookback_days = lookback_months * self._DAYS_PER_MONTH
        skip_days = skip_months * self._DAYS_PER_MONTH

        momentum = Preprocessor.compute_momentum(
            data,
            lookback_days=lookback_days,
            skip_days=skip_days,
        )

        if momentum.empty:
            return pd.Series(dtype=float)

        # Rank and select top N
        top = momentum.nlargest(top_n).index
        signals = pd.Series(0, index=momentum.index, dtype=int)
        signals[top] = 1
        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "momentum_12_1",
            "lookback_months": self._cfg("lookback_months", 12),
            "skip_months": self._cfg("skip_months", 1),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "M"),
        }
