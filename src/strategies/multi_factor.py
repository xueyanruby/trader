"""
Multi-factor strategy combining momentum, low-volatility, and RSI quality.

Theoretical basis
-----------------
Fama & French (1993) "Common risk factors in the returns on stocks and bonds"
Journal of Financial Economics, 33(1), 3-56.

Asness, Moskowitz & Pedersen (2013)
"Value and Momentum Everywhere", Journal of Finance.

Blitz & van Vliet (2007)
"The Volatility Effect: Lower Risk without Lower Return"
Journal of Portfolio Management.

Strategy logic
--------------
1. Compute three factor scores per stock:
   - Momentum  : 12-1 month cumulative return (higher = better)
   - Low-Vol   : inverse of 60-day realised volatility (lower vol = better)
   - Quality   : RSI-14 proximity to 50 (neither overbought nor oversold)
2. Z-score each factor cross-sectionally.
3. Combine with configurable weights.
4. Go long the top_n composite-score stocks.

Adding a new factor from a paper is as easy as:
1. Compute a Series indexed by symbol.
2. Z-score it.
3. Add it to the composite with a weight in config.yaml.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


@register_strategy("multi_factor")
class MultiFactorStrategy(AbstractStrategy):
    """
    Equal or configurable-weight combination of momentum, low-vol, and quality.

    Config keys
    -----------
    factors.momentum  : float  weight for momentum factor  (default 0.4)
    factors.low_vol   : float  weight for low-vol factor   (default 0.3)
    factors.quality   : float  weight for quality factor   (default 0.3)
    top_n             : int    positions to hold            (default 25)
    rebalance_freq    : str    "M" | "W" | "Q"             (default "M")
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        top_n: int = self._cfg("top_n", 25)
        factor_weights: dict = self._cfg("factors", {
            "momentum": 0.4,
            "low_vol": 0.3,
            "quality": 0.3,
        })

        # --- Factor 1: Momentum (12-1 months) ---
        mom = Preprocessor.compute_momentum(data, lookback_days=252, skip_days=21)

        # --- Factor 2: Low volatility (inverse of 60-day vol) ---
        vol = Preprocessor.compute_volatility(data, lookback_days=60)
        low_vol = -vol  # invert: lower vol → higher score

        # --- Factor 3: Quality via RSI proximity to 50 ---
        rsi = Preprocessor.compute_rsi(data, window=14)
        quality = -(rsi - 50).abs()  # closest to 50 = highest quality score

        # Align on common symbol universe
        common = mom.index.intersection(low_vol.index).intersection(rsi.index)
        if len(common) == 0:
            return pd.Series(dtype=float)

        mom, low_vol, quality = mom[common], low_vol[common], quality[common]

        # Cross-sectional z-scores
        z_mom = _zscore(mom)
        z_vol = _zscore(low_vol)
        z_qua = _zscore(quality)

        # Weighted composite
        w_mom = factor_weights.get("momentum", 0.4)
        w_vol = factor_weights.get("low_vol", 0.3)
        w_qua = factor_weights.get("quality", 0.3)

        total_weight = w_mom + w_vol + w_qua
        composite = (
            w_mom * z_mom + w_vol * z_vol + w_qua * z_qua
        ) / total_weight

        # Select top_n
        top = composite.nlargest(top_n).index
        signals = pd.Series(0, index=composite.index, dtype=int)
        signals[top] = 1
        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "multi_factor",
            "factors": self._cfg("factors", {"momentum": 0.4, "low_vol": 0.3, "quality": 0.3}),
            "top_n": self._cfg("top_n", 25),
            "rebalance_freq": self._cfg("rebalance_freq", "M"),
        }
