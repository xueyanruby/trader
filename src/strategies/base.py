"""
Abstract base strategy and the @register_strategy decorator.

Every strategy in this system must:
1. Inherit from AbstractStrategy.
2. Implement generate_signals() and get_params().
3. Decorate the class with @register_strategy("unique_name").

This makes adding paper-sourced strategies a matter of creating a single new
file — no changes to the rest of the system required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class AbstractStrategy(ABC):
    """
    Base class for all trading strategies.

    The strategy contract is intentionally minimal so that any factor model,
    machine-learning model, or rule-based system can be plugged in.

    Signal conventions
    ------------------
    generate_signals() returns a pd.Series with:
      +1  →  BUY / long
      -1  →  SELL / short
       0  →  no change / neutral (excluded from portfolio this rebalance)
    index  → symbol strings
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Parameters
        ----------
        config : dict from config.yaml strategy section, e.g.:
                 {"lookback_months": 12, "skip_months": 1, "top_n": 20}
        """
        self.config = config or {}

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute trading signals from the most recent price snapshot.

        Parameters
        ----------
        data : MultiIndex DataFrame (symbol, field) × date — the full
               history up to the current rebalance date.

        Returns
        -------
        pd.Series  — index=symbol, values in {+1, 0, -1}
                     Only symbols with non-zero signals need to be included.
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Return a flat dict of strategy parameters for reproducibility logging.
        E.g. {"strategy": "momentum_12_1", "lookback_months": 12, "top_n": 20}
        """

    # ------------------------------------------------------------------
    # Optional hooks (override when needed)
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the first rebalance date."""

    def on_end(self) -> None:
        """Called once after the last rebalance date."""

    # ------------------------------------------------------------------
    # Convenience helpers available to subclasses
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Safe config accessor with default value."""
        return self.config.get(key, default)
