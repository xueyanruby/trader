"""
Data preprocessor: cleans raw OHLCV data and engineers features used by
strategies (returns, moving averages, momentum scores, volatility, etc.).

All methods accept and return the same MultiIndex DataFrame schema produced by
the fetchers: columns=(symbol, field), index=DatetimeIndex.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class Preprocessor:
    """
    Stateless collection of data-cleaning and feature-engineering helpers.
    All methods are class methods so you can use them without instantiation,
    or create an instance when you want to chain calls.
    """

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean(prices: pd.DataFrame, min_history_days: int = 252) -> pd.DataFrame:
        """
        Drop symbols with insufficient history or excessive missing data,
        then forward-fill remaining NaNs (up to 5 consecutive days).

        Parameters
        ----------
        prices            : MultiIndex price DataFrame (symbol, field)
        min_history_days  : minimum trading days required per symbol
        """
        close = Preprocessor.get_close(prices)

        # Drop symbols where we have fewer than min_history_days non-NaN closes
        valid_symbols = [
            sym for sym in close.columns
            if close[sym].notna().sum() >= min_history_days
        ]
        if len(valid_symbols) < len(close.columns):
            dropped = len(close.columns) - len(valid_symbols)
            logger.debug(f"Dropped {dropped} symbols with insufficient history")

        prices = prices[valid_symbols]

        # Forward-fill up to 5 days (handles market holidays between markets)
        prices = prices.ffill(limit=5)
        return prices

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @staticmethod
    def get_close(prices: pd.DataFrame) -> pd.DataFrame:
        """Extract close prices: returns a flat DataFrame (date × symbol)."""
        if isinstance(prices.columns, pd.MultiIndex):
            return prices.xs("close", axis=1, level="field")
        return prices

    @staticmethod
    def get_volume(prices: pd.DataFrame) -> pd.DataFrame:
        if isinstance(prices.columns, pd.MultiIndex):
            return prices.xs("volume", axis=1, level="field")
        raise ValueError("prices must be a MultiIndex DataFrame")

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    @staticmethod
    def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Compute simple percentage returns over ``periods`` trading days.
        Returns a flat DataFrame (date × symbol).
        """
        close = Preprocessor.get_close(prices)
        return close.pct_change(periods)

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """Log returns: ln(P_t / P_{t-periods})."""
        close = Preprocessor.get_close(prices)
        return np.log(close / close.shift(periods))

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    @staticmethod
    def compute_momentum(
        prices: pd.DataFrame,
        lookback_days: int = 252,
        skip_days: int = 21,
    ) -> pd.Series:
        """
        Cross-sectional momentum score for the last available date.

        Classic "12-1" momentum: cumulative return over lookback_days,
        excluding the most recent skip_days (short-term reversal avoidance).

        Returns a Series indexed by symbol, values = momentum score.
        """
        close = Preprocessor.get_close(prices)
        if len(close) < lookback_days + skip_days:
            logger.warning("Not enough data for momentum calculation")
            return pd.Series(dtype=float)

        past_price = close.iloc[-(lookback_days + skip_days)]
        recent_price = close.iloc[-skip_days]
        momentum = (recent_price / past_price) - 1
        return momentum.dropna()

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_volatility(
        prices: pd.DataFrame,
        lookback_days: int = 60,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Rolling historical volatility (std of daily log returns) for the
        most recent lookback_days.  Annualised by sqrt(252) if requested.
        Returns a Series indexed by symbol.
        """
        log_ret = Preprocessor.compute_log_returns(prices)
        vol = log_ret.iloc[-lookback_days:].std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol.dropna()

    # ------------------------------------------------------------------
    # Technical indicators (used by mean-reversion strategy)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_z_score(prices: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Z-score of the most recent close vs rolling mean/std over ``window``
        trading days.  Negative z-score = oversold (mean-reversion BUY signal).
        Returns a Series indexed by symbol.
        """
        close = Preprocessor.get_close(prices)
        recent = close.iloc[-window:]
        z = (close.iloc[-1] - recent.mean()) / recent.std()
        return z.dropna()

    @staticmethod
    def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Relative Strength Index for each symbol over the given window.
        Returns a Series indexed by symbol (value 0–100).
        """
        close = Preprocessor.get_close(prices)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1].dropna()

    # ------------------------------------------------------------------
    # Multi-factor features
    # ------------------------------------------------------------------

    @staticmethod
    def compute_all_features(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a feature matrix with one row per symbol, containing:
        momentum_12_1, volatility_60d, rsi_14, z_score_20d.

        Useful for multi-factor strategies.
        """
        features = pd.DataFrame(index=Preprocessor.get_close(prices).columns)
        features["momentum_12_1"] = Preprocessor.compute_momentum(prices)
        features["volatility_60d"] = Preprocessor.compute_volatility(prices)
        features["rsi_14"] = Preprocessor.compute_rsi(prices)
        features["z_score_20d"] = Preprocessor.compute_z_score(prices)
        return features.dropna()

    # ------------------------------------------------------------------
    # Liquidity filter
    # ------------------------------------------------------------------

    @staticmethod
    def filter_liquid(
        prices: pd.DataFrame,
        min_avg_volume: float = 500_000,
        window_days: int = 20,
    ) -> List[str]:
        """
        Return symbols whose average daily volume over the last window_days
        exceeds min_avg_volume.  Useful as a pre-filter before strategy signals.
        """
        volume = Preprocessor.get_volume(prices)
        avg_vol = volume.iloc[-window_days:].mean()
        liquid = avg_vol[avg_vol >= min_avg_volume].index.tolist()
        logger.debug(f"Liquidity filter: {len(liquid)}/{len(avg_vol)} symbols pass")
        return liquid
