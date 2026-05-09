"""
Stock screener: filters a raw universe down to a tradeable pool.

The screener applies a configurable chain of filters before signals are
generated.  This prevents the strategy from seeing illiquid, penny stocks
or newly-listed companies that would introduce survivorship or liquidity bias
in a real trading context.

Filter chain (all optional, controlled via config):
1. Market-cap filter     — exclude micro-caps below min_market_cap
2. Liquidity filter      — minimum average daily dollar volume
3. Price filter          — minimum price (avoid penny stocks)
4. Listing age filter    — minimum days since IPO
5. Sector exclusions     — skip specified sectors (e.g. financials)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger


class Screener:
    """
    Applies a configurable set of filters to reduce a universe to a
    tradeable stock pool.

    Parameters
    ----------
    config : dict of filter parameters (see apply() docstring for keys)
    """

    _DEFAULTS: Dict[str, Any] = {
        "min_market_cap": 500_000_000,       # $500 M
        "min_avg_dollar_volume": 5_000_000,  # $5 M / day
        "min_price": 5.0,                    # $5 minimum price
        "min_listing_days": 252,             # ~1 year listed
        "exclude_sectors": [],               # e.g. ["Financials"]
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self._DEFAULTS, **(config or {})}

    def apply(
        self,
        prices: pd.DataFrame,
        info: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        Run all active filters and return the list of symbols that pass.

        Parameters
        ----------
        prices : MultiIndex price DataFrame (symbol, field) × date
        info   : Optional flat DataFrame with fundamental fields:
                 market_cap, sector, avg_dollar_volume (one row per symbol)

        Returns
        -------
        list of symbol strings that pass all filters
        """
        from ..data.preprocessor import Preprocessor

        all_symbols = (
            prices.columns.get_level_values("symbol").unique().tolist()
            if isinstance(prices.columns, pd.MultiIndex)
            else prices.columns.tolist()
        )

        logger.info(f"Screener: starting with {len(all_symbols)} symbols")

        # --- Price filter ---
        min_price: float = self.config["min_price"]
        close = Preprocessor.get_close(prices)
        last_close = close.iloc[-1]
        price_pass = last_close[last_close >= min_price].index.tolist()
        logger.debug(f"Price filter (>= ${min_price}): {len(price_pass)} pass")

        # --- Listing age filter ---
        min_days: int = self.config["min_listing_days"]
        age_pass = []
        for sym in price_pass:
            if sym in close.columns:
                non_nan_days = close[sym].notna().sum()
                if non_nan_days >= min_days:
                    age_pass.append(sym)
        logger.debug(f"Listing age filter (>= {min_days} days): {len(age_pass)} pass")

        # --- Liquidity filter (dollar volume) ---
        min_dv: float = self.config["min_avg_dollar_volume"]
        liquid_pass = self._filter_dollar_volume(prices, age_pass, min_dv)
        logger.debug(f"Liquidity filter (avg DV >= ${min_dv:,.0f}): {len(liquid_pass)} pass")

        # --- Fundamental filters (requires info DataFrame) ---
        result = liquid_pass
        if info is not None and not info.empty:
            result = self._filter_fundamentals(info, result)

        logger.info(f"Screener: {len(result)} symbols pass all filters")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_dollar_volume(
        prices: pd.DataFrame,
        symbols: List[str],
        min_dv: float,
        window: int = 20,
    ) -> List[str]:
        """Keep symbols with avg daily dollar-volume >= min_dv."""
        from ..data.preprocessor import Preprocessor

        if not isinstance(prices.columns, pd.MultiIndex):
            return symbols

        close = Preprocessor.get_close(prices)
        volume = Preprocessor.get_volume(prices)

        passing = []
        for sym in symbols:
            if sym not in close.columns or sym not in volume.columns:
                continue
            dv = (close[sym] * volume[sym]).iloc[-window:].mean()
            if pd.notna(dv) and dv >= min_dv:
                passing.append(sym)
        return passing

    def _filter_fundamentals(
        self, info: pd.DataFrame, symbols: List[str]
    ) -> List[str]:
        """Apply market-cap and sector filters using the info DataFrame."""
        min_cap: float = self.config["min_market_cap"]
        excl_sectors: List[str] = self.config["exclude_sectors"]

        result = []
        for sym in symbols:
            if sym not in info.index:
                result.append(sym)  # no info available → pass through
                continue

            row = info.loc[sym]

            # Market cap
            cap = row.get("market_cap") or row.get("marketCap")
            if pd.notna(cap) and float(cap) < min_cap:
                continue

            # Sector exclusion
            sector = row.get("sector")
            if sector and excl_sectors and sector in excl_sectors:
                continue

            result.append(sym)

        logger.debug(
            f"Fundamental filters (cap >= ${min_cap/1e6:.0f}M, "
            f"exclude {excl_sectors}): {len(result)} pass"
        )
        return result

    def summary(self) -> Dict[str, Any]:
        """Return current filter configuration."""
        return dict(self.config)
