"""
Abstract base class for all market data fetchers.

Every concrete fetcher (US, HK, CN) must implement this interface so the rest
of the system never has to care which market it is talking to.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


class BaseFetcher(ABC):
    """
    Unified interface for fetching OHLCV price data and basic fundamental
    snapshots from any supported market.

    Subclasses implement ``_fetch_prices`` and ``_fetch_info``.
    This base class adds transparent disk caching on top.
    """

    def __init__(self, cache_dir: str = ".cache/data", cache_enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return a MultiIndex DataFrame: columns=(symbol, field),
        index=DatetimeIndex.  Fields: open, high, low, close, volume.

        Parameters
        ----------
        symbols : list of ticker strings
        start   : "YYYY-MM-DD"
        end     : "YYYY-MM-DD"
        interval: "1d" | "1wk" | "1mo"
        """
        cache_key = self._cache_key("prices", symbols, start, end, interval)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for prices ({len(symbols)} symbols)")
            return cached

        logger.info(f"Fetching prices for {len(symbols)} symbols [{start} → {end}]")
        df = self._fetch_prices(symbols, start, end, interval)
        self._save_cache(cache_key, df)
        return df

    def get_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Return a DataFrame with one row per symbol containing basic
        fundamental / meta fields: market_cap, sector, industry, pe_ratio, etc.
        Available fields depend on the concrete fetcher.
        """
        cache_key = self._cache_key("info", symbols)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for info ({len(symbols)} symbols)")
            return cached

        logger.info(f"Fetching fundamental info for {len(symbols)} symbols")
        df = self._fetch_info(symbols)
        self._save_cache(cache_key, df)
        return df

    # ------------------------------------------------------------------
    # Abstract interface — implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _fetch_prices(
        self, symbols: List[str], start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Fetch raw OHLCV data. Must return MultiIndex DataFrame."""

    @abstractmethod
    def _fetch_info(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch fundamental / meta snapshot. Must return flat DataFrame."""

    @abstractmethod
    def get_universe(self) -> List[str]:
        """Return default tradeable universe for this market."""

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, prefix: str, *args) -> str:
        fingerprint = str(args)
        h = hashlib.md5(fingerprint.encode()).hexdigest()[:10]
        return f"{prefix}_{h}"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                logger.warning(f"Cache read failed ({path}): {exc}")
        return None

    def _save_cache(self, key: str, data: pd.DataFrame) -> None:
        if not self.cache_enabled:
            return
        path = self._cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as exc:
            logger.warning(f"Cache write failed ({path}): {exc}")
