"""
US market data fetcher backed by yfinance.

Provides OHLCV prices and basic fundamental info for NYSE/NASDAQ stocks.
The default universe is the S&P 500 constituent list (fetched from Wikipedia).
"""

from __future__ import annotations

import io
from typing import List

import pandas as pd
import requests
import yfinance as yf
from loguru import logger

from .base_fetcher import BaseFetcher


# A small hard-coded list used as fallback when Wikipedia is unreachable.
_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "UNH", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "KO", "PEP", "AVGO", "COST", "TMO", "MCD", "ACN",
    "BAC", "WMT", "CSCO",
]


class USFetcher(BaseFetcher):
    """Fetch US equity data via yfinance."""

    def get_universe(self) -> List[str]:
        """
        Return S&P 500 tickers scraped from Wikipedia.
        Falls back to a short hard-coded list on network errors.
        """
        try:
            tickers = self._fetch_sp500_universe()
            logger.info(f"Fetched S&P 500 universe: {len(tickers)} tickers")
            return tickers
        except Exception as exc:
            logger.warning(f"S&P 500 universe fetch failed ({exc}); using fallback")
            return _SP500_FALLBACK

    def _fetch_sp500_universe(self) -> List[str]:
        """
        Fetch S&P 500 constituents.

        We prefer Wikipedia for freshness, but it may 403 without a browser-like
        User-Agent. If it fails, fall back to a stable CSV source.
        """
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }

        try:
            resp = requests.get(wiki_url, headers=headers, timeout=10)
            resp.raise_for_status()
            table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
            tickers = (
                table["Symbol"]
                .astype(str)
                .str.strip()
                # Wikipedia uses dots in BRK.B; yfinance wants BRK-B
                .str.replace(".", "-", regex=False)
                .tolist()
            )
            if tickers:
                return tickers
        except Exception as exc:
            logger.debug(f"Wikipedia S&P 500 fetch attempt failed: {exc}")

        # Secondary source: DataHub mirror of S&P 500 constituents.
        # Format: https://datahub.io/core/s-and-p-500-companies
        csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        resp = requests.get(csv_url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))

        symbol_col = "Symbol" if "Symbol" in df.columns else None
        if not symbol_col:
            raise RuntimeError("Unexpected constituents.csv format: missing Symbol column")

        tickers = (
            df[symbol_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        return [t for t in tickers if t]

    def _fetch_prices(
        self, symbols: List[str], start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download OHLCV via yfinance.download() and reshape into a MultiIndex
        DataFrame with columns (symbol, field).
        """
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            logger.warning("yfinance returned empty DataFrame")
            return pd.DataFrame()

        # yfinance returns MultiIndex columns (field, ticker) when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            # Swap levels so outer level is ticker
            df = raw.swaplevel(axis=1).sort_index(axis=1)
        else:
            # Single ticker — add outer level
            ticker = symbols[0] if symbols else "UNKNOWN"
            df = pd.concat({ticker: raw}, axis=1)

        df.columns.names = ["symbol", "field"]
        df.columns = pd.MultiIndex.from_tuples(
            [(sym, field.lower()) for sym, field in df.columns],
            names=["symbol", "field"],
        )
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # Drop symbols with all-NaN close prices
        valid = [
            sym for sym in df.columns.get_level_values("symbol").unique()
            if not df[sym]["close"].isna().all()
        ]
        df = df[valid]
        logger.info(f"Prices fetched: {len(valid)}/{len(symbols)} symbols, "
                    f"{len(df)} rows")
        return df

    def _fetch_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental snapshot (market cap, sector, P/E, etc.) for each
        symbol via yfinance Ticker.info.  Returns one row per symbol.
        """
        records = []
        fields = [
            "marketCap", "sector", "industry",
            "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        for sym in symbols:
            try:
                info = yf.Ticker(sym).info
                row = {"symbol": sym}
                for f in fields:
                    row[f] = info.get(f)
                records.append(row)
            except Exception as exc:
                logger.debug(f"Info fetch failed for {sym}: {exc}")
                records.append({"symbol": sym})

        df = pd.DataFrame(records).set_index("symbol")
        # Normalise column names to snake_case
        df.columns = [
            c[0].lower() + "".join(
                f"_{ch.lower()}" if ch.isupper() else ch for ch in c[1:]
            )
            for c in df.columns
        ]
        return df
