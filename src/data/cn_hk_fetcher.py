"""
HK and China A-share data fetcher backed by akshare.

HK stocks: uses akshare's stock_hk_daily interface.
A-shares : uses akshare's stock_zh_a_hist interface.

Both markets return data in the same MultiIndex DataFrame schema as USFetcher
so the rest of the system is market-agnostic.
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd
from loguru import logger

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None  # type: ignore

from .base_fetcher import BaseFetcher


class CNHKFetcher(BaseFetcher):
    """
    Fetch HK / A-share market data via akshare.

    Parameters
    ----------
    market : "hk" | "cn"
    """

    def __init__(
        self,
        market: Literal["hk", "cn"] = "cn",
        cache_dir: str = ".cache/data",
        cache_enabled: bool = True,
    ):
        super().__init__(cache_dir=cache_dir, cache_enabled=cache_enabled)
        if market not in ("hk", "cn"):
            raise ValueError(f"market must be 'hk' or 'cn', got '{market}'")
        self.market = market
        if ak is None:
            raise ImportError(
                "akshare is not installed. Run: pip install akshare"
            )

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> List[str]:
        """
        HK  : Hang Seng Index constituents.
        CN  : CSI 300 constituents.
        """
        try:
            if self.market == "hk":
                return self._hsi_universe()
            else:
                return self._csi300_universe()
        except Exception as exc:
            logger.warning(f"Universe fetch failed ({exc}); returning empty list")
            return []

    def _hsi_universe(self) -> List[str]:
        df = ak.index_stock_cons_csindex(symbol="HKHSI")
        symbols = df["成分券代码Constituent Code"].astype(str).str.zfill(5).tolist()
        logger.info(f"HSI universe: {len(symbols)} stocks")
        return symbols

    def _csi300_universe(self) -> List[str]:
        df = ak.index_stock_cons(symbol="000300")
        symbols = df["品种代码"].astype(str).tolist()
        logger.info(f"CSI 300 universe: {len(symbols)} stocks")
        return symbols

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def _fetch_prices(
        self, symbols: List[str], start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        if self.market == "hk":
            return self._fetch_hk_prices(symbols, start, end)
        else:
            return self._fetch_cn_prices(symbols, start, end, interval)

    def _fetch_hk_prices(
        self, symbols: List[str], start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for HK stocks via akshare."""
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = ak.stock_hk_daily(symbol=sym, adjust="qfq")
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                df = df[["open", "high", "low", "close", "volume"]]
                mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
                frames[sym] = df.loc[mask]
            except Exception as exc:
                logger.debug(f"HK price fetch failed for {sym}: {exc}")

        if not frames:
            return pd.DataFrame()
        return self._to_multiindex(frames)

    def _fetch_cn_prices(
        self, symbols: List[str], start: str, end: str, interval: str
    ) -> pd.DataFrame:
        """Fetch daily/weekly/monthly OHLCV for A-shares via akshare."""
        period_map = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}
        period = period_map.get(interval, "daily")
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")

        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=sym,
                    period=period,
                    start_date=start_fmt,
                    end_date=end_fmt,
                    adjust="qfq",
                )
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                df = df[["open", "high", "low", "close", "volume"]]
                frames[sym] = df
            except Exception as exc:
                logger.debug(f"CN price fetch failed for {sym}: {exc}")

        if not frames:
            return pd.DataFrame()
        return self._to_multiindex(frames)

    @staticmethod
    def _to_multiindex(frames: dict) -> pd.DataFrame:
        combined = pd.concat(frames, axis=1)
        combined.columns.names = ["symbol", "field"]
        combined.index.name = "date"
        return combined

    # ------------------------------------------------------------------
    # Fundamental info
    # ------------------------------------------------------------------

    def _fetch_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch basic fundamental snapshot via akshare.
        Returns a best-effort DataFrame; missing fields are NaN.
        """
        records = []
        for sym in symbols:
            row = {"symbol": sym}
            try:
                if self.market == "cn":
                    info = ak.stock_individual_info_em(symbol=sym)
                    # info is a 2-column DataFrame: item | value
                    mapping = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
                    row["market_cap"] = mapping.get("总市值")
                    row["sector"] = mapping.get("行业")
                    row["pe_ratio"] = mapping.get("市盈率(动)")
            except Exception as exc:
                logger.debug(f"Info fetch failed for {sym}: {exc}")
            records.append(row)

        return pd.DataFrame(records).set_index("symbol")
