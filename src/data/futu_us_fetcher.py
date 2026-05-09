"""
US market data fetcher backed by futu OpenAPI (FutuOpenD + futu-api).

Contract:
- get_prices() returns MultiIndex DataFrame: columns=(symbol, field), index=DatetimeIndex
- fields: open, high, low, close, volume
"""

from __future__ import annotations

from typing import List, Literal, Optional

import pandas as pd
from loguru import logger

from src.futu.quote_client import get_global_quote_client

try:
    from futu import AuType, KLType  # type: ignore
except ImportError:  # pragma: no cover
    AuType = None  # type: ignore
    KLType = None  # type: ignore

from .base_fetcher import BaseFetcher
from .symbol_normalizer import normalize_symbol


class FutuUSFetcher(BaseFetcher):
    """
    Fetch US equity data via futu OpenAPI.

    Notes:
    - Universe is not provided here; caller should supply a universe (e.g. from USFetcher.get_universe()).
    """

    def __init__(
        self,
        cache_dir: str = ".cache/data",
        cache_enabled: bool = True,
        rehab: Literal["forward", "backward", "none"] = "forward",
    ):
        super().__init__(cache_dir=cache_dir, cache_enabled=cache_enabled)
        self.rehab = rehab
        if KLType is None:
            raise ImportError("futu-api is not installed. Run: pip install futu-api")

    def get_universe(self) -> List[str]:
        return []

    def _fetch_prices(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        ktype_map = {
            "1d": KLType.K_DAY,   # type: ignore[attr-defined]
            "1wk": KLType.K_WEEK, # type: ignore[attr-defined]
            "1mo": KLType.K_MON,  # type: ignore[attr-defined]
        }
        ktype = ktype_map.get(interval)
        if ktype is None:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(ktype_map)}")

        au_type = None
        if AuType is not None:
            # US typically uses split-adjusted prices; keep same mapping as HK/CN
            au_type = {
                "forward": AuType.QFQ,
                "backward": AuType.HFQ,
                "none": AuType.NONE,
            }.get(self.rehab, AuType.QFQ)

        qc = get_global_quote_client()
        frames: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            try:
                futu_code = normalize_symbol(sym, default_market="US").futu_code
                df = qc.request_history_kline(
                    futu_code,
                    start=start,
                    end=end,
                    ktype=ktype,
                    au_type=au_type,
                    max_count=1000,
                )
                if df is None or df.empty:
                    continue

                cols = {str(c).strip().lower(): c for c in df.columns}
                time_col = cols.get("time_key") or cols.get("time") or cols.get("date")
                if time_col is None:
                    logger.debug(f"futu kline missing time column for {sym}: {list(df.columns)}")
                    continue

                def _col(name: str):
                    c = cols.get(name)
                    return df[c] if c is not None else df.get(name)

                out = pd.DataFrame(
                    {
                        "open": _col("open"),
                        "high": _col("high"),
                        "low": _col("low"),
                        "close": _col("close"),
                        "volume": _col("volume"),
                    }
                )
                out.index = pd.to_datetime(df[time_col])
                out.index.name = "date"
                out = out.sort_index()
                out = out[["open", "high", "low", "close", "volume"]]
                frames[sym] = out
            except Exception as exc:
                logger.debug(f"futu price fetch failed for {sym}: {exc}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined.columns.names = ["symbol", "field"]
        combined.index.name = "date"
        return combined

    def _fetch_info(self, symbols: List[str]) -> pd.DataFrame:
        # Keep best-effort empty snapshot (not required for current flows)
        return pd.DataFrame([{"symbol": s} for s in symbols]).set_index("symbol")

