"""
HK and China A-share data fetcher backed by futu OpenAPI (FutuOpenD + futu-api).

Contract:
- get_prices() returns MultiIndex DataFrame: columns=(symbol, field), index=DatetimeIndex
- fields: open, high, low, close, volume
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd
from loguru import logger

from src.futu.quote_client import get_global_quote_client

try:
    from futu import AuType, KLType, RET_OK  # type: ignore
except ImportError:  # pragma: no cover
    AuType = None  # type: ignore
    KLType = None  # type: ignore
    RET_OK = None  # type: ignore

from .base_fetcher import BaseFetcher
from .symbol_normalizer import normalize_symbol


class FutuCNHKFetcher(BaseFetcher):
    """
    Fetch HK / A-share market data via futu OpenAPI.

    Parameters
    ----------
    market : "hk" | "cn"
    rehab : "forward" | "backward" | "none"
    """

    def __init__(
        self,
        market: Literal["hk", "cn"] = "cn",
        cache_dir: str = ".cache/data",
        cache_enabled: bool = True,
        rehab: Literal["forward", "backward", "none"] = "forward",
    ):
        super().__init__(cache_dir=cache_dir, cache_enabled=cache_enabled)
        if market not in ("hk", "cn"):
            raise ValueError(f"market must be 'hk' or 'cn', got '{market}'")
        self.market = market
        self.rehab = rehab

        if KLType is None:
            raise ImportError("futu-api is not installed. Run: pip install futu-api")

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self) -> List[str]:
        # Best-effort: return empty to force explicit config universe when needed,
        # avoiding a hard dependency on plate/index permissions.
        return []

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

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
            au_type = {
                "forward": AuType.QFQ,
                "backward": AuType.HFQ,
                "none": AuType.NONE,
            }.get(self.rehab, AuType.QFQ)

        qc = get_global_quote_client()
        frames: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            try:
                futu_code = normalize_symbol(sym).futu_code
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

    # ------------------------------------------------------------------
    # Fundamental info
    # ------------------------------------------------------------------

    def _fetch_info(self, symbols: List[str]) -> pd.DataFrame:
        """
        Best-effort: use snapshot to fill a minimal set of fields.
        """
        if not symbols:
            return pd.DataFrame()

        qc = get_global_quote_client()
        norm = [normalize_symbol(s) for s in symbols]
        codes = [n.futu_code for n in norm]
        code_to_original = {n.futu_code: n.original for n in norm}

        try:
            ctx = qc._ensure_ctx()  # type: ignore[attr-defined]
            ret, data = ctx.get_market_snapshot(codes)
            if ret != RET_OK:
                return pd.DataFrame([{"symbol": s} for s in symbols]).set_index("symbol")
        except Exception:
            return pd.DataFrame([{"symbol": s} for s in symbols]).set_index("symbol")

        cols = {str(c).strip().lower(): c for c in getattr(data, "columns", [])}
        code_col = cols.get("code")

        records = []
        for _, row in data.iterrows():
            try:
                code = str(row[code_col]) if code_col is not None else ""
                original = code_to_original.get(code, code)
                rec = {"symbol": original}
                for k in ("market_val", "pe_ratio", "turnover_rate", "lot_size"):
                    col = cols.get(k)
                    if col is not None:
                        rec[k] = row.get(col)
                records.append(rec)
            except Exception:
                continue

        if not records:
            records = [{"symbol": s} for s in symbols]
        return pd.DataFrame(records).set_index("symbol")

