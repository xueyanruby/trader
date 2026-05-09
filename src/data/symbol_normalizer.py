from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional


MarketPrefix = Literal["HK", "SH", "SZ", "US"]


@dataclass(frozen=True)
class NormalizedSymbol:
    original: str
    futu_code: str
    market: MarketPrefix


def _strip(s: str) -> str:
    return (s or "").strip()


def _detect_cn_exchange(code6: str) -> MarketPrefix:
    # Minimal heuristic:
    # - 6xxxxx: Shanghai
    # - others : Shenzhen
    # Users can always pass explicit SH./SZ. to override.
    return "SH" if code6.startswith("6") else "SZ"


def normalize_symbol(symbol: str, *, default_market: Optional[MarketPrefix] = None) -> NormalizedSymbol:
    """
    Normalize repository-friendly symbols into futu OpenAPI codes.

    Accepted inputs (examples):
    - HK: "00700", "00700.HK", "HK.00700"
    - CN: "600519", "000001", "SH.600519", "SZ.000001"
    - US: "AAPL", "US.AAPL"

    Returns
    -------
    NormalizedSymbol(original, futu_code, market)
    """
    raw = _strip(symbol)
    up = raw.upper()

    # Already in futu format
    for pfx in ("HK.", "SH.", "SZ.", "US."):
        if up.startswith(pfx) and len(up) > len(pfx):
            market = pfx[:-1]  # strip trailing dot
            return NormalizedSymbol(original=raw, futu_code=f"{market}.{up[len(pfx):]}", market=market)  # type: ignore[arg-type]

    # yfinance-like HK format
    if up.endswith(".HK") and up[:-3].isdigit():
        code = up[:-3].zfill(5)
        return NormalizedSymbol(original=raw, futu_code=f"HK.{code}", market="HK")

    # Pure digits: treat 5-digit as HK, 6-digit as CN
    if up.isdigit():
        if len(up) == 5:
            return NormalizedSymbol(original=raw, futu_code=f"HK.{up.zfill(5)}", market="HK")
        if len(up) == 6:
            market = _detect_cn_exchange(up)
            return NormalizedSymbol(original=raw, futu_code=f"{market}.{up}", market=market)

    # Ticker without prefix: default to US unless user says otherwise
    if default_market is None:
        default_market = "US"
    return NormalizedSymbol(original=raw, futu_code=f"{default_market}.{up}", market=default_market)


def normalize_symbols(
    symbols: Iterable[str],
    *,
    default_market: Optional[MarketPrefix] = None,
) -> List[NormalizedSymbol]:
    return [normalize_symbol(s, default_market=default_market) for s in symbols]

