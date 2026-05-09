from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

from loguru import logger

from src.data.symbol_normalizer import NormalizedSymbol, normalize_symbols

try:
    from futu import KLType, OpenQuoteContext, RET_OK  # type: ignore
except ImportError:  # pragma: no cover
    KLType = None  # type: ignore
    OpenQuoteContext = None  # type: ignore
    RET_OK = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class FutuConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 11111

    @staticmethod
    def from_env() -> "FutuConnectionConfig":
        return FutuConnectionConfig(
            host=os.getenv("FUTU_OPEND_HOST", "127.0.0.1"),
            port=_env_int("FUTU_OPEND_PORT", 11111),
        )


class FutuQuoteClient:
    """
    Thin wrapper around OpenQuoteContext with:
    - Lazy init + reuse
    - Convenience helpers for snapshot prices + history kline
    """

    def __init__(self, config: Optional[FutuConnectionConfig] = None):
        self._config = config or FutuConnectionConfig.from_env()
        self._ctx = None

    @property
    def enabled(self) -> bool:
        return OpenQuoteContext is not None

    def _ensure_ctx(self):
        if not self.enabled:
            raise ImportError("futu-api is not installed. Run: pip install futu-api")
        if self._ctx is None:
            self._ctx = OpenQuoteContext(host=self._config.host, port=self._config.port)  # type: ignore[misc]
        return self._ctx

    def close(self) -> None:
        try:
            if self._ctx is not None:
                self._ctx.close()
        finally:
            self._ctx = None

    # ------------------------------------------------------------------
    # Snapshot (latest price)
    # ------------------------------------------------------------------

    def get_latest_prices(
        self,
        symbols: Sequence[str],
        *,
        default_market: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Return {original_symbol: last_price} for the given input symbols.
        Uses get_market_snapshot (no subscription required).
        """
        if not symbols:
            return {}

        norm = normalize_symbols(symbols, default_market=default_market)  # type: ignore[arg-type]
        codes = [n.futu_code for n in norm]

        ctx = self._ensure_ctx()
        ret, data = ctx.get_market_snapshot(codes)  # type: ignore[call-arg]
        if ret != RET_OK:
            logger.debug(f"futu snapshot failed: {data}; retrying per-code")
            out: Dict[str, float] = {}
            for n in norm:
                try:
                    r, d = ctx.get_market_snapshot([n.futu_code])  # type: ignore[call-arg]
                    if r != RET_OK:
                        continue
                    cols1 = {str(c).strip().lower(): c for c in getattr(d, "columns", [])}
                    code_col1 = cols1.get("code") or cols1.get("stock_code") or cols1.get("security_code")
                    price_col1 = (
                        cols1.get("last_price")
                        or cols1.get("cur_price")
                        or cols1.get("price")
                        or cols1.get("last_close")
                    )
                    if code_col1 is None or price_col1 is None:
                        continue
                    row = d.iloc[0]
                    price = float(row[price_col1])
                    if price > 0:
                        out[n.original] = price
                except Exception:
                    continue
            return out

        # Defensive: futu may change column names slightly across versions.
        cols = {str(c).strip().lower(): c for c in getattr(data, "columns", [])}
        code_col = cols.get("code") or cols.get("stock_code") or cols.get("security_code")
        price_col = (
            cols.get("last_price")
            or cols.get("cur_price")
            or cols.get("price")
            or cols.get("last_close")  # fallback
        )
        if code_col is None or price_col is None:
            logger.warning(f"futu snapshot columns unexpected: {list(getattr(data, 'columns', []))}")
            return {}

        code_to_original = {n.futu_code: n.original for n in norm}
        out: Dict[str, float] = {}
        for _, row in data.iterrows():
            try:
                code = str(row[code_col])
                original = code_to_original.get(code)
                if not original:
                    continue
                price = float(row[price_col])
                if price > 0:
                    out[original] = price
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # History Kline
    # ------------------------------------------------------------------

    def request_history_kline(
        self,
        futu_code: str,
        *,
        start: str,
        end: str,
        ktype,
        au_type=None,
        max_count: int = 1000,
    ):
        """
        Thin wrapper. Returns futu DataFrame on success.
        """
        ctx = self._ensure_ctx()
        ret, data, _page_req = ctx.request_history_kline(  # type: ignore[call-arg]
            code=futu_code,
            start=start,
            end=end,
            ktype=ktype,
            autype=au_type,
            max_count=max_count,
        )
        if ret != RET_OK:
            raise RuntimeError(f"futu history kline failed ({futu_code}): {data}")
        return data


_GLOBAL_CLIENT: Optional[FutuQuoteClient] = None


def get_global_quote_client() -> FutuQuoteClient:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = FutuQuoteClient()
    return _GLOBAL_CLIENT


def close_global_quote_client() -> None:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is not None:
        _GLOBAL_CLIENT.close()
    _GLOBAL_CLIENT = None

