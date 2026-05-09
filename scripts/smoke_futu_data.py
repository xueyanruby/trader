from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smoke_futu_data")

# Ensure repo root is importable (so `import src.*` works no matter where executed)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_prices_shape(df):
    assert df is not None
    assert not df.empty, "prices df is empty"
    assert df.index.name == "date"
    assert getattr(df.columns, "names", None) == ["symbol", "field"]
    required_fields = {"open", "high", "low", "close", "volume"}
    fields = set(df.columns.get_level_values("field").unique().tolist())
    missing = required_fields - fields
    assert not missing, f"missing fields: {missing}, got={sorted(fields)}"


def main() -> None:
    from src.data.futu_cnhk_fetcher import FutuCNHKFetcher
    from src.futu.quote_client import get_global_quote_client

    end = date.today()
    start = end - timedelta(days=45)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    hk_syms = ["00700"]
    cn_syms = ["600519"]

    logger.info(f"HK history: {hk_syms} {start_s} -> {end_s}")
    hk_fetcher = FutuCNHKFetcher(market="hk", cache_enabled=False)
    hk_prices = hk_fetcher.get_prices(hk_syms, start=start_s, end=end_s, interval="1d")
    _check_prices_shape(hk_prices)
    logger.info(f"HK ok: rows={len(hk_prices)} cols={len(hk_prices.columns)}")

    logger.info(f"CN history: {cn_syms} {start_s} -> {end_s}")
    cn_fetcher = FutuCNHKFetcher(market="cn", cache_enabled=False)
    cn_prices = cn_fetcher.get_prices(cn_syms, start=start_s, end=end_s, interval="1d")
    if cn_prices.empty:
        logger.warning(
            "CN prices is empty (likely A-share quote permission missing in FutuOpenD). "
            "You can either enable A-share quote permission, or set config `data.provider.cn: akshare` as fallback."
        )
    else:
        _check_prices_shape(cn_prices)
        logger.info(f"CN ok: rows={len(cn_prices)} cols={len(cn_prices.columns)}")

    qc = get_global_quote_client()
    logger.info("Snapshot latest prices (HK only)")
    snap = qc.get_latest_prices(hk_syms)
    logger.info(f"snapshot={snap}")
    assert snap.get("00700", 0) > 0, "snapshot returned no HK prices"

    print("OK")


if __name__ == "__main__":
    main()

