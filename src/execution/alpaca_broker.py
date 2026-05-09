"""
Alpaca live/paper trading broker.

Uses the Alpaca REST API for US equity trading.
Requires the optional dependency: pip install alpaca-trade-api

Set credentials via environment variables (never hard-code in config):
    export ALPACA_API_KEY=your_key
    export ALPACA_SECRET_KEY=your_secret
    export ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live URL

Reference: https://alpaca.markets/docs/api-documentation/
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from loguru import logger

from .base_broker import BaseBroker, Order, Position


class AlpacaBroker(BaseBroker):
    """
    Alpaca-backed broker for US equity live/paper trading.

    The alpaca-trade-api package must be installed:
        pip install alpaca-trade-api

    Parameters
    ----------
    api_key    : Alpaca API key (falls back to ALPACA_API_KEY env var)
    api_secret : Alpaca secret  (falls back to ALPACA_SECRET_KEY env var)
    base_url   : API endpoint   (falls back to ALPACA_BASE_URL env var)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        key = api_key or os.environ.get("ALPACA_API_KEY", "")
        secret = api_secret or os.environ.get("ALPACA_SECRET_KEY", "")
        url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not key or not secret:
            raise ValueError(
                "Alpaca credentials missing. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is not installed. "
                "Run: pip install alpaca-trade-api"
            )

        self._api = tradeapi.REST(key, secret, url, api_version="v2")
        logger.info(f"AlpacaBroker connected: {url}")

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_cash(self) -> float:
        return float(self._api.get_account().cash)

    def get_equity(self) -> float:
        return float(self._api.get_account().equity)

    def get_positions(self) -> Dict[str, Position]:
        raw_positions = self._api.list_positions()
        result: Dict[str, Position] = {}
        for p in raw_positions:
            result[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_cost=float(p.avg_entry_price),
                current_price=float(p.current_price),
            )
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        kwargs = {
            "symbol": symbol,
            "qty": int(qty),
            "side": side.lower(),
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit" and limit_price:
            kwargs["limit_price"] = str(limit_price)

        raw = self._api.submit_order(**kwargs)
        order = Order(
            symbol=symbol,
            qty=float(raw.qty),
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            order_id=raw.id,
            status=raw.status,
        )
        logger.info(f"Order submitted: {side.upper()} {qty} {symbol} [{order.order_id}]")
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as exc:
            logger.warning(f"Cancel order {order_id} failed: {exc}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        try:
            raw = self._api.get_order(order_id)
            return Order(
                symbol=raw.symbol,
                qty=float(raw.qty),
                side=raw.side,
                order_type=raw.type,
                order_id=raw.id,
                status=raw.status,
                filled_qty=float(raw.filled_qty or 0),
                filled_avg_price=float(raw.filled_avg_price or 0),
            )
        except Exception:
            return None

    def get_open_orders(self) -> List[Order]:
        raw_orders = self._api.list_orders(status="open")
        return [
            Order(
                symbol=o.symbol,
                qty=float(o.qty),
                side=o.side,
                order_type=o.type,
                order_id=o.id,
                status=o.status,
            )
            for o in raw_orders
        ]

    def get_latest_price(self, symbol: str) -> float:
        try:
            bar = self._api.get_latest_trade(symbol)
            return float(bar.price)
        except Exception:
            return 0.0
