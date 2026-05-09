"""
Paper trading broker — full in-memory simulation.

Simulates fills at the last known price plus configurable slippage.
No network calls.  Ideal for strategy development and walk-forward testing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from .base_broker import BaseBroker, Order, Position


class PaperBroker(BaseBroker):
    """
    In-memory paper trading broker.

    Parameters
    ----------
    initial_cash : starting cash balance
    slippage_bps : slippage in basis points applied on each fill
    commission   : commission per dollar traded (e.g. 0.001 = 0.1%)
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        slippage_bps: float = 5.0,
        commission: float = 0.001,
    ):
        self._cash = initial_cash
        self._slippage_bps = slippage_bps
        self._commission = commission
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._latest_prices: Dict[str, float] = {}
        self._fill_log: List[dict] = []
        logger.info(
            f"PaperBroker initialised: cash=${initial_cash:,.2f}, "
            f"slippage={slippage_bps}bps, commission={commission*100:.2f}%"
        )

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Inject latest market prices.  Call this before each rebalance cycle
        so the broker can value positions and fill orders at realistic prices.
        """
        self._latest_prices.update(prices)
        # Update position mark-to-market
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.current_price = prices[sym]

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    def get_cash(self) -> float:
        return self._cash

    def get_equity(self) -> float:
        mv = sum(pos.market_value for pos in self._positions.values())
        return self._cash + mv

    def get_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

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
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            order_type=order_type,
            limit_price=limit_price,
            order_id=order_id,
            submitted_at=datetime.utcnow(),
        )
        self._orders[order_id] = order

        # Immediate market fill simulation
        if order_type == "market":
            self._fill_market_order(order)
        # Limit orders stay pending; call fill_pending_orders() to check

        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == "pending":
            order.status = "cancelled"
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == "pending"]

    def get_latest_price(self, symbol: str) -> float:
        return self._latest_prices.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> dict:
        """Return a snapshot of the portfolio state."""
        pos_data = {
            sym: {
                "qty": pos.qty,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": f"{pos.unrealized_pnl_pct*100:.2f}%",
            }
            for sym, pos in self._positions.items()
        }
        return {
            "cash": self._cash,
            "equity": self.get_equity(),
            "num_positions": len(self._positions),
            "positions": pos_data,
        }

    def fill_log(self) -> List[dict]:
        """Return the full fill history."""
        return list(self._fill_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_market_order(self, order: Order) -> None:
        """Simulate an immediate market fill with slippage."""
        price = self._latest_prices.get(order.symbol, 0.0)
        if price <= 0:
            logger.warning(
                f"No price available for {order.symbol}; order {order.order_id} rejected"
            )
            order.status = "rejected"
            return

        # Apply slippage
        slip = self._slippage_bps / 10_000
        if order.side == "buy":
            fill_price = price * (1 + slip)
        else:
            fill_price = price * (1 - slip)

        trade_value = fill_price * order.qty
        commission = trade_value * self._commission

        if order.side == "buy":
            total_cost = trade_value + commission
            if total_cost > self._cash:
                # Partial fill or reject
                affordable_qty = int(self._cash / (fill_price * (1 + self._commission)))
                if affordable_qty < 1:
                    order.status = "rejected"
                    logger.warning(f"Insufficient cash to buy {order.symbol}")
                    return
                order.qty = affordable_qty
                trade_value = fill_price * order.qty
                commission = trade_value * self._commission
                total_cost = trade_value + commission

            self._cash -= total_cost
            self._update_position_buy(order.symbol, order.qty, fill_price)

        else:  # sell
            if order.symbol not in self._positions:
                order.status = "rejected"
                logger.warning(f"No position in {order.symbol} to sell")
                return
            pos = self._positions[order.symbol]
            sell_qty = min(order.qty, pos.qty)
            proceeds = fill_price * sell_qty - commission
            self._cash += proceeds
            self._update_position_sell(order.symbol, sell_qty)

        order.status = "filled"
        order.filled_qty = order.qty
        order.filled_avg_price = fill_price
        order.filled_at = datetime.utcnow()

        self._fill_log.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "fill_price": fill_price,
            "commission": commission,
            "timestamp": order.filled_at,
        })

        logger.debug(
            f"Fill: {order.side.upper()} {order.qty} {order.symbol} "
            f"@ ${fill_price:.2f} (commission=${commission:.2f})"
        )

    def _update_position_buy(self, symbol: str, qty: float, price: float) -> None:
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_cost = pos.avg_cost * pos.qty + price * qty
            pos.qty += qty
            pos.avg_cost = total_cost / pos.qty
            pos.current_price = price
        else:
            self._positions[symbol] = Position(
                symbol=symbol, qty=qty, avg_cost=price, current_price=price
            )

    def _update_position_sell(self, symbol: str, qty: float) -> None:
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.qty -= qty
        if pos.qty <= 0:
            del self._positions[symbol]
