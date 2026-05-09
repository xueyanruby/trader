"""
Abstract broker interface.

Any concrete broker (paper, Alpaca, IBKR, etc.) must implement this interface.
Strategies and the execution engine only depend on BaseBroker — swapping brokers
requires zero changes to strategy or risk code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Order:
    """Represents a single order submitted to the broker."""

    symbol: str
    qty: float           # positive = buy, negative = sell
    side: str            # "buy" | "sell"
    order_type: str      # "market" | "limit"
    limit_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "pending"  # pending | filled | cancelled | rejected
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    """Represents a current holding."""

    symbol: str
    qty: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


class BaseBroker(ABC):
    """
    Abstract broker interface.

    All monetary values are in the account's base currency.
    """

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    @abstractmethod
    def get_cash(self) -> float:
        """Return available buying power / cash balance."""

    @abstractmethod
    def get_equity(self) -> float:
        """Return total account equity (cash + market value of positions)."""

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Return all open positions keyed by symbol."""

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """
        Submit an order.

        Parameters
        ----------
        symbol      : ticker string
        qty         : number of shares (positive)
        side        : "buy" | "sell"
        order_type  : "market" | "limit"
        limit_price : required when order_type == "limit"
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve the current state of an order by ID."""

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Return all open (pending) orders."""

    # ------------------------------------------------------------------
    # Market data (minimal, for execution purposes)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent trade price for a symbol."""

    # ------------------------------------------------------------------
    # Portfolio rebalancing helper (shared across all brokers)
    # ------------------------------------------------------------------

    def rebalance(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> List[Order]:
        """
        Execute the orders needed to move from current positions to
        the target_weights allocation.

        target_weights : {symbol: fraction_of_equity}  — must sum to <= 1.0
        current_prices : {symbol: price}

        Returns list of submitted Order objects.
        """
        equity = self.get_equity()
        current_positions = self.get_positions()

        orders: List[Order] = []

        # Compute target share counts
        for symbol, weight in target_weights.items():
            price = current_prices.get(symbol) or self.get_latest_price(symbol)
            if price <= 0:
                continue
            target_value = equity * weight
            target_shares = int(target_value / price)

            current_shares = int(current_positions[symbol].qty) if symbol in current_positions else 0
            delta = target_shares - current_shares

            if abs(delta) < 1:
                continue

            side = "buy" if delta > 0 else "sell"
            order = self.place_order(symbol=symbol, qty=abs(delta), side=side)
            orders.append(order)

        # Close positions not in the target (sell everything)
        for symbol, pos in current_positions.items():
            if symbol not in target_weights and pos.qty > 0:
                order = self.place_order(symbol=symbol, qty=int(pos.qty), side="sell")
                orders.append(order)

        return orders
