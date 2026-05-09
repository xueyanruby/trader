"""
Risk manager: translates raw strategy signals into sized, risk-adjusted orders.

Responsibilities
----------------
1. Position sizing: convert +1/-1 signals to target portfolio weights
2. Max position cap: no single stock exceeds max_position_weight
3. Sector concentration limit (optional, requires sector info)
4. Dynamic regime scaling: scale total exposure by market regime (bull/neutral/bear)
5. Stop-loss tracking: flag positions that have breached their stop
6. Portfolio-level circuit breaker: halt trading if max drawdown exceeded
7. Risk dashboard: expose real-time metrics snapshot for notifications

The risk manager is called by both the backtest engine (to compute weights on
each rebalance date) and the paper/live execution loop (to gate orders).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class RiskManager:
    """
    Converts raw signals to sized weights and enforces risk constraints.

    Parameters
    ----------
    config : risk section from config.yaml
    """

    _DEFAULTS: Dict[str, Any] = {
        "max_position_weight": 0.10,
        "max_sector_weight": 0.30,
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.25,
        "max_drawdown_halt": 0.20,
        "commission_rate": 0.001,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self._DEFAULTS, **(config or {})}
        self._peak_equity: float = 0.0
        self._trading_halted: bool = False
        self._entry_prices: Dict[str, float] = {}  # symbol → avg entry price

    # ------------------------------------------------------------------
    # Signal → target weight conversion
    # ------------------------------------------------------------------

    def size_positions(
        self,
        signals: pd.Series,
        prices: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        regime_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Convert a signal Series (values +1/-1) to a dict of target weights.

        Equal-weight allocation among all long signals by default.
        Applies max_position_weight cap, sector constraints, and regime scaling.

        Parameters
        ----------
        signals      : pd.Series index=symbol, values in {+1, 0, -1}
        prices       : optional latest prices (used for logging)
        sector_map   : optional {symbol: sector_name} for sector limits
        regime_scale : market regime exposure multiplier from RegimeDetector
                       (e.g. 0.40 in bull, 0.20 in neutral, 0.05 in bear).
                       Applied as a cap on total portfolio exposure.

        Returns
        -------
        dict {symbol: target_weight}  — weights sum to ≤ regime_scale
        """
        if self._trading_halted:
            logger.warning("Trading halted (max drawdown exceeded). No new positions.")
            return {}

        long_signals = signals[signals > 0]
        if long_signals.empty:
            return {}

        n = len(long_signals)
        max_weight: float = self.config["max_position_weight"]

        # Raw equal weight, capped at max_position_weight
        raw_weight = min(1.0 / n, max_weight)
        weights: Dict[str, float] = {sym: raw_weight for sym in long_signals.index}

        # Sector concentration check
        if sector_map:
            weights = self._apply_sector_cap(weights, sector_map)

        # Re-normalise so weights sum to ≤ 1.0 before regime scaling
        total = sum(weights.values())
        if total > 1.0:
            weights = {sym: w / total for sym, w in weights.items()}

        # Apply regime scale: cap total exposure at regime_scale
        # regime_scale comes from RegimeDetector (bull=0.40, neutral=0.20, bear=0.05)
        if 0 < regime_scale < 1.0:
            current_total = sum(weights.values())
            if current_total > regime_scale:
                scale_factor = regime_scale / current_total
                weights = {sym: w * scale_factor for sym, w in weights.items()}
                logger.debug(
                    f"RiskManager: regime_scale={regime_scale:.0%} applied, "
                    f"portfolio scaled by {scale_factor:.2f}"
                )

        total_invested = sum(weights.values())
        logger.debug(
            f"RiskManager: {len(weights)} positions, "
            f"avg weight={total_invested/max(len(weights), 1):.2%}, "
            f"total invested={total_invested:.2%}  "
            f"(regime_scale={regime_scale:.0%})"
        )
        return weights

    # ------------------------------------------------------------------
    # Stop-loss / take-profit monitoring
    # ------------------------------------------------------------------

    def record_entry(self, symbol: str, price: float) -> None:
        """Record the entry price for stop-loss tracking."""
        self._entry_prices[symbol] = price

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Return True if the position has breached the stop-loss threshold.
        The caller should immediately exit the position.
        """
        entry = self._entry_prices.get(symbol)
        if entry is None or entry == 0:
            return False
        loss_pct = (current_price - entry) / entry
        stop: float = -abs(self.config["stop_loss_pct"])
        if loss_pct <= stop:
            logger.warning(
                f"Stop-loss triggered: {symbol} "
                f"entry=${entry:.2f} current=${current_price:.2f} "
                f"loss={loss_pct:.2%} (threshold={stop:.2%})"
            )
            return True
        return False

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Return True if the position has hit the take-profit target."""
        entry = self._entry_prices.get(symbol)
        if entry is None or entry == 0:
            return False
        gain_pct = (current_price - entry) / entry
        target: float = self.config["take_profit_pct"]
        if gain_pct >= target:
            logger.info(
                f"Take-profit triggered: {symbol} "
                f"entry=${entry:.2f} current=${current_price:.2f} "
                f"gain={gain_pct:.2%} (target={target:.2%})"
            )
            return True
        return False

    def get_stops_to_exit(
        self, current_prices: Dict[str, float]
    ) -> List[str]:
        """
        Scan all tracked positions and return symbols that should be exited
        due to stop-loss OR take-profit triggers.
        """
        exits = []
        for sym, price in current_prices.items():
            if sym not in self._entry_prices:
                continue
            if self.check_stop_loss(sym, price) or self.check_take_profit(sym, price):
                exits.append(sym)
        return exits

    def clear_entry(self, symbol: str) -> None:
        """Remove stop tracking for a closed position."""
        self._entry_prices.pop(symbol, None)

    # ------------------------------------------------------------------
    # Portfolio-level circuit breaker
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """
        Called after each valuation with the current portfolio equity.
        If equity has fallen more than max_drawdown_halt from peak, halt trading.
        """
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / self._peak_equity
            halt_threshold: float = -abs(self.config["max_drawdown_halt"])
            if drawdown <= halt_threshold and not self._trading_halted:
                logger.error(
                    f"CIRCUIT BREAKER: portfolio drawdown {drawdown:.2%} "
                    f"exceeded halt threshold {halt_threshold:.2%}. "
                    "All new orders blocked."
                )
                self._trading_halted = True

    def reset_halt(self) -> None:
        """Manually re-enable trading after a circuit-breaker event."""
        self._trading_halted = False
        logger.info("Circuit breaker reset — trading re-enabled")

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    # ------------------------------------------------------------------
    # Risk dashboard
    # ------------------------------------------------------------------

    def risk_dashboard(self, rt_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return a structured risk dashboard snapshot suitable for notifications.

        Parameters
        ----------
        rt_metrics : dict returned by RealTimeMetrics.update()
                     (keys: volatility, max_drawdown, var_95, hhi, equity)

        Returns
        -------
        dict with all risk dimensions merged for the daily report.
        """
        dashboard: Dict[str, Any] = {
            # Circuit breaker state
            "is_halted": self._trading_halted,
            "peak_equity": self._peak_equity,
            "tracked_positions": len(self._entry_prices),
            # Config limits
            "max_position_weight": self.config["max_position_weight"],
            "max_sector_weight": self.config["max_sector_weight"],
            "stop_loss_pct": self.config["stop_loss_pct"],
            "take_profit_pct": self.config["take_profit_pct"],
            "max_drawdown_halt": self.config["max_drawdown_halt"],
        }
        # Merge real-time metrics if provided
        if rt_metrics:
            dashboard.update({
                "volatility": rt_metrics.get("volatility", 0.0),
                "max_drawdown": rt_metrics.get("max_drawdown", 0.0),
                "var_95": rt_metrics.get("var_95", 0.0),
                "hhi": rt_metrics.get("hhi", 0.0),
                "current_equity": rt_metrics.get("equity", 0.0),
            })
        return dashboard

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_sector_cap(
        self,
        weights: Dict[str, float],
        sector_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Reduce weights so no sector exceeds max_sector_weight."""
        max_sector: float = self.config["max_sector_weight"]

        # Group by sector
        sector_totals: Dict[str, float] = {}
        for sym, w in weights.items():
            sector = sector_map.get(sym, "Unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + w

        result = dict(weights)
        for sector, total in sector_totals.items():
            if total > max_sector:
                scale = max_sector / total
                for sym in weights:
                    if sector_map.get(sym) == sector:
                        result[sym] = weights[sym] * scale
                logger.debug(
                    f"Sector cap applied: {sector} "
                    f"({total:.2%} → {max_sector:.2%})"
                )
        return result

    def summary(self) -> Dict[str, Any]:
        return {
            "peak_equity": self._peak_equity,
            "trading_halted": self._trading_halted,
            "tracked_positions": len(self._entry_prices),
            "config": self.config,
        }
