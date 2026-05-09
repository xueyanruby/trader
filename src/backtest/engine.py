"""
Vectorised backtesting engine.

Design principles
-----------------
- No event-by-event simulation: portfolio weights are computed on each
  rebalance date, then held until the next rebalance.  This is the standard
  approach for monthly/weekly factor strategies.
- Realistic costs: configurable commission and slippage per trade.
- No look-ahead bias: signals are generated using only data available up to
  (but not including) the rebalance date.  The position is opened at the next
  available close price.

Usage
-----
    from src.backtest.engine import BacktestEngine
    from src.strategies import get_strategy

    engine = BacktestEngine(config)
    results = engine.run(prices, strategy_name="momentum_12_1")
    results.plot()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..data.preprocessor import Preprocessor
from ..risk.manager import RiskManager
from ..strategies import get_strategy
from .metrics import performance_report, print_report, drawdown_series


class BacktestResult:
    """Container for backtest output."""

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        portfolio_value: pd.Series,
        weight_history: pd.DataFrame,
        trade_log: pd.DataFrame,
        strategy_params: Dict[str, Any],
    ):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.portfolio_value = portfolio_value
        self.weight_history = weight_history
        self.trade_log = trade_log
        self.strategy_params = strategy_params

    def report(self, risk_free_rate: float = 0.0) -> dict:
        return performance_report(
            self.portfolio_returns,
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=risk_free_rate,
        )

    def print_report(self, risk_free_rate: float = 0.0) -> None:
        print_report(self.report(risk_free_rate))

    def plot(self, figsize=(14, 8)) -> None:
        """Plot cumulative returns and drawdown."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            cum_port = (1 + self.portfolio_returns).cumprod()
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

            # Cumulative return
            axes[0].plot(cum_port.index, cum_port.values, label="Strategy", linewidth=1.5)
            if self.benchmark_returns is not None:
                cum_bench = (1 + self.benchmark_returns.reindex(cum_port.index).fillna(0)).cumprod()
                axes[0].plot(cum_bench.index, cum_bench.values,
                             label="Benchmark", linewidth=1.0, linestyle="--", alpha=0.7)
            axes[0].set_ylabel("Cumulative Return")
            axes[0].legend()
            axes[0].set_title(f"Backtest: {self.strategy_params.get('strategy', 'strategy')}")
            axes[0].grid(alpha=0.3)

            # Drawdown
            dd = drawdown_series(self.portfolio_returns)
            axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
            axes[1].set_ylabel("Drawdown")
            axes[1].set_xlabel("Date")
            axes[1].grid(alpha=0.3)

            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.tight_layout()
            plt.savefig("backtest_result.png", dpi=150)
            plt.show()
            logger.info("Plot saved to backtest_result.png")
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")


class BacktestEngine:
    """
    Vectorised portfolio backtester.

    Parameters
    ----------
    config : dict from config.yaml (backtest + risk + strategy sections)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        backtest_cfg = config.get("backtest", {})
        self.initial_capital: float = backtest_cfg.get("initial_capital", 1_000_000)
        self.slippage_bps: float = backtest_cfg.get("slippage_bps", 5)
        risk_cfg = config.get("risk", {})
        self.commission_rate: float = risk_cfg.get("commission_rate", 0.001)
        self.risk_manager = RiskManager(risk_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        strategy_name: str,
        benchmark_prices: Optional[pd.DataFrame] = None,
        screened_symbols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Run the backtest for the given strategy over the full price history.

        Parameters
        ----------
        prices            : MultiIndex price DataFrame (symbol, field) × date
        strategy_name     : registered strategy name
        benchmark_prices  : optional benchmark price series (single-symbol MultiIndex)
        screened_symbols  : optional pre-filtered symbol list (from Screener)

        Returns
        -------
        BacktestResult
        """
        # Instantiate strategy
        strategy_cls = get_strategy(strategy_name)
        strategy_cfg = (
            self.config.get("strategy", {}).get(strategy_name, {})
        )
        strategy = strategy_cls(config=strategy_cfg)
        strategy.on_start()

        rebalance_freq: str = strategy_cfg.get("rebalance_freq", "M")

        # Determine rebalance dates (end of each period)
        close = Preprocessor.get_close(prices)
        rebalance_dates = self._rebalance_dates(close.index, rebalance_freq)

        logger.info(
            f"Running backtest: strategy={strategy_name}, "
            f"rebalance={rebalance_freq}, "
            f"dates={len(rebalance_dates)}, "
            f"symbols={len(close.columns)}"
        )

        # Limit universe to screened symbols if provided
        if screened_symbols:
            available = [s for s in screened_symbols if s in close.columns]
            if isinstance(prices.columns, pd.MultiIndex):
                prices = prices[available]
            close = Preprocessor.get_close(prices)

        # Benchmark
        bench_returns: Optional[pd.Series] = None
        if benchmark_prices is not None:
            bench_close = Preprocessor.get_close(benchmark_prices)
            bench_col = bench_close.columns[0]
            bench_returns = bench_close[bench_col].pct_change().dropna()

        # Main loop
        weight_records: List[dict] = []
        trade_records: List[dict] = []
        portfolio_values: List[float] = [self.initial_capital]
        portfolio_dates: List[pd.Timestamp] = [close.index[0]]
        portfolio_cash: float = self.initial_capital
        current_weights: Dict[str, float] = {}

        for i, reb_date in enumerate(rebalance_dates):
            # Use all history up to rebalance date (no look-ahead)
            hist = prices.loc[:reb_date]

            # Generate signals
            signals = strategy.generate_signals(hist)
            if signals.empty:
                continue

            # Convert signals to target weights via risk manager
            target_weights = self.risk_manager.size_positions(
                signals=signals,
                prices=Preprocessor.get_close(hist).iloc[-1],
            )

            # Determine next period returns (from reb_date to next reb_date)
            next_date = (
                rebalance_dates[i + 1]
                if i + 1 < len(rebalance_dates)
                else close.index[-1]
            )
            period_slice = close.loc[reb_date:next_date]
            if len(period_slice) < 2:
                continue

            period_return = self._period_return(
                period_slice, target_weights, current_weights
            )

            # Track portfolio value
            prev_val = portfolio_values[-1]
            new_val = prev_val * (1 + period_return)
            portfolio_values.append(new_val)
            portfolio_dates.append(next_date)

            # Log weights
            weight_records.append({
                "date": reb_date,
                **{sym: w for sym, w in target_weights.items()},
            })

            # Log trades (changed positions)
            for sym in set(list(target_weights.keys()) + list(current_weights.keys())):
                prev_w = current_weights.get(sym, 0.0)
                new_w = target_weights.get(sym, 0.0)
                if abs(new_w - prev_w) > 1e-6:
                    trade_records.append({
                        "date": reb_date,
                        "symbol": sym,
                        "direction": "BUY" if new_w > prev_w else "SELL",
                        "weight_change": new_w - prev_w,
                    })

            current_weights = dict(target_weights)

        strategy.on_end()

        # Build output series
        portfolio_value_series = pd.Series(
            portfolio_values,
            index=pd.DatetimeIndex(portfolio_dates),
            name="portfolio_value",
        )
        portfolio_ret_series = portfolio_value_series.pct_change().dropna()
        portfolio_ret_series.name = "portfolio_return"

        weight_df = pd.DataFrame(weight_records).set_index("date") if weight_records else pd.DataFrame()
        trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        result = BacktestResult(
            portfolio_returns=portfolio_ret_series,
            benchmark_returns=bench_returns,
            portfolio_value=portfolio_value_series,
            weight_history=weight_df,
            trade_log=trade_df,
            strategy_params=strategy.get_params(),
        )
        result.print_report()
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """Return end-of-period dates in the index for the given frequency."""
        # Resample to get period-end dates that exist in the actual trading calendar
        series = pd.Series(1, index=index)
        resampled = series.resample(freq).last()
        return [t for t in resampled.index if t in index]

    def _period_return(
        self,
        price_slice: pd.DataFrame,
        target_weights: Dict[str, float],
        prev_weights: Dict[str, float],
    ) -> float:
        """
        Compute equal-weighted portfolio return over a price slice,
        accounting for transaction costs.
        """
        if price_slice.empty or not target_weights:
            return 0.0

        start_prices = price_slice.iloc[0]
        end_prices = price_slice.iloc[-1]

        gross_return = 0.0
        for sym, weight in target_weights.items():
            if sym not in start_prices.index or sym not in end_prices.index:
                continue
            p_start = start_prices[sym]
            p_end = end_prices[sym]
            if pd.isna(p_start) or pd.isna(p_end) or p_start == 0:
                continue
            # Apply slippage on entry
            p_entry = p_start * (1 + self.slippage_bps / 10_000)
            stock_return = (p_end - p_entry) / p_entry
            gross_return += weight * stock_return

        # Transaction costs: commission on turnover
        turnover = sum(
            abs(target_weights.get(s, 0) - prev_weights.get(s, 0))
            for s in set(list(target_weights) + list(prev_weights))
        )
        cost = turnover * self.commission_rate
        return gross_return - cost
