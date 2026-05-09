"""
Performance metrics for backtesting.

All functions accept a pd.Series of portfolio returns (daily or periodic)
and return scalar statistics or a summary dict.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Cumulative return over the full period."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate (CAGR)."""
    n = len(returns)
    if n == 0:
        return 0.0
    total = (1 + returns).prod()
    return float(total ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio = (annualised excess return) / (annualised volatility).
    risk_free_rate is the annualised risk-free rate (e.g. 0.05 for 5%).
    """
    excess = returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    ann_excess = float(excess.mean() * periods_per_year)
    return ann_excess / vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino ratio — like Sharpe but uses downside deviation only.
    """
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_vol = float(downside.std() * np.sqrt(periods_per_year))
    if downside_vol == 0:
        return np.inf
    ann_excess = float(excess.mean() * periods_per_year)
    return ann_excess / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (as a negative fraction).
    E.g. -0.30 means the portfolio fell 30% from its peak at worst.
    """
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.inf
    return annualized_return(returns, periods_per_year) / mdd


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """
    OLS regression of portfolio excess returns on benchmark excess returns.
    Returns (annualised alpha, beta).
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 1.0

    port = aligned.iloc[:, 0].values
    bench = aligned.iloc[:, 1].values

    beta = float(np.cov(port, bench)[0, 1] / np.var(bench))
    alpha_daily = float(np.mean(port) - beta * np.mean(bench))
    alpha_ann = alpha_daily * periods_per_year
    return alpha_ann, beta


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full time-series of drawdowns (useful for plotting)."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    return cum / rolling_max - 1


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def performance_report(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute a comprehensive performance report.

    Parameters
    ----------
    returns           : portfolio periodic returns
    benchmark_returns : optional benchmark returns for alpha/beta
    risk_free_rate    : annualised risk-free rate
    periods_per_year  : 252 for daily, 52 for weekly, 12 for monthly

    Returns
    -------
    dict with all key metrics
    """
    report = {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "num_periods": len(returns),
    }

    if benchmark_returns is not None:
        alpha, beta = alpha_beta(returns, benchmark_returns, periods_per_year)
        report["alpha"] = alpha
        report["beta"] = beta
        # Information ratio vs benchmark
        active = returns - benchmark_returns.reindex(returns.index).fillna(0)
        tracking_error = float(active.std() * np.sqrt(periods_per_year))
        ir = (float(active.mean()) * periods_per_year / tracking_error
              if tracking_error > 0 else 0.0)
        report["information_ratio"] = ir
        report["tracking_error"] = tracking_error

    return report


def print_report(report: dict) -> None:
    """Pretty-print the performance report to stdout."""
    pct = lambda x: f"{x*100:.2f}%"
    print("\n" + "=" * 50)
    print("  PERFORMANCE REPORT")
    print("=" * 50)
    print(f"  Total Return         : {pct(report['total_return'])}")
    print(f"  Annualised Return    : {pct(report['annualized_return'])}")
    print(f"  Annualised Volatility: {pct(report['annualized_volatility'])}")
    print(f"  Sharpe Ratio         : {report['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio        : {report['sortino_ratio']:.2f}")
    print(f"  Max Drawdown         : {pct(report['max_drawdown'])}")
    print(f"  Calmar Ratio         : {report['calmar_ratio']:.2f}")
    print(f"  Win Rate             : {pct(report['win_rate'])}")
    if "alpha" in report:
        print(f"  Alpha (annualised)   : {pct(report['alpha'])}")
        print(f"  Beta                 : {report['beta']:.2f}")
        print(f"  Information Ratio    : {report['information_ratio']:.2f}")
        print(f"  Tracking Error       : {pct(report['tracking_error'])}")
    print(f"  Periods              : {report['num_periods']}")
    print("=" * 50 + "\n")
