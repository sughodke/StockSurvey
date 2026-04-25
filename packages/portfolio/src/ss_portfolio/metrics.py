"""Standard performance metrics on a daily return (or log-return) series.

All accept either a numpy array or a pandas Series and return Python
floats. Implementations use `numpy` (not JAX) — these are post-hoc
reporting helpers, not part of any autograd path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def _to_array(x: np.ndarray | pd.Series) -> np.ndarray:
    return x.values if isinstance(x, pd.Series) else np.asarray(x)


def annualized_sharpe(daily_returns: np.ndarray | pd.Series) -> float:
    """Annualized Sharpe of a daily return series. Assumes 252 trading days."""
    r = _to_array(daily_returns)
    sd = r.std(ddof=0)
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def cagr(daily_returns: np.ndarray | pd.Series) -> float:
    """Compound annual growth rate from a daily simple-return series."""
    r = _to_array(daily_returns)
    if len(r) == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    years = len(r) / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0 if years > 0 else 0.0


def max_drawdown(daily_returns: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the equity curve (negative)."""
    r = _to_array(daily_returns)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sortino(daily_returns: np.ndarray | pd.Series) -> float:
    """Annualized Sortino: mean / downside-deviation."""
    r = _to_array(daily_returns)
    downside = r[r < 0]
    dd = downside.std(ddof=0) if len(downside) else 0.0
    if dd <= 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS))


def calmar(daily_returns: np.ndarray | pd.Series) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(daily_returns)
    if mdd >= 0:
        return 0.0
    return cagr(daily_returns) / abs(mdd)
