"""Vectorbt-backed portfolio backtester.

Wraps `vectorbt.Portfolio.from_orders` with the conventions the regime
strategies use: long-only, periodic rebalance to target weights, flat
per-side commission. Returns a metrics dict with the Sharpe / CAGR /
max-drawdown numbers the trainers actually consume.

vectorbt is the right tool for hyperparameter sweeps because the inner
loop is numba-JIT'd; the first call pays a ~5-20s compile cost,
subsequent calls reuse the cached kernel and run in milliseconds. See
README.md "Nix dev shell" for why this only works inside the nix
environment on Intel macOS Python 3.13.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (used by NaN sentinel)
import pandas as pd


def vbt_backtest(
    prices: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    rebalance_days: int = 20,
    commission_bps: float = 10.0,
    spread_df: pd.DataFrame | None = None,
    init_cash: float = 100_000.0,
    fill_lag: int = 1,
) -> dict[str, float]:
    """Run one backtest and return Sharpe / CAGR / max-drawdown / total return.

    Parameters
    ----------
    prices :
        `(n_dates, n_tickers)` close prices, DatetimeIndex.
    weight_df :
        `(n_dates, n_tickers)` target weights per date — typically the
        output of `select_top_n_matrix`. Need not align in rows; we
        subsample at `::rebalance_days` and forward-fill between.
    rebalance_days :
        Treat every `rebalance_days`-th row of `weight_df` as a rebalance
        signal; hold flat in between.
    commission_bps :
        Per-side commission as basis points of notional turnover.
    spread_df :
        Optional `(n_dates, n_tickers)` Corwin-Schultz relative-spread
        estimates. When provided, per-side fees become
        `commission_bps/10000 + spread/2`, charging each name its own
        liquidity cost on every rebalance. The optimizer then naturally
        prefers configs that pick tradable names instead of needing a
        binary spread filter upstream. When `None`, fees are flat at
        `commission_bps`.
    init_cash :
        Starting capital, in currency units. Sharpe / max-drawdown are
        scale-invariant, so this only matters if you also want absolute
        equity-curve numbers.
    fill_lag :
        Bars between signal and execution. Default 1 means the signal
        computed at close[t] fills at close[t+1] — the realistic
        assumption that you can't trade at a price you only knew once
        the bar closed. Set to 0 for the (optimistic) same-bar fill
        used by most naive backtests; useful for comparison.

    Returns
    -------
    dict with keys: sharpe, cagr, max_drawdown, total_return.
    """
    import vectorbt as vbt

    # Align columns; drop tickers without prices for the requested window.
    common = prices.columns.intersection(weight_df.columns)
    p = prices[common]
    w = weight_df[common]

    # Build a sparse target-percent matrix: only rebalance dates have
    # values; all other rows are NaN. With `size_type='targetpercent'`,
    # vectorbt only places orders on non-NaN rows. If we forward-filled
    # we'd get daily re-targeting (which drifts the portfolio back to
    # weights every bar — a different strategy that earns a rebalancing
    # premium). bt-library's `RunOnDate + WeighTarget` only trades on
    # the listed dates, so we mirror that here.
    #
    # `fill_lag` shifts the chosen rebalance dates forward by N bars so
    # a signal computed at close[t] executes at close[t+N]. Without this
    # shift vbt would fill at the same bar that produced the signal —
    # using a price you didn't know until the close.
    rebal_w = pd.DataFrame(np.nan, index=p.index, columns=common)
    raw_dates = w.index[::rebalance_days].intersection(p.index)
    if fill_lag > 0:
        positions = p.index.get_indexer(raw_dates) + fill_lag
        positions = positions[positions < len(p.index)]
        fill_dates = p.index[positions]
        # Each shifted fill_date corresponds to a raw signal date that
        # still fits in the panel — line them up by position.
        signal_dates = raw_dates[: len(fill_dates)]
        rebal_w.loc[fill_dates] = w.loc[signal_dates].values
    else:
        rebal_w.loc[raw_dates] = w.loc[raw_dates].values

    # Per-side fee = flat commission + half the relative spread (the
    # canonical "cross-half-spread" cost assumption). vectorbt accepts a
    # `(n_dates, n_tickers)` fees matrix, so this charges each ticker
    # its own time-varying spread on entry/exit.
    if spread_df is not None:
        s = spread_df.reindex(index=p.index, columns=common).fillna(0.0)
        fees = commission_bps / 10000.0 + s.values / 2.0
    else:
        fees = commission_bps / 10000.0

    # Build a single grouped portfolio. `targetpercent` size_type takes
    # the weight matrix and computes share quantities for each rebalance.
    pf = vbt.Portfolio.from_orders(
        close=p,
        size=rebal_w,
        size_type='targetpercent',
        fees=fees,
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        freq='1D',
    )

    # vectorbt's default year_freq is 365 calendar days. bt-library uses
    # 252 trading days. Pin to 252 here so the Sharpe we report is
    # directly comparable to the bt-library number that the legacy
    # `regime.research.optimize_regime` script produced.
    sharpe = float(pf.sharpe_ratio(year_freq='252 days'))
    return {
        'sharpe': sharpe if np.isfinite(sharpe) else float('nan'),
        'cagr': float(pf.annualized_return(year_freq='252 days')),
        'max_drawdown': float(pf.max_drawdown()),
        'total_return': float(pf.total_return()),
    }
