"""Strategy-level weights builders.

Higher-level than `weights.py` (which holds primitive ops like
`select_top_n_matrix` and `softmax_weights`) — these functions combine
CWT computation, divergence scoring, screening, and weight selection
into one call.

Currently just `weights_regime` (per-stock CWT-divergence top-N
ranking). Other strategy variants (sector-excess regime,
cross-sectional dispersion gating, etc.) live in their respective
research apps until they justify promotion here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_indicators import get_divergence
from ss_portfolio.screening import apply_nan_mask
from ss_portfolio.weights import select_top_n_matrix
from ss_wavelets import causal_cwt, precompute_windows


def log_returns_matrix(prices_arr: np.ndarray) -> np.ndarray:
    """`(T, N) -> (T, N)` log-returns; first row padded with 0.

    Available behind the `use_log_returns` flag of `weights_regime`
    (default OFF). Daily log returns are roughly stationary (zero-mean,
    vol-clustered but not trended), so feeding them to `causal_cwt`
    instead of raw close prices means the rolling z-norm is doing vol
    normalization rather than trend removal — long-scale wavelet power
    is no longer dominated by persistent multi-year drift in the price
    level.

    NaN propagation: a NaN at price index k makes log-returns NaN at
    indices k and k+1. `apply_nan_mask` downstream still keys off the
    original `prices.values`, so masking still happens at the right
    cells; the transform doesn't introduce NEW NaN regions, only
    widens existing ones by one bar.
    """
    out = np.zeros_like(prices_arr, dtype=np.float64)
    out[1:] = np.log(prices_arr[1:] / prices_arr[:-1])
    return out


def weights_regime(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """Hard-top-N basket ranked by CWT-power-distribution divergence.

    The score per (date, ticker) is the chosen divergence between the
    recent vs historical CWT-power distributions across `scales`. We
    compute scores for all valid dates in one vectorized pass via
    `precompute_windows` + numpy divergence.

    `scale_log_weights = zeros` makes the divergence's internal softmax
    uniform — Optuna chooses *which* scales to include, but each
    included scale contributes equally (matching legacy behavior).

    Picks **highest-divergence** names (`ascending=False`): biggest
    regime shift wins. Direction (price up vs down) doesn't enter — it's
    a momentum-of-volatility-shift idea.

    Liquidity is not filtered here; it enters the objective via per-
    (date, ticker) fees in `vbt_backtest`. Wide-spread names get
    ranked normally and then naturally tank the realized Sharpe of any
    config that picks them.

    CWT input is **raw close** by default. `use_log_returns=True` runs
    on log returns — preserved as a flag for non-ranking research
    (vol forecasting, regime-break detection); empirically worse on
    the cross-sectional ranking objective, see CLAUDE.md "Key findings"
    for the controlled walk-forward eval evidence.
    """
    cwt_input = (log_returns_matrix(prices.values)
                 if use_log_returns else prices.values)
    coeffs = causal_cwt(cwt_input, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)

    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    scores = np.array(div_fn(recent, historical, scale_log_weights),
                      copy=True)
    scores = apply_nan_mask(scores, prices.values, lookback)

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)
