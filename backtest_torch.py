"""
backtest_torch.py : Differentiable regime strategy via scipy L-BFGS-B.

PyTorch is unavailable on Python 3.13+ / Intel macOS (wheels dropped in 2.3).
This module achieves the same goal — gradient-based optimization of the
continuous parameters — using scipy.optimize.minimize with numerical gradients.

For Optuna (black-box TPE), all hyperparams are discrete/categorical and
the search space is O(50-200 trials). Here we isolate the continuous
subspace — scale weights and portfolio temperature — and optimize them
with L-BFGS-B in O(50-100) function evaluations.

Learnable parameters (14 total):
  - scale_log_weights (13,): log-weights over CWT scales, softmax-normalized
  - log_temperature   (1,) : portfolio concentration for soft top-N

Integer hyperparams (lookback, n_tail, top_n, divergence) are passed as
fixed arguments; they can still be searched with Optuna over the outer loop.

The cumsum trick makes score computation O(n_dates) with no Python loop.
The causal CWT uses left-padded np.convolve (strictly causal, no lookahead).

Usage:
    uv run python backtest_torch.py --data-dir ./Nasdaq3347
    uv run python backtest_torch.py --data-dir ./Nasdaq3347 --lookback 229 --n-tail 106
    uv run python backtest_torch.py --data-dir ./Nasdaq3347 --method nelder-mead
    uv run python backtest_torch.py --data-dir ./Nasdaq3347 --save
"""

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from tqdm import tqdm

from backtest_bt import load_price_matrix, corwin_schultz_spread

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

ALL_SCALES = [3, 5, 7, 10, 12, 15, 21, 26, 42, 50, 63, 90, 126]


# ---------------------------------------------------------------------------
# Causal CWT (correctly causal: output[i] depends only on input[:i+1])
# ---------------------------------------------------------------------------

def _ricker_causal(scale, n_dates):
    """One-sided Ricker kernel, t in [-points, 0].

    kernel[k] = ricker((-points+k)/s)  for k in [0, points]
    kernel[0]  = ricker at far past
    kernel[-1] = ricker at t=0 (present)

    Causal convolution: np.convolve(x, kernel, mode='full')[-n_dates:]
    gives output[i] = sum_{lag=0}^{points} kernel[points-lag] * x[i-lag]
                    = ricker(0)*x[i] + ricker(-1/s)*x[i-1] + ...
    """
    points = min(10 * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale)


def causal_cwt_numpy(prices_np, scales, lookback):
    """Causal Ricker CWT for all tickers at all scales.

    Normalization: causal rolling mean/std using cumsum (O(n_dates), no loop).
    Convolution: np.convolve mode='full', take last n_dates elements.

    prices_np : (n_dates, n_tickers) float64
    Returns   : (n_scales, n_dates, n_tickers) float32
    """
    n_dates, n_tickers = prices_np.shape

    # Causal rolling normalization using cumsum
    cs = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices_np]), axis=0)  # (n+1, T)
    cs2 = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices_np ** 2]), axis=0)

    idx = np.arange(n_dates)
    lo = np.maximum(0, idx - lookback + 1)
    counts = idx - lo + 1  # actual window length (capped at lookback)

    rolling_mu = (cs[idx + 1] - cs[lo]) / counts[:, None]
    rolling_mu2 = (cs2[idx + 1] - cs2[lo]) / counts[:, None]
    rolling_std = np.sqrt(np.maximum(rolling_mu2 - rolling_mu ** 2, 1e-9))

    x_norm = (prices_np - rolling_mu) / rolling_std  # (n_dates, n_tickers)

    coeffs = np.zeros((len(scales), n_dates, n_tickers), dtype=np.float32)
    for si, s in enumerate(scales):
        kernel = _ricker_causal(s, n_dates)
        # fftconvolve full then take last n_dates (= causal result)
        full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
        coeffs[si] = full[-n_dates:].astype(np.float32)

    return coeffs  # (n_scales, n_dates, n_tickers)


# ---------------------------------------------------------------------------
# Vectorized regime scores via cumsum (no Python loop over dates)
# ---------------------------------------------------------------------------

def regime_scores_numpy(power, lookback, n_tail, scale_weights=None):
    """Symmetric KL divergence between recent and historical power.

    For each valid date j in [0, n_valid-1] (= real date j + lookback):
      recent[j]     = mean(power[:, j+lookback-n_tail+1 : j+lookback+1])
      historical[j] = mean(power[:, j                  : j+n_hist+1])
      where n_hist = lookback - n_tail

    Computed via cumulative sums — O(n_dates), no Python loop.

    power        : (n_scales, n_dates, n_tickers)
    scale_weights: (n_scales,) softmax-normalized weights (or None for uniform)
    Returns      : (n_valid, n_tickers) divergence scores
    """
    n_scales, n_dates, n_tickers = power.shape
    n_valid = n_dates - lookback
    n_hist = lookback - n_tail

    if scale_weights is not None:
        sw = np.exp(scale_weights)
        sw /= sw.sum()
        power = power * sw[:, None, None]

    # Prepend zeros: cs[s, k, t] = sum(power[s, 0:k, t])
    cs = np.concatenate([np.zeros((n_scales, 1, n_tickers), dtype=power.dtype),
                         np.cumsum(power, axis=1)], axis=1)

    # recent[:, j, :] = (cs[:, j+lookback+1] - cs[:, j+lookback-n_tail+1]) / n_tail
    recent = (cs[:, lookback + 1:, :]
              - cs[:, lookback - n_tail + 1: n_dates - n_tail + 1, :]) / n_tail

    # historical[:, j, :] = (cs[:, j+n_hist+1] - cs[:, j]) / (n_hist+1)
    historical = (cs[:, n_hist + 1: n_valid + n_hist + 1, :]
                  - cs[:, :n_valid, :]) / (n_hist + 1)

    # Normalize to distributions over scales
    eps = 1e-9
    rd = recent / (recent.sum(axis=0, keepdims=True) + eps)
    hd = historical / (historical.sum(axis=0, keepdims=True) + eps)

    kl = 0.5 * np.sum(rd * np.log((rd + eps) / (hd + eps)), axis=0)
    kl += 0.5 * np.sum(hd * np.log((hd + eps) / (rd + eps)), axis=0)
    return kl  # (n_valid, n_tickers)


# ---------------------------------------------------------------------------
# Differentiable Sharpe (continuous, smooth in all params)
# ---------------------------------------------------------------------------

def soft_sharpe(scores, log_temperature, prices_np, lookback,
                spread_mask=None):
    """Soft portfolio Sharpe via temperature-scaled softmax weights.

    Scores with -inf (illiquid) get zero weight via softmax naturally.

    scores        : (n_valid, n_tickers) divergence values
    log_temperature: scalar — exp(.) gives the softmax temperature
    prices_np     : (n_dates, n_tickers) raw prices
    spread_mask   : (n_valid, n_tickers) bool — True = illiquid
    """
    if spread_mask is not None:
        scores = np.where(spread_mask, -np.inf, scores)

    temp = max(np.exp(log_temperature), 1e-3)
    # Numerically stable softmax
    s = scores / temp
    s -= s.max(axis=1, keepdims=True)
    weights = np.exp(s)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-9  # (n_valid, n_tickers)

    # Cap n_valid so we always have n_valid+1 price rows available
    n_valid = min(weights.shape[0], len(prices_np) - lookback - 1)
    weights = weights[:n_valid]
    p = prices_np[lookback: lookback + n_valid + 1]
    log_ret = np.log(np.maximum(p[1:] / np.maximum(p[:-1], 1e-9), 1e-9))

    port_ret = (weights * log_ret).sum(axis=1)  # (n_valid,)
    std = port_ret.std()
    if std < 1e-9 or np.isnan(std):
        return float('-inf')
    return port_ret.mean() / std * (252 ** 0.5)


# ---------------------------------------------------------------------------
# scipy objective and optimizer
# ---------------------------------------------------------------------------

def make_objective(power, prices_np, lookback, n_tail, spread_mask):
    """Return (objective, history_list) for scipy.optimize.minimize."""
    history = []

    def objective(x):
        scale_log_weights = x[:len(ALL_SCALES)]
        log_temperature = x[-1]

        scores = regime_scores_numpy(power, lookback, n_tail, scale_log_weights)
        sharpe = soft_sharpe(scores, log_temperature, prices_np, lookback, spread_mask)

        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = -10.0

        history.append(-sharpe)  # minimizing negative Sharpe
        return -sharpe

    return objective, history


def optimize_scipy(prices, lookback, n_tail, scales,
                   method='L-BFGS-B', max_iter=200,
                   spread_df=None, max_spread=0.02):
    """Optimize scale_weights + temperature with scipy.

    Precomputes the expensive CWT once, then hands the 14-param continuous
    problem to L-BFGS-B (default) or Nelder-Mead.
    """
    n_dates, n_tickers = prices.shape

    print(f'Computing causal CWT ({len(scales)} scales x {n_tickers} tickers '
          f'x {n_dates} dates)...')
    power = causal_cwt_numpy(prices.values, scales, lookback) ** 2
    prices_np = prices.values.astype(np.float64)

    # Spread mask: (n_valid, n_tickers)
    spread_mask = None
    if spread_df is not None:
        n_valid = n_dates - lookback
        spread_mask = spread_df.values[lookback: lookback + n_valid] > max_spread

    objective, history = make_objective(power, prices_np, lookback, n_tail, spread_mask)

    # Initial params: uniform scale weights, temperature=0.5
    x0 = np.zeros(len(scales) + 1)
    x0[-1] = np.log(0.5)

    print(f'Optimizing 14 params with {method} (max_iter={max_iter})...')
    print(f'  Initial Sharpe: {-objective(x0):.4f}')

    pbar = tqdm(total=max_iter, desc=method, unit='eval')
    last_n = [0]

    def callback(x):
        n = len(history)
        pbar.update(n - last_n[0])
        last_n[0] = n
        if history:
            pbar.set_postfix(sharpe=f'{-history[-1]:.4f}')

    opts = {'maxiter': max_iter}
    if method.upper() == 'L-BFGS-B':
        opts['maxfun'] = max_iter * 20
        res = minimize(objective, x0, method='L-BFGS-B',
                       options=opts, callback=callback)
    else:
        opts['maxiter'] = max_iter * 10
        res = minimize(objective, x0, method='Nelder-Mead',
                       options=opts, callback=callback)

    pbar.close()

    best_x = res.x
    best_sharpe = -res.fun
    print(f'  Final Sharpe:   {best_sharpe:.4f}  (converged: {res.success})')

    return best_x, best_sharpe, history


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(best_x, scales, best_sharpe):
    scale_log_w = best_x[:len(scales)]
    log_temp = best_x[-1]

    sw = np.exp(scale_log_w)
    sw /= sw.sum()

    print('\nLearned scale weights:')
    for s, w in zip(scales, sw):
        bar = '█' * max(1, int(w * 50))
        print(f'  scale={s:3d}d  {w:.4f}  {bar}')
    print(f'Temperature: {np.exp(log_temp):.4f}')
    print(f'Best Sharpe: {best_sharpe:.4f}')


def plot_training(history, best_sharpe, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(history)), [-h for h in history], linewidth=0.8)
    ax.axhline(0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_ylabel('Sharpe')
    ax.set_xlabel('Function evaluation')
    ax.set_title(f'Differentiable optimizer convergence  (best Sharpe: {best_sharpe:.4f})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Differentiable regime optimizer via scipy L-BFGS-B')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--lookback', type=int, default=120,
                        help='Historical lookback window in days (default: 120)')
    parser.add_argument('--n-tail', type=int, default=20,
                        help='Recent window size in days (default: 20)')
    parser.add_argument('--method', default='L-BFGS-B',
                        choices=['L-BFGS-B', 'Nelder-Mead'],
                        help='Optimizer (default: L-BFGS-B)')
    parser.add_argument('--max-iter', type=int, default=300,
                        help='Max optimizer iterations (default: 300)')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--max-spread', type=float, default=0.02)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=504,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%}')

    best_x, best_sharpe, history = optimize_scipy(
        prices,
        lookback=args.lookback,
        n_tail=args.n_tail,
        scales=ALL_SCALES,
        method=args.method,
        max_iter=args.max_iter,
        spread_df=spread_df,
        max_spread=args.max_spread,
    )

    print_results(best_x, ALL_SCALES, best_sharpe)

    plot_training(
        history, best_sharpe,
        save_path='Output/backtest-torch-training.png' if args.save else None)
