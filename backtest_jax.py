"""
backtest_jax.py : Differentiable regime strategy with JAX autograd.

Optimizes the continuous parameters of the regime-divergence strategy via
analytic gradients (jax.grad) + optax Adam. Matches backtest_bt.py's regime
semantics so reported Sharpe is comparable to the bt baseline:
  - Rebalance every N trading days (weights held constant between rebalances)
  - Soft top-N via temperature-scaled softmax (differentiable relaxation)
  - Transaction cost = commission_bps * L1 turnover / 2 per rebalance
  - Liquidity mask from Corwin-Schultz spread estimate

Learnable parameters (14):
  scale_log_weights (13,) : softmax weights over CWT scales
  log_temperature   (1,)  : softmax concentration

Speed strategy: the CWT + windowed means + block log-returns are all
*independent* of the learnable params, so we precompute them once in numpy.
The JIT'd forward/backward then only touches (n_scales, n_blocks, n_tickers)
tensors where n_blocks = n_valid/rebal_days — ~150 instead of ~3000 rows.

Numerical stability: power is normalized per-ticker so that cumsum over
~3000 dates doesn't blow past float32 precision (the raw CWT on some
high-priced tickers reaches 1e11+, which breaks cumsum in float32).

Honest evaluation: optimizes on the first train_frac of valid dates, reports
both train and held-out val Sharpe.

Usage:
    uv run python backtest_jax.py --data-dir ./Nasdaq3347
    uv run python backtest_jax.py --data-dir ./Nasdaq3347 --lookback 229 --n-tail 106
    uv run python backtest_jax.py --data-dir ./Nasdaq3347 --save
"""

import argparse
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tqdm import tqdm

from backtest_bt import load_price_matrix, corwin_schultz_spread

warnings.filterwarnings('ignore')

ALL_SCALES = [3, 5, 7, 10, 12, 15, 21, 26, 42, 50, 63, 90, 126]
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Causal CWT (numpy — precomputed once, not differentiated through)
# ---------------------------------------------------------------------------

def _ricker_causal(scale, n_dates):
    """One-sided Ricker kernel on t in [-points, 0]."""
    points = min(4 * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale)


def causal_cwt_numpy(prices_np, scales, lookback):
    """Causal CWT: output[i] depends only on input[:i+1]."""
    n_dates, n_tickers = prices_np.shape

    cs = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices_np]), axis=0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices_np ** 2]), axis=0)

    idx = np.arange(n_dates)
    lo = np.maximum(0, idx - lookback + 1)
    counts = idx - lo + 1

    mu = (cs[idx + 1] - cs[lo]) / counts[:, None]
    mu2 = (cs2[idx + 1] - cs2[lo]) / counts[:, None]
    std = np.sqrt(np.maximum(mu2 - mu ** 2, 1e-4))

    x_norm = (prices_np - mu) / std

    coeffs = np.zeros((len(scales), n_dates, n_tickers), dtype=np.float32)
    for si, s in enumerate(scales):
        kernel = _ricker_causal(s, n_dates)
        full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
        coeffs[si] = full[-n_dates:].astype(np.float32)

    return coeffs


def precompute_windows(power, lookback, n_tail):
    """Precompute recent- and historical-window means of power.

    Uses float64 cumsum for precision, returns float32.
    Normalizes per-ticker so cumsum values are O(n_dates), not O(power_max).
    KL divergence is invariant to per-ticker uniform scaling of power.
    """
    n_scales, n_dates, n_tickers = power.shape
    n_valid = n_dates - lookback
    n_hist = lookback - n_tail

    # Per-ticker normalization: mean power across (scales, time) == 1
    pm = power.mean(axis=(0, 1), keepdims=True)
    power = power / np.maximum(pm, 1e-12)

    cs = np.cumsum(power.astype(np.float64), axis=1)
    cs = np.concatenate([
        np.zeros((n_scales, 1, n_tickers), dtype=np.float64),
        cs,
    ], axis=1)

    recent = (cs[:, lookback + 1:, :]
              - cs[:, lookback - n_tail + 1: n_dates - n_tail + 1, :]) / n_tail
    historical = (cs[:, n_hist + 1: n_valid + n_hist + 1, :]
                  - cs[:, :n_valid, :]) / (n_hist + 1)
    return recent.astype(np.float32), historical.astype(np.float32)


# ---------------------------------------------------------------------------
# JAX: differentiable score + portfolio Sharpe (small tensors only)
# ---------------------------------------------------------------------------

def regime_scores_from_windows(recent_rebal, historical_rebal, scale_log_weights):
    """Symmetric KL between weighted recent/historical distributions.

    recent_rebal, historical_rebal: (n_scales, n_blocks, n_tickers)
    scale_log_weights: (n_scales,) pre-softmax
    returns: (n_blocks, n_tickers)
    """
    sw = jax.nn.softmax(scale_log_weights)
    rw = sw[:, None, None] * recent_rebal          # weighted recent
    hw = sw[:, None, None] * historical_rebal

    eps = 1e-9
    rd = rw / (rw.sum(axis=0, keepdims=True) + eps)
    hd = hw / (hw.sum(axis=0, keepdims=True) + eps)

    kl = 0.5 * jnp.sum(rd * jnp.log((rd + eps) / (hd + eps)), axis=0)
    kl += 0.5 * jnp.sum(hd * jnp.log((hd + eps) / (rd + eps)), axis=0)
    return kl


def block_sharpe_with_costs(rebal_scores, log_temperature, block_log_ret,
                             rebal_mask, rebal_days, commission_frac):
    """Block-level portfolio Sharpe with rebalance-boundary transaction costs.

    rebal_scores:    (n_blocks, n_tickers)
    log_temperature: scalar
    block_log_ret:   (n_blocks, n_tickers) sum of daily log returns across the block
    rebal_mask:      (n_blocks, n_tickers)  1=liquid
    rebal_days:      scalar int (used only for annualization factor)
    commission_frac: per-unit turnover cost (10bps = 1e-3)

    Returns annualized Sharpe.  Block return variance * (TRADING_DAYS/rebal_days)
    equals daily variance under iid, giving the same annualized Sharpe.
    """
    temp = jnp.exp(log_temperature)
    s = rebal_scores / temp + jnp.log(rebal_mask + 1e-12)
    s = s - s.max(axis=1, keepdims=True)
    exp_s = jnp.exp(s) * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)  # (n_blocks, n_tickers)

    port_block_ret = (w * block_log_ret).sum(axis=1)  # (n_blocks,)

    # Turnover costs: entry on first rebal, rebalance on subsequent
    init_cost = 0.5 * jnp.abs(w[0]).sum()
    diff_cost = 0.5 * jnp.abs(w[1:] - w[:-1]).sum(axis=1)
    costs = commission_frac * jnp.concatenate([init_cost[None], diff_cost])
    port_block_ret = port_block_ret - costs

    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    # Annualize: block Sharpe * sqrt(blocks per year)
    return mean / std * jnp.sqrt(TRADING_DAYS / rebal_days)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def optimize_jax(prices, spread_df, lookback, n_tail, rebal_days,
                 commission_bps, max_spread, scales,
                 n_steps, learning_rate, train_frac):
    n_dates, n_tickers = prices.shape
    prices_np = prices.values.astype(np.float64)

    print(f'Computing causal CWT ({len(scales)} scales x {n_tickers} tickers '
          f'x {n_dates} dates)...')
    coeffs = causal_cwt_numpy(prices_np, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)

    print('Precomputing recent/historical windows...')
    recent, historical = precompute_windows(power, lookback, n_tail)
    # shape: (n_scales, n_valid, n_tickers)

    # Block returns: sum daily log returns across each rebal_days window
    log_ret_full = np.zeros_like(prices_np, dtype=np.float32)
    log_ret_full[:-1] = np.log(
        np.maximum(prices_np[1:] / np.maximum(prices_np[:-1], 1e-9), 1e-9))
    log_ret_valid = log_ret_full[lookback:]  # (n_valid, n_tickers)

    liquid = (spread_df.values <= max_spread).astype(np.float32)
    liquid_valid = liquid[lookback:]  # (n_valid, n_tickers)

    n_valid = n_dates - lookback
    n_blocks = n_valid // rebal_days
    n_used = n_blocks * rebal_days

    # Subsample to rebalance dates
    recent_rb = recent[:, :n_used][:, ::rebal_days]         # (n_scales, n_blocks, n_tickers)
    historical_rb = historical[:, :n_used][:, ::rebal_days]
    mask_rb = liquid_valid[:n_used][::rebal_days]           # (n_blocks, n_tickers)

    # Block log returns: sum over each rebal_days window
    block_log_ret = log_ret_valid[:n_used].reshape(
        n_blocks, rebal_days, n_tickers).sum(axis=1)         # (n_blocks, n_tickers)

    # Train/val split on blocks
    n_train_blocks = int(train_frac * n_blocks)
    train_end_date = prices.index[lookback + n_train_blocks * rebal_days - 1]
    val_end_date = prices.index[lookback + n_used - 1]
    print(f'Blocks: {n_blocks} total  (train: {n_train_blocks}, '
          f'val: {n_blocks - n_train_blocks})')
    print(f'Train: {prices.index[lookback].date()} -> {train_end_date.date()}')
    print(f'Val:   {train_end_date.date()} -> {val_end_date.date()}')

    # Move to JAX
    recent_j = jnp.asarray(recent_rb)
    historical_j = jnp.asarray(historical_rb)
    blr_j = jnp.asarray(block_log_ret)
    mask_j = jnp.asarray(mask_rb)
    commission_frac = commission_bps / 1e4

    @jax.jit
    def train_loss(params):
        scores = regime_scores_from_windows(
            recent_j[:, :n_train_blocks],
            historical_j[:, :n_train_blocks],
            params['scale_log_weights'])
        return -block_sharpe_with_costs(
            scores, params['log_temperature'],
            blr_j[:n_train_blocks], mask_j[:n_train_blocks],
            rebal_days, commission_frac)

    @jax.jit
    def val_sharpe(params):
        scores = regime_scores_from_windows(
            recent_j[:, n_train_blocks:],
            historical_j[:, n_train_blocks:],
            params['scale_log_weights'])
        return block_sharpe_with_costs(
            scores, params['log_temperature'],
            blr_j[n_train_blocks:], mask_j[n_train_blocks:],
            rebal_days, commission_frac)

    value_and_grad = jax.jit(jax.value_and_grad(train_loss))

    params = {
        'scale_log_weights': jnp.zeros(len(scales), dtype=jnp.float32),
        'log_temperature': jnp.log(jnp.asarray(0.5, dtype=jnp.float32)),
    }

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    print(f'Optimizing {len(scales) + 1} params with JAX Adam '
          f'(n_steps={n_steps}, lr={learning_rate})...')
    init_train = -float(train_loss(params))
    init_val = float(val_sharpe(params))
    print(f'  Initial   train: {init_train:+.4f}   val: {init_val:+.4f}')

    train_hist, val_hist = [], []
    pbar = tqdm(range(n_steps), desc='JAX-Adam', unit='step')
    for step in pbar:
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        train_hist.append(-float(loss))
        if step % 5 == 0 or step == n_steps - 1:
            vs = float(val_sharpe(params))
            val_hist.append((step, vs))
            pbar.set_postfix(tr=f'{train_hist[-1]:+.3f}', val=f'{vs:+.3f}')

    final_train = train_hist[-1]
    final_val = float(val_sharpe(params))
    print(f'  Final     train: {final_train:+.4f}   val: {final_val:+.4f}')

    return params, train_hist, val_hist, final_train, final_val


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(params, scales, train_sharpe, val_sharpe):
    sw = np.asarray(jax.nn.softmax(params['scale_log_weights']))
    temp = float(jnp.exp(params['log_temperature']))

    print('\nLearned scale weights:')
    for s, w in zip(scales, sw):
        bar = '#' * max(1, int(w * 60))
        print(f'  scale={s:3d}d  {w:.4f}  {bar}')
    print(f'Temperature: {temp:.4f}')
    print(f'Train Sharpe: {train_sharpe:+.4f}')
    print(f'Val   Sharpe: {val_sharpe:+.4f}')


def plot_training(train_hist, val_hist, train_sharpe, val_sharpe, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(train_hist)), train_hist,
            color='tab:blue', linewidth=1.0, label='Train Sharpe')
    if val_hist:
        vs_x, vs_y = zip(*val_hist)
        ax.plot(vs_x, vs_y, color='tab:orange', linewidth=1.0,
                label='Val Sharpe (out-of-sample)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_ylabel('Sharpe (ann.)')
    ax.set_xlabel('Adam step')
    ax.set_title(f'JAX differentiable regime optimizer  '
                 f'(train: {train_sharpe:+.3f}  |  val: {val_sharpe:+.3f})')
    ax.legend(loc='best')
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
        description='Differentiable regime optimizer via JAX + optax Adam')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--lookback', type=int, default=120)
    parser.add_argument('--n-tail', type=int, default=20)
    parser.add_argument('--rebal-days', type=int, default=20,
                        help='Rebalance every N trading days (default: 20)')
    parser.add_argument('--commission-bps', type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--train-frac', type=float, default=0.7)
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

    params, train_hist, val_hist, train_sharpe, val_sharpe = optimize_jax(
        prices, spread_df,
        lookback=args.lookback,
        n_tail=args.n_tail,
        rebal_days=args.rebal_days,
        commission_bps=args.commission_bps,
        max_spread=args.max_spread,
        scales=ALL_SCALES,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        train_frac=args.train_frac,
    )

    print_results(params, ALL_SCALES, train_sharpe, val_sharpe)

    plot_training(
        train_hist, val_hist, train_sharpe, val_sharpe,
        save_path='Output/backtest-jax-training.png' if args.save else None)
