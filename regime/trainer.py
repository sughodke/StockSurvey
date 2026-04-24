"""JAX + optax Adam training loop for the regime strategy."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm

from regime.cwt import causal_cwt, precompute_windows
from regime.strategy import block_sharpe_with_costs, regime_scores


@dataclass
class TrainResult:
    """Output of `train()`.

    Fields
    ------
    params         : learned {'scale_log_weights', 'log_temperature'}.
    train_history  : per-step train Sharpe (length n_steps).
    val_history    : (step_index, val_sharpe) tuples sampled every 5 steps.
    train_sharpe   : final in-sample annualized Sharpe.
    val_sharpe     : final out-of-sample annualized Sharpe.
    scales         : the scale list that was trained on.
    train_dates    : (start, end) timestamps for the train period.
    val_dates      : (start, end) timestamps for the val period.
    """

    params: dict[str, jax.Array]
    train_history: list[float]
    val_history: list[tuple[int, float]]
    train_sharpe: float
    val_sharpe: float
    scales: list[int]
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]


def train(
    prices: pd.DataFrame,
    spread_df: pd.DataFrame,
    *,
    scales: list[int],
    lookback: int,
    n_tail: int,
    rebal_days: int,
    commission_bps: float,
    max_spread: float,
    n_steps: int,
    learning_rate: float,
    train_frac: float,
) -> TrainResult:
    """Optimize the regime strategy via JAX autograd + optax Adam.

    Splits valid blocks into the first `train_frac` for optimization and
    the remainder as held-out validation. The forward pass runs once per
    Adam step on a small (n_blocks, n_tickers) tensor; the heavy CWT and
    windowing work is done once up-front in numpy.
    """
    n_dates, n_tickers = prices.shape
    prices_np = prices.values.astype(np.float64)

    print(f'Computing causal CWT ({len(scales)} scales x {n_tickers} tickers '
          f'x {n_dates} dates)...')
    coeffs = causal_cwt(prices_np, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)

    print('Precomputing recent/historical windows...')
    recent, historical = precompute_windows(power, lookback, n_tail)

    log_ret = np.zeros_like(prices_np, dtype=np.float32)
    log_ret[:-1] = np.log(
        np.maximum(prices_np[1:] / np.maximum(prices_np[:-1], 1e-9), 1e-9))
    log_ret_valid = log_ret[lookback:]

    liquid = (spread_df.values <= max_spread).astype(np.float32)[lookback:]

    n_valid = n_dates - lookback
    n_blocks = n_valid // rebal_days
    n_used = n_blocks * rebal_days

    recent_rb = recent[:, :n_used][:, ::rebal_days]
    historical_rb = historical[:, :n_used][:, ::rebal_days]
    mask_rb = liquid[:n_used][::rebal_days]
    block_log_ret = log_ret_valid[:n_used].reshape(
        n_blocks, rebal_days, n_tickers).sum(axis=1)

    n_train_blocks = int(train_frac * n_blocks)
    train_start = prices.index[lookback]
    train_end = prices.index[lookback + n_train_blocks * rebal_days - 1]
    val_start = prices.index[lookback + n_train_blocks * rebal_days]
    val_end = prices.index[lookback + n_used - 1]
    print(f'Blocks: {n_blocks} total  '
          f'(train: {n_train_blocks}, val: {n_blocks - n_train_blocks})')
    print(f'Train: {train_start.date()} -> {train_end.date()}')
    print(f'Val:   {val_start.date()} -> {val_end.date()}')

    recent_j = jnp.asarray(recent_rb)
    historical_j = jnp.asarray(historical_rb)
    blr_j = jnp.asarray(block_log_ret)
    mask_j = jnp.asarray(mask_rb)
    commission_frac = commission_bps / 1e4
    train_slc = slice(0, n_train_blocks)
    val_slc = slice(n_train_blocks, n_blocks)

    def _sharpe(params, slc):
        scores = regime_scores(
            recent_j[:, slc], historical_j[:, slc], params['scale_log_weights'])
        return block_sharpe_with_costs(
            scores, params['log_temperature'],
            blr_j[slc], mask_j[slc], rebal_days, commission_frac)

    def _train_neg(p):
        return -_sharpe(p, train_slc)

    def _val(p):
        return _sharpe(p, val_slc)

    value_and_grad = jax.jit(jax.value_and_grad(_train_neg))
    val_sharpe_fn = jax.jit(_val)

    params: dict[str, jax.Array] = {
        'scale_log_weights': jnp.zeros(len(scales), dtype=jnp.float32),
        'log_temperature': jnp.log(jnp.asarray(0.5, dtype=jnp.float32)),
    }
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    print(f'Optimizing {len(scales) + 1} params with JAX Adam '
          f'(n_steps={n_steps}, lr={learning_rate})...')
    init_loss, _ = value_and_grad(params)
    print(f'  Initial   train: {-float(init_loss):+.4f}   '
          f'val: {float(val_sharpe_fn(params)):+.4f}')

    train_hist: list[float] = []
    val_hist: list[tuple[int, float]] = []
    pbar = tqdm(range(n_steps), desc='JAX-Adam', unit='step')
    for step in pbar:
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        train_hist.append(-float(loss))
        if step % 5 == 0 or step == n_steps - 1:
            vs = float(val_sharpe_fn(params))
            val_hist.append((step, vs))
            pbar.set_postfix(tr=f'{train_hist[-1]:+.3f}', val=f'{vs:+.3f}')

    final_train = train_hist[-1]
    final_val = float(val_sharpe_fn(params))
    print(f'  Final     train: {final_train:+.4f}   val: {final_val:+.4f}')

    return TrainResult(
        params=params,
        train_history=train_hist,
        val_history=val_hist,
        train_sharpe=final_train,
        val_sharpe=final_val,
        scales=scales,
        train_dates=(train_start, train_end),
        val_dates=(val_start, val_end),
    )
