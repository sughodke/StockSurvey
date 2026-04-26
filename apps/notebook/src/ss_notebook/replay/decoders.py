"""Decoder fits for the CWT-slice reconstruction probe.

All three signatures are `fit(X_train, y_train, X_full, ...) -> yhat_full`,
where `X_full` is the prediction set (may be the same as `X_train`, or a
concatenated train+val set when running zero-shot evaluation). Train z-norm
stats and OLS weights are derived from `X_train` only — `X_full` never
influences the fit.
"""
from __future__ import annotations

import numpy as np


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns `(weights, bias)` minimizing ‖X w + b − y‖².

    Implemented via `np.linalg.lstsq` on `[X | 1]` rather than the normal
    equations — better-conditioned for ill-scaled features.
    """
    Xb = np.column_stack([X, np.ones(len(X), dtype=X.dtype)])
    sol, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return sol[:-1], float(sol[-1])


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_full: np.ndarray,
    *,
    hidden: int = 128,
    n_layers: int = 2,
    n_steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Tiny JAX MLP regressor — `n_layers` hidden ReLU blocks of width
    `hidden`, fit with Adam for `n_steps`. Returns the prediction over
    `X_full` as numpy, lined up with the input row order.

    Features are z-normalized with stats from `X_train` only.
    """
    import jax
    import jax.numpy as jnp
    import optax

    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_tr = ((X_train - mu) / sd).astype(np.float32)
    X_fl = ((X_full - mu) / sd).astype(np.float32)

    key = jax.random.PRNGKey(seed)
    sizes = [X_tr.shape[1]] + [hidden] * n_layers + [1]
    params: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (fan_in, fan_out),
                              dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out, dtype=jnp.float32)
        params.append((W, b))

    def forward(p, x):
        for W, b in p[:-1]:
            x = jax.nn.relu(x @ W + b)
        W, b = p[-1]
        return (x @ W + b).squeeze(-1)

    def loss_fn(p, X, y):
        return jnp.mean((forward(p, X) - y) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, st, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, X, y)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    X_j = jnp.asarray(X_tr)
    y_j = jnp.asarray(y_train.astype(np.float32))
    for _ in range(n_steps):
        params, opt_state, _ = step(params, opt_state, X_j, y_j)

    yhat = forward(params, jnp.asarray(X_fl))
    return np.asarray(yhat, dtype=np.float64)


def fit_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_full: np.ndarray,
    *,
    n_channels_per_lag: int,
    hidden: int = 64,
    kernel: int = 5,
    n_layers: int = 2,
    n_steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """1-D CNN over the trailing-window features.

    Reshapes X from `(n, K * F)` to `(n, K, F)` (lag-axis = sequence,
    F = `n_channels_per_lag` = 2 * n_scales = channels) and applies
    `n_layers` of Conv1D + ReLU with valid padding. A linear head over
    the flattened activations regresses to the target.

    Weight sharing across lags is the right inductive bias for fixed
    linear filters (RSI/MACD): the optimal mixing of lags is the same on
    every bar, so the model learns it with `kernel * F * hidden` params
    regardless of K, where MLP needs `K * F * hidden`.
    """
    if X_train.shape[1] % n_channels_per_lag != 0:
        raise ValueError(
            f'feature count {X_train.shape[1]} not divisible by '
            f'{n_channels_per_lag} (channels per lag) — pass --window-cols '
            f'> 1 and avoid --include-zscore-stats with --decoder cnn '
            f'(z-score stats are not lag-windowed).')
    K = X_train.shape[1] // n_channels_per_lag
    if K <= kernel * n_layers:
        raise ValueError(
            f'window_cols K={K} too small for {n_layers} conv layers of '
            f'kernel size {kernel} (need K > kernel * n_layers).')

    import jax
    import jax.numpy as jnp
    import optax

    F = n_channels_per_lag
    X_tr = X_train.reshape(-1, K, F).astype(np.float32)
    X_fl = X_full.reshape(-1, K, F).astype(np.float32)

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_fl = (X_fl - mu) / sd

    key = jax.random.PRNGKey(seed)
    chs = [F] + [hidden] * n_layers
    conv_params = []
    for in_c, out_c in zip(chs[:-1], chs[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(
            sub, (kernel, in_c, out_c),
            dtype=jnp.float32) * jnp.sqrt(2.0 / (kernel * in_c))
        b = jnp.zeros(out_c, dtype=jnp.float32)
        conv_params.append((W, b))

    out_K = K - n_layers * (kernel - 1)
    head_in = out_K * hidden
    key, sub = jax.random.split(key)
    W_head = jax.random.normal(
        sub, (head_in, 1),
        dtype=jnp.float32) * jnp.sqrt(2.0 / head_in)
    b_head = jnp.zeros(1, dtype=jnp.float32)
    params = (conv_params, (W_head, b_head))

    def conv1d(x, W, b):
        return jax.lax.conv_general_dilated(
            x, W,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        ) + b

    def forward(p, x):
        conv_p, head_p = p
        h = x
        for W, b in conv_p:
            h = jax.nn.relu(conv1d(h, W, b))
        h = h.reshape(h.shape[0], -1)
        Wh, bh = head_p
        return (h @ Wh + bh).squeeze(-1)

    def loss_fn(p, X, y):
        return jnp.mean((forward(p, X) - y) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, st, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, X, y)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    Xj = jnp.asarray(X_tr)
    yj = jnp.asarray(y_train.astype(np.float32))
    for _ in range(n_steps):
        params, opt_state, _ = step(params, opt_state, Xj, yj)

    yhat = forward(params, jnp.asarray(X_fl))
    return np.asarray(yhat, dtype=np.float64)
