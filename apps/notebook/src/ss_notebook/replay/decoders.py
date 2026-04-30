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
    batch_size: int | None = None,
    predict_chunk: int = 32_768,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tiny JAX MLP regressor — `n_layers` hidden ReLU blocks of width
    `hidden`, fit with Adam for `n_steps`. Returns
    `(yhat_full, params_dict)`. `params_dict` is flat numpy arrays
    (`feat_mu`, `feat_sd`, `layer{i}_W`, `layer{i}_b`) suitable for
    `np.savez`; it's everything you need to rerun `forward` from the
    saved file.

    Features are z-normalized with stats from `X_train` only.

    `batch_size=None` is full-batch GD (default — fastest when it fits).
    Set `batch_size=B` for stochastic Adam with batches of size B sampled
    with replacement; needed for large pools where the full feature
    matrix and its activations blow past device memory. Predictions are
    streamed in `predict_chunk`-row pieces regardless of batch_size, so
    the predict pass also stays bounded.
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

    n_train = X_tr.shape[0]
    y_train32 = y_train.astype(np.float32)
    if batch_size is None or batch_size >= n_train:
        X_j = jnp.asarray(X_tr)
        y_j = jnp.asarray(y_train32)
        for _ in range(n_steps):
            params, opt_state, _ = step(params, opt_state, X_j, y_j)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_steps):
            idx = rng.integers(0, n_train, size=batch_size)
            xb = jnp.asarray(X_tr[idx])
            yb = jnp.asarray(y_train32[idx])
            params, opt_state, _ = step(params, opt_state, xb, yb)

    yhat_chunks: list[np.ndarray] = []
    for start in range(0, X_fl.shape[0], predict_chunk):
        chunk = jnp.asarray(X_fl[start:start + predict_chunk])
        yhat_chunks.append(np.asarray(forward(params, chunk)))
    yhat = np.concatenate(yhat_chunks)
    params_dict: dict[str, np.ndarray] = {
        'feat_mu': np.asarray(mu, dtype=np.float32).reshape(-1),
        'feat_sd': np.asarray(sd, dtype=np.float32).reshape(-1),
    }
    for i, (W, b) in enumerate(params):
        params_dict[f'layer{i}_W'] = np.asarray(W, dtype=np.float32)
        params_dict[f'layer{i}_b'] = np.asarray(b, dtype=np.float32)
    return np.asarray(yhat, dtype=np.float64), params_dict


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
    batch_size: int | None = None,
    predict_chunk: int = 32_768,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """1-D CNN over the trailing-window features. Returns
    `(yhat_full, params_dict)`; `params_dict` carries `feat_mu/feat_sd`
    (per `(K, F)` cell), every `conv{i}_W/b`, and the linear head's
    `head_W/head_b` — flat numpy arrays for `np.savez`.

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

    n_train = X_tr.shape[0]
    y_train32 = y_train.astype(np.float32)
    if batch_size is None or batch_size >= n_train:
        Xj = jnp.asarray(X_tr)
        yj = jnp.asarray(y_train32)
        for _ in range(n_steps):
            params, opt_state, _ = step(params, opt_state, Xj, yj)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_steps):
            idx = rng.integers(0, n_train, size=batch_size)
            xb = jnp.asarray(X_tr[idx])
            yb = jnp.asarray(y_train32[idx])
            params, opt_state, _ = step(params, opt_state, xb, yb)

    yhat_chunks: list[np.ndarray] = []
    for start in range(0, X_fl.shape[0], predict_chunk):
        chunk = jnp.asarray(X_fl[start:start + predict_chunk])
        yhat_chunks.append(np.asarray(forward(params, chunk)))
    yhat = np.concatenate(yhat_chunks)
    conv_p, head_p = params
    params_dict: dict[str, np.ndarray] = {
        'feat_mu': np.asarray(mu, dtype=np.float32),
        'feat_sd': np.asarray(sd, dtype=np.float32),
    }
    for i, (W, b) in enumerate(conv_p):
        params_dict[f'conv{i}_W'] = np.asarray(W, dtype=np.float32)
        params_dict[f'conv{i}_b'] = np.asarray(b, dtype=np.float32)
    Wh, bh = head_p
    params_dict['head_W'] = np.asarray(Wh, dtype=np.float32)
    params_dict['head_b'] = np.asarray(bh, dtype=np.float32)
    return np.asarray(yhat, dtype=np.float64), params_dict


def fit_cnn_masked_ae(
    X_train: np.ndarray,
    X_full: np.ndarray,
    *,
    n_channels_per_lag: int,
    hidden: int = 64,
    kernel: int = 5,
    n_layers: int = 2,
    decoder_hidden: int = 256,
    decoder_layers: int = 2,
    n_steps: int = 10_000,
    lr: float = 1e-3,
    seed: int = 0,
    batch_size: int | None = None,
    mask_ratio: float = 0.4,
    predict_chunk: int = 32_768,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Self-supervised pretrain — masked CWT autoencoding.

    Encoder is the same conv stack as `fit_cnn_multihead` (same shape on
    disk so `ss_notebook.scoring.backbone.load_backbone` reads it). A
    small MLP decoder maps the flattened backbone latent
    `(n, K_post * hidden)` back to the full `(n, K * F)` z-normed input.

    Per training step a fresh random mask of shape `(n, K, F)` covers
    `mask_ratio` of the cells in each row. Masked cells are replaced
    with 0 (the z-normed mean) before feeding the encoder; the decoder
    must recover the masked cells from the visible context. Loss is MSE
    over the masked positions only — visible positions don't contribute
    a gradient.

    The encoder doesn't see any per-target supervision, so it has to
    learn whatever statistical structure lets it reconstruct any cell
    from the rest. That includes the multi-scale CWT power dependencies
    that RSI/MACD/vol crudely summarize, plus everything else they
    throw away.

    Returns
    -------
    `(params_dict, train_stats)` where:
      - `params_dict` carries `feat_mu/sd`, `conv{i}_W/b` (the backbone),
        and `head_dec{i}_W/b` (the decoder under the `head_` prefix that
        `load_backbone` strips). Suitable for `np.savez` after
        prefixing with a synthetic target name (`ssl__`).
      - `train_stats` reports final masked + unmasked train MSE
        (post-z-norm units) for diagnostic logging.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f'mask_ratio must be in (0, 1); got {mask_ratio}')
    if X_train.shape[1] % n_channels_per_lag != 0:
        raise ValueError(
            f'feature count {X_train.shape[1]} not divisible by '
            f'{n_channels_per_lag} (channels per lag).')
    K = X_train.shape[1] // n_channels_per_lag
    if K <= kernel * n_layers:
        raise ValueError(
            f'window_cols K={K} too small for {n_layers} conv layers of '
            f'kernel size {kernel} (need K > kernel * n_layers).')
    if decoder_layers < 1:
        raise ValueError(f'decoder_layers must be >= 1, got {decoder_layers}')

    import jax
    import jax.numpy as jnp
    import optax

    F = n_channels_per_lag
    X_tr = X_train.reshape(-1, K, F).astype(np.float32)
    X_fl = X_full.reshape(-1, K, F).astype(np.float32)
    feat_mu = X_tr.mean(axis=0, keepdims=True)
    feat_sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - feat_mu) / feat_sd
    X_fl = (X_fl - feat_mu) / feat_sd

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
    latent_flat = out_K * hidden
    out_flat = K * F
    dec_sizes = [latent_flat] + [decoder_hidden] * (decoder_layers - 1) + [out_flat]
    dec_params: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    for fan_in, fan_out in zip(dec_sizes[:-1], dec_sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(
            sub, (fan_in, fan_out),
            dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out, dtype=jnp.float32)
        dec_params.append((W, b))

    params = (conv_params, dec_params)

    def conv1d(x, W, b):
        return jax.lax.conv_general_dilated(
            x, W,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        ) + b

    def encoder(conv_p, x):
        h = x
        for W, b in conv_p:
            h = jax.nn.relu(conv1d(h, W, b))
        return h.reshape(h.shape[0], -1)

    def decoder(dec_p, z):
        h = z
        for W, b in dec_p[:-1]:
            h = jax.nn.relu(h @ W + b)
        Wf, bf = dec_p[-1]
        return h @ Wf + bf

    def forward_masked(p, x_full, mask):
        """`x_full` is z-normed `(n, K, F)`, `mask` is `(n, K, F)` float
        in {0, 1} where 1 = masked. Encoder sees the masked input;
        decoder predicts the full flattened input."""
        conv_p, dec_p = p
        x_masked = x_full * (1.0 - mask)  # mask token = 0
        z = encoder(conv_p, x_masked)
        return decoder(dec_p, z)

    def masked_mse(p, x_full, mask):
        x_flat = x_full.reshape(x_full.shape[0], -1)
        mask_flat = mask.reshape(mask.shape[0], -1)
        yhat = forward_masked(p, x_full, mask)
        err = (yhat - x_flat) ** 2
        denom = jnp.sum(mask_flat) + 1e-8
        return jnp.sum(err * mask_flat) / denom

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, st, X, key_step):
        m = (jax.random.uniform(key_step, X.shape, dtype=jnp.float32)
             < mask_ratio).astype(jnp.float32)
        loss, grads = jax.value_and_grad(masked_mse)(p, X, m)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    n_train = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    if batch_size is None or batch_size >= n_train:
        Xj = jnp.asarray(X_tr)
        for _ in range(n_steps):
            key, sub = jax.random.split(key)
            params, opt_state, _ = step(params, opt_state, Xj, sub)
    else:
        for _ in range(n_steps):
            idx = rng.integers(0, n_train, size=batch_size)
            Xb = jnp.asarray(X_tr[idx])
            key, sub = jax.random.split(key)
            params, opt_state, _ = step(params, opt_state, Xb, sub)

    # Final diagnostic stats: masked + unmasked recon MSE on X_train.
    @jax.jit
    def eval_unmasked(p, X):
        zero_mask = jnp.zeros_like(X)
        x_flat = X.reshape(X.shape[0], -1)
        yhat = forward_masked(p, X, zero_mask)
        return jnp.mean((yhat - x_flat) ** 2)

    @jax.jit
    def eval_masked(p, X, key_step):
        m = (jax.random.uniform(key_step, X.shape, dtype=jnp.float32)
             < mask_ratio).astype(jnp.float32)
        return masked_mse(p, X, m)

    train_jnp = jnp.asarray(X_tr)
    key, sub = jax.random.split(key)
    train_stats = {
        'train_mse_masked': float(eval_masked(params, train_jnp, sub)),
        'train_mse_unmasked': float(eval_unmasked(params, train_jnp)),
        'mask_ratio': float(mask_ratio),
        'n_train_rows': int(n_train),
        'n_steps': int(n_steps),
    }
    # `X_fl` here is concat(train, val); we surface val stats separately
    # if the caller chose to split. For now we report the pooled MSE so
    # the CLI can subset by ticker offsets.
    if X_fl.shape[0] != X_tr.shape[0]:
        full_jnp = jnp.asarray(X_fl)
        train_stats['full_mse_unmasked'] = float(eval_unmasked(params, full_jnp))

    conv_p_final, dec_p_final = params
    params_dict: dict[str, np.ndarray] = {
        'feat_mu': np.asarray(feat_mu, dtype=np.float32),
        'feat_sd': np.asarray(feat_sd, dtype=np.float32),
    }
    for i, (W, b) in enumerate(conv_p_final):
        params_dict[f'conv{i}_W'] = np.asarray(W, dtype=np.float32)
        params_dict[f'conv{i}_b'] = np.asarray(b, dtype=np.float32)
    # Decoder under `head_` prefix so `load_backbone` skips it (matches
    # the existing per-target head filter convention).
    for i, (W, b) in enumerate(dec_p_final):
        params_dict[f'head_dec{i}_W'] = np.asarray(W, dtype=np.float32)
        params_dict[f'head_dec{i}_b'] = np.asarray(b, dtype=np.float32)
    return params_dict, train_stats


def fit_cnn_multihead(
    X_train: np.ndarray,
    Y_train: dict[str, np.ndarray],
    X_full: np.ndarray,
    *,
    n_channels_per_lag: int,
    hidden: int = 64,
    kernel: int = 5,
    n_layers: int = 2,
    n_steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    batch_size: int | None = None,
    predict_chunk: int = 32_768,
    head_conditioning_train: dict[str, np.ndarray] | None = None,
    head_conditioning_predict: dict[str, np.ndarray] | None = None,
    train_pool_idx: np.ndarray | None = None,
    film_hidden: int = 32,
    frozen_backbone: 'object | None' = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Multi-head 1-D CNN — one shared conv backbone, per-target linear
    heads. Targets are standardized internally (per-target mean/std on
    train) so heads with different output ranges (RSI 0–100, MACD ±10,
    price 10–500) contribute equally to the loss; predictions are
    returned unstandardized.

    For period/parameter-conditioned heads (e.g. RSI(n) trained over
    multiple n), pass `head_conditioning_train[target]` shape
    `(n_train, p_dim)` (typically `p_dim=1`) and the matching
    `head_conditioning_predict[target]` shape `(n_full, p_dim)`.
    Targets without conditioning entries fall back to the latent-only
    head input.

    Conditioned heads use FiLM modulation when `film_hidden > 0`
    (default 32): two 2-layer MLPs take the conditioning vector and
    produce per-latent-feature scale `gamma` and shift `beta`; the
    backbone latent is modulated `gamma * h + beta` before the linear
    head. This gives true latent×cond interaction (each cond value can
    re-weight latent features differently) while keeping the
    `latent -> output` projection linear — same per-ticker memorization
    risk as a plain linear head, since the only path that sees
    per-sample latent stays linear, and the MLPs only ever see the
    cond vector (which carries no ticker info). At init, the last
    layer of both MLPs is zero so `gamma = 1`, `beta = 0` — head is
    identity wrt cond at step 0. With `film_hidden = 0`, falls back
    to the legacy additive-concat path: cond is concatenated to the
    latent and the head's linear weights absorb it (older behavior;
    cannot represent latent×cond interactions).

    `train_pool_idx` enables lazy training-row augmentation. When
    provided, `X_train` is treated as a small *pool* of unique input
    rows and `train_pool_idx[i]` maps logical training row `i` to its
    pool row. `Y_train[t]` and `head_conditioning_train[t]` must have
    length `len(train_pool_idx)` (the logical training set size). Used
    when the same pool row is paired with several target/conditioning
    values (e.g. RSI period grid) — avoids materializing the augmented
    feature matrix, which can be many GB. When `None`, defaults to the
    identity mapping.

    Returns `(yhats, params_per_target)` where:
      - `yhats[target]` is the prediction over `X_full` for that target,
        unstandardized.
      - `params_per_target[target]` is a flat numpy dict with the
        shared input z-norm (`feat_mu/sd`), the shared conv backbone
        (`conv{i}_W/b`), this target's head (`head_W/b`,
        `head_cond_dim`), and this target's output unstandardizer
        (`target_mu/sd`). For conditioned heads, `head_cond_dim`
        records the conditioning width (the head's first
        `head_in - head_cond_dim` weights map the latent; the trailing
        `head_cond_dim` weights map the conditioning vector). Suitable
        for `np.savez`. The shared keys are duplicated across targets
        so each per-target dict is self-contained at inference time —
        callers don't need to know whether a key is shared or per-head.
        All conditioning bookkeeping is under the `head_` prefix so
        downstream loaders that strip per-head artifacts (e.g.
        `ss_notebook.scoring.backbone.load_backbone`) see only the
        shared backbone.
    """
    if X_train.shape[1] % n_channels_per_lag != 0:
        raise ValueError(
            f'feature count {X_train.shape[1]} not divisible by '
            f'{n_channels_per_lag} (channels per lag).')
    K = X_train.shape[1] // n_channels_per_lag
    if K <= kernel * n_layers:
        raise ValueError(
            f'window_cols K={K} too small for {n_layers} conv layers of '
            f'kernel size {kernel} (need K > kernel * n_layers).')
    if not Y_train:
        raise ValueError('fit_cnn_multihead needs at least one target')

    n_pool = X_train.shape[0]
    if train_pool_idx is None:
        pool_idx = np.arange(n_pool, dtype=np.int64)
    else:
        pool_idx = np.asarray(train_pool_idx, dtype=np.int64)
        if pool_idx.ndim != 1:
            raise ValueError(
                f'train_pool_idx must be 1-D; got shape {pool_idx.shape}')
        if pool_idx.size and (pool_idx.min() < 0 or pool_idx.max() >= n_pool):
            raise ValueError(
                f'train_pool_idx values must be in [0, n_pool={n_pool})')
    n_train = pool_idx.size
    for t, y in Y_train.items():
        if y.shape[0] != n_train:
            raise ValueError(
                f'Y_train[{t!r}] has length {y.shape[0]} but logical '
                f'n_train={n_train} (X_train pool size n_pool={n_pool}).')

    cond_train = head_conditioning_train or {}
    cond_predict = head_conditioning_predict or {}
    cond_dim: dict[str, int] = {}
    for t, arr in cond_train.items():
        if t not in Y_train:
            raise ValueError(
                f'head_conditioning_train target {t!r} not in Y_train')
        if arr.ndim != 2 or arr.shape[0] != n_train:
            raise ValueError(
                f'head_conditioning_train[{t!r}] must be (n_train, p_dim); '
                f'got {arr.shape}, n_train={n_train}')
        if t not in cond_predict:
            raise ValueError(
                f'head_conditioning_train[{t!r}] given but no matching '
                f'head_conditioning_predict entry — predictor needs a '
                'parameter to evaluate at.')
        cp = cond_predict[t]
        if cp.ndim != 2 or cp.shape[0] != X_full.shape[0] or \
           cp.shape[1] != arr.shape[1]:
            raise ValueError(
                f'head_conditioning_predict[{t!r}] shape {cp.shape} does '
                f'not match (n_full={X_full.shape[0]}, p_dim={arr.shape[1]}).')
        cond_dim[t] = arr.shape[1]

    import jax
    import jax.numpy as jnp
    import optax

    F = n_channels_per_lag
    X_tr = X_train.reshape(-1, K, F).astype(np.float32)
    X_fl = X_full.reshape(-1, K, F).astype(np.float32)

    # When `frozen_backbone` is given, we use *its* feat_mu/sd and conv
    # weights instead of recomputing/initializing. This keeps the
    # frozen encoder at the exact distribution it saw during pretrain.
    # The backbone forward pass below wraps activations in
    # `jax.lax.stop_gradient`, so gradients don't flow into conv params
    # and adam's updates for them are no-ops (m, v stay at zero
    # forever -> update = 0). Per-target heads + FiLM still train.
    freeze_backbone_flag = frozen_backbone is not None
    if freeze_backbone_flag:
        bb = frozen_backbone
        if bb.K != K:
            raise ValueError(
                f'frozen_backbone K={bb.K} does not match input K={K}')
        if bb.F != F:
            raise ValueError(
                f'frozen_backbone F={bb.F} does not match input F={F}')
        if bb.hidden != hidden:
            raise ValueError(
                f'frozen_backbone hidden={bb.hidden} does not match '
                f'hidden={hidden}')
        if bb.kernel != kernel:
            raise ValueError(
                f'frozen_backbone kernel={bb.kernel} does not match '
                f'kernel={kernel}')
        if bb.n_layers != n_layers:
            raise ValueError(
                f'frozen_backbone n_layers={bb.n_layers} does not match '
                f'n_layers={n_layers}')
        feat_mu = np.asarray(bb.feat_mu, dtype=np.float32)
        feat_sd = np.asarray(bb.feat_sd, dtype=np.float32)
    else:
        feat_mu = X_tr.mean(axis=0, keepdims=True)
        feat_sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - feat_mu) / feat_sd
    X_fl = (X_fl - feat_mu) / feat_sd

    target_names = list(Y_train.keys())
    target_mu = {t: float(np.mean(Y_train[t])) for t in target_names}
    target_sd = {t: float(np.std(Y_train[t]) + 1e-8) for t in target_names}
    Y_std = {t: ((Y_train[t] - target_mu[t]) / target_sd[t]).astype(np.float32)
             for t in target_names}
    cond_tr = {t: arr.astype(np.float32) for t, arr in cond_train.items()}
    cond_fl = {t: arr.astype(np.float32) for t, arr in cond_predict.items()}

    key = jax.random.PRNGKey(seed)
    if freeze_backbone_flag:
        # Reuse the frozen weights verbatim; head/FiLM init still uses
        # `key` below.
        conv_params = [
            (jnp.asarray(W, dtype=jnp.float32),
             jnp.asarray(b, dtype=jnp.float32))
            for W, b in frozen_backbone.conv_params
        ]
    else:
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
    latent_dim = out_K * hidden
    # `use_film[t]` decides per-target whether the conditioning is
    # delivered via FiLM (modulates latent before linear head) or via
    # legacy additive concat (cond appended to latent, absorbed by the
    # head's linear weights). FiLM is on whenever `film_hidden > 0`
    # *and* the head actually has conditioning to apply.
    use_film: dict[str, bool] = {
        t: (film_hidden > 0 and cond_dim.get(t, 0) > 0)
        for t in target_names
    }
    head_params: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}
    for t in target_names:
        # FiLM modulates the latent in-place, so the linear head reads
        # `latent_dim` inputs regardless of cond_dim. The legacy concat
        # path expands the head's input dim by cond_dim.
        head_in = latent_dim if use_film[t] else (
            latent_dim + cond_dim.get(t, 0))
        key, sub = jax.random.split(key)
        Wh = jax.random.normal(
            sub, (head_in, 1),
            dtype=jnp.float32) * jnp.sqrt(2.0 / head_in)
        bh = jnp.zeros(1, dtype=jnp.float32)
        head_params[t] = (Wh, bh)

    # FiLM gamma/beta MLPs: 2-layer ReLU, cond_dim -> film_hidden ->
    # latent_dim. Last layer initialized to zero so gamma starts at 1
    # (we add 1.0 in forward) and beta starts at 0 — identity wrt cond.
    # Each conditioned target gets its own gamma & beta MLPs since
    # different targets generally want different modulation patterns.
    film_params: dict[str, tuple] = {}
    for t in target_names:
        if not use_film[t]:
            continue
        c_dim = cond_dim[t]
        layers = []
        for _which in ('gamma', 'beta'):
            key, sub = jax.random.split(key)
            W0 = jax.random.normal(
                sub, (c_dim, film_hidden),
                dtype=jnp.float32) * jnp.sqrt(2.0 / c_dim)
            b0 = jnp.zeros(film_hidden, dtype=jnp.float32)
            W1 = jnp.zeros((film_hidden, latent_dim), dtype=jnp.float32)
            b1 = jnp.zeros(latent_dim, dtype=jnp.float32)
            layers.append((W0, b0, W1, b1))
        film_params[t] = tuple(layers)  # ((g_W0,g_b0,g_W1,g_b1), (b_W0,...))

    params = (conv_params, head_params, film_params)

    def conv1d(x, W, b):
        return jax.lax.conv_general_dilated(
            x, W,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        ) + b

    def backbone(conv_p, x):
        h = x
        for W, b in conv_p:
            h = jax.nn.relu(conv1d(h, W, b))
        h = h.reshape(h.shape[0], -1)
        if freeze_backbone_flag:
            # Cuts gradient back into conv weights (and into x, but x is
            # also non-trainable input so that's a no-op). Head + FiLM
            # remain fully trainable since they live downstream of `h`.
            h = jax.lax.stop_gradient(h)
        return h

    def _film_mlp(layers, c):
        W0, b0, W1, b1 = layers
        return jax.nn.relu(c @ W0 + b0) @ W1 + b1

    # Note: `cond` is a dict-of-arrays that lives outside `params` — they
    # are training/eval-time *inputs*, not learnable. JAX treats them as
    # part of the traced function's static structure (keys) + dynamic
    # values (arrays), so the per-target film/concat is jit-friendly.
    def forward(p, x, cond):
        conv_p, head_p, film_p = p
        h = backbone(conv_p, x)
        out: dict[str, jax.Array] = {}
        for t, (Wh, bh) in head_p.items():
            if t in film_p:
                g_layers, b_layers = film_p[t]
                gamma = _film_mlp(g_layers, cond[t]) + 1.0
                beta = _film_mlp(b_layers, cond[t])
                h_t = gamma * h + beta
                out[t] = (h_t @ Wh + bh).squeeze(-1)
            elif t in cond:
                h_t = jnp.concatenate([h, cond[t]], axis=-1)
                out[t] = (h_t @ Wh + bh).squeeze(-1)
            else:
                out[t] = (h @ Wh + bh).squeeze(-1)
        return out

    def loss_fn(p, X, Y, C):
        preds = forward(p, X, C)
        per_target = jnp.stack([
            jnp.mean((preds[t] - Y[t]) ** 2) for t in target_names
        ])
        return jnp.mean(per_target)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, st, X, Y, C):
        loss, grads = jax.value_and_grad(loss_fn)(p, X, Y, C)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    # `n_train` here is the *logical* training-row count — equal to
    # `len(pool_idx)`, which is `n_pool * n_replicas` when the caller is
    # augmenting (e.g. RSI period grid). `X_tr` always has only `n_pool`
    # unique rows; we never materialize the augmented feature matrix.
    if batch_size is None or batch_size >= n_train:
        # Full-batch path. Materializes one (n_train, K, F) gather of the
        # pool — for the augmenting cases this is the only large copy and
        # it lives only as long as the loop. If this OOMs the caller
        # should pass `batch_size` to switch to the stochastic path,
        # which never materializes more than `batch_size` augmented rows.
        Xj = jnp.asarray(X_tr[pool_idx])
        Yj = {t: jnp.asarray(Y_std[t]) for t in target_names}
        Cj = {t: jnp.asarray(arr) for t, arr in cond_tr.items()}
        for _ in range(n_steps):
            params, opt_state, _ = step(params, opt_state, Xj, Yj, Cj)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_steps):
            idx = rng.integers(0, n_train, size=batch_size)
            Xb = jnp.asarray(X_tr[pool_idx[idx]])
            Yb = {t: jnp.asarray(Y_std[t][idx]) for t in target_names}
            Cb = {t: jnp.asarray(arr[idx]) for t, arr in cond_tr.items()}
            params, opt_state, _ = step(params, opt_state, Xb, Yb, Cb)

    yhat_chunks: dict[str, list[np.ndarray]] = {t: [] for t in target_names}
    for start in range(0, X_fl.shape[0], predict_chunk):
        chunk = jnp.asarray(X_fl[start:start + predict_chunk])
        chunk_cond = {
            t: jnp.asarray(arr[start:start + predict_chunk])
            for t, arr in cond_fl.items()
        }
        preds = forward(params, chunk, chunk_cond)
        for t in target_names:
            yhat_chunks[t].append(np.asarray(preds[t]))
    yhats: dict[str, np.ndarray] = {}
    for t in target_names:
        yhat_std = np.concatenate(yhat_chunks[t]).astype(np.float64)
        yhats[t] = yhat_std * target_sd[t] + target_mu[t]

    conv_p_final, head_p_final, film_p_final = params
    shared = {
        'feat_mu': np.asarray(feat_mu, dtype=np.float32),
        'feat_sd': np.asarray(feat_sd, dtype=np.float32),
    }
    for i, (W, b) in enumerate(conv_p_final):
        shared[f'conv{i}_W'] = np.asarray(W, dtype=np.float32)
        shared[f'conv{i}_b'] = np.asarray(b, dtype=np.float32)
    params_per_target: dict[str, dict[str, np.ndarray]] = {}
    for t in target_names:
        Wh, bh = head_p_final[t]
        pp = {
            **shared,
            'head_W': np.asarray(Wh, dtype=np.float32),
            'head_b': np.asarray(bh, dtype=np.float32),
            'target_mu': np.array([target_mu[t]], dtype=np.float32),
            'target_sd': np.array([target_sd[t]], dtype=np.float32),
            'head_cond_dim': np.array([cond_dim.get(t, 0)], dtype=np.int32),
        }
        if t in film_p_final:
            (g_layers, b_layers) = film_p_final[t]
            for tag, layers in (('gamma', g_layers), ('beta', b_layers)):
                W0, b0, W1, b1 = layers
                pp[f'head_film_{tag}_W0'] = np.asarray(W0, dtype=np.float32)
                pp[f'head_film_{tag}_b0'] = np.asarray(b0, dtype=np.float32)
                pp[f'head_film_{tag}_W1'] = np.asarray(W1, dtype=np.float32)
                pp[f'head_film_{tag}_b1'] = np.asarray(b1, dtype=np.float32)
            pp['head_film_hidden'] = np.array([film_hidden], dtype=np.int32)
        params_per_target[t] = pp
    return yhats, params_per_target
