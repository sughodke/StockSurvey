"""Decoder fits for the CWT-slice reconstruction probe.

All three signatures are `fit(X_train, y_train, X_full, ...) -> yhat_full`,
where `X_full` is the prediction set (may be the same as `X_train`, or a
concatenated train+val set when running zero-shot evaluation). Train z-norm
stats and OLS weights are derived from `X_train` only — `X_full` never
influences the fit.

Tinygrad notes
--------------
- `Tensor.training = True` must be set before any optimizer step. We
  toggle it inside each fit and restore the previous value on exit.
- Mixed precision is opt-in via `use_bf16=True` (default). Activations
  and weight reads cast to bf16 in the forward; gradients accumulate in
  fp32 (tinygrad backward keeps the source dtype of the requires_grad
  parameters). Falls back to fp32 cleanly if the backend lacks bf16
  shader support (e.g. Metal on Intel Mac) — pass `use_bf16=False`.
- Microbatching: `microbatch_size` lets a logical `batch_size` be split
  into smaller forward/backward passes whose gradients are averaged
  before the optimizer step. Drops VRAM proportional to the split.
"""
from __future__ import annotations

import contextlib

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.nn.optim import Adam

from ss_tg_ops import conv1d_nhc


# ---- shared utilities -------------------------------------------------------

@contextlib.contextmanager
def _training_mode():
    prev = Tensor.training
    Tensor.training = True
    try:
        yield
    finally:
        Tensor.training = prev


def _maybe_bf16(x: Tensor, use_bf16: bool) -> Tensor:
    return x.cast(dtypes.bfloat16) if use_bf16 else x


def _he_normal_np(rng: np.random.Generator, shape: tuple[int, ...],
                  fan_in: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32) * (2.0 / fan_in) ** 0.5


# ---- public fits ------------------------------------------------------------

def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns `(weights, bias)` minimizing ‖X w + b − y‖².

    Implemented via `np.linalg.lstsq` on `[X | 1]` rather than the normal
    equations — better-conditioned for ill-scaled features. Pure numpy,
    no tinygrad.
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
    use_bf16: bool = True,
    microbatch_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tiny tinygrad MLP regressor — `n_layers` hidden ReLU blocks of
    width `hidden`, fit with Adam for `n_steps`. Returns
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

    `microbatch_size` (default = `batch_size`) splits each step's batch
    for gradient accumulation: forward/backward over `microbatch_size`
    rows at a time, average gradients, single Adam step. Lets you keep
    a large *effective* batch on tight VRAM. `use_bf16=True` (default)
    runs the forward in bf16 for ~2x lower activation memory.
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    X_tr = ((X_train - mu) / sd).astype(np.float32)
    X_fl = ((X_full - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(seed)
    sizes = [X_tr.shape[1]] + [hidden] * n_layers + [1]
    params: list[tuple[Tensor, Tensor]] = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W = Tensor(_he_normal_np(rng, (fan_in, fan_out), fan_in),
                   requires_grad=True)
        b = Tensor(np.zeros(fan_out, dtype=np.float32), requires_grad=True)
        params.append((W, b))

    flat_params = [t for layer in params for t in layer]
    opt = Adam(flat_params, lr=lr)

    def forward(x: Tensor) -> Tensor:
        h = _maybe_bf16(x, use_bf16)
        for W, b in params[:-1]:
            Wc = _maybe_bf16(W, use_bf16)
            bc = _maybe_bf16(b, use_bf16)
            h = (h @ Wc + bc).relu()
        W, b = params[-1]
        Wc = _maybe_bf16(W, use_bf16)
        bc = _maybe_bf16(b, use_bf16)
        h = h @ Wc + bc
        return h.cast(dtypes.float32).squeeze(-1)

    n_train = X_tr.shape[0]
    y_train32 = y_train.astype(np.float32)
    eff_batch = batch_size if batch_size is not None else n_train
    eff_batch = min(eff_batch, n_train)
    micro = microbatch_size if microbatch_size is not None else eff_batch
    micro = max(1, min(micro, eff_batch))

    with _training_mode():
        for _ in range(n_steps):
            if batch_size is None or batch_size >= n_train:
                # Full-batch path: still respect microbatch_size for VRAM.
                idx_full = np.arange(n_train)
            else:
                idx_full = rng.integers(0, n_train, size=eff_batch)
            opt.zero_grad()
            n_micro = max(1, len(idx_full) // micro)
            for mi in range(n_micro):
                sub = idx_full[mi * micro:(mi + 1) * micro]
                if len(sub) == 0:
                    continue
                xb = Tensor(X_tr[sub])
                yb = Tensor(y_train32[sub])
                loss = (forward(xb) - yb).square().mean() / n_micro
                loss.backward()
            opt.step()

    yhat_chunks: list[np.ndarray] = []
    Tensor.training = False
    try:
        for start in range(0, X_fl.shape[0], predict_chunk):
            chunk = Tensor(X_fl[start:start + predict_chunk])
            yhat_chunks.append(forward(chunk).numpy())
    finally:
        pass
    yhat = np.concatenate(yhat_chunks).astype(np.float64)

    params_dict: dict[str, np.ndarray] = {
        'feat_mu': mu.astype(np.float32).reshape(-1),
        'feat_sd': sd.astype(np.float32).reshape(-1),
    }
    for i, (W, b) in enumerate(params):
        params_dict[f'layer{i}_W'] = W.numpy().astype(np.float32)
        params_dict[f'layer{i}_b'] = b.numpy().astype(np.float32)
    return yhat, params_dict


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
    use_bf16: bool = True,
    microbatch_size: int | None = None,
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

    F = n_channels_per_lag
    X_tr = X_train.reshape(-1, K, F).astype(np.float32)
    X_fl = X_full.reshape(-1, K, F).astype(np.float32)

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_fl = (X_fl - mu) / sd

    rng = np.random.default_rng(seed)
    chs = [F] + [hidden] * n_layers
    conv_params: list[tuple[Tensor, Tensor]] = []
    for in_c, out_c in zip(chs[:-1], chs[1:]):
        W = Tensor(_he_normal_np(rng, (kernel, in_c, out_c), kernel * in_c),
                   requires_grad=True)
        b = Tensor(np.zeros(out_c, dtype=np.float32), requires_grad=True)
        conv_params.append((W, b))

    out_K = K - n_layers * (kernel - 1)
    head_in = out_K * hidden
    W_head = Tensor(_he_normal_np(rng, (head_in, 1), head_in),
                    requires_grad=True)
    b_head = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    flat_params = [t for layer in conv_params for t in layer] + [W_head, b_head]
    opt = Adam(flat_params, lr=lr)

    def forward(x: Tensor) -> Tensor:
        h = _maybe_bf16(x, use_bf16)
        for W, b in conv_params:
            Wc = _maybe_bf16(W, use_bf16)
            bc = _maybe_bf16(b, use_bf16)
            h = conv1d_nhc(h, Wc, bc).relu()
        h = h.reshape(h.shape[0], -1)
        Wh = _maybe_bf16(W_head, use_bf16)
        bh = _maybe_bf16(b_head, use_bf16)
        return (h @ Wh + bh).cast(dtypes.float32).squeeze(-1)

    n_train = X_tr.shape[0]
    y_train32 = y_train.astype(np.float32)
    eff_batch = batch_size if batch_size is not None else n_train
    eff_batch = min(eff_batch, n_train)
    micro = microbatch_size if microbatch_size is not None else eff_batch
    micro = max(1, min(micro, eff_batch))

    with _training_mode():
        for _ in range(n_steps):
            if batch_size is None or batch_size >= n_train:
                idx_full = np.arange(n_train)
            else:
                idx_full = rng.integers(0, n_train, size=eff_batch)
            opt.zero_grad()
            n_micro = max(1, len(idx_full) // micro)
            for mi in range(n_micro):
                sub = idx_full[mi * micro:(mi + 1) * micro]
                if len(sub) == 0:
                    continue
                xb = Tensor(X_tr[sub])
                yb = Tensor(y_train32[sub])
                loss = (forward(xb) - yb).square().mean() / n_micro
                loss.backward()
            opt.step()

    yhat_chunks: list[np.ndarray] = []
    Tensor.training = False
    for start in range(0, X_fl.shape[0], predict_chunk):
        chunk = Tensor(X_fl[start:start + predict_chunk])
        yhat_chunks.append(forward(chunk).numpy())
    yhat = np.concatenate(yhat_chunks).astype(np.float64)

    params_dict: dict[str, np.ndarray] = {
        'feat_mu': mu.astype(np.float32),
        'feat_sd': sd.astype(np.float32),
    }
    for i, (W, b) in enumerate(conv_params):
        params_dict[f'conv{i}_W'] = W.numpy().astype(np.float32)
        params_dict[f'conv{i}_b'] = b.numpy().astype(np.float32)
    params_dict['head_W'] = W_head.numpy().astype(np.float32)
    params_dict['head_b'] = b_head.numpy().astype(np.float32)
    return yhat, params_dict


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
    use_bf16: bool = True,
    microbatch_size: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Self-supervised pretrain — masked CWT autoencoding.

    Encoder is the same conv stack as `fit_cnn_multihead` (same shape on
    disk so `ss_features.load_backbone` reads it). A
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

    F = n_channels_per_lag
    X_tr = X_train.reshape(-1, K, F).astype(np.float32)
    X_fl = X_full.reshape(-1, K, F).astype(np.float32)
    feat_mu = X_tr.mean(axis=0, keepdims=True)
    feat_sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - feat_mu) / feat_sd
    X_fl = (X_fl - feat_mu) / feat_sd

    rng = np.random.default_rng(seed)
    chs = [F] + [hidden] * n_layers
    conv_params: list[tuple[Tensor, Tensor]] = []
    for in_c, out_c in zip(chs[:-1], chs[1:]):
        W = Tensor(_he_normal_np(rng, (kernel, in_c, out_c), kernel * in_c),
                   requires_grad=True)
        b = Tensor(np.zeros(out_c, dtype=np.float32), requires_grad=True)
        conv_params.append((W, b))

    out_K = K - n_layers * (kernel - 1)
    latent_flat = out_K * hidden
    out_flat = K * F
    dec_sizes = [latent_flat] + [decoder_hidden] * (decoder_layers - 1) + [out_flat]
    dec_params: list[tuple[Tensor, Tensor]] = []
    for fan_in, fan_out in zip(dec_sizes[:-1], dec_sizes[1:]):
        W = Tensor(_he_normal_np(rng, (fan_in, fan_out), fan_in),
                   requires_grad=True)
        b = Tensor(np.zeros(fan_out, dtype=np.float32), requires_grad=True)
        dec_params.append((W, b))

    flat_params = ([t for layer in conv_params for t in layer]
                   + [t for layer in dec_params for t in layer])
    opt = Adam(flat_params, lr=lr)

    def encoder(x: Tensor) -> Tensor:
        h = _maybe_bf16(x, use_bf16)
        for W, b in conv_params:
            Wc = _maybe_bf16(W, use_bf16)
            bc = _maybe_bf16(b, use_bf16)
            h = conv1d_nhc(h, Wc, bc).relu()
        return h.reshape(h.shape[0], -1)

    def decoder_fwd(z: Tensor) -> Tensor:
        h = z
        for W, b in dec_params[:-1]:
            Wc = _maybe_bf16(W, use_bf16)
            bc = _maybe_bf16(b, use_bf16)
            h = (h @ Wc + bc).relu()
        Wf, bf = dec_params[-1]
        Wfc = _maybe_bf16(Wf, use_bf16)
        bfc = _maybe_bf16(bf, use_bf16)
        return h @ Wfc + bfc

    def masked_recon_loss(x_full_np: np.ndarray, mask_np: np.ndarray) -> Tensor:
        x_t = Tensor(x_full_np)
        mask_t = Tensor(mask_np)
        x_masked = x_t * (1.0 - mask_t)
        z = encoder(x_masked)
        yhat = decoder_fwd(z).cast(dtypes.float32)
        x_flat = x_t.reshape(x_t.shape[0], -1)
        mask_flat = mask_t.reshape(mask_t.shape[0], -1)
        err = (yhat - x_flat).square()
        denom = mask_flat.sum() + 1e-8
        return (err * mask_flat).sum() / denom

    n_train = X_tr.shape[0]
    eff_batch = batch_size if batch_size is not None else n_train
    eff_batch = min(eff_batch, n_train)
    micro = microbatch_size if microbatch_size is not None else eff_batch
    micro = max(1, min(micro, eff_batch))

    with _training_mode():
        for _ in range(n_steps):
            if batch_size is None or batch_size >= n_train:
                idx_full = np.arange(n_train)
            else:
                idx_full = rng.integers(0, n_train, size=eff_batch)
            opt.zero_grad()
            n_micro = max(1, len(idx_full) // micro)
            for mi in range(n_micro):
                sub = idx_full[mi * micro:(mi + 1) * micro]
                if len(sub) == 0:
                    continue
                Xb_np = X_tr[sub]
                m_np = (rng.random(Xb_np.shape) < mask_ratio).astype(np.float32)
                loss = masked_recon_loss(Xb_np, m_np) / n_micro
                loss.backward()
            opt.step()

    Tensor.training = False
    # Final diagnostic stats on training pool (and val pool if X_fl differs).
    def _eval_unmasked(X_np: np.ndarray) -> float:
        z = encoder(Tensor(X_np))
        yhat = decoder_fwd(z).cast(dtypes.float32)
        x_flat = Tensor(X_np).reshape(X_np.shape[0], -1)
        return (yhat - x_flat).square().mean().item()

    def _eval_masked(X_np: np.ndarray) -> float:
        m_np = (rng.random(X_np.shape) < mask_ratio).astype(np.float32)
        return masked_recon_loss(X_np, m_np).item()

    train_stats = {
        'train_mse_masked': float(_eval_masked(X_tr)),
        'train_mse_unmasked': float(_eval_unmasked(X_tr)),
        'mask_ratio': float(mask_ratio),
        'n_train_rows': int(n_train),
        'n_steps': int(n_steps),
    }
    if X_fl.shape[0] != X_tr.shape[0]:
        train_stats['full_mse_unmasked'] = float(_eval_unmasked(X_fl))

    params_dict: dict[str, np.ndarray] = {
        'feat_mu': feat_mu.astype(np.float32),
        'feat_sd': feat_sd.astype(np.float32),
    }
    for i, (W, b) in enumerate(conv_params):
        params_dict[f'conv{i}_W'] = W.numpy().astype(np.float32)
        params_dict[f'conv{i}_b'] = b.numpy().astype(np.float32)
    for i, (W, b) in enumerate(dec_params):
        params_dict[f'head_dec{i}_W'] = W.numpy().astype(np.float32)
        params_dict[f'head_dec{i}_b'] = b.numpy().astype(np.float32)
    return params_dict, train_stats


def fit_cnn_multihead(
    X_train: np.ndarray,
    Y_train: dict[str, np.ndarray],
    X_full: 'np.ndarray | None' = None,
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
    use_bf16: bool = True,
    microbatch_size: int | None = None,
) -> 'tuple[dict[str, np.ndarray] | object, dict[str, dict[str, np.ndarray]]]':
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
    head. At init, the last layer of both MLPs is zero so `gamma = 1`
    (we add 1.0 in forward) and `beta = 0` — head is identity wrt cond
    at step 0. With `film_hidden = 0`, falls back to legacy
    additive-concat.

    `train_pool_idx` enables lazy training-row augmentation. When
    provided, `X_train` is treated as a small *pool* of unique input
    rows and `train_pool_idx[i]` maps logical training row `i` to its
    pool row. `Y_train[t]` and `head_conditioning_train[t]` must have
    length `len(train_pool_idx)` (the logical training set size).

    VRAM optimization (tinygrad port)
    ---------------------------------
    Even in full-batch mode, we never materialize `X_train[pool_idx]`
    on device — we sample minibatches every step. The default
    `batch_size=None` is now interpreted as "stream the full
    augmented logical set across `n_micro` minibatches per step,"
    not "load it all at once." This eliminates the O(n_pool ×
    |grid|) replicated tile that crashed the JAX path on Steam Deck
    iGPU. Set `batch_size=B` to switch to stochastic batches.
    `microbatch_size` further splits each batch for gradient
    accumulation.

    Returns `(yhats, params_per_target)` matching the original layout
    when `X_full` is given. When `X_full=None`, training runs without
    a single-batch predict pass and the first return is a `predict_fn`
    closure: `predict_fn(X_chunk, cond_chunk_dict) -> dict[target,
    np.ndarray]` — caller is then expected to drive prediction
    ticker-by-ticker (or whatever chunking they want) so the
    full-feature concatenation never has to live in memory at once.

    **Ownership: this function consumes `X_train`.** Feature
    standardization is applied in place against the caller's buffer
    (a view through `reshape(-1, K, F)`); after this function returns
    the caller's `X_train` is normalized and should not be reused.
    The caller should `del X_train` immediately after the call. If
    the input is not float32 + C-contiguous we fall back to a copy
    rather than mutate.

    See npz schema in `replay/README.md` "Outputs".
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
    streaming_predict = X_full is None
    n_full = 0 if streaming_predict else X_full.shape[0]
    for t, arr in cond_train.items():
        if t not in Y_train:
            raise ValueError(
                f'head_conditioning_train target {t!r} not in Y_train')
        if arr.ndim != 2 or arr.shape[0] != n_train:
            raise ValueError(
                f'head_conditioning_train[{t!r}] must be (n_train, p_dim); '
                f'got {arr.shape}, n_train={n_train}')
        if streaming_predict:
            # Caller drives predict; cond_predict shape is per-chunk and
            # validated inside the predict_fn closure rather than here.
            if t in cond_predict:
                cp = cond_predict[t]
                if cp.ndim != 2 or cp.shape[1] != arr.shape[1]:
                    raise ValueError(
                        f'head_conditioning_predict[{t!r}] shape {cp.shape} '
                        f'must be (n_chunk, p_dim={arr.shape[1]}) when '
                        'X_full=None (streaming predict)')
        else:
            if t not in cond_predict:
                raise ValueError(
                    f'head_conditioning_train[{t!r}] given but no matching '
                    f'head_conditioning_predict entry — predictor needs a '
                    'parameter to evaluate at.')
            cp = cond_predict[t]
            if cp.ndim != 2 or cp.shape[0] != n_full or \
               cp.shape[1] != arr.shape[1]:
                raise ValueError(
                    f'head_conditioning_predict[{t!r}] shape {cp.shape} does '
                    f'not match (n_full={n_full}, p_dim={arr.shape[1]}).')
        cond_dim[t] = arr.shape[1]

    F = n_channels_per_lag
    # Take ownership of `X_train`: at full-pool scale (~300 tickers,
    # K=96, 7 channels per scale) duplicating the training matrix
    # alongside the per-ticker `.features` arrays the caller still
    # holds blows past Modal T4 memory ceilings. We normalize in place
    # via a `(n, K, F)` view of the caller's buffer — the caller is
    # expected to `del X_train` immediately after this function returns
    # (see `reconstruct.fit_and_evaluate`). If the caller hands us a
    # non-float32 or non-contiguous buffer we fall back to a defensive
    # copy.
    X_tr_pool = X_train.reshape(-1, K, F)
    if X_tr_pool.dtype != np.float32 or not X_tr_pool.flags.c_contiguous:
        X_tr_pool = X_tr_pool.astype(np.float32)
    if streaming_predict:
        X_fl = None
    else:
        X_fl = X_full.reshape(-1, K, F).astype(np.float32)

    # Backbone reuse (frozen) — load weights, hold them constant.
    freeze_backbone_flag = frozen_backbone is not None
    if freeze_backbone_flag:
        bb = frozen_backbone
        if bb.K != K:
            raise ValueError(f'frozen_backbone K={bb.K} != input K={K}')
        if bb.F != F:
            raise ValueError(f'frozen_backbone F={bb.F} != input F={F}')
        if bb.hidden != hidden:
            raise ValueError(
                f'frozen_backbone hidden={bb.hidden} != hidden={hidden}')
        if bb.kernel != kernel:
            raise ValueError(
                f'frozen_backbone kernel={bb.kernel} != kernel={kernel}')
        if bb.n_layers != n_layers:
            raise ValueError(
                f'frozen_backbone n_layers={bb.n_layers} != n_layers={n_layers}')
        feat_mu_np = np.asarray(bb.feat_mu, dtype=np.float32)
        feat_sd_np = np.asarray(bb.feat_sd, dtype=np.float32)
    else:
        feat_mu_np = X_tr_pool.mean(axis=0, keepdims=True)
        feat_sd_np = X_tr_pool.std(axis=0, keepdims=True) + 1e-8
    # In-place normalization: `(a - mu) / sd` would allocate two ~60 GB
    # temporaries at full-pool scale; `-=` and `/=` broadcast write
    # against the existing buffer, so peak stays at one X_tr_pool size.
    X_tr_pool -= feat_mu_np
    X_tr_pool /= feat_sd_np
    # X_fl is normalized inside `_predict_chunk` (defined below) — do
    # NOT pre-normalize here, or the predict path double-applies the
    # transform.

    target_names = list(Y_train.keys())
    target_mu = {t: float(np.mean(Y_train[t])) for t in target_names}
    target_sd = {t: float(np.std(Y_train[t]) + 1e-8) for t in target_names}
    Y_std = {t: ((Y_train[t] - target_mu[t]) / target_sd[t]).astype(np.float32)
             for t in target_names}
    cond_tr = {t: arr.astype(np.float32) for t, arr in cond_train.items()}
    if streaming_predict:
        cond_fl = None
    else:
        cond_fl = {t: arr.astype(np.float32) for t, arr in cond_predict.items()}

    rng = np.random.default_rng(seed)
    if freeze_backbone_flag:
        # Backbone stays frozen — `requires_grad=False` so it doesn't
        # enter the optimizer's parameter list and the loss never
        # backprops into it.
        conv_params = [
            (Tensor(np.asarray(W, dtype=np.float32), requires_grad=False),
             Tensor(np.asarray(b, dtype=np.float32), requires_grad=False))
            for W, b in frozen_backbone.conv_params
        ]
    else:
        chs = [F] + [hidden] * n_layers
        conv_params = []
        for in_c, out_c in zip(chs[:-1], chs[1:]):
            W = Tensor(_he_normal_np(rng, (kernel, in_c, out_c),
                                     kernel * in_c),
                       requires_grad=True)
            b = Tensor(np.zeros(out_c, dtype=np.float32), requires_grad=True)
            conv_params.append((W, b))

    out_K = K - n_layers * (kernel - 1)
    latent_dim = out_K * hidden
    use_film: dict[str, bool] = {
        t: (film_hidden > 0 and cond_dim.get(t, 0) > 0)
        for t in target_names
    }
    head_params: dict[str, tuple[Tensor, Tensor]] = {}
    for t in target_names:
        head_in = latent_dim if use_film[t] else (
            latent_dim + cond_dim.get(t, 0))
        Wh = Tensor(_he_normal_np(rng, (head_in, 1), head_in),
                    requires_grad=True)
        bh = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
        head_params[t] = (Wh, bh)

    # FiLM gamma/beta MLPs — last layer initialized to zero so gamma
    # starts at 1 (we add 1.0 in forward) and beta at 0 — head is
    # identity wrt cond at step 0.
    film_params: dict[str, tuple] = {}
    for t in target_names:
        if not use_film[t]:
            continue
        c_dim = cond_dim[t]
        layers_list: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for _which in ('gamma', 'beta'):
            W0 = Tensor(_he_normal_np(rng, (c_dim, film_hidden), c_dim),
                        requires_grad=True)
            b0 = Tensor(np.zeros(film_hidden, dtype=np.float32),
                        requires_grad=True)
            W1 = Tensor(np.zeros((film_hidden, latent_dim), dtype=np.float32),
                        requires_grad=True)
            b1 = Tensor(np.zeros(latent_dim, dtype=np.float32),
                        requires_grad=True)
            layers_list.append((W0, b0, W1, b1))
        film_params[t] = tuple(layers_list)

    # Trainable params for optimizer.
    params_for_opt: list[Tensor] = []
    if not freeze_backbone_flag:
        for W, b in conv_params:
            params_for_opt.extend([W, b])
    for t in target_names:
        Wh, bh = head_params[t]
        params_for_opt.extend([Wh, bh])
    for t, layers in film_params.items():
        for W0, b0, W1, b1 in layers:
            params_for_opt.extend([W0, b0, W1, b1])
    opt = Adam(params_for_opt, lr=lr)

    def backbone(x: Tensor) -> Tensor:
        h = _maybe_bf16(x, use_bf16)
        for W, b in conv_params:
            Wc = _maybe_bf16(W, use_bf16)
            bc = _maybe_bf16(b, use_bf16)
            h = conv1d_nhc(h, Wc, bc).relu()
        h = h.reshape(h.shape[0], -1)
        if freeze_backbone_flag:
            h = h.detach()
        return h

    def _film_mlp(layers, c: Tensor) -> Tensor:
        W0, b0, W1, b1 = layers
        W0c = _maybe_bf16(W0, use_bf16)
        b0c = _maybe_bf16(b0, use_bf16)
        W1c = _maybe_bf16(W1, use_bf16)
        b1c = _maybe_bf16(b1, use_bf16)
        return ((c @ W0c + b0c).relu()) @ W1c + b1c

    def forward(x: Tensor, cond: dict[str, Tensor]) -> dict[str, Tensor]:
        h = backbone(x)
        out: dict[str, Tensor] = {}
        for t, (Wh, bh) in head_params.items():
            Whc = _maybe_bf16(Wh, use_bf16)
            bhc = _maybe_bf16(bh, use_bf16)
            if t in film_params:
                g_layers, b_layers = film_params[t]
                c = _maybe_bf16(cond[t], use_bf16)
                gamma = _film_mlp(g_layers, c) + 1.0
                beta = _film_mlp(b_layers, c)
                h_t = gamma * h + beta
                out[t] = (h_t @ Whc + bhc).cast(dtypes.float32).squeeze(-1)
            elif t in cond:
                c = _maybe_bf16(cond[t], use_bf16)
                h_t = h.cat(c, dim=-1)
                out[t] = (h_t @ Whc + bhc).cast(dtypes.float32).squeeze(-1)
            else:
                out[t] = (h @ Whc + bhc).cast(dtypes.float32).squeeze(-1)
        return out

    eff_batch = batch_size if batch_size is not None else n_train
    eff_batch = min(eff_batch, n_train)
    micro = microbatch_size if microbatch_size is not None else eff_batch
    micro = max(1, min(micro, eff_batch))

    with _training_mode():
        for _ in range(n_steps):
            if batch_size is None or batch_size >= n_train:
                idx_full = np.arange(n_train)
                rng.shuffle(idx_full)
            else:
                idx_full = rng.integers(0, n_train, size=eff_batch)
            opt.zero_grad()
            n_micro = max(1, len(idx_full) // micro)
            for mi in range(n_micro):
                sub = idx_full[mi * micro:(mi + 1) * micro]
                if len(sub) == 0:
                    continue
                # Stream gather: pull only the unique pool rows referenced
                # by `pool_idx[sub]` — never materialize the full
                # n_pool × n_replicas × K × F tile.
                pool_sel = pool_idx[sub]
                Xb = Tensor(X_tr_pool[pool_sel])
                Cb = {t: Tensor(cond_tr[t][sub]) for t in cond_tr}
                preds = forward(Xb, Cb)
                per_target = []
                for t in target_names:
                    yt = Tensor(Y_std[t][sub])
                    per_target.append((preds[t] - yt).square().mean())
                loss = sum(per_target) / len(per_target) / n_micro
                loss.backward()
            opt.step()

    Tensor.training = False

    def _predict_chunk(
        X_in: np.ndarray, cond_in: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the trained model on one chunk of inputs.

        `X_in` is `(n, K * F)` or `(n, K, F)` raw-feature float32; will
        be normalized internally via `feat_mu_np / feat_sd_np`.
        `cond_in` follows the same shape convention as the saved
        `cond_predict` (per-target `(n, p_dim)`); pass `None` for
        unconditioned-only runs.
        Output dict carries unstandardized predictions per target.
        """
        cond_in = cond_in or {}
        # Every conditioned target must get a cond block in this chunk —
        # otherwise the FiLM forward pass below will KeyError deep
        # inside tinygrad. Catch the omission up front with a clear msg.
        missing = [t for t in cond_dim if t not in cond_in]
        if missing:
            raise ValueError(
                f'predict cond missing target(s) {missing!r}; the trained '
                f'model has FiLM heads on {sorted(cond_dim)} and each '
                'requires a per-chunk conditioning array')
        X_arr = np.asarray(X_in)
        if X_arr.ndim == 2:
            X_arr = X_arr.reshape(-1, K, F)
        X_arr = X_arr.astype(np.float32, copy=False)
        X_arr = (X_arr - feat_mu_np) / feat_sd_np
        cond_arrs = {
            t: np.asarray(cond_in[t], dtype=np.float32)
            for t in cond_in if t in cond_dim
        }
        for t, arr in cond_arrs.items():
            if arr.ndim != 2 or arr.shape[1] != cond_dim[t]:
                raise ValueError(
                    f'predict cond[{t!r}] must be (n, p_dim={cond_dim[t]}); '
                    f'got {arr.shape}')
            if arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f'predict cond[{t!r}] rows {arr.shape[0]} != X rows '
                    f'{X_arr.shape[0]}')
        out_chunks: dict[str, list[np.ndarray]] = {t: [] for t in target_names}
        for start in range(0, X_arr.shape[0], predict_chunk):
            stop = start + predict_chunk
            chunk = Tensor(X_arr[start:stop])
            chunk_cond = {
                t: Tensor(arr[start:stop]) for t, arr in cond_arrs.items()
            }
            preds = forward(chunk, chunk_cond)
            for t in target_names:
                out_chunks[t].append(preds[t].numpy())
        out: dict[str, np.ndarray] = {}
        for t in target_names:
            yhat_std = np.concatenate(out_chunks[t]).astype(np.float64)
            out[t] = yhat_std * target_sd[t] + target_mu[t]
        return out

    yhats: 'dict[str, np.ndarray] | object'
    if streaming_predict:
        # Caller drives prediction ticker-by-ticker.
        yhats = _predict_chunk
    else:
        yhats = _predict_chunk(X_fl, cond_fl)

    shared = {
        'feat_mu': feat_mu_np.astype(np.float32),
        'feat_sd': feat_sd_np.astype(np.float32),
    }
    for i, (W, b) in enumerate(conv_params):
        shared[f'conv{i}_W'] = W.numpy().astype(np.float32)
        shared[f'conv{i}_b'] = b.numpy().astype(np.float32)
    params_per_target: dict[str, dict[str, np.ndarray]] = {}
    for t in target_names:
        Wh, bh = head_params[t]
        pp = {
            **shared,
            'head_W': Wh.numpy().astype(np.float32),
            'head_b': bh.numpy().astype(np.float32),
            'target_mu': np.array([target_mu[t]], dtype=np.float32),
            'target_sd': np.array([target_sd[t]], dtype=np.float32),
            'head_cond_dim': np.array([cond_dim.get(t, 0)], dtype=np.int32),
        }
        if t in film_params:
            (g_layers, b_layers) = film_params[t]
            for tag, layers in (('gamma', g_layers), ('beta', b_layers)):
                W0, b0, W1, b1 = layers
                pp[f'head_film_{tag}_W0'] = W0.numpy().astype(np.float32)
                pp[f'head_film_{tag}_b0'] = b0.numpy().astype(np.float32)
                pp[f'head_film_{tag}_W1'] = W1.numpy().astype(np.float32)
                pp[f'head_film_{tag}_b1'] = b1.numpy().astype(np.float32)
            pp['head_film_hidden'] = np.array([film_hidden], dtype=np.int32)
        params_per_target[t] = pp
    return yhats, params_per_target
