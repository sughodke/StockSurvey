"""Tinygrad-Adam training loop for the scoring head.

Pipeline (all heavy work runs once up front, before the Adam loop):
  1. Align list[TickerData] to a common date axis -> AlignedTickers.
  2. Run the frozen backbone over every (date, ticker) feature row ->
     `(D, N, hidden_flat)` representation tensor (kept on host as numpy
     so the head loop can stream Tensor minibatches without retaining
     the full dataset in VRAM).
  3. Build forward log-returns and a liquid mask, subsample both to
     rebalance granularity.
  4. Split rebal bars into train / val by `train_frac`.
  5. Adam loop minimizing `-pearson_rank_ic` on train bars; track val
     IC + val Sharpe (via `block_sharpe`) every 5 steps.

Sharpe is intentionally only an *evaluation* metric — `pearson_rank_ic`
gives a per-rebalance dense gradient signal that converges much
faster than direct Sharpe optimization.

Stage 2 fine-tune drops the precomputed latent (it goes stale once the
backbone unfreezes) and re-runs the encoder on raw features per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad import dtypes
from tinygrad.nn.optim import AdamW

from ss_notebook.replay.features import TickerData
from ss_notebook.scoring.backbone import (
    Backbone, apply_backbone, apply_backbone_pytree, backbone_to_pytree,
)
from ss_notebook.scoring.data import (
    AlignedTickers, align_tickers, forward_log_returns,
)
from ss_notebook.scoring.objectives import block_sharpe, pearson_rank_ic
from ss_notebook.scoring.scorers import get_scorer


@dataclass
class TrainResult:
    """Output of `train_scorer`.

    Fields
    ------
    params              : learned head params dict (numpy arrays).
    scorer              : scorer kind ('linear' or 'mlp').
    train_history       : per-step train rank IC for Stage 1 (head only).
    val_history         : (step, val_ic, val_sharpe) tuples sampled every 5
                          Stage-1 steps.
    finetune_history    : per-step train rank IC for Stage 2. Empty when
                          fine-tuning is disabled.
    finetune_val_history: (step, val_ic, val_sharpe) tuples for Stage 2.
                          Empty when fine-tuning is disabled.
    train_ic            : final in-sample mean rank IC.
    val_ic              : final out-of-sample mean rank IC.
    train_sharpe        : final in-sample annualized Sharpe (eval-only).
    val_sharpe          : final out-of-sample annualized Sharpe (eval-only).
    n_train_bars        : number of rebalance bars used for training.
    n_val_bars          : number of rebalance bars held out for val.
    aligned             : the AlignedTickers used (for inspection / replay).
    backbone_params     : final backbone weights as a numpy-array pytree
                          dict (`feat_mu`, `feat_sd`, `conv`). Identical
                          to the loaded backbone when fine-tuning is
                          disabled, otherwise carries the updated `conv`
                          weights from Stage 2.
    """

    params: dict[str, np.ndarray]
    scorer: str
    train_history: list[float]
    val_history: list[tuple[int, float, float]]
    finetune_history: list[float]
    finetune_val_history: list[tuple[int, float, float]]
    train_ic: float
    val_ic: float
    train_sharpe: float
    val_sharpe: float
    n_train_bars: int
    n_val_bars: int
    aligned: AlignedTickers
    backbone_params: dict


def _params_to_numpy(p: dict[str, Tensor]) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in p.items()}


def precompute_inputs(
    tickers: list[TickerData], backbone: Backbone, *,
    rebal_days: int, max_spread: float | None = None,
    spread: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the frozen backbone over every (date, ticker) and prep tensors.

    Returns a dict with `representation_rb`, `fwd_ret_rb`, `mask_rb` (all
    rebal-subsampled), plus the underlying `aligned` and
    `block_log_ret_rb` for downstream Sharpe eval. All arrays are numpy
    (host-side) — Stage 1 head training pulls Tensor minibatches each
    step. This matches the JAX path's "run encoder once up front" trick
    and keeps the backbone latent off the GPU.

    `spread`, if given, must be aligned to the same dates / tickers as
    the ticker list (caller's responsibility — we do not compute it
    here). Cells with `spread > max_spread` are masked out.
    """
    aligned = align_tickers(tickers, K=backbone.K, F=backbone.F)
    D, N, K, F = aligned.features.shape

    flat = aligned.features.reshape(D * N, K, F).astype(np.float32)
    # Run encoder forward in numpy-out-numpy-in chunks so the latent
    # never stays in VRAM between minibatches. Realize each chunk so the
    # next chunk's allocation can reuse the buffer.
    CHUNK = 8192
    repr_chunks: list[np.ndarray] = []
    Tensor.training = False
    for s in range(0, flat.shape[0], CHUNK):
        x = Tensor(flat[s:s + CHUNK])
        repr_chunks.append(apply_backbone(backbone, x).numpy())
    repr_flat = np.concatenate(repr_chunks, axis=0)
    repr_full = repr_flat.reshape(D, N, backbone.hidden_flat).astype(np.float32)

    fwd_ret = forward_log_returns(aligned.prices, rebal_days=rebal_days)
    daily_log_ret = np.zeros_like(aligned.prices, dtype=np.float64)
    log_p = np.log(np.maximum(aligned.prices, 1e-12))
    daily_log_ret[1:] = log_p[1:] - log_p[:-1]

    base_mask = (aligned.valid & np.isfinite(fwd_ret)
                 & np.isfinite(repr_full).all(axis=-1))
    if spread is not None:
        if spread.shape != aligned.prices.shape:
            raise ValueError(
                f'spread shape {spread.shape} must match prices shape '
                f'{aligned.prices.shape}')
        if max_spread is None:
            raise ValueError('spread provided but max_spread is None')
        base_mask &= spread <= max_spread

    rebal_idx = np.arange(0, D, rebal_days)
    rebal_idx = rebal_idx[rebal_idx + rebal_days < D]
    n_blocks = len(rebal_idx)
    if n_blocks < 4:
        raise ValueError(
            f'only {n_blocks} rebalance blocks fit in {D} aligned dates with '
            f'rebal_days={rebal_days}; need >=4 for a sensible train/val.')

    block_log_ret = np.empty((n_blocks, N), dtype=np.float64)
    for b, i in enumerate(rebal_idx):
        block_log_ret[b] = daily_log_ret[i + 1: i + rebal_days + 1].sum(axis=0)

    repr_rb = np.nan_to_num(repr_full[rebal_idx], nan=0.0).astype(np.float32)
    fwd_rb = np.nan_to_num(fwd_ret[rebal_idx], nan=0.0).astype(np.float32)
    blr_rb = np.nan_to_num(block_log_ret, nan=0.0).astype(np.float32)
    return {
        'aligned': aligned,
        'representation_rb': repr_rb,
        'fwd_ret_rb': fwd_rb,
        'mask_rb': base_mask[rebal_idx].astype(np.float32),
        'block_log_ret_rb': blr_rb,
        'rebal_idx': rebal_idx,
    }


def train_scorer(
    tickers: list[TickerData], backbone: Backbone, *,
    rebal_days: int = 5,
    train_frac: float = 0.7,
    scorer: str = 'linear',
    mlp_hidden: int = 64,
    mlp_layers: int = 1,
    n_steps: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    finetune_steps: int = 0,
    finetune_lr_scale: float = 0.1,
    finetune_batch_bars: int = 8,
    seed: int = 0,
    commission_bps: float = 10.0,
    init_log_temperature: float = 0.0,
    train_temperature: bool = True,
    spread: np.ndarray | None = None,
    max_spread: float | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train a scoring head against rank IC on the (optionally fine-tuned)
    backbone.

    Two-stage training:
      * **Stage 1 (always runs)** — frozen backbone, head-only. The
        backbone forward pass runs once up front in `precompute_inputs`
        (latents materialized on host as numpy) and Tensor minibatches
        of the cached representation are streamed to the head every
        step. Cheap. Loops for `n_steps` Adam updates at
        `learning_rate`.
      * **Stage 2 (optional)** — joint head + backbone. Only runs when
        `finetune_steps > 0`. The cached representation is dropped (it
        goes stale once backbone weights move) and the backbone is
        re-applied to a minibatch of `finetune_batch_bars` rebalance
        bars per step. Backbone gets `learning_rate * finetune_lr_scale`
        (default 0.1×) so the pretrained features are nudged, not
        overwritten. Head and log-temperature stay at full
        `learning_rate`.

    `feat_mu` / `feat_sd` (input z-norm) are *not* optimized in either
    stage — keeping them fixed means the backbone keeps seeing the same
    input distribution it was pretrained on.

    `init_log_temperature=0.0` corresponds to softmax temperature 1.0.
    `train_temperature` lets Adam adjust the softmax temperature used
    inside `block_sharpe`; rank IC itself is scale-invariant so this
    only matters for the Sharpe eval signal.

    `weight_decay > 0` switches the Adam variant to AdamW (decoupled L2
    on params). The 5632-dim flattened latent → 1 linear head is wildly
    under-determined on a few hundred rebalance bars; without decay the
    head memorizes train-cell noise and val IC collapses to ~0. 1e-2 is
    a reasonable starting point. log_temperature gets decayed too but
    that's a benign pull toward the init (softmax temp 1.0).
    """
    pre = precompute_inputs(
        tickers, backbone, rebal_days=rebal_days,
        max_spread=max_spread, spread=spread)
    repr_rb_np = pre['representation_rb']
    fwd_rb_np = pre['fwd_ret_rb']
    mask_rb_np = pre['mask_rb']
    blr_rb_np = pre['block_log_ret_rb']

    n_blocks = repr_rb_np.shape[0]
    n_train = int(train_frac * n_blocks)
    if n_train < 2 or n_blocks - n_train < 2:
        raise ValueError(
            f'train_frac={train_frac} on {n_blocks} blocks gives '
            f'train={n_train} / val={n_blocks - n_train}; need >=2 each.')
    train_slc = slice(0, n_train)
    val_slc = slice(n_train, n_blocks)

    init_fn, apply_fn = get_scorer(scorer)
    rng = np.random.default_rng(seed)
    if scorer == 'mlp':
        head_params = init_fn(
            rng, backbone.hidden_flat, hidden=mlp_hidden, n_layers=mlp_layers)
    else:
        head_params = init_fn(rng, backbone.hidden_flat)

    # `log_temperature` only affects the Sharpe eval (the IC objective is
    # scale-invariant) — its gradient is identically zero either way.
    # The original JAX version put it in optax's state when
    # `train_temperature=True`, but the IC loss never produced a non-zero
    # update for it (Adam state only moved through weight_decay). Keeping
    # it out of the tinygrad optimizer gives the same observable result
    # without the bookkeeping. `train_temperature` stays as a no-op kwarg
    # for caller compatibility.
    _ = train_temperature
    log_temperature = Tensor(np.array(init_log_temperature, dtype=np.float32),
                             requires_grad=False)

    # ----- Stage 1: frozen backbone, head-only Adam loop. -----
    head_param_list = list(head_params.values())
    opt = AdamW(head_param_list, lr=learning_rate, weight_decay=weight_decay)

    def _scores_from_repr(repr_t: Tensor) -> Tensor:
        return apply_fn(head_params, repr_t)

    def _eval_ic(slc: slice) -> float:
        Tensor.training = False
        repr_t = Tensor(repr_rb_np[slc])
        s = _scores_from_repr(repr_t)
        ic = pearson_rank_ic(s, Tensor(fwd_rb_np[slc]),
                             Tensor(mask_rb_np[slc]))
        return float(ic.item())

    def _eval_sharpe(slc: slice) -> float:
        Tensor.training = False
        repr_t = Tensor(repr_rb_np[slc])
        s = _scores_from_repr(repr_t)
        sh = block_sharpe(s, log_temperature, Tensor(blr_rb_np[slc]),
                          Tensor(mask_rb_np[slc]),
                          rebal_days, commission_bps / 1e4)
        return float(sh.item())

    if verbose:
        n_params = sum(int(np.prod(v.shape)) for v in head_param_list)
        print(f'Backbone hidden_flat={backbone.hidden_flat}  '
              f'(K_post={backbone.K_post} x hidden={backbone.hidden})')
        print(f'Head: {scorer}  ({n_params} params)')
        print(f'Rebalance blocks: {n_blocks}  '
              f'(train: {n_train}, val: {n_blocks - n_train})')
        init_ic = _eval_ic(train_slc)
        print(f'  Initial   train IC: {init_ic:+.4f}   '
              f'val IC: {_eval_ic(val_slc):+.4f}   '
              f'val Sharpe: {_eval_sharpe(val_slc):+.3f}')

    train_hist: list[float] = []
    val_hist: list[tuple[int, float, float]] = []
    pbar = tqdm(range(n_steps), desc=f'rank-IC ({scorer}) stage1',
                unit='step', disable=not verbose)
    repr_train_t = Tensor(repr_rb_np[train_slc])
    fwd_train_t = Tensor(fwd_rb_np[train_slc])
    mask_train_t = Tensor(mask_rb_np[train_slc])
    for step in pbar:
        Tensor.training = True
        opt.zero_grad()
        s = _scores_from_repr(repr_train_t)
        loss = -pearson_rank_ic(s, fwd_train_t, mask_train_t)
        loss.backward()
        opt.step()
        train_ic_val = -float(loss.item())
        train_hist.append(train_ic_val)
        if step % 5 == 0 or step == n_steps - 1:
            vi = _eval_ic(val_slc)
            vs = _eval_sharpe(val_slc)
            val_hist.append((step, vi, vs))
            pbar.set_postfix(
                tr_ic=f'{train_ic_val:+.3f}',
                val_ic=f'{vi:+.3f}', val_sh=f'{vs:+.2f}')

    # Pull final head params off device for the result.
    head_params_np = _params_to_numpy(head_params)

    # ----- Stage 2: optional joint head + backbone fine-tune. -----
    bb_pytree = backbone_to_pytree(backbone)   # tinygrad Tensors
    finetune_hist: list[float] = []
    finetune_val_hist: list[tuple[int, float, float]] = []
    if finetune_steps > 0:
        # Drop the precomputed latents — they're stale the moment the
        # backbone moves. Rebuild from raw aligned features per step.
        del repr_train_t   # release VRAM hold on stage-1 cache
        aligned = pre['aligned']
        rebal_idx = pre['rebal_idx']
        feat_rb = np.nan_to_num(aligned.features[rebal_idx],
                                nan=0.0).astype(np.float32)
        K = backbone.K
        F = backbone.F
        N = aligned.features.shape[1]

        # Stage-2 optimizer: backbone at scaled LR, head at full LR.
        # Tinygrad doesn't have optax.multi_transform, so we run two
        # optimizers and `step()` both per training iteration.
        ft_lr_bb = learning_rate * finetune_lr_scale
        bb_conv_params: list[Tensor] = []
        for layer in bb_pytree['conv']:
            bb_conv_params.extend([layer['W'], layer['b']])
        opt_bb = AdamW(bb_conv_params, lr=ft_lr_bb, weight_decay=weight_decay)
        opt_head = AdamW(head_param_list, lr=learning_rate,
                         weight_decay=weight_decay)

        if verbose:
            n_bb_params = sum(int(np.prod(t.shape)) for t in bb_conv_params)
            print(f'  Stage 2 fine-tune: backbone ({n_bb_params} params) '
                  f'unfrozen at lr={ft_lr_bb:.1e}, '
                  f'head at lr={learning_rate:.1e}, '
                  f'batch={finetune_batch_bars} bars / step')

        rng2 = np.random.default_rng(seed + 1)
        train_bars = np.arange(n_train, dtype=np.int64)
        if finetune_batch_bars > n_train:
            finetune_batch_bars = n_train

        def _ft_forward(X_batch: Tensor) -> Tensor:
            B = X_batch.shape[0]
            flat = X_batch.reshape(B * N, K, F)
            repr_flat = apply_backbone_pytree(bb_pytree, flat)
            repr_b = repr_flat.reshape(B, N, backbone.hidden_flat)
            return apply_fn(head_params, repr_b)

        def _ft_eval_ic(slc: slice) -> float:
            Tensor.training = False
            X = Tensor(feat_rb[slc])
            s = _ft_forward(X)
            return float(pearson_rank_ic(
                s, Tensor(fwd_rb_np[slc]), Tensor(mask_rb_np[slc])).item())

        def _ft_eval_sharpe(slc: slice) -> float:
            Tensor.training = False
            X = Tensor(feat_rb[slc])
            s = _ft_forward(X)
            return float(block_sharpe(
                s, log_temperature, Tensor(blr_rb_np[slc]),
                Tensor(mask_rb_np[slc]),
                rebal_days, commission_bps / 1e4).item())

        pbar2 = tqdm(range(finetune_steps),
                     desc=f'rank-IC ({scorer}) stage2',
                     unit='step', disable=not verbose)
        for step in pbar2:
            sel = rng2.choice(train_bars, size=finetune_batch_bars,
                              replace=False)
            Tensor.training = True
            opt_bb.zero_grad()
            opt_head.zero_grad()
            Xb = Tensor(feat_rb[sel])
            s = _ft_forward(Xb)
            loss = -pearson_rank_ic(s, Tensor(fwd_rb_np[sel]),
                                    Tensor(mask_rb_np[sel]))
            loss.backward()
            # Global-norm grad clipping at 1.0 — IC over a small batch is
            # high-variance and Adam's first-moment will blow up otherwise.
            _clip_grads_global_norm(bb_conv_params + head_param_list, 1.0)
            opt_bb.step()
            opt_head.step()
            tic = -float(loss.item())
            finetune_hist.append(tic)
            if step % 5 == 0 or step == finetune_steps - 1:
                vi = _ft_eval_ic(val_slc)
                vs = _ft_eval_sharpe(val_slc)
                finetune_val_hist.append((step, vi, vs))
                pbar2.set_postfix(
                    tr_ic=f'{tic:+.3f}',
                    val_ic=f'{vi:+.3f}', val_sh=f'{vs:+.2f}')

        # Sync `head_params_np` with the fine-tuned head; pull updated
        # backbone conv weights back into the pytree.
        head_params_np = _params_to_numpy(head_params)

    # ----- Final eval -----
    if finetune_steps > 0:
        final_train_ic = _ft_eval_ic(train_slc)
        final_val_ic = _ft_eval_ic(val_slc)
        final_train_sharpe = _ft_eval_sharpe(train_slc)
        final_val_sharpe = _ft_eval_sharpe(val_slc)
    else:
        final_train_ic = float(train_hist[-1]) if train_hist else 0.0
        final_val_ic = _eval_ic(val_slc)
        final_train_sharpe = _eval_sharpe(train_slc)
        final_val_sharpe = _eval_sharpe(val_slc)

    if verbose:
        print(f'  Final     train IC: {final_train_ic:+.4f}   '
              f'val IC: {final_val_ic:+.4f}')
        print(f'            train Sharpe: {final_train_sharpe:+.3f}   '
              f'val Sharpe: {final_val_sharpe:+.3f}')

    # Backbone params back to numpy for the result.
    bb_out = {
        'feat_mu': bb_pytree['feat_mu'].numpy(),
        'feat_sd': bb_pytree['feat_sd'].numpy(),
        'conv': [{'W': layer['W'].numpy(), 'b': layer['b'].numpy()}
                 for layer in bb_pytree['conv']],
    }

    return TrainResult(
        params=head_params_np,
        scorer=scorer,
        train_history=train_hist,
        val_history=val_hist,
        finetune_history=finetune_hist,
        finetune_val_history=finetune_val_hist,
        train_ic=final_train_ic,
        val_ic=final_val_ic,
        train_sharpe=final_train_sharpe,
        val_sharpe=final_val_sharpe,
        n_train_bars=n_train,
        n_val_bars=n_blocks - n_train,
        aligned=pre['aligned'],
        backbone_params=bb_out,
    )


def _clip_grads_global_norm(params: list[Tensor], max_norm: float) -> None:
    """In-place global-norm gradient clip across `params`'s `.grad`s.

    Only clips tensors that have a non-None `.grad`. Realizes the
    scaling factor first so the subsequent grad mutations don't grow a
    massive lazy graph across hundreds of params.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_sq = sum((g * g).sum() for g in grads)
    total_norm = total_sq.sqrt().realize()
    scale = (Tensor([max_norm]) / total_norm.maximum(max_norm)).realize()
    for g in grads:
        g.assign(g * scale)


def predict(
    aligned: AlignedTickers, backbone: Backbone,
    head_params: dict[str, np.ndarray], *,
    scorer: str = 'linear',
) -> np.ndarray:
    """Apply backbone + scoring head over an `AlignedTickers` block.

    Returns scores shape `(D, N)`. NaN where features are NaN. Useful
    for held-out evaluation on a fresh ticker pool without re-running
    the trainer.
    """
    _, apply_fn = get_scorer(scorer)
    D, N, K, F = aligned.features.shape

    # Wrap head_params (numpy) into Tensors for the apply_fn call.
    head_t = {k: Tensor(v) for k, v in head_params.items()}

    Tensor.training = False
    flat = aligned.features.reshape(D * N, K, F).astype(np.float32)
    CHUNK = 8192
    repr_chunks: list[np.ndarray] = []
    for s in range(0, flat.shape[0], CHUNK):
        repr_chunks.append(apply_backbone(backbone, Tensor(flat[s:s + CHUNK])).numpy())
    repr_flat = np.concatenate(repr_chunks, axis=0)
    repr_full = repr_flat.reshape(D, N, backbone.hidden_flat)
    scores = apply_fn(head_t, Tensor(repr_full)).numpy()
    finite = np.isfinite(repr_full).all(axis=-1)
    return np.where(finite, scores, np.nan).astype(np.float64)
