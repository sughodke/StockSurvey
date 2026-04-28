"""JAX-Adam training loop for the scoring head.

Pipeline (all heavy work runs once up front, before the Adam loop):
  1. Align list[TickerData] to a common date axis -> AlignedTickers.
  2. Run the frozen backbone over every (date, ticker) feature row ->
     `(D, N, hidden_flat)` representation tensor.
  3. Build forward log-returns and a liquid mask, subsample both to
     rebalance granularity.
  4. Split rebal bars into train / val by `train_frac`.
  5. Adam loop minimizing `-pearson_rank_ic` on train bars; track val
     IC + val Sharpe (via `block_sharpe_with_costs`) every 5 steps.

Sharpe is intentionally only an *evaluation* metric — `pearson_rank_ic`
gives a per-rebalance dense gradient signal that converges much
faster than direct Sharpe optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

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
    params              : learned head params dict.
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
    backbone_params     : final backbone weights as a pytree dict (`feat_mu`,
                          `feat_sd`, `conv`). Identical to the loaded backbone
                          when fine-tuning is disabled, otherwise carries the
                          updated `conv` weights from Stage 2.
    """

    params: dict[str, jax.Array]
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


def precompute_inputs(
    tickers: list[TickerData], backbone: Backbone, *,
    rebal_days: int, max_spread: float | None = None,
    spread: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the frozen backbone over every (date, ticker) and prep tensors.

    Returns a dict with `representation`, `fwd_ret`, `mask` (all
    rebal-subsampled), plus the underlying `aligned` and `block_log_ret`
    in *daily* form for downstream Sharpe eval.

    `spread`, if given, must be aligned to the same dates / tickers as
    the ticker list (caller's responsibility — we do not compute it
    here). Cells with `spread > max_spread` are masked out.
    """
    aligned = align_tickers(tickers, K=backbone.K, F=backbone.F)
    D, N, K, F = aligned.features.shape

    flat = aligned.features.reshape(D * N, K, F)
    repr_flat = np.asarray(apply_backbone(backbone, jnp.asarray(flat)))
    repr_full = repr_flat.reshape(D, N, backbone.hidden_flat).astype(np.float32)

    fwd_ret = forward_log_returns(aligned.prices, rebal_days=rebal_days)
    daily_log_ret = np.zeros_like(aligned.prices, dtype=np.float64)
    log_p = np.log(np.maximum(aligned.prices, 1e-12))
    daily_log_ret[1:] = log_p[1:] - log_p[:-1]

    base_mask = aligned.valid & np.isfinite(fwd_ret) & np.isfinite(repr_full).all(axis=-1)
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
        and the cached representation tensor is reused every step. Cheap.
        Loops for `n_steps` Adam updates at `learning_rate`. This gives
        the head a good warm-start before any backbone weights move.
      * **Stage 2 (optional)** — joint head + backbone. Only runs when
        `finetune_steps > 0`. Backbone weights enter the gradient; per
        step the backbone is re-applied to a minibatch of
        `finetune_batch_bars` rebalance bars (re-running the backbone
        every step is the cost of letting it learn). Backbone gets
        `learning_rate * finetune_lr_scale` (default 0.1×) so the
        pretrained features are nudged, not overwritten. Head and
        log-temperature stay at full `learning_rate`.

    `feat_mu` / `feat_sd` (input z-norm) are *not* optimized in either
    stage — keeping them fixed means the backbone keeps seeing the same
    input distribution it was pretrained on.

    `init_log_temperature=0.0` corresponds to softmax temperature 1.0.
    `train_temperature` lets Adam adjust the softmax temperature used
    inside `block_sharpe`; rank IC itself is scale-invariant so this
    only matters for the Sharpe eval signal.
    """
    pre = precompute_inputs(
        tickers, backbone, rebal_days=rebal_days,
        max_spread=max_spread, spread=spread)
    repr_rb = jnp.asarray(pre['representation_rb'])
    fwd_rb = jnp.asarray(pre['fwd_ret_rb'])
    mask_rb = jnp.asarray(pre['mask_rb'])
    blr_rb = jnp.asarray(pre['block_log_ret_rb'])

    n_blocks = repr_rb.shape[0]
    n_train = int(train_frac * n_blocks)
    if n_train < 2 or n_blocks - n_train < 2:
        raise ValueError(
            f'train_frac={train_frac} on {n_blocks} blocks gives '
            f'train={n_train} / val={n_blocks - n_train}; need >=2 each.')
    train_slc = slice(0, n_train)
    val_slc = slice(n_train, n_blocks)

    init_fn, apply_fn = get_scorer(scorer)
    key = jax.random.PRNGKey(seed)
    if scorer == 'mlp':
        head_params = init_fn(
            key, backbone.hidden_flat, hidden=mlp_hidden, n_layers=mlp_layers)
    else:
        head_params = init_fn(key, backbone.hidden_flat)

    params: dict = {
        'head': head_params,
        'log_temperature': jnp.asarray(init_log_temperature, dtype=jnp.float32),
    }
    commission_frac = commission_bps / 1e4

    # ----- Stage 1: frozen backbone, head-only Adam loop. -----
    def _scores(params, repr_slc):
        return apply_fn(params['head'], repr_slc)

    def _ic(params, slc):
        s = _scores(params, repr_rb[slc])
        return pearson_rank_ic(s, fwd_rb[slc], mask_rb[slc])

    def _sharpe(params, slc):
        s = _scores(params, repr_rb[slc])
        return block_sharpe(
            s, params['log_temperature'],
            blr_rb[slc], mask_rb[slc], rebal_days, commission_frac)

    def _train_neg(p):
        return -_ic(p, train_slc)

    value_and_grad = jax.jit(jax.value_and_grad(_train_neg))
    val_ic_fn = jax.jit(lambda p: _ic(p, val_slc))
    val_sharpe_fn = jax.jit(lambda p: _sharpe(p, val_slc))
    train_sharpe_fn = jax.jit(lambda p: _sharpe(p, train_slc))

    if not train_temperature:
        labels = jax.tree_util.tree_map(lambda _: 'head', params['head'])
        labels = {'head': labels, 'log_temperature': 'frozen'}
        optimizer = optax.multi_transform(
            {'head': optax.adam(learning_rate),
             'frozen': optax.set_to_zero()},
            labels,
        )
    else:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    if verbose:
        n_params = sum(int(np.prod(v.shape))
                       for v in jax.tree_util.tree_leaves(head_params))
        print(f'Backbone hidden_flat={backbone.hidden_flat}  '
              f'(K_post={backbone.K_post} x hidden={backbone.hidden})')
        print(f'Head: {scorer}  ({n_params} params)')
        print(f'Rebalance blocks: {n_blocks}  '
              f'(train: {n_train}, val: {n_blocks - n_train})')
        init_loss, _ = value_and_grad(params)
        print(f'  Initial   train IC: {-float(init_loss):+.4f}   '
              f'val IC: {float(val_ic_fn(params)):+.4f}   '
              f'val Sharpe: {float(val_sharpe_fn(params)):+.3f}')

    train_hist: list[float] = []
    val_hist: list[tuple[int, float, float]] = []
    pbar = tqdm(range(n_steps), desc=f'rank-IC ({scorer}) stage1',
                unit='step', disable=not verbose)
    for step in pbar:
        loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        train_hist.append(-float(loss))
        if step % 5 == 0 or step == n_steps - 1:
            vi = float(val_ic_fn(params))
            vs = float(val_sharpe_fn(params))
            val_hist.append((step, vi, vs))
            pbar.set_postfix(
                tr_ic=f'{train_hist[-1]:+.3f}',
                val_ic=f'{vi:+.3f}', val_sh=f'{vs:+.2f}')

    # ----- Stage 2: optional joint head + backbone fine-tune. -----
    bb_pytree = backbone_to_pytree(backbone)
    feat_mu = bb_pytree['feat_mu']            # held constant; not in opt
    feat_sd = bb_pytree['feat_sd']
    finetune_hist: list[float] = []
    finetune_val_hist: list[tuple[int, float, float]] = []
    if finetune_steps > 0:
        # Pull rebalance-bar features out of the aligned tensor on demand —
        # we never materialize the full (n_blocks, N, K, F) tensor on
        # device, only minibatches.
        aligned = pre['aligned']
        rebal_idx = pre['rebal_idx']
        # `feat_rb`: (n_blocks, N, K, F). Lives on host; the per-batch slice
        # is jnp.asarray'd inside the loop. `nan_to_num` mirrors what
        # `precompute_inputs` does to the cached representation tensor —
        # warmup-row NaNs survive `aligned.features` and would propagate
        # through the backbone into the loss + gradient on first step.
        # The mask filters them out of the IC sum; this just keeps the
        # forward numerically clean.
        feat_rb = np.nan_to_num(aligned.features[rebal_idx],
                                nan=0.0).astype(np.float32)
        K = backbone.K
        F = backbone.F
        N = aligned.features.shape[1]

        # Optimizer for joint fine-tune. Adam state is fresh — Stage 2 is
        # a different objective surface (jointly over backbone + head) so
        # we don't carry Stage-1 momentum across.
        #
        # Both head and backbone use `learning_rate * finetune_lr_scale`.
        # The head was already trained to its Stage-1 optimum and now
        # needs to *track* the slowly-moving backbone — a full-rate Adam
        # on the head will dominate the gradient and stomp on the
        # backbone's adjustments. log_temperature is frozen during
        # fine-tune (its Stage-1 value is used for Sharpe eval only).
        #
        # Global-norm gradient clipping at 1.0 protects against the
        # noisy per-bar IC signal — minibatches over a small number of
        # rebalance bars give a high-variance gradient that can otherwise
        # blow up Adam's first-moment estimate within a few steps.
        ft_lr = learning_rate * finetune_lr_scale
        ft_params = {
            'head': params['head'],
            'log_temperature': params['log_temperature'],
            'backbone_conv': bb_pytree['conv'],
        }
        labels = {
            'head': jax.tree_util.tree_map(lambda _: 'trainable',
                                           ft_params['head']),
            'log_temperature': 'frozen',
            'backbone_conv': jax.tree_util.tree_map(
                lambda _: 'trainable', ft_params['backbone_conv']),
        }
        ft_optimizer = optax.multi_transform(
            {'trainable': optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(ft_lr)),
             'frozen': optax.set_to_zero()},
            labels,
        )
        ft_opt_state = ft_optimizer.init(ft_params)

        def _ft_forward(ft_p, X_batch):
            """X_batch: (B, N, K, F). Returns scores (B, N)."""
            B = X_batch.shape[0]
            flat = X_batch.reshape(B * N, K, F)
            bb = {'feat_mu': feat_mu, 'feat_sd': feat_sd,
                  'conv': ft_p['backbone_conv']}
            repr_flat = apply_backbone_pytree(bb, flat)
            repr_b = repr_flat.reshape(B, N, backbone.hidden_flat)
            return apply_fn(ft_p['head'], repr_b)

        def _ft_train_neg(ft_p, X_batch, fwd_batch, mask_batch):
            scores = _ft_forward(ft_p, X_batch)
            return -pearson_rank_ic(scores, fwd_batch, mask_batch)

        ft_value_and_grad = jax.jit(jax.value_and_grad(_ft_train_neg))

        # Eval helper: arrays in, scalars out — `slice` objects don't
        # cross the jit boundary, callers materialize the slice before
        # invoking. Re-runs the (now-trained) backbone over the slice in
        # one shot — the val window is small enough not to need batching.
        @jax.jit
        def _ft_eval_arrays(ft_p, X, fwd, mask, blr):
            scores = _ft_forward(ft_p, X)
            ic = pearson_rank_ic(scores, fwd, mask)
            sharpe = block_sharpe(
                scores, ft_p['log_temperature'],
                blr, mask, rebal_days, commission_frac)
            return ic, sharpe

        def ft_eval_jit(ft_p, slc):
            return _ft_eval_arrays(
                ft_p, jnp.asarray(feat_rb[slc]),
                fwd_rb[slc], mask_rb[slc], blr_rb[slc])

        rng = np.random.default_rng(seed + 1)
        train_bars = np.arange(n_train, dtype=np.int64)
        if finetune_batch_bars > n_train:
            finetune_batch_bars = n_train
        if verbose:
            n_bb_params = sum(int(np.prod(v['W'].shape) + np.prod(v['b'].shape))
                              for v in ft_params['backbone_conv'])
            print(f'  Stage 2 fine-tune: backbone ({n_bb_params} params) '
                  f'unfrozen at lr={learning_rate * finetune_lr_scale:.1e}, '
                  f'head/temp at lr={learning_rate:.1e}, '
                  f'batch={finetune_batch_bars} bars / step')
        pbar2 = tqdm(range(finetune_steps),
                     desc=f'rank-IC ({scorer}) stage2',
                     unit='step', disable=not verbose)
        for step in pbar2:
            sel = rng.choice(train_bars, size=finetune_batch_bars,
                             replace=False)
            Xb = jnp.asarray(feat_rb[sel])
            fb = fwd_rb[sel]
            mb = mask_rb[sel]
            loss, grads = ft_value_and_grad(ft_params, Xb, fb, mb)
            updates, ft_opt_state = ft_optimizer.update(
                grads, ft_opt_state, ft_params)
            ft_params = optax.apply_updates(ft_params, updates)
            finetune_hist.append(-float(loss))
            if step % 5 == 0 or step == finetune_steps - 1:
                vi, vs = ft_eval_jit(ft_params, val_slc)
                finetune_val_hist.append(
                    (step, float(vi), float(vs)))
                pbar2.set_postfix(
                    tr_ic=f'{finetune_hist[-1]:+.3f}',
                    val_ic=f'{float(vi):+.3f}',
                    val_sh=f'{float(vs):+.2f}')

        # Sync `params` with the fine-tuned head/temperature; pull the
        # updated backbone conv weights back into the pytree.
        params = {
            'head': ft_params['head'],
            'log_temperature': ft_params['log_temperature'],
        }
        bb_pytree = {**bb_pytree, 'conv': ft_params['backbone_conv']}

    # ----- Final eval — must use the fine-tuned backbone if Stage 2 ran. -----
    if finetune_steps > 0:
        ft_params_final = {
            'head': params['head'],
            'log_temperature': params['log_temperature'],
            'backbone_conv': bb_pytree['conv'],
        }
        train_ic_jax, train_sharpe_jax = ft_eval_jit(ft_params_final, train_slc)
        val_ic_jax, val_sharpe_jax = ft_eval_jit(ft_params_final, val_slc)
        final_train_ic = float(train_ic_jax)
        final_train_sharpe = float(train_sharpe_jax)
        final_val_ic = float(val_ic_jax)
        final_val_sharpe = float(val_sharpe_jax)
    else:
        final_train_ic = float(train_hist[-1])
        final_val_ic = float(val_ic_fn(params))
        final_train_sharpe = float(train_sharpe_fn(params))
        final_val_sharpe = float(val_sharpe_fn(params))

    if verbose:
        print(f'  Final     train IC: {final_train_ic:+.4f}   '
              f'val IC: {final_val_ic:+.4f}')
        print(f'            train Sharpe: {final_train_sharpe:+.3f}   '
              f'val Sharpe: {final_val_sharpe:+.3f}')

    return TrainResult(
        params=params,
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
        backbone_params=bb_pytree,
    )


def predict(
    aligned: AlignedTickers, backbone: Backbone,
    head_params: dict[str, jax.Array], *,
    scorer: str = 'linear',
) -> np.ndarray:
    """Apply backbone + scoring head over an `AlignedTickers` block.

    Returns scores shape `(D, N)`. NaN where features are NaN. Useful
    for held-out evaluation on a fresh ticker pool without re-running
    the trainer.
    """
    _, apply_fn = get_scorer(scorer)
    D, N, K, F = aligned.features.shape
    flat = aligned.features.reshape(D * N, K, F)
    repr_flat = np.asarray(apply_backbone(backbone, jnp.asarray(flat)))
    repr_full = repr_flat.reshape(D, N, backbone.hidden_flat)
    scores = np.asarray(apply_fn(head_params, jnp.asarray(repr_full)))
    finite = np.isfinite(repr_full).all(axis=-1)
    return np.where(finite, scores, np.nan).astype(np.float64)
