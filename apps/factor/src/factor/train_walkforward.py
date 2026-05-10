"""Walk-forward variant of `train_scorer`.

Disambiguates the single-split val IC ≈ 0 result we saw in the smoke
sweep: is it a regime break (some windows generalize, others don't) or
a genuine null (no window's val IC moves)? Both look identical with one
70/30 split.

Mirrors the convention in `apps/regime`'s walk-forward search:
`train_window_blocks` rebal blocks of fit, then `val_window_blocks`
of held-out evaluation, slid forward `step_window_blocks` at a time.
Step defaults to val so consecutive val periods don't overlap.

For our identity backbone the per-window cost is dominated by tinygrad
JIT compile, not by the head's tiny matmul, so total wall scales sub-
linearly in the window count once the kernel cache warms up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW

from ss_features import TickerData, block_windows
from factor.backbone import Backbone
from factor.data import AlignedTickers
from factor.objectives import (
    block_sharpe, block_sharpe_long_short, masked_mse, pearson_rank_ic,
)
from factor.scorers import get_scorer
from factor.train import precompute_inputs


@dataclass(frozen=True)
class WalkForwardWindow:
    """One train/val window's metrics + final head params.

    Indexes refer to rebalance-block positions in the underlying
    `precompute_inputs` output (so `train_block_start` to
    `train_block_end - 1` inclusive is the train slice, etc.).
    """
    window_idx:        int
    train_block_start: int
    train_block_end:   int   # exclusive
    val_block_start:   int
    val_block_end:     int   # exclusive
    train_ic:          float
    val_ic:            float
    train_sharpe:      float
    val_sharpe:        float
    n_train_bars:      int
    n_val_bars:        int
    head_params:       dict[str, np.ndarray]
    train_aux_mse:     float = float('nan')
    val_aux_mse:       float = float('nan')
    train_sharpe_long_short: float = float('nan')
    val_sharpe_long_short:   float = float('nan')


@dataclass
class WalkForwardResult:
    """Aggregate of `train_scorer_walkforward`."""
    scorer:                str
    n_steps:               int
    learning_rate:         float
    weight_decay:          float
    rebal_days:            int
    train_window_blocks:   int
    val_window_blocks:     int
    step_window_blocks:    int
    feature_width:         int
    aligned:               AlignedTickers
    forward_target_kind:   str = 'log_return'
    windows:               list[WalkForwardWindow] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def mean_val_ic(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ic for w in self.windows]))

    @property
    def median_val_ic(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.median([w.val_ic for w in self.windows]))

    @property
    def mean_val_sharpe(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_sharpe for w in self.windows]))

    @property
    def mean_val_sharpe_long_short(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_sharpe_long_short for w in self.windows]))

    @property
    def positive_val_sharpe_long_short_fraction(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean(
            [w.val_sharpe_long_short > 0 for w in self.windows]))

    @property
    def positive_val_ic_fraction(self) -> float:
        """Share of windows with val_ic > 0. A regime-break universe will
        scatter (fraction near 0.5 with both signs); a genuine-null
        universe sits near 0.5 too but with magnitudes near 0; a
        consistently-signal universe lands near 0 or 1."""
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ic > 0 for w in self.windows]))




def train_scorer_walkforward(
    tickers: list[TickerData], backbone: Backbone, *,
    rebal_days: int = 5,
    train_window_blocks: int = 63,    # ~5y at rebal_days=20 (~12.6 blocks/yr)
    val_window_blocks:   int = 39,    # ~3y at rebal_days=20
    step_window_blocks:  int | None = None,  # default = val (no overlap)
    scorer:        str = 'linear',
    mlp_hidden:    int = 64,
    mlp_layers:    int = 1,
    n_steps:       int = 200,
    learning_rate: float = 1e-3,
    weight_decay:  float = 0.0,
    seed:          int = 0,
    commission_bps: float = 10.0,
    init_log_temperature: float = 0.0,
    spread:      np.ndarray | None = None,
    max_spread:  float | None = None,
    forward_target_kind: str = 'log_return',
    aux_weight:  float = 0.0,
    aux_winsor:  tuple[float, float] = (0.01, 0.99),
    verbose:     bool = True,
) -> WalkForwardResult:
    """Walk-forward variant of `train_scorer`.

    Initialization & training of the head is independent per window —
    fresh seed-derived init, fresh AdamW state, `n_steps` updates. The
    backbone is run forward exactly once over the full date range
    (`precompute_inputs`), then per-window we just slice the cached
    representation tensor — no JIT recompile per window.

    Stage 2 fine-tune is intentionally not exposed here. Joint head +
    backbone tuning per window would multiply per-window cost by the
    number of fine-tune steps (the cached representation goes stale on
    every backbone update). Walk-forward eval is cheap *because* we
    precompute. If you want fine-tune-vs-frozen comparison, use
    `train_scorer` and a single split.

    `forward_target_kind` is forwarded to `precompute_inputs` (see its
    docstring). val_ic / train_ic are computed against whatever target
    the loss saw, so they remain comparable to the loss value; val_sharpe
    / train_sharpe are always against actual block log returns.

    Multi-task aux head: setting `aux_weight > 0` requires
    `scorer='mlp_multitask'` and adds a second output head that
    predicts cross-sectionally winsorized + z-scored forward log
    returns via MSE. The shared MLP trunk gets gradients from both
    losses; the primary `Wp` head's `train_ic` / `val_ic` is what's
    reported. `train_aux_mse` / `val_aux_mse` track the aux head as a
    sanity check that it learned something. See
    `factor.scorers.init_mlp_multitask` and
    `factor.objectives.masked_mse`.
    """
    if aux_weight < 0.0:
        raise ValueError(f'aux_weight={aux_weight} must be >= 0')
    if aux_weight > 0.0 and scorer != 'mlp_multitask':
        raise ValueError(
            f'aux_weight > 0 requires scorer=mlp_multitask, got {scorer!r}')
    if aux_weight == 0.0 and scorer == 'mlp_multitask':
        raise ValueError(
            'scorer=mlp_multitask requires aux_weight > 0 (otherwise '
            'the aux head trains nothing — use scorer=mlp instead)')
    is_multitask = scorer == 'mlp_multitask'

    if step_window_blocks is None:
        step_window_blocks = val_window_blocks

    pre = precompute_inputs(
        tickers, backbone, rebal_days=rebal_days,
        max_spread=max_spread, spread=spread,
        forward_target_kind=forward_target_kind,
        aux_target_kind='robust_z' if is_multitask else None,
        aux_winsor=aux_winsor)
    repr_rb_np: np.ndarray = pre['representation_rb']
    fwd_rb_np:  np.ndarray = pre['fwd_ret_rb']
    mask_rb_np: np.ndarray = pre['mask_rb']
    blr_rb_np:  np.ndarray = pre['block_log_ret_rb']
    aligned:    AlignedTickers = pre['aligned']
    aux_rb_np: np.ndarray | None = pre.get('aux_target_rb')

    n_blocks = repr_rb_np.shape[0]
    slices = block_windows(
        n_blocks, train_window_blocks, val_window_blocks, step_window_blocks)
    if not slices:
        raise ValueError(
            f'no walk-forward windows fit: have {n_blocks} rebal blocks but '
            f'each window needs train+val={train_window_blocks + val_window_blocks}')

    init_fn, apply_fn = get_scorer(scorer)
    log_temperature = Tensor(np.array(init_log_temperature, dtype=np.float32),
                             requires_grad=False)

    if verbose:
        aux_tag = (
            f' aux_weight={aux_weight} winsor={aux_winsor}' if is_multitask
            else '')
        print(f'walk-forward: {len(slices)} windows over {n_blocks} blocks  '
              f'(train={train_window_blocks}, val={val_window_blocks}, '
              f'step={step_window_blocks})  scorer={scorer} '
              f'n_steps={n_steps} lr={learning_rate} wd={weight_decay} '
              f'target={forward_target_kind}{aux_tag}')

    result = WalkForwardResult(
        scorer=scorer, n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        feature_width=backbone.F, aligned=aligned,
        forward_target_kind=forward_target_kind,
    )

    pbar = tqdm(slices, desc=f'walk-forward ({scorer})', unit='window',
                disable=not verbose)
    for w_idx, (train_slc, val_slc) in enumerate(pbar):
        # Fresh head per window. Different seed per window so any
        # sensitivity to init is averaged out across the result.
        rng = np.random.default_rng(seed + w_idx)
        if scorer in ('mlp', 'mlp_multitask'):
            head_params = init_fn(
                rng, backbone.hidden_flat, hidden=mlp_hidden, n_layers=mlp_layers)
        else:
            head_params = init_fn(rng, backbone.hidden_flat)

        head_param_list = list(head_params.values())
        opt = AdamW(head_param_list, lr=learning_rate, weight_decay=weight_decay)

        repr_train = Tensor(repr_rb_np[train_slc])
        fwd_train  = Tensor(fwd_rb_np[train_slc])
        mask_train = Tensor(mask_rb_np[train_slc])
        repr_val   = Tensor(repr_rb_np[val_slc])
        fwd_val    = Tensor(fwd_rb_np[val_slc])
        mask_val   = Tensor(mask_rb_np[val_slc])
        blr_train  = Tensor(blr_rb_np[train_slc])
        blr_val    = Tensor(blr_rb_np[val_slc])
        aux_train: Tensor | None = (
            Tensor(aux_rb_np[train_slc]) if is_multitask else None)
        aux_val: Tensor | None = (
            Tensor(aux_rb_np[val_slc]) if is_multitask else None)

        for _ in range(n_steps):
            Tensor.training = True
            opt.zero_grad()
            out = apply_fn(head_params, repr_train)
            if is_multitask:
                s_p, s_a = out
                loss = (
                    -pearson_rank_ic(s_p, fwd_train, mask_train)
                    + aux_weight * masked_mse(s_a, aux_train, mask_train)
                )
            else:
                loss = -pearson_rank_ic(out, fwd_train, mask_train)
            loss.backward()
            opt.step()

        Tensor.training = False
        if is_multitask:
            s_train_p, s_train_a = apply_fn(head_params, repr_train)
            s_val_p, s_val_a = apply_fn(head_params, repr_val)
            s_train, s_val = s_train_p, s_val_p
            train_aux_mse = float(masked_mse(
                s_train_a, aux_train, mask_train).item())
            val_aux_mse = float(masked_mse(
                s_val_a, aux_val, mask_val).item())
        else:
            s_train = apply_fn(head_params, repr_train)
            s_val   = apply_fn(head_params, repr_val)
            train_aux_mse = float('nan')
            val_aux_mse = float('nan')
        train_ic = float(pearson_rank_ic(s_train, fwd_train, mask_train).item())
        val_ic   = float(pearson_rank_ic(s_val,   fwd_val,   mask_val  ).item())
        train_sh = float(block_sharpe(
            s_train, log_temperature, blr_train, mask_train,
            rebal_days, commission_bps / 1e4).item())
        val_sh   = float(block_sharpe(
            s_val,   log_temperature, blr_val,   mask_val,
            rebal_days, commission_bps / 1e4).item())
        train_sh_ls = float(block_sharpe_long_short(
            s_train, blr_train, mask_train,
            rebal_days, commission_bps / 1e4).item())
        val_sh_ls   = float(block_sharpe_long_short(
            s_val,   blr_val,   mask_val,
            rebal_days, commission_bps / 1e4).item())

        if verbose:
            postfix = {
                'tr_ic': f'{train_ic:+.3f}',
                'val_ic': f'{val_ic:+.3f}',
                'val_sh': f'{val_sh:+.2f}',
                'val_sh_ls': f'{val_sh_ls:+.2f}',
            }
            if is_multitask:
                postfix['val_aux'] = f'{val_aux_mse:.3f}'
            pbar.set_postfix(**postfix)

        result.windows.append(WalkForwardWindow(
            window_idx=w_idx,
            train_block_start=train_slc.start, train_block_end=train_slc.stop,
            val_block_start=val_slc.start,     val_block_end=val_slc.stop,
            train_ic=train_ic, val_ic=val_ic,
            train_sharpe=train_sh, val_sharpe=val_sh,
            n_train_bars=train_slc.stop - train_slc.start,
            n_val_bars=val_slc.stop - val_slc.start,
            head_params={k: v.numpy() for k, v in head_params.items()},
            train_aux_mse=train_aux_mse, val_aux_mse=val_aux_mse,
            train_sharpe_long_short=train_sh_ls,
            val_sharpe_long_short=val_sh_ls,
        ))

    if verbose:
        print(f'walk-forward done: {result.n_windows} windows, '
              f'mean val IC={result.mean_val_ic:+.4f}, '
              f'median val IC={result.median_val_ic:+.4f}, '
              f'positive-val-IC fraction={result.positive_val_ic_fraction:.2f}, '
              f'mean val Sharpe long-only={result.mean_val_sharpe:+.3f}, '
              f'mean val Sharpe long-short={result.mean_val_sharpe_long_short:+.3f}')

    return result


__all__ = [
    'WalkForwardWindow',
    'WalkForwardResult',
    'train_scorer_walkforward',
]
