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

from ss_features import TickerData
from factor.backbone import Backbone
from factor.data import AlignedTickers
from factor.objectives import block_sharpe, pearson_rank_ic
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
    def positive_val_ic_fraction(self) -> float:
        """Share of windows with val_ic > 0. A regime-break universe will
        scatter (fraction near 0.5 with both signs); a genuine-null
        universe sits near 0.5 too but with magnitudes near 0; a
        consistently-signal universe lands near 0 or 1."""
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ic > 0 for w in self.windows]))


def _generate_window_slices(
    n_blocks: int, train_w: int, val_w: int, step_w: int,
) -> list[tuple[slice, slice]]:
    """Roll a (train_w, val_w) pair forward by `step_w` blocks at a time.

    Returns one (train_slice, val_slice) per window that fits entirely
    inside `n_blocks`. The last window may not align to the end exactly;
    we don't pad — better to drop a partial window than evaluate on too
    few val blocks.
    """
    if train_w < 2 or val_w < 2:
        raise ValueError(
            f'train_window_blocks={train_w} and val_window_blocks={val_w} '
            'must each be >= 2 for a meaningful IC evaluation')
    if step_w < 1:
        raise ValueError(f'step_window_blocks={step_w} must be >= 1')

    out: list[tuple[slice, slice]] = []
    cursor = 0
    needed = train_w + val_w
    while cursor + needed <= n_blocks:
        train_slc = slice(cursor, cursor + train_w)
        val_slc = slice(cursor + train_w, cursor + needed)
        out.append((train_slc, val_slc))
        cursor += step_w
    return out


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
    """
    if step_window_blocks is None:
        step_window_blocks = val_window_blocks

    pre = precompute_inputs(
        tickers, backbone, rebal_days=rebal_days,
        max_spread=max_spread, spread=spread)
    repr_rb_np: np.ndarray = pre['representation_rb']
    fwd_rb_np:  np.ndarray = pre['fwd_ret_rb']
    mask_rb_np: np.ndarray = pre['mask_rb']
    blr_rb_np:  np.ndarray = pre['block_log_ret_rb']
    aligned:    AlignedTickers = pre['aligned']

    n_blocks = repr_rb_np.shape[0]
    slices = _generate_window_slices(
        n_blocks, train_window_blocks, val_window_blocks, step_window_blocks)
    if not slices:
        raise ValueError(
            f'no walk-forward windows fit: have {n_blocks} rebal blocks but '
            f'each window needs train+val={train_window_blocks + val_window_blocks}')

    init_fn, apply_fn = get_scorer(scorer)
    log_temperature = Tensor(np.array(init_log_temperature, dtype=np.float32),
                             requires_grad=False)

    if verbose:
        print(f'walk-forward: {len(slices)} windows over {n_blocks} blocks  '
              f'(train={train_window_blocks}, val={val_window_blocks}, '
              f'step={step_window_blocks})  scorer={scorer} '
              f'n_steps={n_steps} lr={learning_rate} wd={weight_decay}')

    result = WalkForwardResult(
        scorer=scorer, n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, rebal_days=rebal_days,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        feature_width=backbone.F, aligned=aligned,
    )

    pbar = tqdm(slices, desc=f'walk-forward ({scorer})', unit='window',
                disable=not verbose)
    for w_idx, (train_slc, val_slc) in enumerate(pbar):
        # Fresh head per window. Different seed per window so any
        # sensitivity to init is averaged out across the result.
        rng = np.random.default_rng(seed + w_idx)
        if scorer == 'mlp':
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

        for _ in range(n_steps):
            Tensor.training = True
            opt.zero_grad()
            s = apply_fn(head_params, repr_train)
            loss = -pearson_rank_ic(s, fwd_train, mask_train)
            loss.backward()
            opt.step()

        Tensor.training = False
        s_train = apply_fn(head_params, repr_train)
        s_val   = apply_fn(head_params, repr_val)
        train_ic = float(pearson_rank_ic(s_train, fwd_train, mask_train).item())
        val_ic   = float(pearson_rank_ic(s_val,   fwd_val,   mask_val  ).item())
        train_sh = float(block_sharpe(
            s_train, log_temperature, blr_train, mask_train,
            rebal_days, commission_bps / 1e4).item())
        val_sh   = float(block_sharpe(
            s_val,   log_temperature, blr_val,   mask_val,
            rebal_days, commission_bps / 1e4).item())

        if verbose:
            pbar.set_postfix(
                tr_ic=f'{train_ic:+.3f}',
                val_ic=f'{val_ic:+.3f}',
                val_sh=f'{val_sh:+.2f}')

        result.windows.append(WalkForwardWindow(
            window_idx=w_idx,
            train_block_start=train_slc.start, train_block_end=train_slc.stop,
            val_block_start=val_slc.start,     val_block_end=val_slc.stop,
            train_ic=train_ic, val_ic=val_ic,
            train_sharpe=train_sh, val_sharpe=val_sh,
            n_train_bars=train_slc.stop - train_slc.start,
            n_val_bars=val_slc.stop - val_slc.start,
            head_params={k: v.numpy() for k, v in head_params.items()},
        ))

    if verbose:
        print(f'walk-forward done: {result.n_windows} windows, '
              f'mean val IC={result.mean_val_ic:+.4f}, '
              f'median val IC={result.median_val_ic:+.4f}, '
              f'positive-val-IC fraction={result.positive_val_ic_fraction:.2f}')

    return result


__all__ = [
    'WalkForwardWindow',
    'WalkForwardResult',
    'train_scorer_walkforward',
]
