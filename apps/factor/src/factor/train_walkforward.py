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
    block_ir_vs_ew, block_sharpe, block_sharpe_long_short,
    masked_mse, pearson_rank_ic,
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
    train_ir_vs_ew:    float = float('nan')
    val_ir_vs_ew:      float = float('nan')
    final_log_temperature: float = 0.0
    # MSE of head scores against per-bar cross-sectional alpha targets.
    # Diagnostic on both arms; the load-bearing metric for the
    # `mse_alpha` arm.
    train_mse_alpha:   float = float('nan')
    val_mse_alpha:     float = float('nan')
    # Per-val-bar top-decile-minus-bottom-decile predicted alpha. The
    # sizing-input artifact this harness emits for downstream meta-gate
    # consumption. Shape `(n_val_bars,)`; NaN for bars with fewer than
    # the decile-floor number of valid tickers.
    signal_quality_per_val_bar: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    signal_quality_mean: float = float('nan')
    signal_quality_std:  float = float('nan')
    # ISO date of the first val bar — lets downstream gate code align
    # this window's signal-quality with the macro-state snapshot at
    # the same date.
    val_start_date:    str = ''


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
    def mean_val_ir_vs_ew(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ir_vs_ew for w in self.windows]))

    @property
    def positive_val_ir_vs_ew_fraction(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ir_vs_ew > 0 for w in self.windows]))

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

    @property
    def mean_val_mse_alpha(self) -> float:
        """Mean across windows of val MSE against per-bar cross-sectional
        alpha targets. Load-bearing metric for the `mse_alpha` loss
        path; diagnostic for `rank_ic` arm."""
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_mse_alpha for w in self.windows]))

    @property
    def mean_signal_quality(self) -> float:
        """Mean across windows of per-bar mean signal-quality (top-decile
        − bottom-decile predicted alpha). Magnitude is in score units,
        only directly comparable across arms when both arms use the
        same loss-induced score scale."""
        if not self.windows:
            return float('nan')
        return float(np.mean([w.signal_quality_mean for w in self.windows]))




def _signal_quality_per_bar(
    scores: np.ndarray, mask: np.ndarray, decile_frac: float = 0.10,
) -> np.ndarray:
    """Per-bar (top-decile mean − bottom-decile mean) of `scores` over
    the masked-liquid universe.

    Shape `(n_bars, n_tickers)` input. Returns `(n_bars,)` — NaN for
    bars whose liquid count is below `2 * decile_floor` (otherwise
    the top/bottom slices would overlap or be degenerate).

    The decile-spread is a proxy for "how much sizing information does
    this bar's score vector contain?" — large spread means the head is
    confident there's cross-sectional structure to bet on; near-zero
    means the head sees no usable signal at that bar. A downstream
    meta-gate can use this as a conditioning input alongside macro
    state, without needing access to forward returns (so it's
    computable in real time, unlike ICIR which is retroactive).
    """
    n_bars, _ = scores.shape
    out = np.full(n_bars, np.nan, dtype=np.float64)
    mask_bool = mask.astype(bool)
    for b in range(n_bars):
        m = mask_bool[b]
        n_valid = int(m.sum())
        n_decile = max(1, int(round(decile_frac * n_valid)))
        if n_valid < 2 * n_decile:
            continue
        s = scores[b][m]
        order = np.argsort(s)
        bottom_mean = float(s[order[:n_decile]].mean())
        top_mean = float(s[order[-n_decile:]].mean())
        out[b] = top_mean - bottom_mean
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
    forward_target_kind: str = 'log_return',
    aux_weight:  float = 0.0,
    aux_winsor:  tuple[float, float] = (0.01, 0.99),
    loss_kind:   str = 'rank_ic',
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

    `loss_kind` ∈ `{'rank_ic', 'mse_alpha', 'block_sharpe', 'ir_vs_ew'}`:
      * `'rank_ic'` (default) — scale-invariant Pearson IC on
        cross-sectionally demeaned forward log-returns. Existing
        baseline. Score magnitude uncalibrated.
      * `'mse_alpha'` — scale-calibrated regression. `masked_mse` on
        per-bar alpha targets (`fwd_log_return − cross_sectional_mean`).
        Trains scores to match expected per-ticker alpha in log-return
        units. Built to serve as input to a downstream sizing/gate
        layer rather than a portfolio softmax (see
        `findings/factor-rankic-long-only-mismatch.md` + the
        `TODO/factor-sizing-input-reframe.md` plan).
      * `'block_sharpe'` / `'ir_vs_ew'` — Sharpe-aligned losses,
        kept available but documented as worse than `rank_ic` on
        factor-narrow at the +0.005 IC regime
        (`findings/factor-loss-pivot.md`).

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

    valid_losses = {'rank_ic', 'block_sharpe', 'ir_vs_ew', 'mse_alpha'}
    if loss_kind not in valid_losses:
        raise ValueError(
            f'loss_kind={loss_kind!r} not in {sorted(valid_losses)}')
    if loss_kind != 'rank_ic' and is_multitask:
        raise ValueError(
            f'loss_kind={loss_kind!r} not supported with mlp_multitask '
            f'(aux loss is rank-IC-coupled by construction)')
    # rank_ic and mse_alpha don't see log_temperature (rank_ic is
    # scale-invariant; mse_alpha trains scores directly to alpha
    # magnitude — temperature lives in the softmax constructor, which
    # is eval-only for these arms). block_sharpe and ir_vs_ew tune
    # temperature as part of the loss.
    train_temp = loss_kind in ('block_sharpe', 'ir_vs_ew')

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
    alpha_rb_np: np.ndarray = pre['alpha_target_rb']
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

    if verbose:
        aux_tag = (
            f' aux_weight={aux_weight} winsor={aux_winsor}' if is_multitask
            else '')
        print(f'walk-forward: {len(slices)} windows over {n_blocks} blocks  '
              f'(train={train_window_blocks}, val={val_window_blocks}, '
              f'step={step_window_blocks})  scorer={scorer} '
              f'loss={loss_kind} n_steps={n_steps} lr={learning_rate} '
              f'wd={weight_decay} target={forward_target_kind}{aux_tag}')

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
        # Sharpe / IR losses care about score magnitude (constructor is
        # softmax(score / temp)), so the temperature is part of what
        # the optimizer should tune. Rank-IC is scale-invariant so the
        # temperature is irrelevant to its loss; keep it frozen at
        # init for backwards compatibility with prior runs.
        # Shape (1,) not () so AdamW.assign's shape-match works
        # (tinygrad's broadcast doesn't drop dims).
        log_temperature = Tensor(
            np.array([init_log_temperature], dtype=np.float32),
            requires_grad=train_temp)
        opt_params = list(head_param_list)
        if train_temp:
            opt_params.append(log_temperature)
        opt = AdamW(opt_params, lr=learning_rate, weight_decay=weight_decay)

        repr_train = Tensor(repr_rb_np[train_slc])
        fwd_train  = Tensor(fwd_rb_np[train_slc])
        alpha_train = Tensor(alpha_rb_np[train_slc])
        mask_train = Tensor(mask_rb_np[train_slc])
        repr_val   = Tensor(repr_rb_np[val_slc])
        fwd_val    = Tensor(fwd_rb_np[val_slc])
        alpha_val  = Tensor(alpha_rb_np[val_slc])
        mask_val   = Tensor(mask_rb_np[val_slc])
        blr_train  = Tensor(blr_rb_np[train_slc])
        blr_val    = Tensor(blr_rb_np[val_slc])
        aux_train: Tensor | None = (
            Tensor(aux_rb_np[train_slc]) if is_multitask else None)
        aux_val: Tensor | None = (
            Tensor(aux_rb_np[val_slc]) if is_multitask else None)

        commission_frac = commission_bps / 1e4
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
            elif loss_kind == 'rank_ic':
                loss = -pearson_rank_ic(out, fwd_train, mask_train)
            elif loss_kind == 'mse_alpha':
                # Scale-calibrated regression on per-bar cross-sectional
                # alpha. The head's scalar output is trained to match
                # `fwd_log_return − cross_sectional_mean(fwd_log_return)`
                # in alpha units. Downstream consumers (sizing layer,
                # macro meta-gate) can interpret the magnitude.
                loss = masked_mse(out, alpha_train, mask_train)
            elif loss_kind == 'block_sharpe':
                loss = -block_sharpe(
                    out, log_temperature, blr_train, mask_train,
                    rebal_days, commission_frac)
            elif loss_kind == 'ir_vs_ew':
                loss = -block_ir_vs_ew(
                    out, log_temperature, blr_train, mask_train,
                    rebal_days, commission_frac)
            else:
                raise AssertionError(f'unreachable loss_kind={loss_kind}')
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
        train_mse_alpha = float(masked_mse(
            s_train, alpha_train, mask_train).item())
        val_mse_alpha = float(masked_mse(
            s_val, alpha_val, mask_val).item())
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
        train_ir = float(block_ir_vs_ew(
            s_train, log_temperature, blr_train, mask_train,
            rebal_days, commission_bps / 1e4).item())
        val_ir   = float(block_ir_vs_ew(
            s_val,   log_temperature, blr_val,   mask_val,
            rebal_days, commission_bps / 1e4).item())
        final_log_temp = float(np.asarray(log_temperature.numpy()).reshape(-1)[0])

        # Per-val-bar signal-quality (top-decile − bottom-decile of head
        # scores). Computed on numpy after Tensor.training=False — no
        # gradient implications, just a cheap eval summary stat. The
        # full per-bar vector is preserved for downstream gate use; the
        # mean/std are convenience reductions.
        s_val_np = s_val.numpy()
        sq_per_bar = _signal_quality_per_bar(
            s_val_np, mask_rb_np[val_slc])
        sq_mean = float(np.nanmean(sq_per_bar)) if sq_per_bar.size else float('nan')
        sq_std = float(np.nanstd(sq_per_bar)) if sq_per_bar.size else float('nan')

        # ISO date of the first val rebal bar — used by downstream
        # meta-gate harnesses to align this window's signal-quality
        # with macro-state at the same calendar moment.
        val_start_idx = aligned.rebal_idx[val_slc.start]
        val_start_date = str(np.asarray(aligned.dates[val_start_idx]).astype(
            'datetime64[D]'))

        if verbose:
            postfix = {
                'tr_ic': f'{train_ic:+.3f}',
                'val_ic': f'{val_ic:+.3f}',
                'val_sh': f'{val_sh:+.2f}',
                'val_ir': f'{val_ir:+.2f}',
                'val_sh_ls': f'{val_sh_ls:+.2f}',
            }
            if train_temp:
                postfix['logT'] = f'{final_log_temp:+.2f}'
            if is_multitask:
                postfix['val_aux'] = f'{val_aux_mse:.3f}'
            if loss_kind == 'mse_alpha':
                postfix['val_mse_a'] = f'{val_mse_alpha:.3e}'
            postfix['sq'] = f'{sq_mean:+.3e}'
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
            train_ir_vs_ew=train_ir, val_ir_vs_ew=val_ir,
            final_log_temperature=final_log_temp,
            train_mse_alpha=train_mse_alpha, val_mse_alpha=val_mse_alpha,
            signal_quality_per_val_bar=sq_per_bar,
            signal_quality_mean=sq_mean,
            signal_quality_std=sq_std,
            val_start_date=val_start_date,
        ))

    if verbose:
        print(f'walk-forward done: {result.n_windows} windows, '
              f'mean val IC={result.mean_val_ic:+.4f}, '
              f'median val IC={result.median_val_ic:+.4f}, '
              f'positive-val-IC fraction={result.positive_val_ic_fraction:.2f}, '
              f'mean val Sharpe long-only={result.mean_val_sharpe:+.3f}, '
              f'mean val Sharpe long-short={result.mean_val_sharpe_long_short:+.3f}, '
              f'mean val MSE-alpha={result.mean_val_mse_alpha:.4e}, '
              f'mean signal-quality={result.mean_signal_quality:+.4e}')

    return result


__all__ = [
    'WalkForwardWindow',
    'WalkForwardResult',
    'train_scorer_walkforward',
]
