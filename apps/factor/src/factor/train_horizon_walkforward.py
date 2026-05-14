"""Walk-forward trainer for the endogenous-horizon scoring head.

Mirrors `train_scorer_walkforward` but with the mixture-of-horizons
loss (`objectives.horizon_mixture_loss`) and the dual-head scorer
(`scorers.init_mlp_horizon` / `apply_mlp_horizon`). The model emits
both per-ticker scores and a per-bar K-way horizon distribution `π_t`
at every fine-grid rebal bar; deployment follows `argmax(π_t)` and
holds flat between rebals.

Eval is on the **daily PnL stream** under the model-emitted irregular
cadence (`horizon.simulate_irregular_daily_pnl`) — *not* on the
`block_sharpe` used elsewhere, because block_sharpe's
`sqrt(252/rebal_days)` annualization is undefined when the cadence is
endogenous. Per-window output includes:

  - Endogenous net Sharpe (the model's actual deployment).
  - Fixed-horizon baselines at every `h_k` (null N2/N3 check —
    endogenous must beat each).
  - Random-π baseline (null N4 — endogenous must beat horizon picked
    uniformly at random from the same K bins).
  - π entropy histogram + argmax-bin counts (null N1 — π collapse).

The fine rebal grid is `h_min` (smallest horizon). The encoder runs
once on the dense grid; per-horizon forward returns are sliced from a
shared `(K, D, N)` panel; per-horizon masks tighten the base liquidity
mask with each horizon's edge-NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW

from ss_features import TickerData, block_windows
from factor.backbone import Backbone, apply_backbone
from factor.data import (
    AlignedTickers, align_tickers_at_rebal, forward_log_returns_multi,
)
from factor.horizon import (
    IrregularRunResult, simulate_fixed_horizon_daily_pnl,
    simulate_irregular_daily_pnl,
)
from factor.objectives import horizon_mixture_loss
from factor.scorers import apply_mlp_horizon, init_mlp_horizon


@dataclass(frozen=True)
class HorizonWindow:
    """One train/val window's metrics for the endogenous-horizon trainer.

    Endogenous + baseline net Sharpes are all measured under the same
    daily-PnL framework on the same val daily slice, so they're
    directly comparable.
    """
    window_idx:        int
    train_block_start: int
    train_block_end:   int   # exclusive (fine-grid bar indices)
    val_block_start:   int
    val_block_end:     int
    val_daily_start:   int
    val_daily_end:     int
    train_loss:        float
    val_endog_sharpe:  float
    val_endog_mean_holding: float
    val_endog_n_rebals: int
    val_endog_avg_turnover: float
    val_fixed_sharpes: dict[int, float]      # `{h: sharpe}` per fixed-h baseline
    val_random_sharpe: float                  # random-π baseline
    val_pi_entropy_mean: float                # null N1 diagnostic
    val_pi_argmax_counts: dict[int, int]      # null N1 diagnostic
    head_params:       dict[str, np.ndarray]
    val_start_date:    str = ''


@dataclass
class HorizonWalkForwardResult:
    """Aggregate of `train_scorer_horizon_walkforward`."""
    n_steps:               int
    learning_rate:         float
    weight_decay:          float
    horizons:              tuple[int, ...]
    train_window_blocks:   int
    val_window_blocks:     int
    step_window_blocks:    int
    feature_width:         int
    aligned:               AlignedTickers
    entropy_weight:        float = 0.0
    windows:               list[HorizonWindow] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def mean_val_endog_sharpe(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_endog_sharpe for w in self.windows]))

    @property
    def mean_val_random_sharpe(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_random_sharpe for w in self.windows]))

    def mean_fixed_sharpe(self, h: int) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean(
            [w.val_fixed_sharpes.get(h, float('nan')) for w in self.windows]))

    @property
    def best_fixed_horizon(self) -> tuple[int, float]:
        """Return `(h, mean_sharpe)` for the best fixed-h baseline across
        windows. The null-rejection criterion is `mean_val_endog_sharpe
        >= best_fixed + 0.10`."""
        scores = {h: self.mean_fixed_sharpe(h) for h in self.horizons}
        h_best = max(scores, key=lambda h: scores[h])
        return h_best, scores[h_best]


def _precompute_horizon_inputs(
    tickers: list[TickerData], backbone: Backbone, *,
    horizons: tuple[int, ...],
    spread: np.ndarray | None = None,
    max_spread: float | None = None,
) -> dict:
    """Run the encoder once over the fine rebal grid; compute multi-horizon
    forward returns + per-horizon masks + daily log returns for eval.

    Fine grid is `min(horizons)`. The encoder pass is identical to
    `train.precompute_inputs` but the post-processing is multi-horizon:
    we keep a `(K, n_bars, N)` forward-return panel, a matching
    `(K, n_bars, N)` mask panel (base liquidity AND per-horizon
    finite-target), and a `(D, N)` daily log-return panel that the
    daily-PnL simulator needs.
    """
    if not horizons:
        raise ValueError('horizons must be non-empty')
    h_min = min(horizons)
    aligned = align_tickers_at_rebal(
        tickers, K=backbone.K, F=backbone.F, rebal_days=h_min)
    rebal_idx = aligned.rebal_idx
    D = len(aligned.dates)
    Dp, N, K_w, F = aligned.features.shape

    # Encoder forward (chunked) — same pattern as train.precompute_inputs.
    flat_rb = aligned.features.reshape(Dp * N, K_w, F).astype(
        np.float32, copy=False)
    repr_rb_flat = np.empty(
        (Dp * N, backbone.hidden_flat), dtype=np.float32)
    CHUNK = 8192
    Tensor.training = False
    for s in range(0, flat_rb.shape[0], CHUNK):
        x = Tensor(flat_rb[s:s + CHUNK])
        chunk_out = apply_backbone(backbone, x).numpy()
        repr_rb_flat[s:s + chunk_out.shape[0]] = chunk_out
    repr_rb_full = repr_rb_flat.reshape(Dp, N, backbone.hidden_flat)

    # Daily log returns for the simulator. Drop the first row (no prior
    # bar to diff against) — sim's `daily_log_ret[d]` is the move
    # closing on day `d`, so daily_log_ret[0] is undefined; zero is the
    # standard convention and the simulator's segment-sums tolerate it.
    log_p = np.log(np.maximum(aligned.prices, 1e-12))
    daily_log_ret = np.zeros_like(aligned.prices, dtype=np.float64)
    daily_log_ret[1:] = log_p[1:] - log_p[:-1]

    # Multi-horizon forward returns on the full daily axis, sliced to
    # the fine rebal grid. (K, D, N) → (K, n_bars, N).
    fwd_multi_daily = forward_log_returns_multi(
        aligned.prices, horizons=horizons)        # (K, D, N)
    fwd_multi_rb = fwd_multi_daily[:, rebal_idx, :]   # (K, n_bars, N)

    # Base mask at rebal positions: liquid + finite encoder output.
    # Per-horizon target finite-ness gets ANDed in below.
    base_mask_rb = (
        aligned.valid[rebal_idx]
        & np.isfinite(repr_rb_full).all(axis=-1)
    )
    if spread is not None:
        if spread.shape != aligned.prices.shape:
            raise ValueError(
                f'spread shape {spread.shape} must match prices shape '
                f'{aligned.prices.shape}')
        if max_spread is None:
            raise ValueError('spread provided but max_spread is None')
        base_mask_rb &= spread[rebal_idx] <= max_spread

    mask_multi_rb = np.empty(fwd_multi_rb.shape, dtype=bool)
    for k in range(len(horizons)):
        mask_multi_rb[k] = base_mask_rb & np.isfinite(fwd_multi_rb[k])

    repr_rb = np.nan_to_num(repr_rb_full, nan=0.0).astype(np.float32)
    fwd_multi_rb = np.nan_to_num(fwd_multi_rb, nan=0.0).astype(np.float32)

    return {
        'aligned': aligned,
        'representation_rb': repr_rb,                  # (n_bars, N, hidden_flat)
        'fwd_multi_rb': fwd_multi_rb,                  # (K, n_bars, N) float32
        'mask_multi_rb': mask_multi_rb.astype(np.float32),  # (K, n_bars, N)
        'base_mask_rb': base_mask_rb.astype(np.float32),    # (n_bars, N)
        'daily_log_ret': daily_log_ret,                # (D, N) float64
        'rebal_idx': rebal_idx,                        # (n_bars,) int64
    }


def train_scorer_horizon_walkforward(
    tickers: list[TickerData], backbone: Backbone, *,
    horizons: tuple[int, ...] = (5, 10, 20, 40, 60),
    train_window_blocks: int = 252,   # fine bars; with h_min=5 that's ~5y
    val_window_blocks:   int = 156,   # ~3y at h_min=5
    step_window_blocks:  int | None = None,
    mlp_hidden:    int = 64,
    mlp_layers:    int = 1,
    n_steps:       int = 200,
    learning_rate: float = 1e-3,
    weight_decay:  float = 0.0,
    entropy_weight: float = 0.0,
    seed:          int = 0,
    commission_bps: float = 10.0,
    temperature: float = 1.0,
    spread:      np.ndarray | None = None,
    max_spread:  float | None = None,
    verbose:     bool = True,
) -> HorizonWalkForwardResult:
    """Walk-forward train + eval the endogenous-horizon head.

    Per window: fresh seeded init, AdamW on `horizon_mixture_loss`. At
    eval, run the head over val bars and:
      1. Endogenous deployment: `simulate_irregular_daily_pnl` with
         `argmax(π_t)` horizon choice. This is the load-bearing metric.
      2. K fixed-h baselines: `simulate_fixed_horizon_daily_pnl` at
         each `h ∈ horizons` using the same scores. The endogenous arm
         must beat the best of these by ≥ 0.10 to count (null N2/N3).
      3. Random-π baseline: `simulate_irregular_daily_pnl` with
         `horizon_picker='sample'`. Endogenous must beat (null N4).

    `train_window_blocks` / `val_window_blocks` count **fine** rebal
    bars (at `min(horizons)` spacing), not days. With `h_min=5` and the
    defaults, train = ~5y, val = ~3y.
    """
    if step_window_blocks is None:
        step_window_blocks = val_window_blocks

    pre = _precompute_horizon_inputs(
        tickers, backbone, horizons=horizons,
        spread=spread, max_spread=max_spread)
    repr_rb_np: np.ndarray = pre['representation_rb']
    fwd_multi_np: np.ndarray = pre['fwd_multi_rb']
    mask_multi_np: np.ndarray = pre['mask_multi_rb']
    base_mask_np: np.ndarray = pre['base_mask_rb']
    daily_log_ret: np.ndarray = pre['daily_log_ret']
    rebal_idx: np.ndarray = pre['rebal_idx']
    aligned: AlignedTickers = pre['aligned']

    n_bars = repr_rb_np.shape[0]
    K = len(horizons)
    slices = block_windows(
        n_bars, train_window_blocks, val_window_blocks, step_window_blocks)
    if not slices:
        raise ValueError(
            f'no walk-forward windows fit: have {n_bars} fine rebal bars '
            f'but each window needs train+val='
            f'{train_window_blocks + val_window_blocks}')

    if verbose:
        print(f'horizon walk-forward: {len(slices)} windows over {n_bars} '
              f'fine bars (h_min={min(horizons)}, K={K}). '
              f'train={train_window_blocks}, val={val_window_blocks}, '
              f'step={step_window_blocks}. n_steps={n_steps} lr={learning_rate} '
              f'wd={weight_decay} ent_w={entropy_weight}')

    result = HorizonWalkForwardResult(
        n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, horizons=tuple(horizons),
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        feature_width=backbone.F, aligned=aligned,
        entropy_weight=entropy_weight,
    )

    pbar = tqdm(slices, desc='horizon walk-forward', unit='window',
                disable=not verbose)
    for w_idx, (train_slc, val_slc) in enumerate(pbar):
        rng = np.random.default_rng(seed + w_idx)
        head_params = init_mlp_horizon(
            rng, backbone.hidden_flat,
            n_horizons=K, hidden=mlp_hidden, n_layers=mlp_layers)
        opt = AdamW(list(head_params.values()),
                    lr=learning_rate, weight_decay=weight_decay)

        repr_train = Tensor(repr_rb_np[train_slc])
        fwd_train  = Tensor(fwd_multi_np[:, train_slc])      # (K, n_train, N)
        mask_train = Tensor(mask_multi_np[:, train_slc])     # (K, n_train, N)
        base_mask_train = Tensor(base_mask_np[train_slc])    # (n_train, N)

        final_loss = float('nan')
        for _ in range(n_steps):
            Tensor.training = True
            opt.zero_grad()
            scores, pi = apply_mlp_horizon(
                head_params, repr_train, base_mask_train)
            loss = horizon_mixture_loss(
                scores, fwd_train, mask_train, pi,
                entropy_weight=entropy_weight)
            loss.backward()
            opt.step()
            final_loss = float(loss.item())

        # --- Eval on val window. ---
        Tensor.training = False
        repr_val = Tensor(repr_rb_np[val_slc])
        base_mask_val = Tensor(base_mask_np[val_slc])
        scores_val_t, pi_val_t = apply_mlp_horizon(
            head_params, repr_val, base_mask_val)
        scores_val = scores_val_t.numpy()    # (n_val, N)
        pi_val = pi_val_t.numpy()            # (n_val, K)

        # Daily window: from rebal_idx[val_slc.start] through
        # rebal_idx[val_slc.stop-1] + max_horizon (truncated by D).
        # Simulator truncates at daily_end automatically.
        D = daily_log_ret.shape[0]
        val_daily_start = int(rebal_idx[val_slc.start])
        # End at the last fine bar + max horizon, capped at D.
        last_bar = val_slc.stop - 1
        val_daily_end = min(int(rebal_idx[last_bar]) + max(horizons), D)

        # The simulator takes fine-grid scores/π/mask over the full
        # n_bars range and indexes into the val window itself, so we
        # pass the raw arrays + start/end daily bounds.
        scores_full = np.nan_to_num(
            scores_val, nan=0.0).astype(np.float64)
        pi_full = np.nan_to_num(pi_val, nan=1.0 / K).astype(np.float64)
        # Renormalize pi just in case nan_to_num shifted rowsums.
        pi_full = pi_full / np.maximum(
            pi_full.sum(axis=1, keepdims=True), 1e-12)
        mask_full = base_mask_np[val_slc].astype(np.float64)
        rebal_idx_val = rebal_idx[val_slc]

        endog = simulate_irregular_daily_pnl(
            scores=scores_full, pi=pi_full, mask=mask_full,
            daily_log_ret=daily_log_ret, rebal_idx=rebal_idx_val,
            horizons=horizons,
            daily_start=val_daily_start, daily_end=val_daily_end,
            commission_bps=commission_bps, temperature=temperature,
            horizon_picker='argmax')

        fixed_results: dict[int, float] = {}
        for h in horizons:
            fr = simulate_fixed_horizon_daily_pnl(
                scores=scores_full, mask=mask_full,
                daily_log_ret=daily_log_ret, rebal_idx=rebal_idx_val,
                horizon=int(h),
                daily_start=val_daily_start, daily_end=val_daily_end,
                commission_bps=commission_bps, temperature=temperature)
            fixed_results[int(h)] = fr.sharpe

        # Random-π baseline: same simulator, sample from π_t. Use a
        # seed derived from window so the comparison is reproducible.
        rand_rng = np.random.default_rng(seed + w_idx + 10_000)
        random_pi = np.ones_like(pi_full) / K
        rand = simulate_irregular_daily_pnl(
            scores=scores_full, pi=random_pi, mask=mask_full,
            daily_log_ret=daily_log_ret, rebal_idx=rebal_idx_val,
            horizons=horizons,
            daily_start=val_daily_start, daily_end=val_daily_end,
            commission_bps=commission_bps, temperature=temperature,
            horizon_picker='sample', rng=rand_rng)

        # Diagnostics for null N1 (π collapse).
        # Entropy of π_t (per bar) and argmax bin counts.
        log_pi = np.log(np.maximum(pi_full, 1e-12))
        ent_per_bar = -(pi_full * log_pi).sum(axis=1)
        ent_mean = float(np.mean(ent_per_bar)) if ent_per_bar.size else 0.0
        argmax_bins = np.argmax(pi_full, axis=1)
        argmax_counts = {
            int(horizons[k]): int(np.sum(argmax_bins == k))
            for k in range(K)
        }

        val_start_idx = aligned.rebal_idx[val_slc.start]
        val_start_date = str(np.asarray(aligned.dates[val_start_idx]).astype(
            'datetime64[D]'))

        if verbose:
            best_fixed = max(fixed_results.values())
            pbar.set_postfix(
                loss=f'{final_loss:+.3f}',
                endog=f'{endog.sharpe:+.2f}',
                best_fixed=f'{best_fixed:+.2f}',
                random=f'{rand.sharpe:+.2f}',
                ent=f'{ent_mean:.2f}')

        result.windows.append(HorizonWindow(
            window_idx=w_idx,
            train_block_start=train_slc.start,
            train_block_end=train_slc.stop,
            val_block_start=val_slc.start,
            val_block_end=val_slc.stop,
            val_daily_start=val_daily_start,
            val_daily_end=val_daily_end,
            train_loss=final_loss,
            val_endog_sharpe=endog.sharpe,
            val_endog_mean_holding=endog.mean_holding_days,
            val_endog_n_rebals=endog.n_rebals,
            val_endog_avg_turnover=endog.avg_turnover,
            val_fixed_sharpes=fixed_results,
            val_random_sharpe=rand.sharpe,
            val_pi_entropy_mean=ent_mean,
            val_pi_argmax_counts=argmax_counts,
            head_params={k: v.numpy() for k, v in head_params.items()},
            val_start_date=val_start_date,
        ))

    if verbose:
        h_best, best_score = result.best_fixed_horizon
        endog_mean = result.mean_val_endog_sharpe
        rand_mean = result.mean_val_random_sharpe
        delta = endog_mean - best_score
        print(
            f'horizon walk-forward done: {result.n_windows} windows. '
            f'mean endog Sharpe = {endog_mean:+.3f}, '
            f'best fixed (h={h_best}) = {best_score:+.3f} '
            f'(delta = {delta:+.3f}), '
            f'random-π = {rand_mean:+.3f}')

    return result


__all__ = [
    'HorizonWindow',
    'HorizonWalkForwardResult',
    'train_scorer_horizon_walkforward',
]
