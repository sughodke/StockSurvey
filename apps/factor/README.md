# `apps/factor` — cross-sectional rank-IC scorer

Per-(date, ticker) score → per-bar Pearson correlation with forward log
returns ("information coefficient") → cross-sectional alpha factor for
portfolio weighting. Tinygrad trainer; numpy I/O. Two input paths share
the same head, the same objective, and the same output:

1. **Pretrained CWT backbone.** Load the SSL-pretrained CNN backbone
   produced by `apps/notebook` (`ss-replay --decoder cnn`), run it
   forward over a universe of `TickerData`, and train a small linear /
   MLP head on the flattened latent. Optional Stage 2 unfreezes the
   backbone for joint fine-tuning at a separate LR.

2. **Deterministic indicator stack.** Skip the encoder. Build a wide
   flat feature vector per (date, ticker) by sweeping technical
   indicators across many parameter values — strided RSI/CCI grids,
   MACD over a fast-period grid, realized vol over a window grid — and
   feed it to the same head through `identity_backbone(K=1, F=…)`. Use
   as an ablation against the pretrained backbone, or as a standalone
   scorer when no SSL pretrain is available.

## Layout

```
src/factor/
├── __init__.py            public surface (re-exports)
├── backbone.py            tinygrad runtime: identity_backbone,
│                          compute_input_stats, apply_backbone,
│                          apply_backbone_pytree, backbone_to_pytree.
│                          Re-exports Backbone + load_backbone from
│                          ss_features (numpy-only npz I/O).
├── data.py                AlignedTickers + align_tickers + forward_log_returns
├── objectives.py          pearson_rank_ic (training), block_sharpe (eval)
├── scorers.py             Linear / MLP head builders + apply
├── indicator_features.py  IndicatorGridConfig + the no-backbone path
└── train.py               train_scorer (Stage 1 + optional Stage 2)
tests/                     (empty placeholder; add tests here)
```

## Quick start

### Pretrained backbone (Stage 1, head only)

```python
from factor import load_backbone, train_scorer
from ss_features import load_ticker

backbone, meta = load_backbone('Output/replay-cnn-2026-05-XX.npz')
tickers = [
    load_ticker(t, stooq_dir='./StooqData', start='2013-01-29', end='2025-12-11',
                **{k: meta[k] for k in (
                    'scales', 'lookback', 'window_cols',
                    'include_zscore_stats', 'include_returns', 'decoder',
                    'rsi_n', 'macd_fast', 'macd_slow', 'macd_signal',
                )})
    for t in ['AAPL', 'MSFT', 'NVDA', '...']  # universe of N tickers
]
res = train_scorer(
    tickers, backbone,
    rebal_days=20, train_frac=0.7,
    scorer='linear', n_steps=500, learning_rate=1e-3, weight_decay=1e-2,
    finetune_steps=0,           # set > 0 to unfreeze backbone in Stage 2
)
print(f'val IC: {res.val_ic:+.4f}   val Sharpe (eval): {res.val_sharpe:+.3f}')
```

### Deterministic indicator stack (no encoder)

```python
from factor import IndicatorGridConfig, load_ticker_indicators, train_scorer_indicators

cfg = IndicatorGridConfig()    # default 74 channels, ~820-bar warmup
tickers = [
    load_ticker_indicators(t, stooq_dir='./StooqData',
                           start='2010-01-01', end='2025-12-11', cfg=cfg)
    for t in UNIVERSE
]
res = train_scorer_indicators(
    tickers, cfg,
    rebal_days=20, train_frac=0.7,
    scorer='linear', n_steps=500, learning_rate=1e-2, weight_decay=1e-3,
    finetune_steps=0,           # no-op here — identity backbone has no conv weights
)
```

Inspect which channels carry the head's weight:

```python
import numpy as np
W = res.params['W']                      # shape (F,)
order = np.argsort(-np.abs(W))
for i in order[:10]:
    print(f'{cfg.channel_names()[i]:>20s}  {W[i]:+.4f}')
```

## Objective

`pearson_rank_ic` (`objectives.py`) is the training signal — per-rebalance
Pearson correlation of head scores with forward log returns, masked to
the liquid universe at that bar, averaged across bars. Loss is
`-pearson_rank_ic`. Despite the name, the implementation correlates raw
scores (Pearson IC), not ranks (Spearman IC); both sides would need an
`argsort` to be true rank-IC.

`block_sharpe` is **eval-only**: annualized portfolio Sharpe at
rebalance granularity with one-sided turnover costs at
`commission_frac`. IC gives a per-decision dense gradient signal that
converges much faster than direct Sharpe optimization.

### IC as a proxy for Sharpe

The formal link is Grinold's Fundamental Law of Active Management:

```
IR ≈ IC · sqrt(BR)
```

where IR is the information ratio (≈ Sharpe of the active component) and
BR is *effective* breadth — independent bets per year. For the baseline
walk-forward run below — 297 names × ~12.6 rebals/year ≈ 3,740 nominal
bets — naive IC=0.012 → IR ≈ 0.74. That's broadly in line with the mean
realized val Sharpe of ~+0.44 across the 6 windows, after sector/beta
correlations deflate "nominal breadth" to "effective breadth."

**Why we train on IC anyway:**
- Monotonic with Sharpe under reasonable assumptions — better IC almost
  always means better Sharpe at fixed sizing/costs.
- Dense per-decision gradient (one signal per (date, ticker)) vs
  Sharpe's sparse per-bar return signal → much lower-variance
  gradients, much faster convergence.
- Scale-invariant — the head can output anything, IC is invariant under
  affine rescaling.

**Where IC and Sharpe diverge (so `block_sharpe` stays as eval):**
1. **IC ignores position sizing.** A scorer with great IC but extreme
   tail concentration can have worse Sharpe than a meh-IC scorer with
   diversified weights. Top-N + per-name cap matter at deployment time.
2. **IC ignores costs.** Turnover, commission, spread are invisible to
   IC. A signal that flips sign every rebal can have high IC and
   negative net Sharpe.
3. **IC is a *mean of correlations*, not a Sharpe of cumulative
   returns.** Time aggregation differs: high IC concentrated in low-vol
   regimes can give lower realized Sharpe than IC predicts.
4. **`pearson_rank_ic` is actually Pearson on raw scores, not Spearman
   on ranks.** Sensitive to outliers in either scores or returns. True
   rank-IC would be more robust but isn't what the code computes.
5. **Effective breadth ≪ N · rebals.** Cross-sectional bets are
   correlated through market beta and sector exposures, so the sqrt(BR)
   multiplier above is optimistic; effective IR is typically a fraction
   of the naive number.

**Bottom line.** IC is the right *training* objective and a reasonable
*ordering* signal — a +0.012 IC head will almost always beat a +0.005 IC
head on realized Sharpe. But IC is not Sharpe, and the val Sharpe range
in the linear baseline (-1.00 to +1.24 across windows, mean ~+0.44) is
the number that would actually trade.

## Where the moving parts live

- `Backbone` dataclass + `load_backbone` (numpy npz I/O) live in
  `packages/features` (`ss_features`) so both this app and
  `apps/notebook` (which writes the npz and runs SSL probes via
  `replay.reconstruct`) can read the format without depending on each
  other. The runtime forward pass (`apply_backbone`, conv stack, etc.)
  is here in `factor.backbone`.
- `TickerData`, `load_prices`, `realized_vol`, `log_returns`, the CWT
  feature builders (`compute_scalogram`, `build_features_and_targets`,
  `load_ticker`, etc.), and `TARGET_NAMES` all live in `ss_features`
  too. `apps/notebook` re-exports the full set through
  `ss_notebook.replay.features` + `ss_notebook.scalogram` for
  back-compat with existing callers.

## Baseline results (deterministic indicator path)

The deterministic stack — strided RSI/CCI grids, MACD over a fast-period
grid, realized vol over a window grid, and rolling Pearson coherence
between short/long realized vol — is the **null-hypothesis baseline** for
the SSL backbone path. If the pretrained CWT encoder cannot beat these
numbers on the same universe / objective / head, it is not earning its
keep.

**Setup.** Stooq US universe, 297 tickers (after `min_history_bars=6500`
drops 15 short-history names so `align_tickers`' strict intersection
keeps a ~26-year common axis). 74 channels at default
`IndicatorGridConfig`. Walk-forward: `train=63 blocks` (~5y),
`val=39 blocks` (~3y), `step=39` (no val overlap). Rebal cadence 20 bars.
6 windows fit. AdamW, `n_steps=200`, `lr=1e-2`, `weight_decay=1e-3`.

| scorer | mean val IC | median val IC | pos-val-IC frac | mean train IC |
|---|---|---|---|---|
| linear | **+0.0120**  | +0.0168 | **5/6** | ~+0.10 |
| mlp    | +0.0081      | +0.0075 | 4/6     | ~+0.36 |

**Reading the result.**
- **Genuine null is rejected** — train IC is robustly positive on every
  window for both heads (linear ~+0.10, mlp ~+0.36). The features carry
  *some* learnable cross-sectional structure, not noise.
- **One-time regime decay is rejected for the linear head** — 5/6 val
  windows positive rules out a single-window edge that vanished.
- **Capacity hurts here** — the MLP triples train IC over the linear
  head but val IC is *lower* and 2/6 val windows go negative. Classic
  overfitting signature; the nonlinear capacity is fitting noise that
  doesn't transfer.
- **Magnitude is small.** Val IC ~+0.012 is a weak alpha — useful in a
  multi-factor portfolio but not a standalone edge.

Artifacts (regenerate via the `walkforward` entrypoint in
`apps/factor/scripts/modal/train_indicator.py`):
- `Output/walkforward-comparison.png` — per-window train vs val IC bars,
  one panel per scorer.
- `Output/walkforward-{linear,mlp}-s200-wd0.001-windows.npz` — per-window
  head params, train/val IC, train/val Sharpe, block bounds.
- `Output/walkforward-summary.json` — aggregate stats per scorer.

**The bar for the backbone path.** Whatever pretrained CWT encoder we
plug into `train_scorer` should clear **mean val IC > +0.012** and
**positive-val-IC fraction ≥ 5/6** on the *same* universe and walk-forward
config, ideally without the linear→MLP overfitting gap. If it can't,
the encoder isn't learning anything that the trailing-window indicator
grid doesn't already encode in closed form.

## SSL backbone results (apples-to-apples, 2026-05-03)

Backbone:
`Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz`,
produced by `apps/replay/scripts/modal/train_cnn_multihead.py
--bundle cwt-only --steps 1000` on the same 297-ticker universe used
by the deterministic baseline above (1000 steps, FiLM heads on
rsi/cci/vol/macd, no zscore-stats, no returns input — see "Why
cwt-only" below). Resulting encoder: K=96 lags × F=30 channels →
2 conv layers (kernel=5, hidden=64) → flat representation **5632 dims
per (date, ticker)**.

Walk-forward evaluated via `train_scorer_walkforward` on the same
6-window protocol (train=63 / val=39 / step=39 blocks, AdamW
`n_steps=200 lr=1e-2 wd=1e-3`, scorers `linear` + `mlp`).

| path | scorer | mean val IC | median val IC | pos-val-IC frac | mean train IC |
|---|---|---|---|---|---|
| **deterministic** | linear | **+0.0120** | +0.0168 | **5/6** | ~+0.10 |
| deterministic    | mlp    | +0.0081     | +0.0075 | 4/6     | ~+0.36 |
| SSL backbone     | linear | -0.0047     | -0.0109 | 2/6     | ~+0.55 |
| SSL backbone     | mlp    | +0.0076     | +0.0099 | 4/6     | **~+0.91** |

**Verdict: the SSL backbone does not earn its keep at this
configuration.** Both SSL paths underperform `deterministic + linear`
on every aggregate (mean / median val IC, positive-val-IC fraction).
The diagnostic is sharp: train IC scales monotonically with head
capacity (`0.10 → 0.36 → 0.55 → 0.91` as we go det-lin → det-mlp →
SSL-lin → SSL-mlp) while val IC stays pinned near zero. Classic
under-constrained-head overfitting — 5632 input dims × ~9400 train
decisions/window = the head can memorize train without finding any
generalizable structure.

This is **not a death sentence for the SSL path**, but it does
identify the next experiments to try, in increasing order of cost:

1. **Crank weight_decay** (1e-3 → 1e-1 or 1e0). Pure CLI flip on the
   same harness. Expect the highest-impact change because the
   train→val gap is so wide.
2. **PCA the 5632-dim representation to ~100 dims** before the head.
   Forces the head to use only the high-variance directions of the
   encoder. Code change in `factor.train`.
3. **Stage 2 fine-tune** (unfreeze backbone, train head + backbone
   jointly per window). `train_scorer_walkforward` deliberately
   doesn't expose this — extension required. Highest potential
   impact if the issue is that the SSL reconstruction objective
   doesn't align with cross-sectional IC.

Artifacts (regenerate via the `walkforward` entrypoint in
`apps/factor/scripts/modal/train_ssl_walkforward.py`):
- `Output/ssl-walkforward-comparison.png` — per-window train vs val IC
  bars, one panel per scorer.
- `Output/ssl-walkforward-{linear,mlp}-s200-wd0.001-windows.npz`
  — per-window head params + IC + Sharpe + block bounds.
- `Output/ssl-walkforward-summary.json` — aggregate stats per scorer
  + the backbone metadata it used.

### Why `cwt-only`, not the documented `full` recipe

The `full` bundle in `apps/replay`
(`--include-zscore-stats --include-returns`) was tuned for
*reconstruction R²* on the SSL pretrain task. Including raw returns
as an input channel lets the price head shortcut via
`price_t = price_{t-1} * exp(return_t)` and the vol head shortcut via
`vol = std(returns over window)` — two of the five reconstruction
heads can essentially ignore CWT, and reconstruction R² goes way up.

For an SSL → factor pipeline that quality is poison: it means the
encoder is mostly a passthrough for returns, and the latent the
factor head sees is "returns with a thin CWT layer," which contains
nothing the deterministic indicator stack didn't already encode in
closed form. `cwt-only` strips both shortcuts and forces the encoder
to learn from wavelet features only.

`Output/ssl-attention-comparison.png` makes this visible: AAPL FiLM
input-attention saliency for the rsi head under the `full` bundle
(top, smoke run) shows the entire long-period saliency mass landing
on the single `return` channel; under `cwt-only` (bottom, full
pretrain) the saliency spreads across CWT coeff and power scales,
with high-frequency scales dominating short-period RSI(7,1) and
low-frequency scales dominating long-period RSI(17,10) — exactly
what we want from a CWT encoder.

## Caveats

- **Default indicator grid needs ~820 bars of history.** The largest
  CCI cell at the trimmed default (`n=40, w=21`) needs `(n-1)·w + 1 =
  820` bars before the row is fully valid. The pre-trim default
  (`n=80, w=63`) needed 4978 bars and was choking walk-forward windows
  on shorter universes — re-enable those cells in `cci_n_grid` /
  `cci_w_grid` only if the universe has the history for it.
- **Pool z-norm only.** `make_indicator_backbone` uses
  `compute_input_stats` for per-channel z-norm across all valid (date,
  ticker) rows. This handles wildly different indicator scales (RSI in
  [0, 100], CCI ~±200, vol ~0.01) but does *not* standardize within
  each rebalance bar across the cross-section. Per-bar standardization
  would match the IC objective more strictly but needs trainer-loop
  changes.
- **Finetune Stage 2 is a no-op for the indicator path.** The identity
  backbone has no conv weights to update; only the head trains. Pass
  `finetune_steps=0` to skip the wasted second pass.
- **No CLI.** Driven from notebooks and the four scripts under
  `apps/factor/scripts/` (`no_backbone_baseline*.py` and
  `colab/{ssl,stage1}_ic_scorer.py`). Add an `[project.scripts]` entry
  to `pyproject.toml` if a CLI becomes useful.
