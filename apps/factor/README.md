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
from ss_notebook.replay import load_ticker

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

cfg = IndicatorGridConfig()    # default ≈ 79 channels, ~5000-bar warmup
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

## Where the moving parts live

- `Backbone` dataclass + `load_backbone` (numpy npz I/O) live in
  `packages/features` (`ss_features`) so both this app and
  `apps/notebook` (which writes the npz and runs SSL probes via
  `replay.reconstruct`) can read the format without depending on each
  other. The runtime forward pass (`apply_backbone`, conv stack, etc.)
  is here in `factor.backbone`.
- `TickerData` + `realized_vol` + `load_prices` are in `ss_features`
  too; `apps/notebook` re-exports them through `ss_notebook.replay.features`
  and `ss_notebook.scalogram` for back-compat.

## Caveats

- **Default indicator grid needs ~5000 bars of history.** The largest
  CCI cell (`w=63, n=80`) needs `(n-1)·w + 1 = 4978` bars before the
  row is fully valid. Shrink `cci_w_grid` / `cci_n_grid` for shorter
  universes.
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
  `apps/notebook/scripts/` (`no_backbone_baseline*.py` and
  `colab/{ssl,stage1}_ic_scorer.py`). Add an `[project.scripts]` entry
  to `pyproject.toml` if a CLI becomes useful.
