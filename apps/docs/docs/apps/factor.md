# Factor

Cross-sectional rank-IC scorer (tinygrad). Two input paths feed the
same head + objective:

1. The SSL-pretrained CNN backbone produced by `ss-replay --decoder
   cnn`, loaded via `ss_features.load_backbone`. The replay trainer's
   per-target reconstruction heads are discarded; only the shared
   conv stack survives into the npz.
2. `IndicatorGridConfig` — a 74-channel deterministic stack of strided
   RSI/CCI grids, MACD over a fast-period grid, realized vol over a
   window grid, and rolling Pearson coherence between short/long
   realized vol — fed through `identity_backbone(K=1, F=74)`.

Both paths share the rest of the stack: a fresh **linear or MLP head**
initialized per run, an **AdamW loop minimizing `-pearson_rank_ic`**
against forward log-returns at rebalance granularity, and a per-window
walk-forward wrapper that rolls a fresh head across rolling
train/val splits. Sharpe is logged via `block_sharpe` for evaluation
only — rank-IC gives a per-rebalance dense gradient signal that
converges much faster than direct Sharpe optimization.

## How factor uses the replay backbone

`factor.train.train_scorer` is two-stage:

- **Stage 1 (always runs).** The frozen backbone is forward-passed
  over every (date, ticker) feature row once up front
  (`precompute_inputs`); latents materialize on host as numpy. A
  fresh linear or MLP head streams Tensor minibatches of that cached
  representation and runs `n_steps` AdamW updates against rank-IC.
  Cheap — the backbone forward is amortized.
- **Stage 2 (optional, off by default).** When `finetune_steps > 0`,
  the cached representation is dropped (it goes stale once weights
  move) and the encoder is re-applied per-step on raw features.
  Backbone params get `learning_rate * finetune_lr_scale` (default
  0.1×) so the pretrained features get nudged, not overwritten; head
  stays at full lr.

`feat_mu` / `feat_sd` (input z-norm) are not optimized in either
stage — the backbone keeps seeing the same input distribution it was
pretrained on. The **input bundle factor builds at runtime** is the
[polar Morlet bundle](../TODO/polar-morlet-migration.md) from
`ss_features.load_ticker` (`build_features_and_targets`) — 7 channels
per scale (`|c|, |c|², cos(arg), sin(arg), g, g², log_L2_amp`) over
a `window_cols`-bar lag stack. Fresh runs against the new-bundle
backbones in `Output/` are end-to-end consistent; older 2-channel
backbones in `Output/` are stale and will fail with a shape mismatch
on the first conv (worth deleting before they confuse anyone).

Walk-forward eval (`train_scorer_walkforward`) is the OOS protocol;
the rolling-train-and-fresh-head loop is the only honest answer to
overfitting in this regime.

## The deterministic indicator baseline

![74-channel IndicatorGridConfig walk-forward — linear vs MLP heads](images/factor-indicator-grid.png)

What 74 hand-designed channels can buy you on a 297-ticker universe at
20-day horizon: [mean val IC of **+0.0120**](../findings/factor-indicator-baseline.md)
with a linear head, 5/6 windows positive. The MLP head triples train
IC and then gives most of it back on val — the classic overfit
signature. The +0.012 number is the line every later experiment has
been trying to clear, and the [Leaderboard](../leaderboard.md) records
every arm that didn't.

## SSL backbone — does the encoder beat the indicators?

![SSL walk-forward across 6 windows](images/factor-ssl-walkforward.png)

The promise of SSL pretrain was simple: an encoder that's seen the
full structure of the CWT bundle should expose return-predictive
geometry the linear-on-indicators path can't reach. The promise is
testable. The figure above is the test — six rolling windows, val IC
overlaid against the [indicator
baseline](../findings/factor-indicator-baseline.md). The takeaway is
gentle and honest: the SSL-pretrained latent does not lift val IC off
the +0.012 ceiling. The data is the bottleneck, not the encoder. Why this
matters for the broader research program is unpacked in the
[Notes](../notes.md#self-supervised-pretrain-why-and-how) and the
[supervision-is-binding finding](../notes.md#what-we-already-know-about-supervision-being-the-binding-constraint).

## What the SSL latent attends to

![SSL backbone attention compared across heads](images/factor-ssl-attention.png)

A learned-attention readout of which slices of the CWT bundle each
indicator head reaches into. Worth lingering on: the heads partition
the latent space cleanly — RSI activates the high-frequency end, vol
activates the long-scale end, MACD lands in the middle. The encoder
*has* learned the structure of the indicator family. It just can't
turn that structure into return alpha at this universe and horizon —
which is itself a finding, not a failure.

## The numbers in tabular form

The aggregate verdicts and per-window stats live on the
[Leaderboard](../leaderboard.md) and the
[factor indicator-IC baseline](../findings/factor-indicator-baseline.md)
page. The list of pivots that tested *whether the +0.012 is
horizon-bound, universe-bound, or supervision-bound* is in the
[Notes](../notes.md) closing section.
