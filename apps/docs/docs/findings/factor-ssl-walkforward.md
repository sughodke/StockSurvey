# Factor SSL backbone walk-forward — polar Morlet bundle does not clear the indicator baseline

**Operational rule: the polar-Morlet-pretrained replay backbone +
freshly-initialized rank-IC head does not beat the
[deterministic-indicator baseline](factor-indicator-baseline.md)
of mean val IC = +0.012 on the 297-ticker stooq_us_long /
20-day-horizon protocol.** Confirmed by direct rerun on the new
polar Morlet bundle (commit `3451900`) — same conclusion the
prior 2-channel-bundle run reached, so the bundle migration didn't
change the headline factor verdict.

## Setup

- Backbone: `Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz`,
  frozen. K=96 (window cols), F=105 (= 7 channels per scale × 15
  scales = polar Morlet bundle), hidden=64, n_layers=2,
  hidden_flat=5632. Trained by `apps/replay --decoder cnn` on a
  295-ticker stooq_us_long subset with 4 reconstruction targets
  (rsi, macd, vol, cci).
- Universe: 297 stooq_us_long tickers post-`min_history_bars=6500`
  filter, date range `2000-01-03 → 2026-04-01`.
- Walk-forward: 6 rolling windows over 311 rebal-blocks, train=63 /
  val=39 / step=39 blocks at `rebal_days=20`. Fresh AdamW (head
  only, frozen backbone) per window.
- Heads: `linear` (5632 → 1) and `mlp` (5632 → hidden=64 → 1, 1
  hidden layer), both 200 AdamW steps at `lr=1e-2`,
  `weight_decay=1e-3`. Backbone fine-tune disabled (Stage 1 only).
- Modal `cpu=4, gpu=T4, memory=192GB` after three OOM-bumps
  (`cac94e1` → `ed4043f` → memory + copy-elim in `3451900`).

## Walk-forward result (2026-05-09)

![SSL backbone vs indicator baseline, 6 rolling windows](images/factor-ssl-walkforward.png)

Per-window IC summary:

| Window | linear train_ic | linear val_ic | mlp train_ic | mlp val_ic |
|--------|----------------|---------------|--------------|------------|
| 0      | +0.497         | -0.000        | +0.765       | -0.022     |
| 1      | +0.511         | +0.001        | +0.747       | -0.025     |
| 2      | +0.561         | +0.003        | +0.827       | -0.035     |
| 3      | +0.577         | -0.002        | +0.894       | +0.010     |
| 4      | +0.534         | +0.010        | +0.835       | -0.012     |
| 5      | +0.555         | +0.006        | +0.884       | +0.012     |

Aggregates:

| Head   | n_steps | wd   | mean val IC | median val IC | pos-val frac | mean val Sharpe |
|--------|---------|------|-------------|---------------|--------------|-----------------|
| linear | 200     | 1e-3 | **+0.0031** | +0.0020       | 0.67 (4/6)   | +0.41           |
| mlp    | 200     | 1e-3 | **-0.0120** | -0.0171       | 0.33 (2/6)   | (negative)      |
| [indicator baseline](factor-indicator-baseline.md) — linear | 200 | 1e-3 | **+0.0120** | +0.0168 | 0.83 (5/6) | ~+0.44 |
| [indicator baseline](factor-indicator-baseline.md) — mlp    | 200 | 1e-3 | +0.0081     | +0.0075       | 0.67 (4/6)   | — |

## Read

The SSL linear head lands at ~1/4 of the indicator linear head's
val IC. The SSL MLP head goes *negative* on average — worse than
random — even though its train IC averages +0.83 (so the head is
learning something, just nothing that generalizes).

This is the same conclusion the prior factor SSL run reached on
the legacy 2-channel bundle. The bundle migration to polar Morlet
(`ss_features.causal_polar_morlet_matrix`, 7 channels per scale)
did not move the needle on this benchmark. Two readings:

1. **The bottleneck isn't the encoder.** Train IC of +0.50–0.89
   shows the heads can fit the latent → forward-return mapping
   easily; the val IC of ~0.003–0.012 (linear) or ~−0.012 (mlp)
   shows the mapping doesn't survive the 39-block validation
   window. That's data-noise / supervision-binding, not encoder
   capacity.
2. **The polar Morlet representation is informative for SSL
   reconstruction targets but not for cross-sectional return
   prediction at this horizon.** This matches the same split we
   see on the relational side: the bundle improves analog kNN val
   Sharpe by +0.17 on stooq_us_long (see
   [relational-morlet-failure](relational-morlet-failure.md)) —
   different objective, different result. Cross-sectional 20-day
   return prediction at this universe size is a hard ceiling that
   neither the deterministic indicator stack nor the SSL latent
   has crossed.

## Notes

- Modal memory budget was bumped to 192 GB after two OOMs
  (64 → 128 → 192). Even 192 wasn't enough until `3451900`
  removed three redundant copies in `factor.train.precompute_inputs`
  (`.astype(np.float32, copy=False)` × 2, plus pre-allocate
  `repr_full` instead of `np.concatenate(chunks_list)`). The
  alignment step's natural peak (~156 GB during the
  `align_tickers` panel copy) is the binding constraint now;
  refactoring `align_tickers` to consume the input list in place
  would drop peak to ~80 GB and unlock smaller Modal instances.
- Reproducibility: `uvx modal run apps/factor/scripts/modal/
  train_ssl_walkforward.py` after the polar-Morlet backbone npz
  is on local disk under `Output/`.
- Artifacts: `Output/ssl-walkforward-{summary.json,
  comparison.png, linear-s200-wd0.001-windows.npz,
  mlp-s200-wd0.001-windows.npz}`.

## Outstanding question

Whether a Stage 2 fine-tune (joint head + backbone, backbone lr
scaled 0.1×) lifts val IC. The walk-forward harness intentionally
doesn't expose Stage 2 because it would multiply per-window cost
by the fine-tune step count (cached representation goes stale on
every backbone update). If the supervision-is-binding hypothesis is
right, fine-tuning shouldn't help — the data ceiling is the data
ceiling. If it lifts val IC noticeably, the encoder *was* the
bottleneck and the freeze-and-fresh-head pattern is leaving signal
on the table. Worth a single-window probe before deciding whether
to wire it.
