# Diagnose why w=1 row underperforms in the FiLM (w, n) head

The FiLM-conditioned (w, n) RSI head trained on
`n ∈ {5,7,9,13,17,21,25} × w ∈ {1,5,10,21}` zero-shot on CSCO produces
a smooth surface where every cell hits R² ≥ 0.65, but the w=1 row
plateaus at 0.70–0.89 while w=7 (off-grid interp between trained 5
and 10) reaches 0.86–0.95 — the *off-grid* row beats every in-grid
row at small n. Sweep snapshot (R² for n=7 across w):

| w  | in-grid | R²    |
|----|---------|-------|
| 1  | yes     | 0.80  |
| 3  | no      | 0.84  |
| 5  | yes     | 0.89  |
| 7  | no      | 0.93  |
| 10 | yes     | 0.93  |
| 21 | yes     | 0.92  |
| 25 | no      | 0.87  |

This is striking because RSI(n=7, w=1) ≡ canonical RSI(7), and a
prior single-target run on the same backbone hit R²=0.97 for that
exact target. Two compounding hypotheses (see chat 2026-04-29):

1. **Latent-frequency mismatch.** The bundle's CWT power is dominated
   by long scales (the regime-app finding); the w=7 sweet spot is
   exactly where the target's smoothing matches the latent's dominant
   representation. Daily RSI(w=1) is high-frequency oscillation that
   reads off the *minority* latent channels (small CWT scales 1/2/3 +
   the single log-returns channel).
2. **MLP smoothness penalty / cond-space asymmetry.** With 4 trained
   w-points normalized to `w/max(w)` = `[0.048, 0.238, 0.476, 1.0]`,
   w=1 sits as the spatial outlier near zero. The FiLM γ/β MLPs fit a
   smooth function across cond-space; smoothness means w=1 can't get
   a sharp local specialization without hurting other cells.

Two cheap experiments to disentangle (each is one knob, no
architecture change):

| Run | knob                                                      | tests       |
|-----|-----------------------------------------------------------|-------------|
| L   | Switch w_norm to `log(w)/log(max(w))` (instead of `w/max`). New trained points: `[0, 0.529, 0.756, 1.0]` — distributes spacing more evenly toward the small-w end. | factor 2 only |
| S   | Train with `--rsi-w-grid 1` and the same n-grid (degenerates to 1-D n-only conditioning, FiLM still on). | factor 1 only |

If L lifts (w=1) toward the w=5+ range while leaving the w=5/10/21
rows roughly intact, the cond-spacing was the dominant cause and the
fix is just a normalization tweak. If L doesn't help much but S
recovers RSI(7) to ≥0.95, the latent-frequency cause dominates and
the only paths forward are (a) accept that the multi-w head pays a
fixed tax on daily RSI, or (b) bias the latent toward higher
frequencies (e.g. drop long CWT scales, shorten the rolling z-score
lookback, or oversample short scales in the bundle).

**Implementation:**

- L is a one-line change in `apps/notebook/src/ss_notebook/replay/reconstruct.py`
  around line 148 (`w_max = float(max(rsi_w_grid)); w_values = ...`).
  Add `--rsi-w-norm {linear,log}` flag (default `linear` to preserve
  current behavior) and thread through `fit_and_evaluate`. Inference
  cell needs the matching change in the cond_vec construction, so
  the npz should record the chosen normalization in `_meta` for
  loaders to mirror.
- S is just a CLI flag tweak — no code change.

**Out of scope** for the same diagnostic:

- Larger FiLM hidden width (`--cnn-film-hidden 64+`). If both L and S
  fail, that's the next lever — gives the cond MLPs more capacity to
  represent sharper per-cell specialization.
- Multi-resolution latent (separate small-scale-only pretraining
  branch concat'd with the long-scale latent). Architectural; tackle
  only if S confirms latent-frequency is the dominant cause.
