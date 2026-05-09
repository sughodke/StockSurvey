# Relational analog k-NN — raw polar Morlet bundle overfits the Phase-2 OOS gate

**Operational rule: do NOT swap `Output/relational-analog.json` to
`wavelet='morlet'` *yet*.** The canonical Phase-2 analog cross_ticker
scoring stays on the real Ricker kernel until at least one of the
regularization-flavor follow-up experiments listed below clears the
gate.

The polar Morlet infrastructure
(`ss_features.causal_polar_morlet_matrix`,
`scalogram_cache(wavelet='morlet')`,
`weights_regime_analog(wavelet='morlet')`) is preserved as research
infrastructure. **The result reported here is "raw bundle on N=21
overfits", not "the bundle is information-theoretically bad."**
That distinction matters — the same bundle is the SSL trainer's
canonical input ([replay-dwt-compression](replay-dwt-compression.md))
where dense per-bar reconstruction targets and weight-decay
regularization keep the extra channels honest. A signal that hurts
raw L2 distance on N=21 can still be useful through a learned head
on N=300, or through a regularized fingerprint on the same N=21.

## Setup

Phase-2, 2013-01-29 → 2025-12-11, 21 tickers, analog cross_ticker
pool, top-10 rebal-20d, 10bps commission,
`k=50, h=20, fp_window=21, lookback=120,
scales=[5,7,10,12,21,26,50,90], min_sep=21`.

The two arms differ only in the wavelet. Same scores routine
(`analog_knn_scores_fast` with `n_workers=20` on Modal), same kNN
machinery, same downstream selection.

| Arm                | Wavelet           | Channels per scale | fp_dim |
|--------------------|-------------------|--------------------|--------|
| `analog-ricker`    | real Ricker       | 1 (signed coeff)   | 168    |
| `analog-morlet`    | polar Morlet + g  | 4 (`\|c\|, cos, sin, g`) | 672    |

The Morlet bundle adds the bandpass amplitude / phase pair plus the
Gaussian (lowpass) trend companion, computed on cumulative
log-returns so growth stays additive across train→val. fp_dim grows
4× — kNN matmul cost grows linearly.

## Walk-forward result (2026-05-09, 3-arm)

![Phase-2 equity, analog ricker / Morlet / Morlet-DWT-L1](images/relational-morlet-phase2-equity.png)

Canonical Phase-2 train (2013-01-29 → 2020-12-31) / val (2021-01-01 →
2025-12-11) split. Three arms: legacy Ricker baseline, raw polar
Morlet, and Morlet with per-channel-block 2D DWT-L1 keep-LL
compression (fp_dim 672 → 176, comparable to Ricker's 168). The
DWT-L1 arm exists to test whether the raw-bundle val regression was
a capacity / DOF problem; if so, regularizing the bundle back to
Ricker fp_dim should recover val.

| Arm                   | Window | Sharpe | Sortino | CAGR    | MaxDD   |
|-----------------------|--------|--------|---------|---------|---------|
| `analog-ricker`       | full   | 1.0560 | 1.2693  | +20.62% | -37.06% |
| `analog-ricker`       | train  | 1.0104 | 1.1410  | +19.87% | -37.06% |
| `analog-ricker`       | val    | **1.1456** | 1.5480  | +22.13% | -31.44% |
| `analog-morlet`       | full   | 1.0804 | 1.3178  | +22.43% | -44.14% |
| `analog-morlet`       | train  | **1.2401** | 1.4085  | +26.39% | -32.88% |
| `analog-morlet`       | val    | **0.8358** | 1.1469  | +16.54% | -44.14% |
| `analog-morlet-dwtL1` | full   | 0.9257 | 1.1144  | +18.61% | -41.03% |
| `analog-morlet-dwtL1` | train  | 0.9926 | 1.1303  | +20.35% | -36.56% |
| `analog-morlet-dwtL1` | val    | **0.8242** | 1.0931  | +16.05% | -41.03% |

Delta blocks:

|        | morlet − ricker | morlet-dwtL1 − ricker | morlet-dwtL1 − morlet |
|--------|-----------------|------------------------|------------------------|
| full   | +0.024          | -0.130                 | -0.155                 |
| train  | **+0.230**      | -0.018                 | **-0.247**             |
| val    | **-0.310**      | **-0.321**             | -0.012                 |

(Ricker numbers shifted by ~0.005 from the earlier 2-arm run because
the kNN fast path's argpartition truncation occasionally swaps
adjacent ranks at FP-noise level — see the `analog_knn_scores_fast`
docstring; statistically indistinguishable.)

## Read

Two findings stack:

**Raw Morlet overfits.** Train +0.23 / val −0.31 vs Ricker — the
classic train>val sign-flip every other Phase-2 arm exhibited in the
earlier 8-arm DWT A/B (cross_ticker Ricker was the lone reversal
anomaly there; see [DWT failure](relational-dwt-failure.md)).

**DWT-L1 fixes the overfit but doesn't recover val.** Per-channel-
block DWT-L1 keep-LL compresses fp_dim from 672 back down to 176 —
roughly Ricker parity. Train Sharpe drops from 1.24 to 0.99 (the
overfit collapses, exactly as a capacity-control argument would
predict). But **val stays at 0.82** — almost identical to the raw
Morlet's 0.84. Both Morlet arms cluster at val ≈ 0.83 regardless of
their wildly different train scores.

That is the load-bearing observation. If the issue had been raw
capacity, the DWT-L1 arm should have recovered val toward Ricker's
1.15. It didn't. Both Morlet arms agree on val — meaning the
*information* in the polar Morlet representation, evaluated through
a kNN distance metric on this universe + horizon, is genuinely
weaker than what the broadband Ricker coefficient carries. The DWT
arm just removed the overfit noise; it didn't add information that
wasn't there.

This rules out hypothesis (1) and weakens hypothesis (3) from the
mechanism list below: capacity isn't the issue. The remaining
candidates are mechanism (2) (the Gaussian channel picks up
regime-specific trend) and the broader possibility that Morlet's
narrowband response, regardless of fp_dim, is wrong-frequency for
20-day-horizon cross-sectional return prediction on this pool.

The mega-cap-specific finding from
[relational-universe-shift](relational-universe-shift.md) is also
relevant: Phase-2 cross_ticker Ricker val Sharpe of 1.146 was already
suspect (it collapses to ~0.48 off mega-caps). The Morlet arms
landing at val 0.83 — between the mega-cap Ricker number (1.15) and
the wider-universe Ricker number (~0.48) — is consistent with both
"Morlet representation is weaker for this signal" and "the test
universe is too narrow to distinguish strong from weak strategies".

## Possible mechanisms

Three non-exclusive hypotheses for why the bandpass-with-phase
representation overfits the Phase-2 train window:

1. **Phase channels are noisy on a small universe.** With N=21 the
   cross_ticker candidate pool at any rebalance bar is ~21 × t
   pairs, sparser than the wider stooq_us_long pool. Adding 2× more
   distance dimensions (cos, sin) into a sparse pool means the kNN
   nearest-50 grow more idiosyncratic, fitting train particulars
   that don't recur at val.
2. **The Gaussian companion picks up regime-specific trend.**
   2013-2020 was a long bull run; cumulative log-returns drift
   monotonically up. The Gaussian channel `g` over that window has
   a strong DC component that the kNN treats as a similarity axis.
   2021-2025 includes 2022's bear and the AI mega-cap rotation —
   the same `g` channel reads "different macro" and erases the
   match.
3. **Morlet narrowband response amplifies long-scale specificity.**
   Morlet at `omega0=6` is narrowband at `1/scale`; Ricker is
   broadband around `1/scale`. The 50d / 90d scales (where the
   regime trainer's strongest signal lives, per the JAX-Adam
   finding) become more discriminating under Morlet — which also
   means more easily memorized.

Distinguishing these would need a 3-arm A/B (Morlet without `g`,
Morlet without phase, Morlet with both). Not run — the operational
verdict (don't pin Morlet on `analog`) doesn't depend on which
mechanism dominates.

## Notes

- Implementation: `ss_features.causal_polar_morlet_matrix`
  (matrix-form 4-channel panel),
  `relational.scalogram_cache.load_or_compute_cwt(wavelet='morlet')`
  (cache key includes the wavelet name so Ricker / Morlet panels
  coexist), `RelationalCheckpoint.wavelet` field (defaults
  `'ricker'`). Live dispatch in
  `relational.inference._build_weights_panel` raises
  `NotImplementedError` if any of the five other strategies is
  loaded with `wavelet='morlet'` — paper-trade-safe.
- Reproducibility: `uvx modal run apps/relational/scripts/modal/
  relational_morlet_phase2.py` after the
  `prep_phase2_prices.py` prep step.
- Artifacts: `Output/relational-morlet-phase2-{equity.png,
  stats.txt, walkforward.csv, walkforward.txt}`.
- Master walk-forward log:
  [Leaderboard](../leaderboard.md) — once the row lands it carries
  the [`reversed-OOS`](../leaderboard.md#verdict-labels) verdict
  alongside the existing analog cross_ticker rows.
- The polar Morlet bundle remains the canonical SSL CNN input for
  `apps/replay`. The CNN learns the bundle differently from a kNN
  distance metric — a signal that hurts metric-space matching can
  still help a learned head. The replay reconstruction R² result
  ([replay-dwt-compression](replay-dwt-compression.md)) was
  independent of this kNN finding.

## What this result is — and isn't

After the DWT-L1 arm: capacity-control is ruled out as the
explanation for the val regression on Phase-2 — both Morlet arms
land at val ≈ 0.83. The remaining ambiguity is whether the
information gap is universal (the Morlet representation is just
weaker for kNN-distance-based cross-sectional return prediction
regardless of pool) or pool-specific (Phase-2 is so narrow that
*any* representation lands close to the mega-cap floor of 0.48–1.15
once the model can't pattern-match aggressively).

The wider-universe rerun (gating experiment 2 below) is the deciding
test. It's been wired and is in flight as of 2026-05-09.

What this result still does *not* show, even after the 3-arm:

- That the bundle is uninformative for cross-sectional return
  prediction at all.
- That the same bundle through a *learned* head (the
  `apps/factor` linear/MLP path) would also fail — that path has
  weight decay + dense per-bar IC targets, which are exactly the
  regularizers raw kNN distance lacks.
- That the bundle would fail on a wider candidate pool. Phase-2 was
  always going to be a low-information test; `stooq_us_long`
  (N=312) is the same Modal harness on a 15× larger pool and
  produces a directly-comparable train/val split.

## Gating experiments before judging the bundle

Three follow-ups, ordered by cost (cheapest first). The migration to
the other five relational strategies is paused on these — not
abandoned. Either of (1) or (2) clearing would be enough to
reverse the operational verdict.

### (1) Regularize the bundle: 3-arm Phase-2 with DWT-L1 — RAN, did not flip the verdict

`analog-morlet-dwtL1` ran in the same Modal entrypoint as a third
arm. Per-channel-block 2D Haar keep-LL collapses fp_dim 672 → 176.
**Train Sharpe dropped from 1.24 to 0.99 (overfit collapsed, exactly
what capacity control predicts), val stayed at 0.82** — virtually
identical to the raw Morlet's 0.84. Both Morlet arms cluster at val
≈ 0.83, so capacity is not the explanation for the regression.

Two consequences for the migration:

- **Capacity is not the issue on Phase-2.** A pure regularization
  fix doesn't work; the val signal is what's weak. Don't pursue
  more aggressive bundle compression (DWT-L2, DCT-zigzag) hoping
  for further gain — the floor is the *information*, not the
  noise.
- **Phase-2 isn't a sharp test for the bundle.** Mega-cap val
  Sharpe ranges 0.48–1.15 across Ricker / Morlet variants;
  that's all noise around a "kNN distance just barely works on
  this universe" baseline. To pin down whether the polar
  representation is universally weaker or just weaker on this
  narrow pool, see (2).

### (2) Wider universe: rerun analog A/B on `stooq_us_long` — IN FLIGHT

`relational_morlet_stooq_long.py` (Modal, with the persistent
`ss-relational-cwt-cache` volume) runs the same three arms on the
312-ticker `apps/notebook/data/stooq_us_long` universe. Walk-forward
split is the same canonical Phase-2 dates so train/val numbers are
directly comparable. Outputs land at `Output/relational-morlet-
stooq-long-{equity.png, stats.txt, walkforward.csv,
walkforward.txt}`.

Predicted outcomes:

  - If Morlet val ≥ Ricker val on this pool → bundle is
    informative, Phase-2 was just too narrow to distinguish — the
    operational verdict flips and the bundle's a candidate for the
    other five strategies.
  - If Morlet val ≪ Ricker val on this pool too → the bundle is
    universally weaker for kNN-distance cross-sectional return
    prediction; halt the migration to the other strategies as a
    distance-metric primitive (the SSL CNN path remains unaffected
    — different objective, different regularization).

### (3) Channel ablation: 3-arm A/B isolating which channel hurts

Three sub-arms — `analog-morlet-no-g` (drop the Gaussian
companion, keep `(|c|, cos, sin)`),
`analog-morlet-no-phase` (drop phase, keep `(|c|, g)`), and the
existing `analog-morlet` baseline. Distinguishes Mechanism 2
(Gaussian picks up regime-specific trend) from Mechanism 1 / 3
(phase or narrowband response). Lowest priority — useful for the
*why*, but doesn't change the operational verdict on its own. Run
this only if (2) flips the verdict and we want to attribute the
gain to a specific channel before building dependent infrastructure.

The migration to the other five relational strategies (`empirical`,
`gmm`, `farthest`, `diversified`, `velocity`) waits on (2). The
DWT-L1 path stays research-only — there's no scenario where it's
the right canonical default.
