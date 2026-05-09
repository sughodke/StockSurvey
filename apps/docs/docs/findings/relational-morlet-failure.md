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

## Walk-forward result (2026-05-09)

![Phase-2 equity, analog ricker vs polar Morlet](images/relational-morlet-phase2-equity.png)

Canonical Phase-2 train (2013-01-29 → 2020-12-31) / val (2021-01-01 →
2025-12-11) split:

| Arm             | Window | Sharpe | Sortino | CAGR    | MaxDD   |
|-----------------|--------|--------|---------|---------|---------|
| `analog-ricker` | full   | 1.0614 | 1.2753  | +20.75% | -37.06% |
| `analog-ricker` | train  | 1.0191 | 1.1497  | +20.07% | -37.06% |
| `analog-ricker` | val    | **1.1456** | 1.5480  | +22.13% | -31.44% |
| `analog-morlet` | full   | 1.0804 | 1.3178  | +22.43% | -44.14% |
| `analog-morlet` | train  | 1.2401 | 1.4085  | +26.39% | -32.88% |
| `analog-morlet` | val    | **0.8358** | 1.1469  | +16.54% | -44.14% |

Delta morlet − ricker:

|        | Δsharpe | Δsortino | Δcagr   | Δmaxdd   |
|--------|---------|----------|---------|----------|
| full   | +0.019  | +0.043   | +0.017  | -0.071   |
| train  | **+0.221** | +0.259  | +0.063  | +0.042   |
| val    | **-0.310** | -0.401  | -0.056  | -0.127   |

## Read

Classic train>val sign-flip overfit signature:

- Ricker: train 1.02 → val **1.15** (val beats train by +0.13 — the
  anomaly that made Phase-2 cross_ticker the canonical winner in the
  earlier 8-arm DWT A/B; see [DWT failure](relational-dwt-failure.md)
  for the reference baseline).
- Morlet: train **1.24** → val 0.84 (val below train by -0.40 — the
  *typical* reversal pattern every other Phase-2 arm exhibited).

The polar Morlet bundle's extra channels (phase pair + Gaussian
companion) give the kNN distance metric more degrees of freedom to
discriminate between historical fingerprints, which improves the fit
on 2013-2020 (+0.22 train Sharpe) but hurts OOS in 2021-2025
(-0.31 val Sharpe). The full-period number (+0.019) hides the
divergence — judging on full-period alone would have shipped a
strictly worse strategy.

The mega-cap-specific finding from
[relational-universe-shift](relational-universe-shift.md) is also
relevant: Phase-2 cross_ticker val Sharpe of 1.146 was already
suspect (it collapses to ~0.48 off mega-caps). Morlet appears to
amplify whatever was specifically working on the Phase-2 mega-cap
basket and lose the rest.

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

A hard "the bundle is bad" conclusion would require one of:
**(a)** the same overfit signature on a wider universe (rules out
small-N as the cause), or **(b)** the same signature with a
regularized variant of the bundle (rules out raw-capacity as the
cause). Neither has been run yet. Until then, this finding is
narrowly: *the raw polar Morlet bundle, used as a one-shot kNN
distance metric on the Phase-2 21-ticker universe, overfits its
2013-2020 train slice and gives back the train edge plus more in
2021-2025.*

What it does *not* yet show:

- That the bundle is uninformative for cross-sectional return
  prediction at all.
- That the same bundle through a *learned* head (the
  `apps/factor` linear/MLP path) would also fail — that path has
  weight decay + dense per-bar IC targets, which are exactly the
  regularizers raw kNN distance lacks.
- That the bundle would fail on a wider candidate pool. The
  small-N hypothesis (Mechanism 1 above) predicts the OOS gap
  shrinks or reverses on `stooq_us_long` (N=312); that's not
  tested here.

## Gating experiments before judging the bundle

Three follow-ups, ordered by cost (cheapest first). The migration to
the other five relational strategies is paused on these — not
abandoned. Either of (1) or (2) clearing would be enough to
reverse the operational verdict.

### (1) Regularize the bundle: 3-arm Phase-2 with DWT-L1

Add `analog-morlet-dwtL1` as a third arm of the existing Modal
script: same polar Morlet panel, but pass
`Compression(kind='dwt', levels=1, wavelet='haar',
pad_mode='periodization')` through `extract_fingerprints` per
channel block (DWT must NOT mix channels, since `|c|` /
`cos` / `sin` / `g` are heterogeneous). fp_dim drops `672 → ~176`
— back to the same ballpark as Ricker's 168, while keeping the
phase + trend information the polar bundle adds. If the DWT-L1
val Sharpe matches or beats Ricker, the verdict flips: the bundle
*is* informative, the raw L2 distance just needed the same
low-pass denoise that Ricker effectively gets for free from its
single broadband channel.

### (2) Wider universe: rerun analog A/B on `stooq_us_long`

Mirror `relational_morlet_phase2.py` against the
`stooq_us_long`-prepped panel (N=312, ~15× larger candidate pool).
This is the direct test of Mechanism 1 (sparse-pool overfit).
Expectation if the hypothesis is right: train Sharpe gap
shrinks, val Sharpe gap shrinks or reverses. If the same
sign-flip happens at N=312, the universe-size argument is wrong
and the bundle has a deeper problem.

### (3) Channel ablation: 3-arm A/B isolating which channel hurts

Three sub-arms — `analog-morlet-no-g` (drop the Gaussian
companion, keep `(|c|, cos, sin)`),
`analog-morlet-no-phase` (drop phase, keep `(|c|, g)`), and the
existing `analog-morlet` baseline. Distinguishes Mechanism 2
(Gaussian picks up regime-specific trend) from Mechanism 1 / 3
(phase or narrowband response). Lowest priority — useful for the
*why*, but doesn't change the operational verdict on its own.

The migration to the other five relational strategies (`empirical`,
`gmm`, `farthest`, `diversified`, `velocity`) waits on (1) or (2)
clearing. If neither does, *then* the original "halt" verdict
applies and the bundle stays research-only for distance scorers.
