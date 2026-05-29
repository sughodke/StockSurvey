# `apps/ensemble` — learned 2-leg DCA + vol_v3 ensemble

Production orchestrator for the learned blend that beats the
deterministic (DCA + 2x vol_v3) recipe per
[`findings/learned-ensemble-beats-deterministic`](../docs/docs/findings/learned-ensemble-beats-deterministic.md).

## What this app is

A thin orchestrator over the existing `dca.live` and `vol.live` rails.
It owns:

- **`EnsembleCheckpoint`** — JSON serialisation of `(w_dca, w_vol)`,
  the train window the weights were fit on, and the paths to the
  per-leg checkpoints the live runner dispatches into.
- **`ss-ensemble train`** — fits the blend on a strictly-prior window
  via either closed-form diagonal mean-variance or gradient ascent on
  Sharpe (default: gradient ascent — more interpretable scale).
- **`ss-ensemble live --dry-run | --live`** — dispatches both legs
  with the learned scales (`gross_scale=w_dca` to DCA;
  `vega_scale=w_vol` to vol). Inherits each leg's existing risk rails
  and adds one ensemble-level kill switch + one shared dry-run gate.
- **`ss-ensemble inspect`** — prints the checkpoint contents for the
  operator.

## Risk rails

The ensemble runner uses the **union** of:

1. **Per-leg rails (5 each):** DCA's kill switch + freshness +
   cadence/drift + position cap + dry-run; vol's kill switch +
   freshness + cadence + position cap + dry-run, plus
   [vol rail #6 realized-friction monitor](../docs/docs/findings/vol-v3-sleeve-sizing.md).
2. **Ensemble-level (2):**
   - `~/.ensemble-killswitch` — touch this file to abort both legs
     simultaneously before any per-leg dispatch.
   - `--dry-run` default — `--live` is opt-in per dispatch, even when
     a `--live` was previously used.

Per-leg kill switches (`~/.dca-killswitch`, `~/.vol-killswitch`,
broker-specific) still fire independently — the ensemble kill switch
is the operator's single "halt everything" lever; it does not bypass
the per-leg rails it composes with.

## Training the checkpoint

```bash
uv run ss-ensemble train \
    --dca-close-pkl Output/cfr_phase4d_multiasset_close.pkl \
    --vol-v3-npz Output/vol-v3-dolthub-oos-c200-returns.npz \
    --train-start 2023-08-02 \
    --train-end 2025-12-11 \
    --learner grad_sharpe \
    --dca-checkpoint Output/dca-phase4d.json \
    --vol-checkpoint Output/vol-v3-canonical.json \
    --out Output/learned-ensemble-v1.json \
    --name learned-ensemble-v1
```

The trainer reads the DCA basket close pickle and the vol_v3 alpha
NPZ, aligns the vol stream to daily by spreading each rebal's alpha
evenly across the rebal window, fits the blend on `[train_start,
train_end]`, and writes a versioned JSON checkpoint.

## Running live

```bash
# dry-run dispatches both legs without submitting
uv run ss-ensemble live --params Output/learned-ensemble-v1.json

# inspect what's in the checkpoint
uv run ss-ensemble inspect --params Output/learned-ensemble-v1.json

# go live — opt-in per dispatch, even after a prior --live
uv run ss-ensemble live --params Output/learned-ensemble-v1.json --live

# halt everything mid-rebal
touch ~/.ensemble-killswitch
```

## Ship plan — required operator steps

1. **DCA leg.** Build a `DCACheckpoint` per `apps/dca` instructions
   (the 13-ETF Phase 4d basket). Verify dry-run reaches paper Alpaca.
2. **Vol leg.** Pick options broker (Tradier $10/mo recommended per
   the post-2020 finding), set credentials, build a `VolCheckpoint`
   per `apps/vol` instructions.
3. **Ensemble checkpoint.** Run `ss-ensemble train` against the two
   leg checkpoints. Output: `(w_dca, w_vol)`. Confirm they match the
   range in the
   [finding](../docs/docs/findings/learned-ensemble-beats-deterministic.md)
   (around `w_dca ≈ 0.05, w_vol ≈ 2.24` on the full vol_v3 window).
4. **Dry-run cycle.** `ss-ensemble live --params ... --dry-run` for
   at least **6 rebal cycles** before going live. Monitor:
   - `~/.vol-friction-history.csv` — rail #6 must hold (mean rolling-3
     `c_options_bps ≤ 250`).
   - Each leg's `LiveRunResult` for `aborted_reason` / `rejected_orders`.
5. **Live cutover.** Add `--live` once dry-run is clean. Start with a
   small notional cap (e.g. $5K total).
6. **Rebuild cadence.** Re-run `ss-ensemble train` quarterly with the
   latest `train_end` so weights track recent σ/α structure.

## Caveats — important honesty

**The learned weights concentrate risk on vol_v3.** Mean-variance
optimization does not penalize concentration; the deterministic (1,
2) recipe's DCA presence was partly a risk-shape choice not captured
by Sharpe alone. Operationally that means:

- vol_v3 outage = portfolio outage in this recipe.
- vol_v3's measured Sharpe is academic-clean (no options-broker
  friction beyond the declared 10 bps DoltHub model). Rail #6 is the
  enforcement gate.
- vol_v3 substrate has only 33 monthly rebal observations. The MV
  optimum at this sample length is genuinely the MV optimum, but the
  σ/α structure could shift; the learner has no built-in robustness
  to non-stationarity. Quarterly re-training mitigates but does not
  eliminate this risk.
- The deterministic 2x recipe is the conservative fallback; a hand-
  edited checkpoint with `w_dca = 1.0, w_vol = 2.0` reproduces it.

## Layout

```
src/ensemble/
    __init__.py
    persist.py    — EnsembleCheckpoint JSON I/O
    train.py      — build_streams, fit_mv_closed_form, fit_grad_sharpe
    live.py       — run_live orchestrator + kill-switch + dry-run gate
    cli.py        — ss-ensemble entrypoint (train / live / inspect)
tests/
    test_persist.py
```
