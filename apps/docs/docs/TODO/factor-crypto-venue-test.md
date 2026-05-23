# Crypto venue port — factor indicator-grid walk-forward on CryptoCompare top-50

**Status: `pending` — scaffold built, smoke-tested locally, Modal eval not yet kicked off.**

The single most-promising re-point from
`.research-venue-fit.md` (at repo root): port
the 74-channel deterministic indicator grid + tinygrad linear/MLP head
that delivered val IC +0.0212 (6/6 windows) on US equities at
`rebal_days=5` to **CryptoCompare daily OHLCV, top-50 crypto** at the
same horizon. Mechanism: Liu-Tsyvinski (2022) documents a 3-factor
crypto cross-section (market/size/momentum) ~3-4× stronger than equity
momentum; if the indicator grid extracts cross-sectional alpha at all
in equities, crypto's larger inefficiency should lift it.

## Pre-registered falsification bar (locked, do not adjust post-hoc)

`mean val IC > +0.025 ; 4/5 positive windows ; DSR deflated-t > +1.5`

Stored in the script docstring AND written to the result NPZ as
`pre_registered_bar`. Recorded honestly per CLAUDE.md's "Recording
findings" protocol regardless of outcome.

## Files (commit `04bf48d`, on master)

- `apps/factor/scripts/modal/train_indicator_crypto.py` — Modal
  entrypoint (linear + MLP scorers, walk-forward, dumps NPZ with
  `periods_per_year=73.0` for crypto's 7-day week, all required DSR
  keys).
- `apps/factor/scripts/prep_crypto_universe.py` — local CryptoCompare
  fetch → `Output/crypto_universe_panel.pkl`. Inlines a v2-compatible
  CryptoCompare fetcher (the workspace's `ss_loaders.load_cryptocompare`
  is broken — see open question #1 below).

## Local smoke-test result (n_steps=50, linear only)

```
5 windows  mean val IC = +0.0504  pos-val-IC frac = 0.80
mean val Sharpe long-only = −0.108   long-short = +0.541
```

**IC reads as encouraging vs the bar** (+0.0504 > +0.025; 4/5 positive)
— **but at n_steps=50 it's not the real eval**, and no DSR deflation
has been applied yet. Do not read this as a verdict. The real eval at
n_steps=200 is what gets a leaderboard row.

## Modal launch command (when you want to spend ~$0.10)

```bash
# One-time data fetch (~4 min, ~6MB):
uv run python apps/factor/scripts/prep_crypto_universe.py

# Real eval on Modal T4 (~10-20 min, < $0.10):
uvx modal run apps/factor/scripts/modal/train_indicator_crypto.py::walkforward \
    --scorers linear,mlp --n-steps 200 --weight-decay 1e-3
```

## Honest caveats (preserved from build report; address before/during eval)

1. **`ss_loaders.load_cryptocompare` is broken.** It hits CryptoCompare's
   retired v1 endpoint whose response shape (`Data` as a top-level
   list with a `time` column) was replaced — the v2 endpoint nests the
   data at `Data.Data`. The v1 path raises
   `AttributeError: 'DataFrame' object has no attribute 'time'`. The
   prep script inlines a v2-compatible fetcher; **the library function
   should be fixed in a separate small PR** — out of scope for this
   experiment.
2. **Top-50 → 42 effective tickers**: TON, SHIB, ICP, APT, ARB, OP,
   GRT, FLOW are all 2021+ vintage and starve the walk-forward common
   axis under strict date-intersection alignment. Recorded in the NPZ
   as `universe_label='cryptocompare_top50_v1'`. **This is documented,
   not a bug** — same survivorship-shape problem we already document
   for equities.
3. **`commission_bps=10` is the equity baseline.** Real crypto: top-tier
   1-5 bps, long-tail alts 50 bps+. Documented in the docstring; left
   as a deflation-friendly first cut so the result is comparable with
   the existing factor row. **Follow-up arm**: re-run with a crypto-realistic
   cost grid (5 bps top / 25 bps mid / 50 bps long-tail) to see how
   much of any positive result is bps-dependent.
4. **Spot-only — `borrow_bps_yr=0` explicit.** The L/S stream is a
   conceptual market-neutral construct, not a deployable trade on spot.
   Perp-based deployment needs separate funding-rate accounting. If
   the L/S clears the bar, the next test is a perp construction with
   funding-rate cost charged in the stream (data: Binance funding-rate
   history is free).

## Close-out protocol after the Modal eval finishes

In one pass, per CLAUDE.md's "Executing a TODO end-to-end" section:

1. **Append a leaderboard row** to `apps/docs/docs/leaderboard.md`
   with the pre-reg-bar verdict (`confirmed-OOS` if all three gates
   clear, `partial-OOS` if some clear, `confirmed-null` if none).
2. **Add an ArcSpec** to `apps/docs/scripts/compute_dsr.py` for both
   the long-only and L/S streams. Set `sharpe_std_ann` conservatively
   for crypto: **0.35-0.50 annualized** (crypto cross-trial dispersion
   is likely higher than equities' empirical 0.245 — top-N alts have
   wider Sharpe variation across windows than US equity factors).
3. **Re-rank the DSR ladder** and update the table in
   `apps/docs/docs/leaderboard.md`.
4. **Write a finding page** `apps/docs/docs/findings/factor-crypto-venue.md`
   with the standard structure (lede operational rule, eval setup,
   per-window numbers, mechanism discussion, master walk-forward log
   pointer).
5. **Update this TODO**: either mark `Done` with a pointer to the
   finding (if the workstream closes here) or rewrite to the next
   experiment (e.g. cost-grid arm, perp construction, expanding to
   top-100).
6. **Add the finding to**:
   - `apps/docs/mkdocs.yml` Findings nav
   - `apps/docs/docs/findings/index.md`
7. **Commit** the docs changes as a single follow-up commit referencing
   `04bf48d` (the scaffold commit).

## Hypothesis decision tree (what each verdict implies for the next arc)

| outcome | next experiment |
|---|---|
| `confirmed-OOS` (all 3 gates clear; deflated-t > +1.5) | (a) cost-grid stress test to bound the bps-dependence; (b) perp construction with funding-rate accounting; (c) extend to top-100; (d) if all still hold, scope a `ss-crypto live` adapter alongside the existing Alpaca path. |
| `partial-OOS` (1-2 gates clear; e.g. IC bar yes but DSR no) | Pivot to per-arm investigation: which windows carry the signal? Is it 2022-23 (post-FTX dispersion) vs 2024-25 (concentrated rally)? Stratify by realized cross-sectional vol regime. |
| `confirmed-null` (no gates clear) | The repo's "cross-section is bound on free public-equity data" claim extends to crypto too — a meaningful negative result. Reframe to **perp funding-rate cash-and-carry** as the next venue-fit arc (the independent structural-premium recommendation in `.research-venue-fit.md` #5). |

## Cross-links

- Origin: `.research-venue-fit.md` (at repo root)
  — the venue-fit brief, where crypto > forex >> prediction-markets.
- Mechanism reference: Liu, Tsyvinski (2022) "Risks and Returns of
  Cryptocurrency", *Review of Financial Studies*.
- Reused harness: [`factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md)
  — the equity result this is porting (val IC +0.0212 at 5d).
- Methodology: [`deflated-sharpe-leaderboard`](../findings/deflated-sharpe-leaderboard.md)
  — the DSR-deflation framework new arcs are ranked on.
