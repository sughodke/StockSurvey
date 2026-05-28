# CNC followups — venue port is the only adjacent path

**Status:** the original two pre-locked followups (basis-tracking-error
stress + funding-regime gate) landed 2026-05-28. Both closed the
deployability question on Hyperliquid with hard negatives. See
[`findings/cnc-funding-rate-carry`](../findings/cnc-funding-rate-carry.md#stress--gate-followups-2026-05-28).

## What was tested and closed

| Followup | Verdict | Why it closed |
|---|---|---|
| (A) Basis-tracking-error stress | **`friction-fragile`** | Break-even basis drag at Sh=+1.0 is ~4.07 bps/d; below the 5 bps/d "real-world OKX cross-basis tracking" target and far below the 10 bps/d "HL→OKX cross-venue execution" target. |
| (B) Funding-regime gate | **falsified** at all thresholds {0.5, 1.0, 2.0, 5.0} bps/d | Higher thresholds protect 2026 but drag 2024-25 mean Sh down by ≥2.08 vs the ≤1.0 ceiling. Cross-coin trailing-mean signal is too coarse — top-K already lives in the high-funding alt-tail while BTC/ETH zeros pull the universe mean below the gate. |

## What remains open — venue port

The CNC arc is closed at the substrate level (`confirmed-OOS`) and at
the deployment level (`deployment-falsified`). The only adjacent path
that could re-open deployability is **a venue with materially higher
gross funding yield than current-regime Hyperliquid**, where the
break-even basis drag would move from 4 bps/d out to ≥10 bps/d.

### Pre-reg hypothesis (locked before any port begins)

**Hypothesis:** On a venue where gross daily funding averages ≥ 8
bps/d over the eval span (vs HL's pooled ~3.3 bps/d, 2024-2026), the
pre-reg cell (K=5, rebal=1d, sign=positive, trail=30d) clears the
**deployment-robust bar of net Sharpe ≥ +1.0 at 10 bps/d basis-drag
stress**.

**Mechanism (steel-man for confirmation):** Binance / Bybit run the
deepest perpetuals order books; OKX is the third pole. Institutional
basis-trade arb capital is concentrated there and would have driven
funding *down* not up vs HL — *except* that HL's DEX architecture
attracts mostly retail leverage, while CEX perp markets are net
shorter-funded by perp-hedged spot ETFs (post-2024 BTC/ETH ETF
launch). Net direction is empirical; both signs are arguable.

**Failure modes to pre-name:**

- Venue funding is *lower* than HL (institutional arb compressed it
  more) → confirms HL was already at the upper edge of the yield
  distribution. Arc closes for real, no more venues to try.
- Venue funding is comparable to HL → same 4 bps/d break-even, same
  deployment-falsified verdict. Arc closes.
- Venue funding is materially higher than HL → reopen, run the same
  stress + gate followups on the new substrate, and if the new
  break-even drag is ≥10 bps/d, the deployment-robust verdict
  unlocks.

### Test design

| Field | Value |
|---|---|
| Venues to attempt (in priority order) | (1) OKX paid history feed; (2) Binance via VPN-routed service or paid mirror; (3) Bybit via paid mirror |
| Universe | top-20 by current-snapshot $-volume per venue, joint history floor 180d |
| Folds | calendar years available on the venue; HL eval used 2024 / 2025 / 2026YTD; ports must span at least 2 calendar years for the verdict to hold |
| Pre-reg cell | K=5, rebal_days=1, sign=positive, trailing_window=30 |
| Sweep | reuse the existing 36-cell × 4-drag-level grid for robustness |
| Bar (deployment-robust) | net Sharpe ≥ +1.0 at 10 bps/d basis drag |
| Bar (confirmation of HL substrate result) | net Sharpe ≥ +1.0 at 0 bps/d basis drag (proves the venue has a non-trivial gross-yield carry) |
| Bar (deployment-falsified) | net Sharpe < +1.0 at 5 bps/d basis drag |

### Implementation gating

Cannot start without one of:

- A paid OKX historical-funding feed (most likely Coinglass / Glassnode / Kaiko)
- VPN or proxy access to Binance `/fapi/v1/fundingRate` (currently HTTP 451 from this host)
- Same for Bybit (currently HTTP 403)

When data access is provisioned, add a `packages/loaders/src/ss_loaders/<venue>.py`
loader following the `hyperliquid.py` template (panel builders +
on-disk parquet cache). The `cnc` app code itself needs no changes —
`apps/cnc/scripts/run_walkforward.py` and `run_stress_and_gate.py`
both accept the venue-panel via `cnc.data.build_panels`'s output, so
swapping the loader is sufficient.

## Closed sub-bullets (do not re-open)

- ~~(A) Basis-tracking-error stress~~ — landed 2026-05-28, friction-fragile.
- ~~(B) Funding-regime gate~~ — landed 2026-05-28, falsified at all thresholds.
- ~~Improve the gate to per-coin or top-K trailing mean~~ — not worth
  it; the (A) result makes (B) moot since even the ungated strategy
  fails the deployable bar regardless of which regime gate is on
  top.
