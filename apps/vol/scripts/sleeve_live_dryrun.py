"""DCA + vol_v3 sleeve dry-run operational driver.

Prints the would-be vol_v3 sleeve orders for today's rebalance, sized as a
sleeve fraction on top of canonical DCA capital. Does NOT submit anything.

Per the 2026-05-24 vol-sleeve-sizing finding:
  - Pre-reg verdict: partial-OOS
  - Recommended sizing: vega_scale = 2.0 (overlay multiplier on per-rebal
    block returns; under the capital-free overlay model this maps to a
    moderate notional vega budget — see GAP_TO_LIVE_TRADING.md notes
    below).
  - Friction tolerance: holds CI-excludes-0 up to c_options_bps ~= 200;
    collapses at c_options_bps = 400.

This script is the OPERATIONAL ARTIFACT — what you'd cron at 09:35 ET each
trading day to see what vol_v3 *would* trade if a real options broker were
wired. It reuses `vol.live.run_live` with dry_run=True and prints the
strangle legs + sleeve sizing math.

Run from repo root:
  uv run python apps/vol/scripts/sleeve_live_dryrun.py \\
      --vol-params Output/vol-v3.json \\
      [--max-total-vega 5000]

Gap to live trading (documented inline below — also see the finding
page's "Operational rule" section):

  1. ALPACA OPTIONS — `vol.live` already wires `OptionHistoricalDataClient`
     + `TradingClient` (options endpoints). Paper-options is supported
     but multi-leg submission flow is `submit_strangles` and is currently
     gated behind `dry_run=False`. This script DOES NOT flip that flag.

  2. PAID DATA — real-time chain quotes (bid/ask/OI) for the top-200 OI
     universe across 1000+ underlyings is rate-limited on the free
     Alpaca tier. For production, expect to need either:
       - Alpaca Algo Trader Plus (~$99/mo) for higher options-chain
         throughput, OR
       - Tradier sandbox/production ($10/mo; full options chains; no
         pattern-day-trader limits since it's options not equity), OR
       - IBKR with the options-quote bundle (most expensive, most
         flexible).
     None of these are wired today. The `_default_alpaca_clients()`
     helper in `vol/live.py` assumes Alpaca.

  3. CREDENTIALS — needed env vars (already declared in the codebase):
       ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL (paper:
       https://paper-api.alpaca.markets). The user must arrange these.

  4. SLEEVE SIZING — `vol/ensemble.py`'s `run_ensemble` already computes
     the joint DCA + vol sleeve. The decision the operator must lock at
     deployment time is the dollar-vega budget (today defaulted to
     $5,000 via --max-total-vega). The finding recommends sizing this
     as ~2.0x the per-rebal DCA-block dollar exposure. Translating
     vega_scale=2.0 into dollar-vega requires the operator's account
     equity at rebal-time — that's done by `vol.live` itself, not here.

  5. NO COMMIT WIRING — this script doesn't and won't ever call
     `submit_strangles`. The handoff from "print what we'd do" to
     "actually trade" requires the operator to:
       a) review printed orders by eye,
       b) flip `dry_run=False` in `ss-vol live --live`,
       c) confirm paper-vs-live env routing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


# Sleeve-sizing recommendation from
# apps/docs/docs/findings/vol-sleeve-sizing.md
RECOMMENDED_VEGA_SCALE = 2.0
RECOMMENDED_C_BPS_CEILING = 200
DEFAULT_VEGA_BUDGET_USD = 5000.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--vol-params', default='Output/vol-v3.json',
                   help='Path to a VolCheckpoint JSON (default vol-v3.json)')
    p.add_argument('--max-total-vega', type=float,
                   default=DEFAULT_VEGA_BUDGET_USD,
                   help='Portfolio-level cap on total |net vega| in USD')
    p.add_argument('--max-data-age-days', type=int, default=3)
    p.add_argument('--killswitch', default=None,
                   help='Override the kill-switch file path')
    p.add_argument('--no-broker', action='store_true',
                   help='Skip the broker call entirely; just print the '
                        'sleeve sizing recommendation and exit')
    args = p.parse_args()

    print(f'\n=== vol_v3 sleeve dry-run ===')
    print(f'  vol checkpoint    : {args.vol_params}')
    print(f'  recommended vega  : {RECOMMENDED_VEGA_SCALE}x DCA per-rebal block')
    print(f'  c_bps tolerance   : c_options_bps <= {RECOMMENDED_C_BPS_CEILING}')
    print(f'  vega budget cap   : ${args.max_total_vega:,.0f}')

    if args.no_broker:
        print(f'\n  --no-broker set; skipping broker call.')
        print(f'  Per the sleeve-sizing finding, deploy at vega_scale='
              f'{RECOMMENDED_VEGA_SCALE} with the v3 architecture '
              f'(VIX gate + top-200 OI liquidity filter).')
        print(f'\n  See apps/docs/docs/findings/vol-sleeve-sizing.md '
              f'for the eval and apps/docs/docs/apps/vol.md for the per-app '
              f'overview.')
        return 0

    # --- Run dry-run live (no broker calls if Alpaca creds missing) ---
    try:
        from vol.live import DEFAULT_KILLSWITCH, format_run, run_live
    except ImportError as e:
        print(f'\n  cannot import vol.live: {e}', file=sys.stderr)
        print(f'  ensure ss-vol is installed via `uv sync --all-packages`',
              file=sys.stderr)
        return 2

    ckpt_path = REPO_ROOT / args.vol_params if not Path(args.vol_params).is_absolute() else Path(args.vol_params)
    if not ckpt_path.exists():
        print(f'\n  checkpoint not found at {ckpt_path}', file=sys.stderr)
        return 2

    try:
        result = run_live(
            ckpt_path,
            dry_run=True,
            max_total_vega_usd=args.max_total_vega,
            max_data_age_days=args.max_data_age_days,
            killswitch_path=args.killswitch or DEFAULT_KILLSWITCH,
        )
    except NotImplementedError as e:
        print(f'\n  run_live raised NotImplementedError (likely an '
              f'unwired sub-step):\n    {e}', file=sys.stderr)
        print(f'\n  This is expected today — the chain-query and '
              f'multi-leg submit paths are scaffolded but not all sub-steps '
              f'are wired against a live options broker.', file=sys.stderr)
        return 3
    except Exception as e:
        print(f'\n  run_live raised {type(e).__name__}: {e}', file=sys.stderr)
        print(f'\n  Most likely cause: ALPACA_API_KEY / ALPACA_SECRET_KEY '
              f'env vars not set, or the account does not have options '
              f'entitlements enabled. Set credentials per `vol/live.py`s '
              f'_default_alpaca_clients() helper to proceed.',
              file=sys.stderr)
        return 4

    print('\n' + format_run(result))

    # --- Sleeve-sizing audit ---
    print(f'\n=== Sleeve sizing audit ===')
    print(f'  At recommended vega_scale={RECOMMENDED_VEGA_SCALE}, the '
          f'dollar-vega budget should be sized ~= {RECOMMENDED_VEGA_SCALE}x '
          f'the per-rebal DCA dollar block exposure.')
    print(f'  Operator MUST: review printed strangles by eye, confirm '
          f'paper-vs-live env routing, and flip --live only after the '
          f'dry-run has been clean for at least one full VIX-gate-fired '
          f'rebal cycle.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
