"""12-1 cross-sectional momentum — the canonical durable cross-sectional
anomaly the repo never tested at its proper horizon.

Pre-registered single hypothesis (NOT an Optuna sweep over horizon — that
would be the Sullivan-Timmermann-White data-snooping trap the cross-sectional
+ search-based literature warns against). Signal: at each rebal date, score
each name by its cumulative log-return over the [t-252, t-21] formation window
(12 months, skip the most-recent 1 month to drop short-term reversal, per
Jegadeesh-Titman 1993). Long winners / short losers, dollar-neutral, via the
validated `factor.objectives.block_port_returns_long_short_np` constructor.

Evaluated on the full point-in-time history (the rule is fixed — no fitted
parameters, so there is no train/val split; the whole series is OOS) and run
through the cross-arc Deflated-Sharpe harness. Because this is pre-registered
(a named academic factor, not a search winner), the PSR (n_trials=1) is the
honest primary metric — it corrects for higher moments + sample length but not
selection, because we did not select. A deflated read (n_trials=6, the
term-structure {1mo, 12-1, 36mo} ladder it belongs to) is reported alongside.

    uv run python apps/factor/scripts/momentum_12_1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from factor.objectives import block_port_returns_long_short_np
from ss_portfolio import standardize_oos

REPO = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
OUT = REPO / 'Output'
PPY = 252.0 / 21.0  # monthly rebal blocks


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end', default='2026-04-01')
    p.add_argument('--formation', type=int, default=252)  # 12 months
    p.add_argument('--skip', type=int, default=21)         # drop recent 1 month
    p.add_argument('--hold', type=int, default=21)         # monthly rebal/hold
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--borrow-bps-yr', type=float, default=50.0)
    args = p.parse_args()

    man = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    rows = man['tickers'] if isinstance(man, dict) else man
    names = sorted(t['ticker'].upper() for t in rows
                   if t.get('n_bars', 0) >= args.min_history_bars)
    print(f'universe: {len(names)} names (n_bars>={args.min_history_bars})')

    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        str(STOOQ_SUBSET), tickers=names, start_date=args.start, end_date=args.end,
        min_history=args.formation + args.hold + 10)
    panel = closes.to_numpy()
    T, N = panel.shape
    print(f'panel: {T} dates x {N} tickers')

    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(np.isfinite(panel) & (panel > 0), np.log(panel), np.nan)

    F, S, H = args.formation, args.skip, args.hold
    # Non-overlapping monthly rebal dates with full formation history + a
    # forward holding window in range.
    rebal_idx = np.arange(F, T - H, H)
    R = len(rebal_idx)

    scores = np.full((R, N), np.nan)
    blr = np.full((R, N), np.nan)
    for r, i in enumerate(rebal_idx):
        # 12-1 formation: cum log-return from t-F to t-S (skip recent S).
        scores[r] = logp[i - S] - logp[i - F]
        # realized forward holding-period log-return.
        blr[r] = logp[i + H] - logp[i]
    mask = (np.isfinite(scores) & np.isfinite(blr)).astype(np.float64)
    scores = np.nan_to_num(scores, nan=0.0)
    blr = np.nan_to_num(blr, nan=0.0)
    print(f'rebal blocks (monthly, non-overlapping): {R}; '
          f'avg names/block: {mask.sum(axis=1).mean():.0f}')

    cf = args.commission_bps / 1e4
    net = block_port_returns_long_short_np(scores, blr, mask, cf)
    # Borrow on the gross short leg (~0.5 of L1=1), per block.
    net = net - (args.borrow_bps_yr / 1e4) * (1.0 / PPY) * 0.5

    sd = net.std(ddof=0)
    ann = float(net.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0
    psr = standardize_oos(net, periods_per_year=PPY, n_trials=1)
    ladder = standardize_oos(net, periods_per_year=PPY, n_trials=6)

    # Per-window consistency: 6 equal time-chunks (regime-robustness proxy).
    chunks = np.array_split(net, 6)
    chunk_sh = [float(c.mean() / (c.std(ddof=0) + 1e-12) * np.sqrt(PPY)) for c in chunks]
    pos_chunks = int(np.sum(np.array(chunk_sh) > 0))
    hit = float((net > 0).mean())

    print(f'\n--- 12-1 momentum L/S (factor-narrow {N}, monthly, {R} blocks) ---')
    print(f'  ann Sharpe       {ann:+.3f}   (per-block mean {net.mean():+.5f})')
    print(f'  skew {psr.skew:+.2f}  kurt {psr.kurtosis:.2f}  block hit-rate {hit:.2f}')
    print(f'  PSR (n_trials=1, pre-registered):  DSR {psr.dsr:.3f}  t {psr.deflated_tstat:+.3f}')
    print(f'  deflated (n_trials=6, ladder):     DSR {ladder.dsr:.3f}  t {ladder.deflated_tstat:+.3f}')
    print(f'  6-chunk Sharpes: {[round(x,2) for x in chunk_sh]}  '
          f'({pos_chunks}/6 positive)')

    np.savez(OUT / 'momentum-12-1-returns.npz',
             ls_block_returns=net.astype(np.float64),
             periods_per_year=np.float64(PPY))
    (OUT / 'momentum-12-1-summary.json').write_text(json.dumps({
        'universe_size': int(N), 'formation': F, 'skip': S, 'hold': H,
        'n_blocks': int(R), 'ann_sharpe': ann,
        'psr_dsr': psr.dsr, 'psr_tstat': psr.deflated_tstat,
        'ladder_dsr': ladder.dsr, 'ladder_tstat': ladder.deflated_tstat,
        'skew': psr.skew, 'kurtosis': psr.kurtosis, 'block_hit_rate': hit,
        'chunk_sharpes': chunk_sh, 'positive_chunks': pos_chunks,
    }, indent=2))
    print(f'\n-> {OUT / "momentum-12-1-returns.npz"}')


if __name__ == '__main__':
    main()
