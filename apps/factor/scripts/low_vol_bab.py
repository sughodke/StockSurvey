"""Low-volatility cross-sectional factor — the BAB / low-vol family.

Pre-registered single hypothesis (NOT an Optuna sweep). The structural
anti-edge: high-volatility / high-beta names persistently *underperform* on a
risk-adjusted basis because leverage-constrained and lottery-seeking investors
bid them up (Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014 BAB;
Bali-Cakici-Whitelaw 2011 anti-MAX). Taking the other side — long low-vol,
short high-vol — is one of the most robust cross-sectional premia, and it is
the durable survivor the repo never tested (it tested momentum/reversal/
indicators, not the low-vol family). This is also the honest, structural form
of "invert the reliably-bad investments": shorting what naive demand overpays
for, where the badness is genuine negative *gross* alpha (so it survives the
inversion cost-algebra), not churn.

Signal at each rebal date: score = -(trailing realized volatility of daily
log-returns over the formation window). Long lowest-vol / short highest-vol,
dollar-neutral, via the validated `factor.objectives.block_port_returns_long_short_np`.

CAVEAT (flagged in the row): a dollar-neutral long-low-vol/short-high-vol book
carries a NEGATIVE net-beta tilt (low-vol names are low-beta), so it loses on
beta in bull markets. True beta-neutral BAB levers the legs to neutralize
beta; this clean dollar-neutral version is the first pass — a weak/negative
result is partly the beta drag, not only the anomaly failing.

    uv run python apps/factor/scripts/low_vol_bab.py
"""
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end', default='2026-04-01')
    p.add_argument('--vol-lookback', type=int, default=252)  # 12mo trailing vol
    p.add_argument('--hold', type=int, default=21)            # monthly rebal/hold
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
        min_history=args.vol_lookback + args.hold + 10)
    panel = closes.to_numpy()
    T, N = panel.shape
    print(f'panel: {T} dates x {N} tickers')

    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(np.isfinite(panel) & (panel > 0), np.log(panel), np.nan)
    dlr = np.diff(logp, axis=0)  # (T-1, N) daily log-returns

    V, H = args.vol_lookback, args.hold
    rebal_idx = np.arange(V + 1, T - H, H)
    R = len(rebal_idx)

    scores = np.full((R, N), np.nan)
    blr = np.full((R, N), np.nan)
    for r, i in enumerate(rebal_idx):
        # trailing realized vol over the V daily returns ending at bar i-1
        # (known at i); score = -vol so low-vol gets the long (high) score.
        window = dlr[i - 1 - V:i - 1]
        with np.errstate(invalid='ignore'):
            vol = np.nanstd(window, axis=0)
        scores[r] = -vol
        blr[r] = logp[i + H] - logp[i]
    mask = (np.isfinite(scores) & np.isfinite(blr)).astype(np.float64)
    scores = np.nan_to_num(scores, nan=0.0)
    blr = np.nan_to_num(blr, nan=0.0)
    print(f'rebal blocks (monthly, non-overlapping): {R}; '
          f'avg names/block: {mask.sum(axis=1).mean():.0f}')

    cf = args.commission_bps / 1e4
    net = block_port_returns_long_short_np(scores, blr, mask, cf)
    net = net - (args.borrow_bps_yr / 1e4) * (1.0 / PPY) * 0.5

    sd = net.std(ddof=0)
    ann = float(net.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0
    psr = standardize_oos(net, periods_per_year=PPY, n_trials=1)
    ladder = standardize_oos(net, periods_per_year=PPY, n_trials=6)
    chunks = np.array_split(net, 6)
    chunk_sh = [float(c.mean() / (c.std(ddof=0) + 1e-12) * np.sqrt(PPY)) for c in chunks]
    pos_chunks = int(np.sum(np.array(chunk_sh) > 0))
    hit = float((net > 0).mean())

    print(f'\n--- low-vol L/S (long low-vol / short high-vol, factor-narrow {N}, '
          f'monthly, {R} blocks) ---')
    print(f'  ann Sharpe       {ann:+.3f}   (per-block mean {net.mean():+.5f})')
    print(f'  skew {psr.skew:+.2f}  kurt {psr.kurtosis:.2f}  block hit-rate {hit:.2f}')
    print(f'  PSR (n=1, pre-registered):   DSR {psr.dsr:.3f}  t {psr.deflated_tstat:+.3f}')
    print(f'  deflated (n=6, family):      DSR {ladder.dsr:.3f}  t {ladder.deflated_tstat:+.3f}')
    print(f'  6-chunk Sharpes: {[round(x,2) for x in chunk_sh]}  ({pos_chunks}/6 positive)')

    np.savez(OUT / 'low-vol-bab-returns.npz',
             ls_block_returns=net.astype(np.float64),
             periods_per_year=np.float64(PPY))
    (OUT / 'low-vol-bab-summary.json').write_text(json.dumps({
        'universe_size': int(N), 'vol_lookback': V, 'hold': H, 'n_blocks': int(R),
        'ann_sharpe': ann, 'psr_dsr': psr.dsr, 'psr_tstat': psr.deflated_tstat,
        'ladder_dsr': ladder.dsr, 'ladder_tstat': ladder.deflated_tstat,
        'skew': psr.skew, 'kurtosis': psr.kurtosis, 'block_hit_rate': hit,
        'chunk_sharpes': chunk_sh, 'positive_chunks': pos_chunks,
        'note': 'dollar-neutral (carries negative net-beta tilt); beta-neutral BAB is the follow-up',
    }, indent=2))
    print(f'\n-> {OUT / "low-vol-bab-returns.npz"}')


if __name__ == '__main__':
    main()
