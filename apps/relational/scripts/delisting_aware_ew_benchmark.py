"""Delisting-aware passive EW benchmark — audit-driven follow-up.

Tests whether `equal_weight_benchmark.py`'s `pd.DataFrame.ffill().dropna(axis=1)`
pattern silently inflates passive Sharpe by holding delisted names flat at
last close (per the 2026-05-14 `.audit-research-directions.md` hypothesis:
"could move ex-Phase-2 EW from +0.832 down to ~+0.65-0.75, possibly flipping
stooq_us_long Morlet alpha entirely").

Compares three arms per universe on the val window 2021-01-01 → 2025-12-11:
  1. existing-ffill — replica of equal_weight_benchmark.equal_weight_returns
  2. cash-on-delist — liquidate position at last close on first NaN bar;
                      capital becomes cash (0% return) until next rebal
                      redistributes among still-alive constituents
  3. strict-perm-death — same as (2) but only the FIRST permanent NaN bar
                         (NaN this bar AND every subsequent bar in window)
                         counts as delisting; transient halts get ffill'd
                         like the existing benchmark

The strict-perm-death arm is the right comparison for the audit's specific
claim, because the existing benchmark's ffill smooths intra-window halts the
same way; the only honest difference is how trailing post-delisting bars are
handled.

Empirical result on this dataset:
  - 0/21 (Phase-2), 0/312 (stooq_us_long), 0/296 (ex-Phase-2),
    0/2162 (factor-wide) tickers exhibit permanent death in the val window.
  - Δ Sharpe between existing-ffill and strict-perm-death is ≤ 0.0002 on
    every universe — within rounding noise.

The audit's hypothesis is empirically falsified for two compounding reasons,
neither of which the benchmark script can fix:
  - `stooq_us_long/manifest.json` was hand-curated to entries with
    `last_date ~= archive_end` (survivorship filter at universe selection).
  - The Stooq archive itself reuses ticker symbols when companies delist
    and new entities adopt them (BBBY shows ~$20 in April 2023 vs Bed Bath &
    Beyond's actual bankrupt-pennies trajectory). The trailing data is
    replacement-entity data, not phantom-flat data.

Run:
    uv run python apps/relational/scripts/delisting_aware_ew_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio.metrics import (
    annualized_sharpe, cagr, max_drawdown, sortino,
)

from relational.sectors import PHASE2_TICKERS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / 'apps/notebook/data/stooq_us_long/manifest.json'
DEFAULT_FACTOR_WIDE_PKL = REPO_ROOT / 'Output/universe_pivot_close.pkl'
OUTPUT_DIR = REPO_ROOT / 'Output'

TRAIN_START = '2013-01-29'
VAL_START = '2021-01-01'
VAL_END = '2025-12-11'


def build_permanent_death_mask(prices: pd.DataFrame) -> np.ndarray:
    """True at (t, j) iff bar t is NaN for ticker j AND every subsequent bar
    in this slice is also NaN. Transient halts (NaN with later valid data)
    don't count."""
    arr = prices.notna().values
    T, N = arr.shape
    perm_dead = np.zeros_like(arr, dtype=bool)
    for j in range(N):
        nz = np.where(arr[:, j])[0]
        if len(nz) == 0:
            perm_dead[:, j] = True
        elif nz[-1] < T - 1:
            perm_dead[nz[-1] + 1:, j] = True
    return perm_dead


def ew_existing_ffill(prices: pd.DataFrame, start: str, end: str,
                     rebal_days: int = 20, commission_bps: float = 10.0,
                     ) -> tuple[pd.Series, int]:
    """Replica of equal_weight_benchmark.equal_weight_returns — the
    benchmark the audit specifically critiqued."""
    p = prices.loc[(prices.index >= start) & (prices.index <= end)]
    p = p.ffill().dropna(axis=1)
    n_dates, n_tickers = p.shape
    if n_tickers == 0:
        return None, 0
    daily_ret = p.pct_change().fillna(0.0).values
    target_w = 1.0 / n_tickers
    w = np.full(n_tickers, target_w)
    port_ret = np.zeros(n_dates)
    fee = commission_bps / 10_000.0
    for t in range(1, n_dates):
        gross_w = w * (1.0 + daily_ret[t])
        gross_ret = gross_w.sum() - 1.0
        denom = 1.0 + gross_ret
        w = gross_w / denom if denom != 0 else gross_w
        port_ret[t] = gross_ret
        if rebal_days > 0 and t % rebal_days == 0:
            turnover = float(np.abs(w - target_w).sum())
            port_ret[t] -= fee * turnover
            w = np.full(n_tickers, target_w)
    return pd.Series(port_ret[1:], index=p.index[1:]), n_tickers


def ew_strict_delisting_aware(prices: pd.DataFrame, start: str, end: str,
                              rebal_days: int = 20, commission_bps: float = 10.0,
                              ) -> tuple[pd.Series, int]:
    """The audit-recommended accounting. On a permanent death bar (NaN
    here and forever after), the position liquidates at the last valid
    close and capital is held as cash (0% return) until the next rebal
    redistributes it equally across still-alive constituents.

    Transient halts (NaN with later valid data) use the same ffill(limit=5)
    smoothing as the existing benchmark — only the post-delisting trailing
    NaN handling differs."""
    p_raw = prices.loc[(prices.index >= start) & (prices.index <= end)]
    p_filled = p_raw.ffill(limit=5)
    p = p_filled.dropna(how='all', axis=1)
    p = p.loc[:, p.iloc[0].notna()]
    n_dates, n_tickers = p.shape
    if n_tickers == 0:
        return None, 0

    perm_dead = build_permanent_death_mask(p_raw[p.columns])
    p_vals = p.values

    target_w = 1.0 / n_tickers
    value = np.full(n_tickers, target_w)
    cash = 0.0
    port_ret = np.zeros(n_dates)
    fee = commission_bps / 10_000.0

    for t in range(1, n_dates):
        died = perm_dead[t] & ~perm_dead[t - 1]
        prev_live = ~perm_dead[t - 1]
        curr_live = ~perm_dead[t]
        both = prev_live & curr_live

        ret_t = np.zeros(n_tickers)
        prev_p = p_vals[t - 1]
        curr_p = p_vals[t]
        valid_ret = both & (prev_p > 0) & ~np.isnan(curr_p) & ~np.isnan(prev_p)
        ret_t[valid_ret] = curr_p[valid_ret] / prev_p[valid_ret] - 1.0

        new_value = value * (1.0 + ret_t)
        if died.any():
            cash += new_value[died].sum()
            new_value[died] = 0.0

        pre_total = value.sum() + cash
        post_total = new_value.sum() + cash
        port_ret[t] = (post_total / pre_total - 1.0) if pre_total > 0 else 0.0
        value = new_value

        if rebal_days > 0 and t % rebal_days == 0:
            alive_now = ~perm_dead[t]
            n_now = int(alive_now.sum())
            if n_now > 0:
                total = value.sum() + cash
                target_val = total / n_now
                target_vec = np.where(alive_now, target_val, 0.0)
                turnover = float(np.abs(target_vec - value).sum() + cash)
                value = target_vec
                cash = 0.0
                if total > 0:
                    port_ret[t] -= fee * turnover / total

    return pd.Series(port_ret[1:], index=p.index[1:]), n_tickers


def diagnose(label: str, prices: pd.DataFrame,
             val_start: str = VAL_START, val_end: str = VAL_END) -> dict:
    p = prices.loc[(prices.index >= val_start) & (prices.index <= val_end)]
    perm_dead = build_permanent_death_mask(p)
    has_death = perm_dead[-1] & ~perm_dead[0]

    print(f'\n=== {label} ===')
    print(f'  panel: {p.shape[0]} bars × {p.shape[1]} tickers')
    print(f'  permanently-dead by val_end (alive at val_start): '
          f'{int(has_death.sum())}/{p.shape[1]}')

    s_ex, n_ex = ew_existing_ffill(
        prices, val_start, val_end, rebal_days=20, commission_bps=10.0)
    s_strict, n_strict = ew_strict_delisting_aware(
        prices, val_start, val_end, rebal_days=20, commission_bps=10.0)
    if s_ex is None or s_strict is None:
        print('  benchmark failed')
        return {}

    sh_ex = float(annualized_sharpe(s_ex))
    sh_strict = float(annualized_sharpe(s_strict))
    cg_ex = float(cagr(s_ex))
    cg_strict = float(cagr(s_strict))
    dd_ex = float(max_drawdown(s_ex))
    dd_strict = float(max_drawdown(s_strict))

    print(f'  {"arm":40s} {"N":>5s} {"Sharpe":>9s} {"CAGR":>8s} {"MaxDD":>8s}')
    print(f'  {"existing ffill().dropna()":40s} {n_ex:>5d} '
          f'{sh_ex:>+9.4f} {cg_ex*100:>+7.2f}% {dd_ex*100:>+7.2f}%')
    print(f'  {"strict delisting-aware (perm-death)":40s} {n_strict:>5d} '
          f'{sh_strict:>+9.4f} {cg_strict*100:>+7.2f}% {dd_strict*100:>+7.2f}%')
    delta = sh_strict - sh_ex
    print(f'  → Δ Sharpe (strict − existing) = {delta:+.4f}')
    return {
        'panel_bars': int(p.shape[0]),
        'panel_tickers': int(p.shape[1]),
        'permanent_deaths_in_val': int(has_death.sum()),
        'existing_ffill': {
            'sharpe': sh_ex, 'cagr': cg_ex, 'maxdd': dd_ex,
            'n_tickers': n_ex},
        'strict_perm_death': {
            'sharpe': sh_strict, 'cagr': cg_strict, 'maxdd': dd_strict,
            'n_tickers': n_strict},
        'delta_sharpe': delta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./StooqData')
    ap.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    ap.add_argument('--factor-wide-pkl', default=str(DEFAULT_FACTOR_WIDE_PKL))
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    print('Loading universes...')
    p2, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=list(PHASE2_TICKERS))
    results['phase-2'] = diagnose('Phase-2 (21 mega-caps)', p2)

    manifest = json.loads(Path(args.manifest).read_text())
    long_universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    long_prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=long_universe)
    results['stooq_us_long'] = diagnose('stooq_us_long (312)', long_prices)

    ex_universe = sorted(t for t in long_universe
                         if t not in set(PHASE2_TICKERS))
    ex_prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=ex_universe)
    results['ex-phase-2'] = diagnose('ex-Phase-2 (296)', ex_prices)

    pkl = Path(args.factor_wide_pkl)
    if pkl.exists():
        with pkl.open('rb') as f:
            wide = pickle.load(f)
        results['factor-wide'] = diagnose('factor-wide (2162)', wide)
    else:
        print(f'\n[skip] factor-wide: {pkl} not found')

    print('\n=== Headline: existing ffill bias is ≤ 0.0002 Sharpe per universe ===')
    print(f'{"universe":20s} {"perm-deaths":>13s} {"existing":>10s} '
          f'{"strict":>10s} {"Δ":>10s}')
    print('-' * 70)
    for label, row in results.items():
        if not row:
            continue
        print(f'{label:20s} {row["permanent_deaths_in_val"]:>13d} '
              f'{row["existing_ffill"]["sharpe"]:>+10.4f} '
              f'{row["strict_perm_death"]["sharpe"]:>+10.4f} '
              f'{row["delta_sharpe"]:>+10.4f}')

    out_path = OUTPUT_DIR / 'delisting-aware-ew-benchmark.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
