"""Window-5 decomposition for the HRP + gross-exposure-modulator finding.

Reconstructs the modulator's gate-scalar time series across window 5
(2020-07-07 -> 2023-08-10) for the `single` linkage and compares
HRP-standalone vs HRP+modulator daily returns sub-period-by-sub-period.
This answers whether the +0.586 w5 alpha is:

- genuine regime-collapse detection (gate fires during 2022 drawdown)
- coincidental drawup capture (gate idle, alpha is just compounding)
- mixed.

Sub-periods within w5:
  - 2020-07-07 -> 2020-12-31  (post-COVID rebound tail)
  - 2021-01-01 -> 2021-12-31  (calm bull)
  - 2022-01-01 -> 2022-12-31  (rate-cycle drawdown)
  - 2023-01-01 -> 2023-08-10  (recovery)

Outputs:
  Output/hrp-modulator-w5-decomposition.json
  apps/docs/docs/findings/images/hrp-modulator-w5-decomposition.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ss_loaders import load_stooq_matrix
from lie.hrp import weights_hrp
from lie.symmetry_rank import gross_exposure_modulator, trailing_effective_rank


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
OUTPUT = REPO_ROOT / 'Output'
IMG_DIR = REPO_ROOT / 'apps' / 'docs' / 'docs' / 'findings' / 'images'

LOOKBACK = 120
REBAL_DAYS = 20
COMMISSION_FRAC = 10.0 / 1e4
LINKAGE = 'single'
MOD_FLOOR = 0.25

# Walk-forward: 1260 train, 780 val, 780 step (per JSON). w5 start = 1260+5*780 = 5160
W5_TRAIN_START_IDX = 5 * 780
W5_TRAIN_END_IDX = W5_TRAIN_START_IDX + 1260
W5_VAL_END_IDX = W5_TRAIN_END_IDX + 780


def build_panel(prices: pd.DataFrame, *, use_modulator: bool,
                gate_log: list | None = None) -> pd.DataFrame:
    n_t, n_n = prices.shape
    out = np.zeros((n_t, n_n), dtype=np.float64)
    px = prices.to_numpy(dtype=np.float64)
    for t in range(LOOKBACK, n_t):
        if (t - LOOKBACK) % REBAL_DAYS != 0:
            continue
        sub = px[t - LOOKBACK:t + 1]
        try:
            w = weights_hrp(sub, lookback=LOOKBACK, linkage_method=LINKAGE)
        except Exception:
            continue
        scalar = 1.0
        eff = float('nan')
        if use_modulator:
            eff = trailing_effective_rank(sub, lookback=LOOKBACK)
            n_active = int((w > 0).sum())
            if n_active > 0:
                scalar = gross_exposure_modulator(
                    eff, n_assets=n_active, floor=MOD_FLOOR)
                w = w * scalar
        if gate_log is not None:
            gate_log.append({
                't_idx': t,
                'date': prices.index[t],
                'eff_rank': eff,
                'scalar': scalar,
                'n_active': int((w > 0).sum()),
            })
        out[t] = w
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def basket_daily_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    common = prices.index[LOOKBACK:]
    px = prices.loc[common]
    w_arr = weights.loc[common].to_numpy()
    daily_ret = px.pct_change().fillna(0.0).values
    n_t, n_n = w_arr.shape
    rebal_idxs = [i for i in range(n_t) if np.any(w_arr[i] != 0)]
    held = np.zeros((n_t, n_n))
    for k, ridx in enumerate(rebal_idxs):
        end = rebal_idxs[k + 1] if k + 1 < len(rebal_idxs) else n_t
        held[ridx:end] = w_arr[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_idxs):
        if k == 0:
            cost[ridx] = COMMISSION_FRAC * np.abs(w_arr[ridx]).sum()
        else:
            prev = rebal_idxs[k - 1]
            cost[ridx] = COMMISSION_FRAC * 0.5 * np.abs(
                w_arr[ridx] - w_arr[prev]).sum()
    return pd.Series(port_ret - cost, index=common)


def passive_ew_daily_returns(prices: pd.DataFrame) -> pd.Series:
    common = prices.index[LOOKBACK:]
    px = prices.loc[common]
    n_t, n_n = px.shape
    daily_ret = px.pct_change().fillna(0.0).values
    valid = (~px.isna()).values.astype(float)
    held = np.zeros((n_t, n_n))
    w_panel = np.zeros((n_t, n_n))
    rebal_idxs = list(range(0, n_t, REBAL_DAYS))
    for ridx in rebal_idxs:
        v = valid[ridx]
        s = v.sum()
        if s > 0:
            w_panel[ridx] = v / s
    for k, ridx in enumerate(rebal_idxs):
        end = rebal_idxs[k + 1] if k + 1 < len(rebal_idxs) else n_t
        held[ridx:end] = w_panel[ridx]
    held_lag = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
    port_ret = (held_lag * daily_ret).sum(axis=1)
    cost = np.zeros(n_t)
    for k, ridx in enumerate(rebal_idxs):
        if k == 0:
            cost[ridx] = COMMISSION_FRAC * np.abs(w_panel[ridx]).sum()
        else:
            prev = rebal_idxs[k - 1]
            cost[ridx] = COMMISSION_FRAC * 0.5 * np.abs(
                w_panel[ridx] - w_panel[prev]).sum()
    return pd.Series(port_ret - cost, index=common)


def ann_sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=np.float64)
    if r.size < 5 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252.0))


def main() -> None:
    print('Loading stooq_us_long universe ...')
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    prices, _, _, _ = load_stooq_matrix(
        str(STOOQ_SUBSET), min_history=150,
        start_date='2000-01-01', end_date='2025-12-11',
        tickers=universe)
    print(f'  {prices.shape[1]} tickers, {prices.index[0].date()} -> '
          f'{prices.index[-1].date()} ({len(prices)} bars)')

    # Walk-forward windows
    n = len(prices)
    train_w, val_w, step = 1260, 780, 780
    windows = []
    start = 0
    while start + train_w + val_w <= n:
        windows.append((start, start + train_w, start + train_w + val_w))
        start += step
    print(f'  windows={len(windows)}')
    w5_lo, w5_mid, w5_hi = windows[5]
    w5_val_start = prices.index[w5_mid]
    w5_val_end = prices.index[w5_hi - 1]
    print(f'  w5 val: {w5_val_start.date()} -> {w5_val_end.date()}')

    print('Building panels (no-mod + mod, single linkage) ...')
    gate_log: list = []
    w_nomod = build_panel(prices, use_modulator=False)
    w_mod = build_panel(prices, use_modulator=True, gate_log=gate_log)

    print('Computing daily returns ...')
    r_nomod = basket_daily_returns(w_nomod, prices)
    r_mod = basket_daily_returns(w_mod, prices)
    r_ew = passive_ew_daily_returns(prices)

    # Restrict to w5 val window
    mask = (r_nomod.index >= w5_val_start) & (r_nomod.index <= w5_val_end)
    common = r_nomod.index[mask]
    r_nomod_w5 = r_nomod.loc[common]
    r_mod_w5 = r_mod.loc[common]
    r_ew_w5 = r_ew.loc[common]

    # Gate series restricted to w5
    gate_df = pd.DataFrame(gate_log)
    gate_df['date'] = pd.to_datetime(gate_df['date'])
    gate_w5 = gate_df[(gate_df['date'] >= w5_val_start) &
                      (gate_df['date'] <= w5_val_end)].reset_index(drop=True)
    print(f'  w5 rebalance count: {len(gate_w5)}')
    print(f'  gate scalar: min={gate_w5["scalar"].min():.3f}, '
          f'mean={gate_w5["scalar"].mean():.3f}, '
          f'max={gate_w5["scalar"].max():.3f}')
    print(f'  fraction of rebals with scalar < 1.0: '
          f'{(gate_w5["scalar"] < 1.0).mean():.2%}')
    print(f'  fraction with scalar at floor ({MOD_FLOOR}): '
          f'{(gate_w5["scalar"] <= MOD_FLOOR + 1e-6).mean():.2%}')

    # Sub-period decomposition
    sub_periods = [
        ('2020-rebound-tail', '2020-07-07', '2020-12-31'),
        ('2021-calm-bull',    '2021-01-01', '2021-12-31'),
        ('2022-rate-cycle',   '2022-01-01', '2022-12-31'),
        ('2023-recovery',     '2023-01-01', '2023-08-10'),
    ]
    rows = []
    for name, s, e in sub_periods:
        s_ts = pd.Timestamp(s)
        e_ts = pd.Timestamp(e)
        m = (common >= s_ts) & (common <= e_ts)
        sub_nm = r_nomod_w5.values[m]
        sub_md = r_mod_w5.values[m]
        sub_ew = r_ew_w5.values[m]
        gm = (gate_w5['date'] >= s_ts) & (gate_w5['date'] <= e_ts)
        gate_sub = gate_w5[gm]
        s_nm = ann_sharpe(sub_nm)
        s_md = ann_sharpe(sub_md)
        s_ew = ann_sharpe(sub_ew)
        rows.append({
            'sub_period': name,
            'start': s,
            'end': e,
            'n_days': int(m.sum()),
            'n_rebals': int(len(gate_sub)),
            'hrp_sharpe': s_nm,
            'hrp_mod_sharpe': s_md,
            'ew_sharpe': s_ew,
            'alpha_nomod_vs_ew': s_nm - s_ew,
            'alpha_mod_vs_ew': s_md - s_ew,
            'mod_lift_vs_nomod': s_md - s_nm,
            'gate_mean': float(gate_sub['scalar'].mean()) if len(gate_sub) else 1.0,
            'gate_min': float(gate_sub['scalar'].min()) if len(gate_sub) else 1.0,
            'gate_active_pct': float((gate_sub['scalar'] < 1.0).mean()) if len(gate_sub) else 0.0,
            'eff_rank_mean': float(gate_sub['eff_rank'].mean()) if len(gate_sub) else float('nan'),
        })
    decomp = pd.DataFrame(rows)
    print('\nSub-period decomposition:')
    print(decomp.to_string(index=False))

    # Full-window check vs the +0.586 finding
    full_alpha_nm = ann_sharpe(r_nomod_w5.values) - ann_sharpe(r_ew_w5.values)
    full_alpha_md = ann_sharpe(r_mod_w5.values) - ann_sharpe(r_ew_w5.values)
    print(f'\nFull-w5 alpha standalone: {full_alpha_nm:+.3f} '
          f'(finding: -0.262)')
    print(f'Full-w5 alpha modulated:  {full_alpha_md:+.3f} '
          f'(finding: +0.324; lift {full_alpha_md - full_alpha_nm:+.3f}, '
          f'finding lift +0.586)')

    # PNG: 3 panels - gate scalar, eff_rank, cumulative delta
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 1: gate scalar over time, with floor reference
    axes[0].step(gate_w5['date'], gate_w5['scalar'], where='post',
                 color='C0', linewidth=1.5, label='gate scalar')
    axes[0].axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='full exposure')
    axes[0].axhline(MOD_FLOOR, color='red', linestyle=':', alpha=0.5,
                    label=f'floor ({MOD_FLOOR})')
    axes[0].set_ylabel('gross-exposure\nscalar')
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_title('Window 5 (2020-07-07 -> 2023-08-10): modulator gate decomposition')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: effective rank
    axes[1].plot(gate_w5['date'], gate_w5['eff_rank'], color='C2', linewidth=1.5)
    axes[1].set_ylabel('trailing\neffective rank')
    axes[1].grid(alpha=0.3)

    # Panel 3: cumulative log return delta (mod - nomod) and EW reference cum-ret
    cum_mod = (1.0 + r_mod_w5).cumprod()
    cum_nomod = (1.0 + r_nomod_w5).cumprod()
    cum_ew = (1.0 + r_ew_w5).cumprod()
    axes[2].plot(common, cum_mod.values, color='C0', label='HRP + modulator', linewidth=1.4)
    axes[2].plot(common, cum_nomod.values, color='C1', label='HRP standalone', linewidth=1.4)
    axes[2].plot(common, cum_ew.values, color='gray', label='passive EW', linewidth=1.2, alpha=0.7)
    axes[2].set_ylabel('cumulative\nreturn')
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(alpha=0.3)

    # Sub-period shading on all axes
    sub_colors = ['#fff3e0', '#e8f5e9', '#ffebee', '#e3f2fd']
    for (name, s, e), c in zip(sub_periods, sub_colors):
        for ax in axes:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.4, zorder=0)

    fig.autofmt_xdate()
    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMG_DIR / 'hrp-modulator-w5-decomposition.png'
    fig.savefig(img_path, dpi=110)
    print(f'\nWrote {img_path}')

    # JSON dump
    out_json = OUTPUT / 'hrp-modulator-w5-decomposition.json'
    payload = {
        'window': 'w5',
        'val_start': str(w5_val_start.date()),
        'val_end': str(w5_val_end.date()),
        'linkage': LINKAGE,
        'lookback': LOOKBACK,
        'rebal_days': REBAL_DAYS,
        'modulator_floor': MOD_FLOOR,
        'full_window': {
            'hrp_sharpe': ann_sharpe(r_nomod_w5.values),
            'hrp_mod_sharpe': ann_sharpe(r_mod_w5.values),
            'ew_sharpe': ann_sharpe(r_ew_w5.values),
            'alpha_nomod': full_alpha_nm,
            'alpha_mod': full_alpha_md,
            'mod_lift_vs_nomod': full_alpha_md - full_alpha_nm,
            'n_rebals': int(len(gate_w5)),
            'gate_scalar_mean': float(gate_w5['scalar'].mean()),
            'gate_scalar_min': float(gate_w5['scalar'].min()),
            'gate_active_pct': float((gate_w5['scalar'] < 1.0).mean()),
            'gate_at_floor_pct': float((gate_w5['scalar'] <= MOD_FLOOR + 1e-6).mean()),
        },
        'sub_periods': rows,
        'gate_timeseries': [
            {'date': str(d.date()), 'scalar': float(s),
             'eff_rank': float(e), 'n_active': int(n)}
            for d, s, e, n in zip(
                gate_w5['date'], gate_w5['scalar'],
                gate_w5['eff_rank'], gate_w5['n_active'])
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f'Wrote {out_json}')


if __name__ == '__main__':
    main()
