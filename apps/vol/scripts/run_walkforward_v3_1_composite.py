"""v3.1 — composite regime gate (next-experiment per the v3 oracle finding).

The v3 hindsight oracle revealed a +2.86 fired-Sharpe headroom over the
best heuristic (126d-VIX rolling median: +2.01 → oracle: +4.87). The
binding constraint is the gate feature, not the architecture. v3.1
tests whether a richer gate — VIX + cross-sectional IV dispersion +
universe-mean IV-over-HV — captures meaningful headroom over single-
feature 126d-VIX.

Six gate arms on the same walk-forward windows as v3:

  - vix-126d        single-feature VIX vs 126d rolling median (v3 best)
  - disp-126d       cross-sectional std of iv_over_hv20 vs 126d median
  - mean-vrp-126d   universe-mean iv_over_hv20 vs 126d median
  - vix-or-disp     OR composite of vix-126d and disp-126d
  - vix-and-disp    AND composite
  - lr-composite    logistic regression on (vix, disp, mean-vrp, vix-change)
                    trained on per-rebal realized alpha sign, expanding window
  - oracle          hindsight greedy (fire iff realized alpha > 0)

Pre-reg cuts (locked before run):
  PASS:        composite-gate fired-α Sh ≥ +0.30 AND fire-rate ∈ [20%, 80%]
               AND ≥ 4/5 fired-positive
  STRONG-PASS: fired-α Sh ≥ +3.0 (captures ≥ 50% of v3 oracle headroom)

Run from repo root:
    uv run python apps/vol/scripts/run_walkforward_v3_1_composite.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (
    FEATURE_NAMES, build_vol_features, evaluate_portfolio_short_vol,
    forward_iv_rv_gap, load_gauss314_full, predict, train_predictor,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_window_slices_by_date(
    dates: pd.DatetimeIndex, train_days: int, val_days: int, step_days: int,
):
    if len(dates) < train_days + val_days:
        return []
    out = []
    i = 0
    while i + train_days + val_days <= len(dates):
        out.append((
            dates[i], dates[i + train_days - 1],
            dates[i + train_days], dates[i + train_days + val_days - 1],
        ))
        i += step_days
    return out


def apply_oi_filter(merged: pd.DataFrame, raw: pd.DataFrame,
                    oi_top_n: int) -> pd.DataFrame:
    oi_panel = raw[['date', 'symbol',
                    'puts_open_interest',
                    'calls_open_interest']].copy()
    oi_panel['total_oi'] = (
        oi_panel['puts_open_interest'].fillna(0)
        + oi_panel['calls_open_interest'].fillna(0))
    oi_panel['oi_rank'] = oi_panel.groupby('date')['total_oi'].rank(
        method='first', ascending=False)
    keep = oi_panel[oi_panel['oi_rank'] <= oi_top_n][
        ['date', 'symbol']].copy()
    keep['keep_flag'] = True
    out = merged.merge(keep, on=['date', 'symbol'], how='left')
    return out[out['keep_flag'].fillna(False)].drop(columns='keep_flag')


def build_per_date_aggregates(merged: pd.DataFrame) -> pd.DataFrame:
    """For each date, compute cross-sectional aggregates from the
    OI-restricted per-(date, symbol) feature panel. These are point-
    in-time (no look-ahead): the values at date `t` use only the
    cross-section of symbols observable on date `t`.

    Returns a DataFrame indexed by date with columns:
      vix              — per-date VIX level (same across symbols → first)
      mean_vrp         — cross-sectional mean of iv_over_hv20 (the VRP signal)
      disp_iv_over_hv20 — cross-sectional std of iv_over_hv20
      disp_skew_otm    — cross-sectional std of skew_otm
      vix_change_5d    — VIX[t] − VIX[t−5]
    """
    # Need raw VIX too (it's per-row in gauss314, same per date).
    df = merged.groupby('date').agg({
        'iv_over_hv20': ['mean', 'std'],
        'skew_otm':     ['std'],
    })
    df.columns = ['_'.join(c) for c in df.columns]
    df = df.rename(columns={
        'iv_over_hv20_mean': 'mean_vrp',
        'iv_over_hv20_std':  'disp_iv_over_hv20',
        'skew_otm_std':      'disp_skew_otm',
    })
    return df


def build_vix_series(raw: pd.DataFrame) -> pd.Series:
    """Per-date VIX series."""
    s = raw[['date', 'VIX']].drop_duplicates('date').sort_values('date')
    return s.set_index('date')['VIX']


def threshold_gate_from_series(series: pd.Series, lookback: int,
                                ) -> dict[pd.Timestamp, bool]:
    """Generic rolling-median threshold gate: fire iff series[t] > rolling
    median over `lookback` days using past data only."""
    rolling = series.rolling(window=lookback, min_periods=lookback // 2).median()
    fired = series > rolling
    return {d: bool(f) for d, f in fired.items() if not pd.isna(f)}


def compose_gates(g1: dict, g2: dict, mode: str) -> dict[pd.Timestamp, bool]:
    """Boolean composition of two binary gates. `mode` ∈ {'or', 'and'}."""
    all_dates = set(g1) | set(g2)
    out = {}
    for d in all_dates:
        f1 = g1.get(d)
        f2 = g2.get(d)
        # If either gate doesn't have a value at date d (insufficient
        # history), defer to the one that does. If both missing, skip.
        if f1 is None and f2 is None:
            continue
        if f1 is None:
            out[d] = f2
        elif f2 is None:
            out[d] = f1
        else:
            out[d] = (f1 or f2) if mode == 'or' else (f1 and f2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=300)
    p.add_argument('--val-window-days',   type=int, default=120)
    p.add_argument('--step-window-days',  type=int, default=120)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--oi-top-n', type=int, default=200)
    p.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    p.add_argument('--gate-lookback', type=int, default=126,
                   help='Single rolling-median lookback for all rule-based '
                        'gate variants (matches v3 operational-best 126d).')
    p.add_argument('--lr-l2', type=float, default=1.0,
                   help='L2 regularization strength for the LR-composite arm '
                        '(higher = stronger; ~15 train rebals per window so '
                        'a strong prior is appropriate).')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading gauss314 full schema...', flush=True)
    t0 = time.perf_counter()
    raw = load_gauss314_full()
    print(f'  {len(raw):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)

    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)

    merged = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap'])
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        merged[col] = merged[col].clip(
            lower=-args.clip_iv_hv_ratio, upper=args.clip_iv_hv_ratio)
    print(f'  usable rows after merge + clip: {len(merged):,}', flush=True)

    print(f'\nApplying OI-top-{args.oi_top_n} filter per date...', flush=True)
    merged = apply_oi_filter(merged, raw, args.oi_top_n)
    print(f'  retained: {len(merged):,} rows', flush=True)

    # Per-date aggregates from the OI-restricted panel.
    print('\nBuilding per-date cross-sectional aggregates...', flush=True)
    aggregates = build_per_date_aggregates(merged)
    vix_series = build_vix_series(raw).reindex(aggregates.index)
    aggregates['vix'] = vix_series
    aggregates['vix_change_5d'] = aggregates['vix'].diff(5)
    print(f'  per-date aggregate panel: {aggregates.shape[0]} dates × '
          f'{aggregates.shape[1]} features', flush=True)
    print(f'  columns: {list(aggregates.columns)}', flush=True)

    # Rule-based gate dicts (all at the single 126d lookback by default).
    lb = args.gate_lookback
    gate_vix_only = threshold_gate_from_series(aggregates['vix'], lb)
    gate_disp_only = threshold_gate_from_series(
        aggregates['disp_iv_over_hv20'].dropna(), lb)
    gate_mean_vrp_only = threshold_gate_from_series(
        aggregates['mean_vrp'].dropna(), lb)
    gate_vix_or_disp = compose_gates(gate_vix_only, gate_disp_only, mode='or')
    gate_vix_and_disp = compose_gates(gate_vix_only, gate_disp_only, mode='and')

    rule_based_arms = {
        f'vix-{lb}d':          gate_vix_only,
        f'disp-{lb}d':         gate_disp_only,
        f'mean-vrp-{lb}d':     gate_mean_vrp_only,
        'vix-or-disp':         gate_vix_or_disp,
        'vix-and-disp':        gate_vix_and_disp,
    }
    print('\nFire rates (rule-based arms, all-dates global):', flush=True)
    for name, gate in rule_based_arms.items():
        n_fire = sum(1 for v in gate.values() if v)
        n_total = len(gate)
        print(f'  {name:<20} {n_fire}/{n_total} ({100*n_fire/n_total:.0f}%)',
              flush=True)

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    windows = _build_window_slices_by_date(
        dates, args.train_window_days, args.val_window_days,
        args.step_window_days)
    print(f'\nWalk-forward: {len(windows)} windows', flush=True)

    # Per-window cache for the oracle + lr-composite arms (reused).
    per_window_cache: list[dict] = []

    print('\nTraining predictor + caching per-rebal (g_pnl, u_pnl, '
          'aggregate features) per window ...', flush=True)
    for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
        train = merged[(merged['date'] >= ts_tr_lo) &
                       (merged['date'] <= ts_tr_hi)]
        val   = merged[(merged['date'] >= ts_va_lo) &
                       (merged['date'] <= ts_va_hi)]
        if len(train) < 300 or len(val) < 100:
            continue
        X_tr = train[FEATURE_NAMES].values
        y_tr = train['iv_rv_gap'].values
        X_va = val[FEATURE_NAMES].values
        y_va = val['iv_rv_gap'].values

        pred = train_predictor(X_tr, y_tr, FEATURE_NAMES)
        val_pred = predict(pred, X_va)
        val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])

        val_with_pred = val[['date', 'symbol']].copy()
        val_with_pred['pred_gap'] = val_pred
        val_with_realized = val[['date', 'symbol', 'iv_rv_gap']].copy()

        # Also evaluate the predictor on TRAIN to get per-rebal alphas
        # for LR composite training (we need both gated_arm and
        # universe_arm at train rebals).
        train_pred = predict(pred, X_tr)
        train_with_pred = train[['date', 'symbol']].copy()
        train_with_pred['pred_gap'] = train_pred
        train_with_realized = train[['date', 'symbol', 'iv_rv_gap']].copy()

        gated_train = evaluate_portfolio_short_vol(
            train_with_pred, train_with_realized,
            top_k=args.top_k, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='gated_train')
        universe_train = evaluate_portfolio_short_vol(
            train_with_pred, train_with_realized,
            top_k=0, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='universe_train')
        gated_val = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=args.top_k, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='gated_val')
        universe_val = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=0, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='universe_val')

        gt_map = dict(zip(gated_train.per_rebal_dates,
                          gated_train.per_rebal_pnl_vol_points))
        ut_map = dict(zip(universe_train.per_rebal_dates,
                          universe_train.per_rebal_pnl_vol_points))
        gv_map = dict(zip(gated_val.per_rebal_dates,
                          gated_val.per_rebal_pnl_vol_points))
        uv_map = dict(zip(universe_val.per_rebal_dates,
                          universe_val.per_rebal_pnl_vol_points))
        common_train = sorted(set(gt_map) & set(ut_map))
        common_val   = sorted(set(gv_map) & set(uv_map))

        per_window_cache.append({
            'window_idx': w_idx,
            'val_start':  str(ts_va_lo.date()),
            'val_end':    str(ts_va_hi.date()),
            'val_r':      val_corr,
            'train_common': common_train,
            'train_g':    [gt_map[d] for d in common_train],
            'train_u':    [ut_map[d] for d in common_train],
            'val_common': common_val,
            'val_g':      [gv_map[d] for d in common_val],
            'val_u':      [uv_map[d] for d in common_val],
        })
        print(f'  w{w_idx}: train rebals={len(common_train)} '
              f'val rebals={len(common_val)} val r={val_corr:+.3f}',
              flush=True)

    # Build LR-composite gate per window. Logistic regression
    # (regularized) on per-rebal (vix, disp, mean_vrp, vix_change_5d)
    # → fire iff realized alpha (train) > 0. At eval time, predict on
    # val rebal features.
    print('\nTraining LR-composite gate per window ...', flush=True)
    lr_features_cols = ['vix', 'disp_iv_over_hv20', 'mean_vrp',
                        'vix_change_5d']

    def _features_at_dates(dates_list):
        """Look up per-date aggregate features for given dates. Returns
        (n_dates, n_features) matrix with NaN-rows filtered."""
        idx = pd.DatetimeIndex(pd.to_datetime(dates_list))
        sub = aggregates.reindex(idx)[lr_features_cols]
        return sub.values, sub.notna().all(axis=1).values

    lr_per_window: list[dict] = []
    for w in per_window_cache:
        # Train features + target.
        X_tr, mask_tr = _features_at_dates(w['train_common'])
        y_tr_alpha = np.asarray(w['train_g'], dtype=float) - \
                     np.asarray(w['train_u'], dtype=float)
        y_tr = (y_tr_alpha > 0).astype(int)
        keep_tr = mask_tr & np.isfinite(y_tr_alpha)
        X_tr = X_tr[keep_tr]
        y_tr_lr = y_tr[keep_tr]
        # Val features.
        X_va, mask_va = _features_at_dates(w['val_common'])

        if len(X_tr) < 5 or len(np.unique(y_tr_lr)) < 2:
            # Degenerate train sample → fall back to "fire always".
            lr_pred = np.ones(len(X_va), dtype=bool)
            train_p0 = np.nan
        else:
            # z-score using train stats only.
            mu = X_tr.mean(axis=0)
            sd = X_tr.std(axis=0) + 1e-9
            X_tr_z = (X_tr - mu) / sd
            X_va_z = (X_va - mu) / sd
            X_va_z = np.nan_to_num(X_va_z, nan=0.0)
            # Manual L2-regularized logistic regression via Newton-Raphson
            # (numpy-only; ~10 iterations).
            n, k = X_tr_z.shape
            Xb = np.hstack([X_tr_z, np.ones((n, 1))])  # add bias
            w_lr = np.zeros(k + 1)
            for _ in range(20):
                z = Xb @ w_lr
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                W = p * (1 - p)
                # gradient + hessian with L2 (skip bias)
                grad = Xb.T @ (p - y_tr_lr) + args.lr_l2 * np.concatenate(
                    [w_lr[:-1], [0.0]])
                H = Xb.T @ (W[:, None] * Xb) + args.lr_l2 * np.eye(k + 1)
                H[-1, -1] -= args.lr_l2  # no L2 on bias
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    break
                w_lr = w_lr - step
                if np.abs(step).max() < 1e-7:
                    break
            # Predict val fire probability; threshold at 0.5.
            Xb_va = np.hstack([X_va_z, np.ones((len(X_va_z), 1))])
            z_va = Xb_va @ w_lr
            p_va = 1.0 / (1.0 + np.exp(-np.clip(z_va, -30, 30)))
            lr_pred = (p_va >= 0.5) & mask_va
            train_p0 = float(np.mean(p_va))
        lr_per_window.append({
            'window_idx': w['window_idx'],
            'fired': list(lr_pred),
            'val_common': w['val_common'],
            'mean_val_p': train_p0,
        })

    # Build oracle gate per window: fire iff realized val alpha > 0.
    oracle_per_window = []
    for w in per_window_cache:
        alpha = np.asarray(w['val_g'], dtype=float) - \
                np.asarray(w['val_u'], dtype=float)
        oracle_per_window.append({
            'window_idx': w['window_idx'],
            'fired': (alpha > 0).tolist(),
        })

    # Evaluate each arm by walking the per-window cache.
    arms = list(rule_based_arms.keys()) + ['lr-composite', 'oracle']
    print(f'\n=== Per-window fired-α Sharpe by arm ===', flush=True)
    hdr = f'{"win":>3s} {"val period":>25s} '
    hdr += ''.join(f'{a:>15s}' for a in arms)
    print(hdr, flush=True)
    print('-' * (3 + 1 + 25 + 1 + 15 * len(arms)), flush=True)

    per_arm_summary: dict[str, dict] = {a: {
        'fired_alpha_pool': [],
        'full_alpha_pool': [],
        'per_window_fired_sh': [],
        'per_window_n_fired': [],
        'per_window_fire_rate': [],
    } for a in arms}

    ann = float(np.sqrt(252.0 / args.rebal_days))
    for w_idx_pos, w in enumerate(per_window_cache):
        common = w['val_common']
        g_pnls = w['val_g']
        u_pnls = w['val_u']
        row = f'{w["window_idx"]:>3d} {w["val_start"]}→{w["val_end"]} '
        for arm in arms:
            if arm == 'lr-composite':
                fired_flags = lr_per_window[w_idx_pos]['fired']
            elif arm == 'oracle':
                fired_flags = oracle_per_window[w_idx_pos]['fired']
            else:
                gate = rule_based_arms[arm]
                fired_flags = [
                    bool(gate.get(pd.Timestamp(d), False)) for d in common]
            fired_alpha = []
            full_alpha = []
            for d, g, u, f in zip(common, g_pnls, u_pnls, fired_flags):
                if f:
                    fired_alpha.append(g - u)
                    full_alpha.append(g - u)
                else:
                    full_alpha.append(0.0)
            if len(fired_alpha) > 1:
                a = np.asarray(fired_alpha, dtype=float)
                sd = float(a.std(ddof=1))
                fired_alpha_sh = (a.mean() / sd * ann) if sd > 1e-12 else 0.0
            else:
                fired_alpha_sh = 0.0
            n_fired = sum(fired_flags)
            fire_rate = n_fired / len(common) if common else 0.0
            per_arm_summary[arm]['fired_alpha_pool'].extend(fired_alpha)
            per_arm_summary[arm]['full_alpha_pool'].extend(full_alpha)
            per_arm_summary[arm]['per_window_fired_sh'].append(fired_alpha_sh)
            per_arm_summary[arm]['per_window_n_fired'].append(n_fired)
            per_arm_summary[arm]['per_window_fire_rate'].append(fire_rate)
            row += f'{fired_alpha_sh:>+11.3f}({n_fired:>2d})'
        print(row, flush=True)

    print('\n=== Aggregates by arm ===', flush=True)
    print(f'{"arm":<20} {"fired α Sh":>11s} {"full α Sh":>10s} '
          f'{"fire rate":>10s} {"pos windows":>12s}', flush=True)
    arm_aggs = {}
    for arm in arms:
        a_fired = np.asarray(per_arm_summary[arm]['fired_alpha_pool'],
                             dtype=float)
        a_full = np.asarray(per_arm_summary[arm]['full_alpha_pool'],
                            dtype=float)
        fired_sh = (a_fired.mean() / a_fired.std(ddof=1) * ann
                    if a_fired.size > 1 and a_fired.std(ddof=1) > 1e-12
                    else 0.0)
        full_sh = (a_full.mean() / a_full.std(ddof=1) * ann
                   if a_full.size > 1 and a_full.std(ddof=1) > 1e-12
                   else 0.0)
        total_fired = sum(per_arm_summary[arm]['per_window_n_fired'])
        total_rebals = sum(len(w['val_common']) for w in per_window_cache)
        fire_rate = total_fired / max(total_rebals, 1)
        pos_w = sum(1 for s in per_arm_summary[arm]['per_window_fired_sh']
                    if s > 0)
        n_w = len(per_arm_summary[arm]['per_window_fired_sh'])
        arm_aggs[arm] = {
            'pooled_fired_alpha_sharpe': fired_sh,
            'pooled_full_alpha_sharpe':  full_sh,
            'fire_rate': fire_rate,
            'fired_positive_windows': pos_w,
            'total_windows': n_w,
        }
        print(f'{arm:<20} {fired_sh:>+11.3f} {full_sh:>+10.3f} '
              f'{fire_rate*100:>9.1f}% {pos_w:>5d}/{n_w}', flush=True)

    print('\n=== Pre-reg verdict (composite arms vs v3 PASS bar) ===',
          flush=True)
    composite_arms = ['disp-126d', 'mean-vrp-126d', 'vix-or-disp',
                      'vix-and-disp', 'lr-composite']
    for arm in composite_arms:
        a = arm_aggs[arm]
        in_band = 0.20 <= a['fire_rate'] <= 0.80
        if (a['pooled_fired_alpha_sharpe'] >= 3.0
                and in_band and a['fired_positive_windows'] >= 4):
            v = 'STRONG-PASS'
        elif (a['pooled_fired_alpha_sharpe'] >= 0.30
              and in_band and a['fired_positive_windows'] >= 4):
            v = 'PASS'
        elif a['pooled_fired_alpha_sharpe'] >= 0.10 or in_band:
            v = 'MARGINAL'
        else:
            v = 'FAIL'
        print(f'  {arm:<20}: {v}', flush=True)

    # Per-arm headroom vs 126d baseline + oracle ceiling capture.
    vix_baseline = arm_aggs[f'vix-{lb}d']['pooled_fired_alpha_sharpe']
    oracle_ceil = arm_aggs['oracle']['pooled_fired_alpha_sharpe']
    print(f'\n  vix-{lb}d baseline = {vix_baseline:+.3f}', flush=True)
    print(f'  oracle ceiling    = {oracle_ceil:+.3f}', flush=True)
    print(f'  oracle headroom   = {oracle_ceil - vix_baseline:+.3f}',
          flush=True)
    print(f'\n  capture vs oracle (composite − baseline) / '
          f'(oracle − baseline):', flush=True)
    for arm in composite_arms:
        d = arm_aggs[arm]['pooled_fired_alpha_sharpe'] - vix_baseline
        denom = max(oracle_ceil - vix_baseline, 1e-9)
        print(f'    {arm:<20} delta vs baseline {d:>+8.3f}  '
              f'({100*d/denom:>5.1f}% of oracle headroom)', flush=True)

    summary = {
        'oi_top_n': args.oi_top_n,
        'top_k': args.top_k,
        'rebal_days': args.rebal_days,
        'gate_lookback': args.gate_lookback,
        'lr_l2': args.lr_l2,
        'arms_aggregate': arm_aggs,
        'per_window_cache': [
            {
                'window_idx': w['window_idx'],
                'val_start':  w['val_start'],
                'val_end':    w['val_end'],
                'val_r':      w['val_r'],
                'n_val_rebals': len(w['val_common']),
            }
            for w in per_window_cache
        ],
    }
    out_path = output / 'vol-walkforward-v3-1-composite-summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
