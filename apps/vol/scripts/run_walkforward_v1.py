"""v1 walk-forward — per-rebal portfolio Sharpe + costs in the loop.

v1 changes vs `run_walkforward.py` (v0):

  - **Metric**: per-rebal portfolio Sharpe (annualized) instead of
    pooled per-cell Sharpe. v0's metric was flagged as "a weak metric"
    in `findings/vol-surface-v0.md` — per-cell discards temporal
    structure and ignores friction.
  - **Headline alpha**: annualized Sharpe of the
    `gated_per_rebal_PnL − universe_per_rebal_PnL` series. Friction
    cancels in the difference (both arms pay equal per-pick friction),
    so alpha is friction-invariant. ABSOLUTE per-arm Sharpe is also
    reported alongside (inflated by within-basket averaging — useful
    relative-magnitude reference, not directly comparable to equity
    portfolio Sharpe).
  - **Costs in the loop**: model options friction at multiple bps
    levels (only affects the absolute net Sharpe figures; alpha is
    invariant).
  - **Pick size**: top-K=100 fixed per rebal (tradable basket size),
    not top-20% quantile (~700 picks, untradable on options).
  - **Rebal cadence**: 20 trading days (matches the iv_rv_gap forward
    horizon — clean non-overlapping rebals).

Pre-registered cuts (v1, on alpha series):
  - PASS:     alpha Sharpe ≥ +0.30 (annualized), ≥ 4/5 windows
              with positive per-window alpha
  - MARGINAL: alpha Sharpe ∈ [+0.10, +0.30] AND ≥ 3/5 positive
  - FAIL:     alpha Sharpe < +0.10 OR ≤ 2/5 positive

The alpha-Sharpe formulation answers the actual deployment question:
"does the predicted top-100 basket outperform the universe equal-
weighted baseline by enough to clear noise?" Friction cancels in the
alpha because both arms trade at the same per-pick cost. Absolute
basket Sharpe of either arm conflates "vrp captured by being net-
short vol" (the trivial NO_OPTIONS.md baseline) with "vrp captured by
prediction skill", which is exactly the cross-app `passive-EW gate`
operational rule applied to vol.

Reuses v0's data loader / features / linear OLS predictor. The MLP
head (audit-flagged v1 item) is deferred to v2 if v1 shows alpha to
extract; if the linear head fails the honest metric, the next probe
is signal availability (DoltHub extension to 2026, OI filter), not
nonlinear capacity.

Run from repo root:
    uv run python apps/vol/scripts/run_walkforward_v1.py
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
from vol.predictor import evaluate_r2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_window_slices_by_date(
    dates: pd.DatetimeIndex,
    train_days: int, val_days: int, step_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
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


def _pooled_annualized_sharpe(
    per_window_rebal_pnls: list[list[float]],
    friction_bps_roundtrip: float,
    rebal_days: int,
) -> tuple[float, float, list[float]]:
    """Pool per-rebal PnL series across windows, compute annualized
    Sharpe. Also returns the per-window net Sharpes for the 'positive
    in K of N' criterion.
    """
    all_pnls: list[float] = []
    per_window_sharpe: list[float] = []
    friction = friction_bps_roundtrip / 10_000.0
    ann_factor = float(np.sqrt(252.0 / rebal_days))

    for pnls in per_window_rebal_pnls:
        a = np.asarray(pnls, dtype=float)
        if a.size == 0:
            per_window_sharpe.append(0.0)
            continue
        net = a - friction
        all_pnls.extend(net.tolist())
        if a.size > 1:
            sd = float(a.std(ddof=1))
            sh = (net.mean() / sd * ann_factor) if sd > 1e-12 else 0.0
        else:
            sh = 0.0
        per_window_sharpe.append(sh)

    if not all_pnls:
        return 0.0, 0.0, per_window_sharpe
    a = np.asarray(all_pnls, dtype=float)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    pooled_sh = (a.mean() / sd * ann_factor) if sd > 1e-12 else 0.0
    pooled_mean_pnl = float(a.mean())
    return pooled_sh, pooled_mean_pnl, per_window_sharpe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=300)
    p.add_argument('--val-window-days',   type=int, default=120)
    p.add_argument('--step-window-days',  type=int, default=120)
    p.add_argument('--rebal-days', type=int, default=20,
                   help='Trading-day cadence between rebals (matches '
                        'iv_rv_gap forward horizon by default).')
    p.add_argument('--top-k', type=int, default=100,
                   help='Number of cells per rebal (top-K by predicted '
                        'gap). 0 = universe baseline.')
    p.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    p.add_argument('--friction-bps-headline', type=float, default=100.0,
                   help='Pre-registered friction level for PASS/MARGINAL/'
                        'FAIL decision.')
    p.add_argument('--friction-bps-sweep', type=float, nargs='+',
                   default=[0.0, 100.0, 250.0, 500.0],
                   help='Friction levels to report (sensitivity).')
    p.add_argument('--shuffle-control-seeds', type=int, default=10,
                   help='Number of random-predictor shuffle seeds to run '
                        'as a basket-averaging-artifact sanity control. '
                        'Real-predictor alpha mean PnL must materially '
                        'exceed shuffle alpha mean PnL.')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading gauss314 full schema...')
    t0 = time.perf_counter()
    raw = load_gauss314_full()
    print(f'  {len(raw):,} rows in {time.perf_counter()-t0:.1f}s')

    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)

    merged = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap'])
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        merged[col] = merged[col].clip(
            lower=-args.clip_iv_hv_ratio, upper=args.clip_iv_hv_ratio)
    print(f'  usable rows after merge + clip: {len(merged):,}')

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    print(f'  dates: {dates[0].date()} → {dates[-1].date()} ({len(dates)})')

    windows = _build_window_slices_by_date(
        dates, args.train_window_days, args.val_window_days,
        args.step_window_days)
    if not windows:
        raise SystemExit(
            f'no windows fit: have {len(dates)} dates but need '
            f'train+val={args.train_window_days + args.val_window_days}')
    print(f'  walk-forward: {len(windows)} windows '
          f'(rebal_days={args.rebal_days}, top_k={args.top_k})')

    print('\n' + '=' * 120, flush=True)
    print(f'{"win":>3s} {"val period":>25s} {"val r":>7s} '
          f'{"n_reb":>5s} '
          f'{"gated PnL":>10s} {"univ PnL":>10s} {"alpha PnL":>10s} '
          f'{"gated Sh":>9s} {"univ Sh":>9s} {"ALPHA Sh":>10s} {"pos":>3s}',
          flush=True)
    print('-' * 120, flush=True)

    per_window_results = []
    per_window_gated_pnls: list[list[float]] = []
    per_window_universe_pnls: list[list[float]] = []
    per_window_alpha_pnls: list[list[float]] = []
    per_window_val_r: list[float] = []
    per_window_alpha_sharpe: list[float] = []

    for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
        train = merged[(merged['date'] >= ts_tr_lo) &
                       (merged['date'] <= ts_tr_hi)]
        val   = merged[(merged['date'] >= ts_va_lo) &
                       (merged['date'] <= ts_va_hi)]
        if len(train) < 1000 or len(val) < 500:
            continue

        X_tr = train[FEATURE_NAMES].values
        y_tr = train['iv_rv_gap'].values
        X_va = val[FEATURE_NAMES].values
        y_va = val['iv_rv_gap'].values

        pred = train_predictor(X_tr, y_tr, FEATURE_NAMES)
        val_pred = predict(pred, X_va)
        val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])
        val_r2 = evaluate_r2(val_pred, y_va)

        val_with_pred = val[['date', 'symbol']].copy()
        val_with_pred['pred_gap'] = val_pred
        val_with_realized = val[['date', 'symbol', 'iv_rv_gap']].copy()

        gated = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=args.top_k, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='gated')
        # Universe baseline: top_k=0 → use all valid cells per rebal.
        universe = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=0, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='universe')

        # Align per-rebal series by date (both arms use the same rebal
        # date grid by construction, so should be parallel — but guard).
        g_dates = set(gated.per_rebal_dates)
        u_dates = set(universe.per_rebal_dates)
        common_dates = sorted(g_dates & u_dates)
        g_map = dict(zip(gated.per_rebal_dates,
                         gated.per_rebal_pnl_vol_points))
        u_map = dict(zip(universe.per_rebal_dates,
                         universe.per_rebal_pnl_vol_points))
        g_aligned = [g_map[d] for d in common_dates]
        u_aligned = [u_map[d] for d in common_dates]
        alpha_aligned = [g - u for g, u in zip(g_aligned, u_aligned)]

        a = np.asarray(alpha_aligned, dtype=float)
        ann_factor = float(np.sqrt(252.0 / args.rebal_days))
        if a.size > 1:
            sd = float(a.std(ddof=1))
            alpha_sharpe = (a.mean() / sd * ann_factor) if sd > 1e-12 else 0.0
        else:
            alpha_sharpe = 0.0

        per_window_gated_pnls.append(g_aligned)
        per_window_universe_pnls.append(u_aligned)
        per_window_alpha_pnls.append(alpha_aligned)
        per_window_val_r.append(val_corr)
        per_window_alpha_sharpe.append(alpha_sharpe)

        # Per-friction sharpe of gated arm (for sensitivity table).
        sh_at_friction = {}
        for fbps in args.friction_bps_sweep:
            fric = fbps / 10_000.0
            g_net = a + 0.0  # alpha is friction-invariant in difference;
                             # but we ALSO want absolute gated net sh.
            g_pnls_net = np.asarray(g_aligned) - fric
            if len(g_pnls_net) > 1:
                sd_g = float(np.asarray(g_aligned).std(ddof=1))
                sh_g = (g_pnls_net.mean() / sd_g * ann_factor) if sd_g > 1e-12 else 0.0
            else:
                sh_g = 0.0
            sh_at_friction[str(int(fbps))] = sh_g

        per_window_results.append({
            'window_idx': w_idx,
            'train_start': str(ts_tr_lo.date()),
            'train_end':   str(ts_tr_hi.date()),
            'val_start':   str(ts_va_lo.date()),
            'val_end':     str(ts_va_hi.date()),
            'val_r2': val_r2,
            'val_pearson_r': val_corr,
            'n_rebals': len(common_dates),
            'top_k': args.top_k,
            'gated_mean_pnl_vol_pts': float(np.mean(g_aligned)) if g_aligned else 0.0,
            'universe_mean_pnl_vol_pts': float(np.mean(u_aligned)) if u_aligned else 0.0,
            'alpha_mean_pnl_vol_pts': float(np.mean(alpha_aligned)) if alpha_aligned else 0.0,
            'gated_sharpe_gross_annualized': gated.annualized_sharpe,
            'universe_sharpe_gross_annualized': universe.annualized_sharpe,
            'alpha_sharpe_annualized': alpha_sharpe,
            'gated_sharpe_at_friction': sh_at_friction,
            'per_rebal_dates': common_dates,
            'per_rebal_gated_pnls': g_aligned,
            'per_rebal_universe_pnls': u_aligned,
            'per_rebal_alpha_pnls': alpha_aligned,
        })

        pos_mark = '+' if alpha_sharpe > 0 else '-'
        print(f'{w_idx:>3d} {ts_va_lo.date()}→{ts_va_hi.date()} '
              f'{val_corr:>+7.4f} {len(common_dates):>5d} '
              f'{float(np.mean(g_aligned)):>+10.4f} '
              f'{float(np.mean(u_aligned)):>+10.4f} '
              f'{float(np.mean(alpha_aligned)):>+10.4f} '
              f'{gated.annualized_sharpe:>+9.3f} '
              f'{universe.annualized_sharpe:>+9.3f} '
              f'{alpha_sharpe:>+10.3f} '
              f'{pos_mark:>3s}', flush=True)

    print('\n' + '=' * 120, flush=True)
    if not per_window_results:
        print('No usable windows.', flush=True)
        return

    # Pooled alpha across windows — headline metric.
    all_alpha = [a for w in per_window_alpha_pnls for a in w]
    arr_alpha = np.asarray(all_alpha, dtype=float)
    ann_factor = float(np.sqrt(252.0 / args.rebal_days))
    if arr_alpha.size > 1:
        sd_alpha = float(arr_alpha.std(ddof=1))
        pooled_alpha_sharpe = (arr_alpha.mean() / sd_alpha * ann_factor
                               if sd_alpha > 1e-12 else 0.0)
    else:
        pooled_alpha_sharpe = 0.0
    pooled_alpha_mean = float(arr_alpha.mean()) if arr_alpha.size else 0.0
    n_pos_windows = sum(1 for sh in per_window_alpha_sharpe if sh > 0)
    n_total = len(per_window_alpha_sharpe)

    # Absolute gated Sharpe sensitivity (friction matters for absolute,
    # NOT for alpha — alpha is friction-invariant by construction).
    summary: dict = {
        'horizon': args.horizon,
        'train_window_days': args.train_window_days,
        'val_window_days': args.val_window_days,
        'step_window_days': args.step_window_days,
        'rebal_days': args.rebal_days,
        'top_k': args.top_k,
        'clip_iv_hv_ratio': args.clip_iv_hv_ratio,
        'feature_names': FEATURE_NAMES,
        'n_windows': n_total,
        'mean_val_pearson_r': float(np.mean(per_window_val_r)),
        'pooled_alpha_sharpe_annualized': pooled_alpha_sharpe,
        'pooled_alpha_mean_pnl_vol_pts': pooled_alpha_mean,
        'positive_alpha_windows': n_pos_windows,
        'positive_alpha_windows_total': n_total,
        'per_window_alpha_sharpe': per_window_alpha_sharpe,
        'absolute_gated_friction_sensitivity': {},
        'friction_bps_headline': args.friction_bps_headline,
        'per_window': per_window_results,
    }

    print(f'mean val Pearson r = {summary["mean_val_pearson_r"]:+.4f}',
          flush=True)
    print(f'\n=== HEADLINE: pooled alpha Sharpe (annualized) = '
          f'{pooled_alpha_sharpe:+.3f} ; positive in {n_pos_windows}/{n_total} '
          f'windows ===', flush=True)
    print(f'  pooled alpha mean PnL per rebal = '
          f'{pooled_alpha_mean:+.4f} vol pts', flush=True)
    print(f'  alpha is friction-invariant (gated and universe arms pay '
          f'matching per-pick costs)', flush=True)

    print(f'\nFriction sensitivity on absolute gated Sharpe (informational, '
          f'NOT the pre-reg metric):', flush=True)
    print(f'  {"friction":>10s} {"pooled Sh":>10s} {"mean PnL":>10s}',
          flush=True)
    print('  ' + '-' * 36, flush=True)
    for fbps in args.friction_bps_sweep:
        all_gated = [pnl - fbps / 10_000.0
                     for w in per_window_gated_pnls for pnl in w]
        a = np.asarray(all_gated, dtype=float)
        if a.size > 1:
            sd_g = float(a.std(ddof=1))
            sh_g = (a.mean() / sd_g * ann_factor) if sd_g > 1e-12 else 0.0
        else:
            sh_g = 0.0
        summary['absolute_gated_friction_sensitivity'][str(int(fbps))] = {
            'pooled_annualized_sharpe_net': sh_g,
            'pooled_mean_pnl_per_rebal_net': float(a.mean()) if a.size else 0.0,
        }
        marker = ' ← headline' if abs(fbps - args.friction_bps_headline) < 1e-9 else ''
        print(f'  {int(fbps):>8d} bp {sh_g:>+10.3f} '
              f'{(a.mean() if a.size else 0.0):>+10.4f}{marker}', flush=True)

    # Shuffle control: replace predictions with random Gaussians, re-run
    # the top-K selection per rebal. Tests whether the alpha is from the
    # predictor's signal or just a basket-averaging artifact (which would
    # affect both real and shuffle equally).
    if args.shuffle_control_seeds > 0:
        print(f'\nShuffle control: {args.shuffle_control_seeds} random-pred '
              f'seeds per window...', flush=True)
        shuffle_alpha_means: list[float] = []
        shuffle_alpha_sharpes: list[float] = []
        for seed in range(args.shuffle_control_seeds):
            rng = np.random.default_rng(seed)
            all_shuffle_alphas = []
            for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
                val_s = merged[(merged['date'] >= ts_va_lo) &
                               (merged['date'] <= ts_va_hi)]
                if len(val_s) < 500:
                    continue
                vp = val_s[['date', 'symbol']].copy()
                vp['pred_gap'] = rng.standard_normal(len(val_s))
                vr = val_s[['date', 'symbol', 'iv_rv_gap']].copy()
                g_s = evaluate_portfolio_short_vol(
                    vp, vr, top_k=args.top_k, friction_bps_roundtrip=0.0,
                    rebal_days=args.rebal_days)
                u_s = evaluate_portfolio_short_vol(
                    vp, vr, top_k=0, friction_bps_roundtrip=0.0,
                    rebal_days=args.rebal_days)
                # Align on common dates
                u_map = dict(zip(u_s.per_rebal_dates,
                                 u_s.per_rebal_pnl_vol_points))
                g_pnls = g_s.per_rebal_pnl_vol_points
                for d, g_pnl in zip(g_s.per_rebal_dates, g_pnls):
                    if d in u_map:
                        all_shuffle_alphas.append(g_pnl - u_map[d])
            a = np.asarray(all_shuffle_alphas, dtype=float)
            sh = (a.mean() / a.std(ddof=1) * ann_factor
                  if a.size > 1 and a.std(ddof=1) > 1e-12 else 0.0)
            shuffle_alpha_means.append(float(a.mean()))
            shuffle_alpha_sharpes.append(float(sh))
        shuffle_mean = float(np.mean(shuffle_alpha_means))
        shuffle_mean_std = float(np.std(shuffle_alpha_means))
        shuffle_sh_mean = float(np.mean(shuffle_alpha_sharpes))
        ratio = (pooled_alpha_mean / abs(shuffle_mean)
                 if abs(shuffle_mean) > 1e-9 else float('inf'))
        print(f'  shuffle alpha mean = {shuffle_mean:+.4f} ± {shuffle_mean_std:.4f}',
              flush=True)
        print(f'  real alpha mean    = {pooled_alpha_mean:+.4f}',
              flush=True)
        print(f'  ratio (real / |shuffle|) = {ratio:.1f}x',
              flush=True)
        print(f'  shuffle alpha Sharpe mean = {shuffle_sh_mean:+.3f} '
              f'(real = {pooled_alpha_sharpe:+.3f})', flush=True)
        summary['shuffle_control'] = {
            'n_seeds': args.shuffle_control_seeds,
            'shuffle_alpha_mean_pooled': shuffle_mean,
            'shuffle_alpha_mean_pooled_std': shuffle_mean_std,
            'shuffle_alpha_sharpe_pooled_mean': shuffle_sh_mean,
            'real_to_shuffle_mean_ratio': ratio,
            'per_seed_alpha_means': shuffle_alpha_means,
            'per_seed_alpha_sharpes': shuffle_alpha_sharpes,
        }

    # Pre-registered verdict on the alpha Sharpe.
    if pooled_alpha_sharpe >= 0.30 and n_pos_windows >= int(np.ceil(0.8 * n_total)):
        verdict = (f'PASS — alpha Sharpe {pooled_alpha_sharpe:+.3f} ≥ +0.30 '
                   f'with {n_pos_windows}/{n_total} positive windows')
    elif pooled_alpha_sharpe >= 0.10 and n_pos_windows >= int(np.ceil(0.6 * n_total)):
        verdict = (f'MARGINAL — alpha Sharpe {pooled_alpha_sharpe:+.3f} in '
                   f'[+0.10, +0.30] with {n_pos_windows}/{n_total} positive')
    else:
        verdict = (f'FAIL — alpha Sharpe {pooled_alpha_sharpe:+.3f} < +0.10 '
                   f'OR positive windows {n_pos_windows}/{n_total} insufficient')
    summary['verdict'] = verdict
    print(f'\npre-reg verdict: {verdict}', flush=True)

    out_path = output / 'vol-walkforward-v1-summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
