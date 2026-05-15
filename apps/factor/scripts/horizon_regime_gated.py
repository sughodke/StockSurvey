"""Regime-gated horizon selection — does VIX state pick horizons better than
the learned mixture head?

Pre-registered in conversation 2026-05-14 (closing the endogenous-horizon
arc): test whether a hand-engineered macro regime gate on horizon choice
beats both the best fixed-h baseline (+0.401) AND the learned mixture
head at α=0 (+0.448). The hypothesis applies the workspace's
strongest operational rule ("regime filter > richer predictor", per
[`prediction-problem-pivot-arc`](../../docs/docs/findings/prediction-problem-pivot-arc.md))
to the deployment side of `apps/factor`.

Architecture:
  - Score head: vanilla rank-IC linear head, trained per walk-forward
    window at `rebal_days=5` fine grid on the 74-channel
    `IndicatorGridConfig`. No horizon head.
  - Horizon selector: deterministic function `regime_state(t) → h_t`.
    Two regime-mapping arms:
      * `inverted-vol`: VIX high → h=5 (responsive); VIX low → h=60
        (low-turnover). Tests "stress = signal moves faster" prior.
      * `same-vol`: opposite mapping, null-direction check.
  - Plus three baselines for the null table:
      * `fixed-h5` — always-h=5 deployment of the same scores.
      * `fixed-h60` — always-h=60 (best fixed from the entropy sweep).
      * `random-h` — uniform random horizon per bar (sanity floor).
  - Deployment: `simulate_irregular_daily_pnl` with a deterministic
    one-hot π_t per bar (the same simulator the mixture and entropy
    sweep ran). All arms eval on the same daily-PnL metric.

Null hypotheses (pre-registered):
  N1: inverted-vol beats best-fixed (`fixed-h60`) by ≥ +0.10
  N2: inverted-vol beats same-vol by ≥ +0.05 (direction validation)
  N3: inverted-vol beats learned mixture α=0 (+0.448) by ≥ +0.05
  N4: inverted-vol beats random-h (sanity floor)

Cost: local-only, ~2-5 min wall on the Intel Mac (rank-IC training is
tiny on the 74-channel indicator stack; deployment sim is the existing
numpy primitive).

Run:
    uv run python apps/factor/scripts/horizon_regime_gated.py
    uv run python apps/factor/scripts/horizon_regime_gated.py --max-tickers 30   # smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'
MACRO_CACHE = REPO_ROOT / '.macro-cache'


def _resolve_ticker_list(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}', flush=True)
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run', flush=True)
    return names


def _build_one_ticker(args):
    ticker, stooq_dir, cfg, start, end = args
    try:
        from factor import build_indicator_features
        from ss_features import TickerData, load_prices
        series = load_prices(ticker, stooq_dir=stooq_dir, start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        feats, valid = build_indicator_features(prices, cfg)
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _vix_state_per_bar(
    vix_series: pd.Series,
    bar_dates: np.ndarray,
    *,
    lookback_days: int = 126,
    min_lookback: int = 30,
) -> np.ndarray:
    """Classify VIX state at each bar as 'high' or 'low' vs trailing
    `lookback_days` rolling median (using only past data — no look-ahead).

    `bar_dates`: numpy datetime64 array of bar dates (rebal positions).
    `vix_series`: FRED `VIXCLS` daily series.

    Returns an array of strings of length `len(bar_dates)`. Bars with
    fewer than `min_lookback` past observations get `'unknown'` — the
    caller's regime mapping should default to a sensible horizon for
    those (default: h=60, the best-fixed baseline).
    """
    # Align VIX to the trading-bar dates via reindex + ffill.
    bar_idx = pd.DatetimeIndex(bar_dates)
    vix_aligned = vix_series.reindex(bar_idx).ffill().values.astype(np.float64)

    states = np.empty(len(bar_dates), dtype=object)
    for t in range(len(bar_dates)):
        # Lookback window is strictly *past* — `[t - lookback, t)`.
        lo = max(0, t - lookback_days)
        window = vix_aligned[lo:t]
        window = window[np.isfinite(window)]
        if len(window) < min_lookback:
            states[t] = 'unknown'
            continue
        median = np.median(window)
        current = vix_aligned[t]
        if not np.isfinite(current):
            states[t] = 'unknown'
        else:
            states[t] = 'high' if current >= median else 'low'
    return states


def _simulate_arm(
    arm_name: str,
    mapping: dict[str, int],
    val_scores: np.ndarray,
    val_mask: np.ndarray,
    val_rebal_idx: np.ndarray,
    val_states: np.ndarray,
    horizons: tuple[int, ...],
    daily_log_ret: np.ndarray,
    val_daily_start: int,
    val_daily_end: int,
    commission_bps: float,
    temperature: float,
    rng: np.random.Generator | None = None,
):
    """Build a deterministic π_t per bar from the regime → horizon mapping
    and call `simulate_irregular_daily_pnl`.

    For random-h, `mapping` is empty and we sample horizon uniformly per
    bar. For all others, `mapping['unknown']` defaults to h=60 (the
    best-fixed baseline) when VIX history is insufficient.
    """
    from factor import simulate_irregular_daily_pnl

    K = len(horizons)
    h_to_k = {h: k for k, h in enumerate(horizons)}
    pi = np.zeros((len(val_states), K), dtype=np.float64)
    if arm_name == 'random-h':
        # Uniform random horizon per bar.
        if rng is None:
            raise ValueError("rng required for arm 'random-h'")
        for t in range(len(val_states)):
            pi[t] = 1.0 / K
        return simulate_irregular_daily_pnl(
            scores=val_scores, pi=pi, mask=val_mask,
            daily_log_ret=daily_log_ret,
            rebal_idx=val_rebal_idx,
            horizons=horizons,
            daily_start=val_daily_start, daily_end=val_daily_end,
            commission_bps=commission_bps, temperature=temperature,
            horizon_picker='sample', rng=rng,
        )

    default_h = mapping.get('unknown', 60)
    for t, state in enumerate(val_states):
        h = mapping.get(state, default_h)
        k = h_to_k[h]
        pi[t, k] = 1.0
    return simulate_irregular_daily_pnl(
        scores=val_scores, pi=pi, mask=val_mask,
        daily_log_ret=daily_log_ret,
        rebal_idx=val_rebal_idx,
        horizons=horizons,
        daily_start=val_daily_start, daily_end=val_daily_end,
        commission_bps=commission_bps, temperature=temperature,
        horizon_picker='argmax',
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2026-04-01')
    p.add_argument('--horizons', default='5,10,20,40,60')
    p.add_argument('--train-window-blocks', type=int, default=252)
    p.add_argument('--val-window-blocks',   type=int, default=156)
    p.add_argument('--step-window-blocks',  type=int, default=156)
    p.add_argument('--n-steps', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--vix-lookback-days', type=int, default=126,
                   help='Rolling window for the VIX-vs-median state.')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers', type=int, default=0)
    p.add_argument('--n-workers', type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in args.horizons.split(','))
    K = len(horizons)
    h_min = min(horizons)
    assert h_min == 5, 'driver assumes h_min=5 fine grid'

    from factor import (
        IndicatorGridConfig, train_scorer_indicators_walkforward, predict,
        make_indicator_backbone, simulate_oracle_daily_pnl,
    )
    from ss_macro import load_fred_series

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}', flush=True)
    print(f'horizons = {horizons}  (h_min={h_min}, K={K})', flush=True)

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)
    print(f'\nbuilding features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...', flush=True)
    t0 = time.perf_counter()
    pool_args = [(n, str(STOOQ_SUBSET), cfg, args.start, args.end)
                 for n in names]
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(_build_one_ticker, pool_args)
    else:
        results = [_build_one_ticker(a) for a in pool_args]
    ticker_data = []
    failed = []
    for name, td, err in results:
        if td is None:
            failed.append((name, err))
        else:
            ticker_data.append(td)
    print(f'built {len(ticker_data)} / {len(names)} tickers in '
          f'{time.perf_counter() - t0:.1f}s', flush=True)

    # ---------- Step 1: train vanilla rank-IC linear head at h=5. ----------
    print(f'\n--- training rank-IC head at rebal_days={h_min}, '
          f'walk-forward (train={args.train_window_blocks}, '
          f'val={args.val_window_blocks}, step={args.step_window_blocks}) ---',
          flush=True)
    t1 = time.perf_counter()
    res = train_scorer_indicators_walkforward(
        ticker_data, cfg=cfg,
        rebal_days=h_min,
        train_window_blocks=args.train_window_blocks,
        val_window_blocks=args.val_window_blocks,
        step_window_blocks=args.step_window_blocks,
        scorer='linear',
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        commission_bps=args.commission_bps,
        verbose=True,
    )
    print(f'training wall: {time.perf_counter() - t1:.1f}s', flush=True)

    aligned = res.aligned
    rebal_idx = aligned.rebal_idx
    n_bars = len(rebal_idx)
    print(f'\nfine grid: {n_bars} bars at h_min={h_min}; '
          f'{res.n_windows} walk-forward windows', flush=True)

    # Daily log returns for the simulator.
    log_p = np.log(np.maximum(aligned.prices, 1e-12))
    daily_log_ret = np.zeros_like(aligned.prices, dtype=np.float64)
    daily_log_ret[1:] = log_p[1:] - log_p[:-1]
    D = daily_log_ret.shape[0]

    # ---------- Step 2: VIX state at each rebal bar. ----------
    print(f'\n--- VIX state classification (rolling median, lookback='
          f'{args.vix_lookback_days}d) ---', flush=True)
    vix = load_fred_series('VIXCLS', cache_dir=MACRO_CACHE)
    bar_dates = aligned.dates[rebal_idx]
    vix_states = _vix_state_per_bar(
        vix, bar_dates, lookback_days=args.vix_lookback_days)
    n_high = int((vix_states == 'high').sum())
    n_low = int((vix_states == 'low').sum())
    n_unknown = int((vix_states == 'unknown').sum())
    print(f'  bar states: high={n_high}, low={n_low}, unknown={n_unknown} '
          f'(/{n_bars})', flush=True)

    # ---------- Step 3: per-window scoring + per-arm deployment. ----------
    backbone = make_indicator_backbone(ticker_data, cfg)
    print(f'\n--- per-window deployment across {res.n_windows} windows ---',
          flush=True)

    arms_config = {
        'fixed-h5':     {'high': 5,  'low': 5,  'unknown': 5},
        'fixed-h60':    {'high': 60, 'low': 60, 'unknown': 60},
        'inverted-vol': {'high': 5,  'low': 60, 'unknown': 60},
        'same-vol':     {'high': 60, 'low': 5,  'unknown': 60},
        'random-h':     {},
        'oracle':       {},  # special: hindsight greedy
    }
    per_arm_per_window_sharpe: dict[str, list[float]] = {a: [] for a in arms_config}
    per_arm_per_window_holding: dict[str, list[float]] = {a: [] for a in arms_config}
    per_arm_per_window_rebals: dict[str, list[int]] = {a: [] for a in arms_config}
    per_arm_per_window_argmax_counts: dict[str, list[dict[int, int]]] = {
        a: [] for a in arms_config}
    per_window_val_start_date: list[str] = []
    per_window_vix_high_share: list[float] = []

    rng_master = np.random.default_rng(args.seed)
    rng_random_h = np.random.default_rng(args.seed + 1)

    for w in res.windows:
        # Per-window head → scores at every rebal bar.
        all_scores = predict(
            aligned, backbone, w.head_params, scorer='linear')
        # all_scores is shape (n_rebal_bars, n_tickers).
        val_slc = slice(w.val_block_start, w.val_block_end)
        val_scores = np.nan_to_num(all_scores[val_slc], nan=0.0).astype(np.float64)
        val_rebal_idx = rebal_idx[val_slc]
        val_mask = (aligned.valid[val_rebal_idx]
                    & np.isfinite(all_scores[val_slc])).astype(np.float64)
        val_states = vix_states[val_slc.start:val_slc.stop]

        # Daily window for the simulator.
        val_daily_start = int(val_rebal_idx[0])
        val_daily_end = min(int(val_rebal_idx[-1]) + max(horizons), D)

        # Record per-window VIX state share for diagnostics.
        n_high_w = int((val_states == 'high').sum())
        n_known_w = int((val_states != 'unknown').sum())
        vix_high_share_w = n_high_w / max(n_known_w, 1)
        per_window_val_start_date.append(w.val_start_date)
        per_window_vix_high_share.append(vix_high_share_w)

        for arm_name, mapping in arms_config.items():
            if arm_name == 'oracle':
                r = simulate_oracle_daily_pnl(
                    scores=val_scores, mask=val_mask,
                    daily_log_ret=daily_log_ret,
                    rebal_idx=val_rebal_idx,
                    horizons=horizons,
                    daily_start=val_daily_start, daily_end=val_daily_end,
                    commission_bps=args.commission_bps,
                    temperature=args.temperature,
                )
            else:
                r = _simulate_arm(
                    arm_name=arm_name,
                    mapping=mapping,
                    val_scores=val_scores, val_mask=val_mask,
                    val_rebal_idx=val_rebal_idx,
                    val_states=val_states,
                    horizons=horizons,
                    daily_log_ret=daily_log_ret,
                    val_daily_start=val_daily_start,
                    val_daily_end=val_daily_end,
                    commission_bps=args.commission_bps,
                    temperature=args.temperature,
                    rng=rng_random_h if arm_name == 'random-h' else None,
                )
            per_arm_per_window_sharpe[arm_name].append(r.sharpe)
            per_arm_per_window_holding[arm_name].append(r.mean_holding_days)
            per_arm_per_window_rebals[arm_name].append(r.n_rebals)
            counts = {h: 0 for h in horizons}
            for _, _, h_chosen in r.rebal_log:
                counts[h_chosen] = counts.get(h_chosen, 0) + 1
            per_arm_per_window_argmax_counts[arm_name].append(counts)

    # ---------- Step 4: per-window + aggregate tables. ----------
    print('\n=== per-window val Sharpe by arm ===', flush=True)
    arms_order = ['fixed-h5', 'fixed-h60', 'inverted-vol', 'same-vol',
                  'random-h', 'oracle']
    header = f'{"win":>3}  {"date":>10}  {"vix-hi%":>7}  ' + '  '.join(
        f'{a:>13}' for a in arms_order)
    print(header, flush=True)
    for i in range(len(per_window_val_start_date)):
        row = f'{i:>3}  {per_window_val_start_date[i]:>10}  '
        row += f'{per_window_vix_high_share[i]*100:>6.1f}%  '
        for arm in arms_order:
            row += f'{per_arm_per_window_sharpe[arm][i]:>+13.3f}  '
        print(row, flush=True)

    print('\n=== aggregates ===', flush=True)
    aggregates: dict[str, float] = {}
    for arm in arms_order:
        mean_s = float(np.mean(per_arm_per_window_sharpe[arm]))
        mean_h = float(np.mean(per_arm_per_window_holding[arm]))
        aggregates[arm] = mean_s
        print(f'  {arm:<15} mean val Sharpe = {mean_s:+.3f}  '
              f'(mean holding days = {mean_h:.1f})', flush=True)

    # Oracle's global horizon-pick distribution — diagnostic for which
    # horizons hindsight prefers.
    oracle_global = {h: 0 for h in horizons}
    for counts in per_arm_per_window_argmax_counts['oracle']:
        for h, c in counts.items():
            oracle_global[h] += c
    oracle_total = sum(oracle_global.values())
    if oracle_total > 0:
        print('\n  oracle argmax shares: '
              f'{ {h: round(oracle_global[h]/oracle_total, 2) for h in horizons} }',
              flush=True)

    # ---------- Step 5: pre-registered null-rejection table. ----------
    # The pre-registered comparisons are inverted-vol vs each baseline.
    inv = aggregates['inverted-vol']
    same = aggregates['same-vol']
    fix60 = aggregates['fixed-h60']
    fix5 = aggregates['fixed-h5']
    rand_h = aggregates['random-h']

    # Compare to learned mixture α=0 if we have it on disk.
    mixture_a0_path = output / 'horizon-mixture-a0-windows.npz'
    if mixture_a0_path.exists():
        mix_blob = np.load(mixture_a0_path, allow_pickle=True)
        mix_per_window = mix_blob['val_endog_sharpe']
        mix_mean = float(np.mean(mix_per_window))
        print(f'  learned-mixture (α=0)  mean val Sharpe = {mix_mean:+.3f}  '
              f'(from {mixture_a0_path.name})', flush=True)
    else:
        mix_mean = 0.448  # from the leaderboard 2026-05-14 row
        print(f'  learned-mixture (α=0)  mean val Sharpe = '
              f'{mix_mean:+.3f}  (hard-coded from leaderboard, '
              f'{mixture_a0_path.name} not present)', flush=True)

    delta_n1 = inv - fix60
    delta_n2 = inv - same
    delta_n3 = inv - mix_mean
    delta_n4 = inv - rand_h

    verdict_n1 = delta_n1 >= 0.10
    verdict_n2 = delta_n2 >= 0.05
    verdict_n3 = delta_n3 >= 0.05
    verdict_n4 = delta_n4 > 0.0
    verdict_pass = verdict_n1 and verdict_n2 and verdict_n3 and verdict_n4

    print('\n=== null-rejection checks (inverted-vol arm) ===', flush=True)
    print(f'  N1 (beats fixed-h60 by ≥ +0.10):    delta {delta_n1:+.3f}  '
          f'{"PASS" if verdict_n1 else "FAIL"}', flush=True)
    print(f'  N2 (beats same-vol by ≥ +0.05):     delta {delta_n2:+.3f}  '
          f'{"PASS" if verdict_n2 else "FAIL"}', flush=True)
    print(f'  N3 (beats learned mixture by ≥ +0.05): delta {delta_n3:+.3f}  '
          f'{"PASS" if verdict_n3 else "FAIL"}', flush=True)
    print(f'  N4 (beats random-h sanity floor):   delta {delta_n4:+.3f}  '
          f'{"PASS" if verdict_n4 else "FAIL"}', flush=True)

    # Pre-registered verdict-label logic.
    if verdict_pass:
        label = 'confirmed-OOS'
    elif verdict_n1 and verdict_n3 and verdict_n4:
        label = 'confirmed-OOS (direction prior reversed)'  # N2 failed
    elif verdict_n3 and verdict_n4:
        label = 'partial-OOS'
    elif verdict_n2 and not verdict_n1:
        label = 'diagnostic'
    else:
        label = 'confirmed-null'
    print(f'\nOverall verdict label: {label}', flush=True)

    # ---------- Step 6: pack artifacts. ----------
    blob: dict[str, np.ndarray] = {
        'window_idx': np.arange(len(per_window_val_start_date), dtype=np.int32),
        'val_start_date': np.array(per_window_val_start_date),
        'vix_high_share': np.array(per_window_vix_high_share, dtype=np.float32),
    }
    for arm in arms_order:
        blob[f'sharpe_{arm.replace("-", "_")}'] = np.array(
            per_arm_per_window_sharpe[arm], dtype=np.float32)
        blob[f'mean_holding_{arm.replace("-", "_")}'] = np.array(
            per_arm_per_window_holding[arm], dtype=np.float32)
        blob[f'n_rebals_{arm.replace("-", "_")}'] = np.array(
            per_arm_per_window_rebals[arm], dtype=np.int32)
    blob['_summary'] = np.array(json.dumps({
        'horizons': list(horizons),
        'vix_lookback_days': args.vix_lookback_days,
        'n_windows': len(per_window_val_start_date),
        'universe_size': len(ticker_data),
        'feature_width': F,
        'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'commission_bps': args.commission_bps,
        'temperature': args.temperature,
        'aggregates': aggregates,
        'learned_mixture_a0_mean': mix_mean,
        'deltas': {
            'N1_inv_vs_fixed_h60': delta_n1,
            'N2_inv_vs_same_vol': delta_n2,
            'N3_inv_vs_learned_mixture': delta_n3,
            'N4_inv_vs_random_h': delta_n4,
        },
        'verdicts': {
            'N1': verdict_n1, 'N2': verdict_n2,
            'N3': verdict_n3, 'N4': verdict_n4,
            'overall_pass': verdict_pass,
            'label': label,
        },
        'vix_global_state_counts': {
            'high': n_high, 'low': n_low, 'unknown': n_unknown,
        },
    }, indent=2))
    out_path = output / 'horizon-regime-gated-windows.npz'
    np.savez(out_path, **blob)
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
