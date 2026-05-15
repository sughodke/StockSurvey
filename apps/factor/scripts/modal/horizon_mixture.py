"""Modal entrypoint for the endogenous-horizon mixture walk-forward.

Mirrors `train_indicator.py::walkforward` but invokes
`train_scorer_horizon_walkforward` with the multi-horizon mixture loss
and the dual-head scorer. Returns per-window endog Sharpe, fixed-h
baselines, random-π baseline, π entropy + argmax-bin histogram, all
packed into a single npz + a summary json.

Pre-registered hypothesis (see commit history): state-conditional
horizon selection beats best-fixed-h by ≥ 0.10 Sharpe AND beats
random-π. Verdict labels follow the leaderboard vocabulary
(confirmed-OOS / partial-OOS / confirmed-null).

Usage
-----
Smoke (~3 min wall, ~$0.04 at T4 prices):
    uvx modal run apps/factor/scripts/modal/horizon_mixture.py \\
        --max-tickers 30 --n-steps 50

Full 297-ticker walkforward (~15-25 min wall, ~$0.30-0.50):
    uvx modal run apps/factor/scripts/modal/horizon_mixture.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'
STOOQ_SUBSET_REL = 'apps/notebook/data/stooq_us_long'
STOOQ_SUBSET = f'{REMOTE_REPO}/{STOOQ_SUBSET_REL}'


image = (
    modal.Image.from_registry(
        'nvidia/cuda:12.4.0-devel-ubuntu22.04',
        add_python='3.12',
    )
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .env({'PYTHONUNBUFFERED': '1'})
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            # `.claude/` carries the scheduled-tasks lock + agent transcripts;
            # Modal aborts the build with "modified during build process" if
            # the harness writes a wakeup mid-upload.
            '.claude/**',
            '.modal_metadata/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            '.edgar-cache/**',
            '.macro-cache/**',
            'apps/relational/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            'apps/replay/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('factor-horizon-mixture', image=image)


@app.function(gpu='T4', cpu=4, memory=8192, timeout=2 * 60 * 60)
def train_horizon_walkforward_remote(
    horizons_csv: str,
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    entropy_weights_csv: str,
    deployment_reward_weights_csv: str,
    config_variant: str,
    mlp_hidden: int,
    mlp_layers: int,
    commission_bps: float,
    temperature: float,
    train_window_blocks: int,
    val_window_blocks: int,
    step_window_blocks: int,
    seed: int,
    tickers: str,
    start: str,
    end: str,
    max_tickers: int,
    min_history_bars: int,
) -> dict[str, bytes]:
    """Remote: build features once on the universe, then sweep
    `entropy_weights_csv` (one or more α values), running a fresh
    walk-forward per α. Bundle one npz + one plot per α plus a
    cross-α sweep summary."""
    import os
    import subprocess
    os.makedirs(f'{REMOTE_REPO}/Output', exist_ok=True)
    output = Path(f'{REMOTE_REPO}/Output')

    os.environ['CUDA'] = '1'

    print('=== Step 1/4: uv sync workspace deps ===', flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'factor', '--inexact'],
        cwd=REMOTE_REPO, check=True)

    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.12/site-packages')

    from tinygrad import Device
    if Device.DEFAULT != 'CUDA':
        raise RuntimeError(
            f'tinygrad picked Device.DEFAULT={Device.DEFAULT!r}, expected CUDA')
    print(f'  tinygrad Device.DEFAULT = {Device.DEFAULT}', flush=True)

    print('\n=== Step 2/4: load tickers + build deterministic indicator features ===',
          flush=True)
    ticker_list = _resolve_ticker_list(tickers, max_tickers, min_history_bars)
    print(f'  universe: {len(ticker_list)} tickers '
          f'(first 5: {ticker_list[:5]} ...)')

    from factor import (
        IndicatorGridConfig, make_indicator_backbone,
        train_scorer_horizon_walkforward,
    )
    from ss_features import TickerData

    # Config variants. `default` is the 2026-05-14 74-channel baseline;
    # `horizon-aligned` is the 2026-05-15 expansion that adds 30 cells
    # at periods matching the horizon set {5, 10, 20, 40, 60} (see
    # TODO/factor-horizon-aligned-grid.md for rationale).
    if config_variant == 'default':
        cfg = IndicatorGridConfig()
    elif config_variant == 'horizon-aligned':
        cfg = IndicatorGridConfig(
            rsi_n_grid=(5, 7, 10, 14, 20, 21, 30, 40, 60),
            rsi_w_grid=(1, 5, 10, 21, 63),
            cci_n_grid=(10, 14, 20, 40),
            cci_w_grid=(1, 5, 10, 21),
            vol_n_grid=(5, 10, 20, 40, 60, 120, 252),
            macd_fast_grid=(5, 8, 10, 12, 20, 21, 34, 40, 55, 60),
            coherence_window_grid=(5, 10, 20, 40, 60, 120),
        )
    else:
        raise ValueError(
            f"unknown config_variant={config_variant!r}; "
            f"expected 'default' or 'horizon-aligned'")
    F = cfg.feature_width()
    print(f'  config_variant = {config_variant!r}')
    print(f'  cfg.feature_width() = {F} channels')

    horizons = tuple(int(h) for h in horizons_csv.split(','))
    print(f'  horizons = {horizons} (h_min={min(horizons)}, K={len(horizons)})')

    import multiprocessing as mp
    n_workers = max(1, int(os.environ.get('FACTOR_FEATURE_WORKERS',
                                          os.cpu_count() or 4)))
    print(f'  parallelizing feature build across {n_workers} workers')

    t0 = time.perf_counter()
    ticker_data: list[TickerData] = []
    skipped: list[str] = []
    work_args = [(t, STOOQ_SUBSET, cfg, start, end) for t in ticker_list]
    with mp.Pool(n_workers) as pool:
        for i, (ticker, result) in enumerate(
                pool.imap_unordered(_build_one_ticker, work_args)):
            if isinstance(result, TickerData):
                ticker_data.append(result)
            else:
                skipped.append(f'{ticker} {result}')
            if (i + 1) % 50 == 0:
                print(f'  built {i + 1}/{len(ticker_list)}  '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
    ticker_data.sort(key=lambda td: td.name)
    print(f'  feature build done: {len(ticker_data)} usable / '
          f'{len(skipped)} skipped  ({time.perf_counter()-t0:.0f}s)')
    if len(ticker_data) < 4:
        raise RuntimeError(
            f'only {len(ticker_data)} tickers built — too few for IC training')

    print('\n=== Step 3/4: horizon-mixture walk-forward (sweep) ===',
          flush=True)
    backbone = make_indicator_backbone(ticker_data, cfg)
    entropy_weights = [float(a) for a in entropy_weights_csv.split(',') if a]
    deployment_reward_weights = [
        float(x) for x in deployment_reward_weights_csv.split(',') if x.strip()
    ]
    # Sweep is the cross-product (α, λ). For the canonical bilevel sweep
    # you pass α='0.0' so we only sweep λ; for legacy entropy-only sweeps
    # you pass deployment_reward_weights='0.0' so we only sweep α. Either
    # way, the smaller sweep stays len-1 and the inner loop just runs the
    # outer's full set.
    print(f'  sweeping entropy_weight ∈ {entropy_weights}', flush=True)
    print(f'  sweeping deployment_reward_weight ∈ {deployment_reward_weights}', flush=True)

    import numpy as np
    sweep_summary: list[dict] = []

    def _alpha_tag(a: float) -> str:
        """File-name-safe tag (e.g. 0.05 → 'a0p05', 0.0 → 'a0')."""
        s = f'{a:g}'.replace('.', 'p').replace('-', 'n')
        return f'a{s}'

    def _lambda_tag(lam: float) -> str:
        """File-name-safe tag (e.g. 0.25 → 'lam0p25', 0.0 → 'lam0')."""
        s = f'{lam:g}'.replace('.', 'p').replace('-', 'n')
        return f'lam{s}'

    for alpha in entropy_weights:
      for lam in deployment_reward_weights:
        # Single-arm runs (the default canonical sweep is α=0 fixed
        # while λ varies) use a clean tag. Cross-product runs get a
        # compound tag.
        if len(entropy_weights) == 1 and len(deployment_reward_weights) > 1:
            tag = _lambda_tag(lam)
        elif len(deployment_reward_weights) == 1 and len(entropy_weights) > 1:
            tag = _alpha_tag(alpha)
        else:
            tag = f'{_alpha_tag(alpha)}-{_lambda_tag(lam)}'
        print(f'\n  --- α={alpha} λ={lam} ({tag}) ---', flush=True)
        t1 = time.perf_counter()
        res = train_scorer_horizon_walkforward(
            ticker_data, backbone,
            horizons=horizons,
            train_window_blocks=train_window_blocks,
            val_window_blocks=val_window_blocks,
            step_window_blocks=step_window_blocks,
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
            n_steps=n_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            entropy_weight=alpha,
            deployment_reward_weight=lam,
            commission_bps=commission_bps,
            temperature=temperature,
            seed=seed,
            verbose=True,
        )
        wall = time.perf_counter() - t1
        print(f'    walk-forward wall: {wall:.1f}s', flush=True)

        endog_mean = res.mean_val_endog_sharpe
        random_mean = res.mean_val_random_sharpe
        h_best, best_fixed_mean = res.best_fixed_horizon
        delta_best = endog_mean - best_fixed_mean
        delta_rand = endog_mean - random_mean

        all_counts = np.zeros(len(horizons), dtype=np.int64)
        total_bars = 0
        for w in res.windows:
            for k, h in enumerate(horizons):
                all_counts[k] += w.val_pi_argmax_counts[h]
            total_bars += sum(w.val_pi_argmax_counts.values())
        pi_global = all_counts / max(total_bars, 1)
        collapse_max = float(np.max(pi_global))
        verdict_n1 = collapse_max <= 0.90
        verdict_n2 = endog_mean > res.mean_fixed_sharpe(max(horizons))
        verdict_n3 = delta_best >= 0.10
        verdict_n4 = delta_rand > 0.0
        verdict_pass = verdict_n1 and verdict_n2 and verdict_n3 and verdict_n4
        verdict_label = (
            'confirmed-OOS' if verdict_pass else
            'partial-OOS' if (verdict_n1 and verdict_n2 and verdict_n4) else
            'confirmed-null')

        # Mean per-window entropy across windows (sanity for whether α is
        # actually pulling π off the one-hot attractor).
        mean_entropy = float(np.mean(
            [w.val_pi_entropy_mean for w in res.windows]))

        print(f'    endog={endog_mean:+.3f}  best-fix h={h_best} '
              f'={best_fixed_mean:+.3f}  Δ={delta_best:+.3f}  '
              f'random={random_mean:+.3f}  Δr={delta_rand:+.3f}',
              flush=True)
        print(f'    argmax shares={ {h: round(float(pi_global[k]), 2) for k, h in enumerate(horizons)} }  '
              f'mean H(π)={mean_entropy:.2f}', flush=True)
        print(f'    N1={"P" if verdict_n1 else "F"} '
              f'N2={"P" if verdict_n2 else "F"} '
              f'N3={"P" if verdict_n3 else "F"} '
              f'N4={"P" if verdict_n4 else "F"} → {verdict_label}',
              flush=True)

        # Per-α npz.
        blob: dict[str, np.ndarray] = {
            'window_idx':           np.array([w.window_idx for w in res.windows], dtype=np.int32),
            'train_block_start':    np.array([w.train_block_start for w in res.windows], dtype=np.int32),
            'train_block_end':      np.array([w.train_block_end for w in res.windows], dtype=np.int32),
            'val_block_start':      np.array([w.val_block_start for w in res.windows], dtype=np.int32),
            'val_block_end':        np.array([w.val_block_end for w in res.windows], dtype=np.int32),
            'val_daily_start':      np.array([w.val_daily_start for w in res.windows], dtype=np.int32),
            'val_daily_end':        np.array([w.val_daily_end for w in res.windows], dtype=np.int32),
            'train_loss':           np.array([w.train_loss for w in res.windows], dtype=np.float32),
            'val_endog_sharpe':     np.array([w.val_endog_sharpe for w in res.windows], dtype=np.float32),
            'val_random_sharpe':    np.array([w.val_random_sharpe for w in res.windows], dtype=np.float32),
            'val_endog_mean_holding': np.array([w.val_endog_mean_holding for w in res.windows], dtype=np.float32),
            'val_endog_n_rebals':   np.array([w.val_endog_n_rebals for w in res.windows], dtype=np.int32),
            'val_endog_avg_turnover': np.array([w.val_endog_avg_turnover for w in res.windows], dtype=np.float32),
            'val_pi_entropy_mean':  np.array([w.val_pi_entropy_mean for w in res.windows], dtype=np.float32),
            'val_start_date':       np.array([w.val_start_date for w in res.windows]),
        }
        for h in horizons:
            blob[f'val_fixed_sharpe_h{h}'] = np.array(
                [w.val_fixed_sharpes[h] for w in res.windows], dtype=np.float32)
            blob[f'val_argmax_count_h{h}'] = np.array(
                [w.val_pi_argmax_counts[h] for w in res.windows], dtype=np.int32)
        blob['_summary'] = np.array(json.dumps({
            'horizons':             list(horizons),
            'config_variant':       config_variant,
            'feature_width':        F,
            'entropy_weight':       alpha,
            'deployment_reward_weight': lam,
            'mean_endog_sharpe':    endog_mean,
            'mean_random_sharpe':   random_mean,
            'best_fixed_horizon':   h_best,
            'best_fixed_sharpe':    best_fixed_mean,
            'delta_vs_best_fixed':  delta_best,
            'delta_vs_random':      delta_rand,
            'pi_argmax_global_shares': {
                int(h): float(pi_global[k]) for k, h in enumerate(horizons)},
            'mean_pi_entropy':      mean_entropy,
            'verdict_n1':           verdict_n1,
            'verdict_n2':           verdict_n2,
            'verdict_n3':           verdict_n3,
            'verdict_n4':           verdict_n4,
            'verdict_pass':         verdict_pass,
            'verdict_label':        verdict_label,
            'n_windows':            res.n_windows,
            'universe_size':        len(ticker_data),
            'feature_width':        F,
            'mlp_hidden':           mlp_hidden,
            'mlp_layers':           mlp_layers,
            'n_steps':              n_steps,
            'learning_rate':        learning_rate,
            'weight_decay':         weight_decay,
            'commission_bps':       commission_bps,
            'temperature':          temperature,
            'train_window_blocks':  train_window_blocks,
            'val_window_blocks':    val_window_blocks,
            'step_window_blocks':   step_window_blocks,
            'wall_seconds':         round(wall, 1),
        }, indent=2))
        # File-name prefix:
        #   - `horizon-align` for the 2026-05-15 horizon-aligned config sweep
        #     (regardless of λ/α dims, the config-variant is the load-bearing axis).
        #   - `horizon-bilevel` for the 2026-05-15 default-config λ sweep.
        #   - `horizon-mixture` for legacy default-config α-only sweeps.
        if config_variant == 'horizon-aligned':
            prefix = 'horizon-align'
        elif len(deployment_reward_weights) > 1:
            prefix = 'horizon-bilevel'
        else:
            prefix = 'horizon-mixture'
        npz_path = output / f'{prefix}-{tag}-windows.npz'
        np.savez(npz_path, **blob)
        plot_path = output / f'{prefix}-{tag}-comparison.png'
        _plot_horizon_mixture(res, horizons, plot_path)
        print(f'    -> {npz_path.name}, {plot_path.name}', flush=True)

        sweep_summary.append({
            'config_variant':    config_variant,
            'feature_width':     F,
            'entropy_weight':    alpha,
            'deployment_reward_weight': lam,
            'mean_endog_sharpe': endog_mean,
            'mean_random_sharpe': random_mean,
            'best_fixed_horizon': h_best,
            'best_fixed_sharpe': best_fixed_mean,
            'delta_vs_best_fixed': delta_best,
            'delta_vs_random':   delta_rand,
            'pi_argmax_global_shares': {
                int(h): float(pi_global[k]) for k, h in enumerate(horizons)},
            'mean_pi_entropy':   mean_entropy,
            'collapse_max':      collapse_max,
            'verdict_n1':        verdict_n1,
            'verdict_n2':        verdict_n2,
            'verdict_n3':        verdict_n3,
            'verdict_n4':        verdict_n4,
            'verdict_pass':      verdict_pass,
            'verdict_label':     verdict_label,
            'per_window_endog_sharpe': [w.val_endog_sharpe for w in res.windows],
            'per_window_best_fixed_sharpe': [
                max(w.val_fixed_sharpes.values()) for w in res.windows],
            'per_window_entropy': [w.val_pi_entropy_mean for w in res.windows],
            'wall_seconds':      round(wall, 1),
        })

    # ---------- Step 4: cross-arm sweep summary ----------
    print('\n=== Step 4/4: cross-arm sweep summary ===', flush=True)
    print(f'{"α":>6}  {"λ":>6}  {"endog":>7}  {"best-fix":>9}  {"Δ-fix":>6}  '
          f'{"Δ-rand":>7}  {"H(π)":>5}  {"verdict":>15}', flush=True)
    for s in sweep_summary:
        print(f'{s["entropy_weight"]:>6.3g}  '
              f'{s["deployment_reward_weight"]:>6.3g}  '
              f'{s["mean_endog_sharpe"]:>+7.3f}  '
              f'{s["best_fixed_sharpe"]:>+9.3f}  '
              f'{s["delta_vs_best_fixed"]:>+6.3f}  '
              f'{s["delta_vs_random"]:>+7.3f}  '
              f'{s["mean_pi_entropy"]:>5.2f}  '
              f'{s["verdict_label"]:>15}', flush=True)

    # Best-arm heuristic: highest delta_vs_best_fixed among those passing N1+N2+N4.
    eligible = [s for s in sweep_summary
                if s['verdict_n1'] and s['verdict_n2'] and s['verdict_n4']]
    if eligible:
        best = max(eligible, key=lambda s: s['delta_vs_best_fixed'])
        print(f'\n  best arm (passes N1/N2/N4, max Δ-fix): '
              f'α={best["entropy_weight"]} λ={best["deployment_reward_weight"]}  '
              f'Δ-fix={best["delta_vs_best_fixed"]:+.3f}  '
              f'verdict={best["verdict_label"]}', flush=True)
    else:
        print('\n  no arm passes N1+N2+N4 — architecture confirmed-null '
              'under the discrete mixture-of-horizons-IC + deployment-reward '
              'objective.', flush=True)

    # Sweep summary path keyed by config_variant — horizon-aligned
    # writes to its own summary so it doesn't overwrite the prior
    # default-config sweeps' files.
    if config_variant == 'horizon-aligned':
        sweep_path = output / 'horizon-align-sweep-summary.json'
    elif len(deployment_reward_weights) > 1:
        sweep_path = output / 'horizon-bilevel-sweep-summary.json'
    else:
        sweep_path = output / 'horizon-mixture-sweep-summary.json'
    sweep_path.write_text(json.dumps({
        'config_variant': config_variant,
        'feature_width':  F,
        'sweep_alphas':   entropy_weights,
        'sweep_lambdas':  deployment_reward_weights,
        'arms':           sweep_summary,
    }, indent=2))
    print(f'  -> {sweep_path.name}')

    artifacts: dict[str, bytes] = {}
    for p in sorted(output.iterdir()):
        if p.is_file() and (p.name.startswith('horizon-mixture')
                            or p.name.startswith('horizon-bilevel')
                            or p.name.startswith('horizon-align')):
            artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts')
    return artifacts


def _build_one_ticker(args):
    ticker, stooq_subset, cfg, start, end = args
    import numpy as np
    from factor import build_indicator_features
    from ss_features import TickerData, load_prices
    try:
        series = load_prices(ticker, stooq_dir=stooq_subset,
                             start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        feats, valid = build_indicator_features(prices, cfg)
        if not valid.any():
            return ticker, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        )
    except Exception as e:
        return ticker, f'({type(e).__name__}: {e})'


def _resolve_ticker_list(
    tickers: str, max_tickers: int, min_history_bars: int = 0,
) -> list[str]:
    manifest_path = Path(STOOQ_SUBSET) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if tickers:
        requested = {t.strip().upper() for t in tickers.split(',') if t.strip()}
        entries = [t for t in manifest['tickers'] if t['ticker'].upper() in requested]
    else:
        entries = list(manifest['tickers'])
    if min_history_bars > 0:
        before = len(entries)
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
        dropped = before - len(entries)
        if dropped:
            print(f'  min_history_bars={min_history_bars}: '
                  f'dropped {dropped} short-history tickers')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
    return names


def _plot_horizon_mixture(res, horizons, out_path: Path) -> None:
    """Three-panel plot:
      1. Per-window Sharpes (endog vs each fixed-h vs random-π).
      2. Per-window argmax-bin histogram (which horizons the model picked).
      3. Mean π entropy per window (proxy for confidence vs uncertainty).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not res.windows:
        return
    n = res.n_windows
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, n * 0.4), 11),
                             sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1]})

    # Panel 1: Sharpes.
    ax1 = axes[0]
    endog = [w.val_endog_sharpe for w in res.windows]
    ax1.bar(x, endog, color='crimson', label='endog (argmax π)', alpha=0.85)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(horizons)))
    for k, (h, c) in enumerate(zip(horizons, colors)):
        ys = [w.val_fixed_sharpes[h] for w in res.windows]
        ax1.plot(x, ys, marker='o', markersize=4, color=c, label=f'fix h={h}',
                 linewidth=1.2)
    rand = [w.val_random_sharpe for w in res.windows]
    ax1.plot(x, rand, marker='x', markersize=5, color='gray',
             label='random-π', linestyle='--', linewidth=1.2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Sharpe (daily PnL, net of costs)')
    ax1.set_title(
        f'Horizon-mixture walk-forward — mean endog={res.mean_val_endog_sharpe:+.3f}  '
        f'best-fix(h={res.best_fixed_horizon[0]})={res.best_fixed_horizon[1]:+.3f}  '
        f'random-π={res.mean_val_random_sharpe:+.3f}')
    ax1.legend(fontsize=8, ncol=3)

    # Panel 2: argmax-bin histogram per window.
    ax2 = axes[1]
    bottom = np.zeros(n)
    for k, (h, c) in enumerate(zip(horizons, colors)):
        heights = np.array([w.val_pi_argmax_counts[h] for w in res.windows],
                           dtype=float)
        ax2.bar(x, heights, bottom=bottom, color=c,
                label=f'h={h}', width=0.85)
        bottom += heights
    ax2.set_ylabel('argmax(π_t) bar counts')
    ax2.legend(fontsize=8, ncol=len(horizons))

    # Panel 3: mean entropy per window.
    ax3 = axes[2]
    ent = [w.val_pi_entropy_mean for w in res.windows]
    ax3.plot(x, ent, marker='o', color='steelblue', linewidth=1.2)
    ax3.axhline(np.log(len(horizons)), color='gray', linestyle=':',
                label=f'log(K)={np.log(len(horizons)):.2f}')
    ax3.set_ylabel('mean H(π_t)')
    ax3.set_xlabel('walk-forward window')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'w{i}' for i in range(n)], fontsize=7,
                       rotation=45 if n > 10 else 0)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  -> {out_path.name}')


@app.local_entrypoint()
def main(
    horizons: str = '5,10,20,40,60',
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    entropy_weights: str = '0.0',
    deployment_reward_weights: str = '0.0',
    config_variant: str = 'default',
    mlp_hidden: int = 32,
    mlp_layers: int = 1,
    commission_bps: float = 10.0,
    temperature: float = 1.0,
    train_window_blocks: int = 252,
    val_window_blocks: int = 156,
    step_window_blocks: int = 156,
    seed: int = 0,
    tickers: str = '',
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    max_tickers: int = 0,
    min_history_bars: int = 6500,
) -> None:
    """Local entrypoint: kicks off remote walk-forward sweep and downloads
    artifacts.

    Pass a comma-separated `entropy_weights` to sweep entropy regularization
    (the 2026-05-14 confirmed-null sweep), or `deployment_reward_weights`
    for the 2026-05-15 bilevel objective sweep. Examples:

        --entropy-weights '0.0,0.05,0.1,0.2,0.3'                     (legacy α-sweep)
        --deployment-reward-weights '0.0,0.25,0.5,1.0,2.0'           (bilevel λ-sweep)

    Both can be provided together for a cross-product sweep, but the
    canonical bilevel sweep keeps α=0 fixed.
    """
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'launching factor horizon-mixture sweep on Modal '
          f'(horizons={horizons}, n_steps={n_steps}, lr={learning_rate}, '
          f'wd={weight_decay}, entropy_weights={entropy_weights}, '
          f'deployment_reward_weights={deployment_reward_weights}, '
          f'config_variant={config_variant!r}, '
          f'train/val/step blocks={train_window_blocks}/{val_window_blocks}/'
          f'{step_window_blocks}, max_tickers={max_tickers}, '
          f'min_history_bars={min_history_bars})')
    artifacts = train_horizon_walkforward_remote.remote(
        horizons_csv=horizons,
        n_steps=n_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        entropy_weights_csv=entropy_weights,
        deployment_reward_weights_csv=deployment_reward_weights,
        config_variant=config_variant,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
        commission_bps=commission_bps,
        temperature=temperature,
        train_window_blocks=train_window_blocks,
        val_window_blocks=val_window_blocks,
        step_window_blocks=step_window_blocks,
        seed=seed,
        tickers=tickers,
        start=start, end=end,
        max_tickers=max_tickers,
        min_history_bars=min_history_bars,
    )
    for name, data in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(data)
        print(f'  wrote {out}  ({len(data) // 1024}KB)')
    print(f'done — {len(artifacts)} files in {LOCAL_OUTPUT_DIR}/')
