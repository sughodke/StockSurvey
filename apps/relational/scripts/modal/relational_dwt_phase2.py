"""Modal entrypoint: 8-arm Phase-2 head-to-head for DWT-LL fingerprint
compression on three distance-based scorers.

Arms:
  analog                — k-NN cross_ticker pool, full-resolution fingerprint
  analog-dwt-L1         — same kNN, 2D Haar keep-LL fingerprint (1 level)
  analog-pt             — k-NN per_ticker pool, full-resolution fingerprint
  analog-pt-dwt-L1      — same per_ticker kNN, DWT-L1 fingerprint
  farthest              — centroid-distance scoring, full-resolution
  farthest-dwt-L1       — same, DWT-L1
  diversified           — greedy farthest-first thinning, full-resolution
  diversified-dwt-L1    — same, DWT-L1

CWT scalogram is shared across arms via `relational.scalogram_cache`
(same prices / scales / lookback hash → same npz), so the per-arm cost
after the first arm is just the kNN / centroid / thinning loops. Pure
numpy + bt; no GPU needed, plain CPU instance is fine.

The Phase-2 universe (21 mega-cap tickers) is not fully present in the
baked-in `stooq_us_long` subset (CRM / GOOGL / META / NFLX / TSLA
post-date its 2000-01-01 cutoff), so prices are prepped locally via
`prep_phase2_prices.py` and shipped over RPC as a pickle blob. The
remote function unpickles and runs the 8 arms.

Usage
-----
    uv run python apps/relational/scripts/modal/prep_phase2_prices.py
    uvx modal run apps/relational/scripts/modal/relational_dwt_phase2.py

Wall-time estimate: ~25-30 min (one ~7-min baseline arm worth of CWT
build amortised across 8 arms; arms 2-8 are ~2-3 min each).
"""
from __future__ import annotations

from pathlib import Path

import modal


try:
    REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    REPO_ROOT = Path('/root/StockSurvey')
LOCAL_OUTPUT_DIR = REPO_ROOT / 'Output'
REMOTE_REPO = '/root/StockSurvey'

DEFAULT_PRICES_PKL = Path('/tmp/phase2-prices.pkl')

# Plain Python image — no GPU needed, the relational backtest is pure
# numpy + pandas + bt. Slim base + uv keeps cold start fast.
image = (
    modal.Image.debian_slim(python_version='3.13')
    .apt_install('git', 'curl', 'build-essential', 'clang')
    .pip_install('uv')
    .add_local_dir(
        REPO_ROOT.as_posix(),
        remote_path=REMOTE_REPO,
        ignore=[
            '.git/**',
            '.venv/**',
            '.iv-cache/**',
            '.claude/**',
            'Output/**',
            'StooqData/**',
            'Nasdaq3347/**',
            'apps/factor/src/**',
            'apps/replay/src/**',
            'apps/regime/src/**',
            'apps/v1/src/**',
            '**/__pycache__/**',
            '**/*.pyc',
        ],
    )
)

app = modal.App('ss-relational-dwt-phase2', image=image)


# Arms run sequentially in one container; total ~25-30 min with the
# CWT cached after the first call. Bumped timeout to 90 min for safety.
@app.function(cpu=8, memory=16384, timeout=60 * 90)
def run_arms(prices_pkl: bytes) -> dict[str, bytes]:
    """Run the 8-arm head-to-head and return artifacts as
    {filename: bytes}."""
    import os
    import pickle
    import subprocess
    import warnings
    warnings.filterwarnings('ignore')

    print('=== Step 1/3: uv sync workspace deps (one-time per cold start) ===',
          flush=True)
    subprocess.run(
        ['uv', 'sync', '--package', 'relational', '--extra', 'research',
         '--inexact'],
        cwd=REMOTE_REPO, check=True)

    # Activate the editable-installed venv so we can import workspace pkgs.
    import site
    site.addsitedir(f'{REMOTE_REPO}/.venv/lib/python3.13/site-packages')

    import bt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ss_features import Compression
    from ss_portfolio.bt_helpers import build_strategy

    from relational.analog_knn import weights_regime_analog
    from relational.diversify import weights_regime_diversified
    from relational.farthest import weights_regime_farthest

    bundle = pickle.loads(prices_pkl)
    prices = bundle['prices']
    start = bundle['start']
    end = bundle['end']
    print(f'\n=== Step 2/3: loaded prices {prices.shape} '
          f'({start} → {end}) ===', flush=True)

    lookback = 120
    top_n = 10
    fp_window = 21
    rebal_days = 20
    commission_bps = 10.0
    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    comp_l1 = Compression(kind='dwt', levels=1, wavelet='haar',
                          pad_mode='periodization')

    arms: list[tuple[str, callable]] = [
        ('analog', lambda: weights_regime_analog(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, k_neighbors=50, forward_horizon=20,
            min_sep_days=21, pool_mode='cross_ticker')),
        ('analog-dwt-L1', lambda: weights_regime_analog(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, k_neighbors=50, forward_horizon=20,
            min_sep_days=21, pool_mode='cross_ticker',
            compression=comp_l1)),
        ('analog-pt', lambda: weights_regime_analog(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, k_neighbors=50, forward_horizon=20,
            min_sep_days=21, pool_mode='per_ticker')),
        ('analog-pt-dwt-L1', lambda: weights_regime_analog(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, k_neighbors=50, forward_horizon=20,
            min_sep_days=21, pool_mode='per_ticker',
            compression=comp_l1)),
        ('farthest', lambda: weights_regime_farthest(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window)),
        ('farthest-dwt-L1', lambda: weights_regime_farthest(
            prices, lookback=lookback, top_n=top_n, scales=scales,
            fp_window=fp_window, compression=comp_l1)),
        ('diversified', lambda: weights_regime_diversified(
            prices, lookback=lookback, scales=scales,
            n_tail=20, k_keep=top_n, top_pool=20, divergence='kl',
            fp_window=fp_window)),
        ('diversified-dwt-L1', lambda: weights_regime_diversified(
            prices, lookback=lookback, scales=scales,
            n_tail=20, k_keep=top_n, top_pool=20, divergence='kl',
            fp_window=fp_window, compression=comp_l1)),
    ]

    weights_by_arm = {}
    for name, fn in arms:
        print(f'\n[{name}] computing weights ...', flush=True)
        weights_by_arm[name] = fn()
        print(f'[{name}] done — '
              f'{(weights_by_arm[name].sum(axis=1) > 0).sum()} '
              f'rebalances rows', flush=True)

    print(f'\n=== Step 3/3: bt backtests ===', flush=True)
    strategies = [
        build_strategy(name, prices, w,
                       rebal_days=rebal_days, commission_bps=commission_bps)
        for name, w in weights_by_arm.items()
    ]
    result = bt.run(*strategies)
    result.display()

    output = Path(REMOTE_REPO) / 'Output'
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    result.plot(ax=ax)
    ax.set_title(
        f'Relational distance scorers — DWT-L1 vs full-res fingerprint '
        f'(Phase-2, {start} → {end}, top-{top_n}, rebal={rebal_days}d)')
    fig.tight_layout()
    fig_path = output / 'relational-dwt-phase2-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    stats_path = output / 'relational-dwt-phase2-stats.txt'
    stats_path.write_text(str(result.stats))

    artifacts: dict[str, bytes] = {}
    for p in [fig_path, stats_path]:
        artifacts[p.name] = p.read_bytes()
    print(f'\nbundling {len(artifacts)} artifacts', flush=True)
    return artifacts


@app.local_entrypoint()
def main(prices_pkl_path: str = str(DEFAULT_PRICES_PKL)) -> None:
    """Read the Phase-2 prices pickle (made by `prep_phase2_prices.py`)
    and ship it to the remote container."""
    src = Path(prices_pkl_path)
    if not src.exists():
        raise SystemExit(
            f'{src} not found — run `uv run python '
            f'apps/relational/scripts/modal/prep_phase2_prices.py` first.')
    pkl = src.read_bytes()
    print(f'>>> shipping {src} ({len(pkl):,} bytes) to Modal ...')
    artifacts = run_arms.remote(pkl)
    LOCAL_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'\n=== Writing {len(artifacts)} artifacts to '
          f'{LOCAL_OUTPUT_DIR} ===')
    for name, blob in artifacts.items():
        out = LOCAL_OUTPUT_DIR / name
        out.write_bytes(blob)
        print(f'  ← {out.name}  ({len(blob):,} bytes)')
    print('\nDone.')
