"""Build the canonical RelationalCheckpoint JSONs for the six
scoreboard winners, ready to feed `ss-relational live --params ...`.

Six strategies, two universes:

  Phase-2 mega-cap (21 names, PHASE2_TICKERS):
    * empirical   — k-means clusters of CWT fingerprints
                    (NO_OPTIONS phase 2: long-only Sharpe 1.07-1.13)
    * gmm         — soft-cluster GMM replacement (+0.03 over empirical)
    * analog      — k-NN analog forecasting on fingerprints (idea-B)
    * farthest    — centroid-distance scoring (idea-C)
    * diversified — greedy farthest-first thinning (idea-D)

  Wide universe (~312 names, apps/notebook/data/stooq_us_long):
    * velocity    — fingerprint-space directed motion (Phase-11,
                    designed for wider universes specifically)

Universes are resolved at script-runtime so the JSON pins the actual
list of names (rather than a placeholder). The Phase-2 list is
literal; the wide universe is enumerated by `load_stooq_matrix`.

KNOWN CONCENTRATION RISK on the Phase-2 strategies: phase-8 of the
NO_OPTIONS arc showed all four ideas (A/B/C/D) degrade from
Sharpe ~1.1 to ~0.4 when run on the 312-ticker `stooq_us_long`
universe instead. The Phase-2 wins are mega-cap-specific. The
checkpoint metadata records this honestly via train_sharpe /
val_sharpe; operators should not extrapolate the result to other
universes without retraining.

Usage:
    uv run python apps/relational/scripts/build_canonical_checkpoints.py

Writes to `Output/relational-{strategy}.json`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from relational.persist import (
    CHECKPOINT_VERSION,
    RelationalCheckpoint,
    save_checkpoint,
)
from relational.sectors import PHASE2_TICKERS

# Canonical scale grid used by all five idea_*.py + diagnostic_velocity.py.
# Spans 1-week (5d) through 4-month (90d). Different from `ss_wavelets.ALL_SCALES`
# which is broader (15 scales up to 126d) — the relational scripts pin a
# subset. Keep these two in sync if either is changed.
RELATIONAL_SCALES: list[int] = [5, 7, 10, 12, 21, 26, 50, 90]

# Phase-2 hyperparameters that won the scoreboard for ideas A-D.
PHASE2_HPARAMS = dict(
    lookback=120,
    top_n=10,
    n_tail=20,
    fp_window=21,
    divergence='kl',
)

# Provenance windows. Phase-2 backtests run 2013-01-29 -> 2025-12-11
# with a default split; transcribe the exact dates rather than recomputing.
PHASE2_TRAIN_START = '2013-01-29'
PHASE2_TRAIN_END = '2020-12-31'
PHASE2_VAL_START = '2021-01-01'
PHASE2_VAL_END = '2025-12-11'

# Phase-11 (velocity) backtest range, per
# `apps/relational/src/relational/research/diagnostic_velocity.py`.
PHASE11_TRAIN_START = '2010-01-01'
PHASE11_TRAIN_END = '2020-12-31'
PHASE11_VAL_START = '2021-01-01'
PHASE11_VAL_END = '2025-12-11'


def _common(strategy: str) -> dict:
    """Fields shared across all checkpoints."""
    return dict(
        version=CHECKPOINT_VERSION,
        strategy=strategy,
        scales=RELATIONAL_SCALES,
        rebal_days=20,
        max_spread=0.05,
        commission_bps=10.0,
        trained_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
    )


def build_phase2_checkpoints() -> list[RelationalCheckpoint]:
    """Five mega-cap-specific strategies, all on PHASE2_TICKERS."""
    universe = list(PHASE2_TICKERS)
    return [
        # idea-A — Sharpe 1.07 long-only (NO_OPTIONS phase-2)
        RelationalCheckpoint(
            **_common('empirical'),
            universe=universe,
            **{k: PHASE2_HPARAMS[k] for k in ('lookback', 'top_n')},
            train_start=PHASE2_TRAIN_START, train_end=PHASE2_TRAIN_END,
            val_start=PHASE2_VAL_START, val_end=PHASE2_VAL_END,
            train_sharpe=1.07, val_sharpe=1.07,
            strategy_kwargs=dict(
                n_tail=PHASE2_HPARAMS['n_tail'],
                divergence=PHASE2_HPARAMS['divergence'],
                fp_window=PHASE2_HPARAMS['fp_window'],
                k_clusters=11,
                refit_days=252,
            ),
        ),
        # GMM soft-cluster replacement — phase-10, +0.03 over empirical
        RelationalCheckpoint(
            **_common('gmm'),
            universe=universe,
            **{k: PHASE2_HPARAMS[k] for k in ('lookback', 'top_n')},
            train_start=PHASE2_TRAIN_START, train_end=PHASE2_TRAIN_END,
            val_start=PHASE2_VAL_START, val_end=PHASE2_VAL_END,
            train_sharpe=1.10, val_sharpe=1.10,
            strategy_kwargs=dict(
                n_tail=PHASE2_HPARAMS['n_tail'],
                divergence=PHASE2_HPARAMS['divergence'],
                fp_window=PHASE2_HPARAMS['fp_window'],
                n_components=11,
                refit_days=252,
            ),
        ),
        # idea-B — k-NN analog forecasting on full-resolution
        # fingerprints (NO compression). The earlier 2026-05-07 in-
        # sample head-to-head suggested DWT-L1 helped (Daily Sharpe
        # 1.11 vs 1.07 over 12y in-sample), but the segmented
        # walk-forward eval showed the train edge does NOT survive
        # OOS — full 8-arm Modal A/B over (analog cross/per_ticker ×
        # farthest × diversified) × ±DWT-L1 confirmed the reversal:
        # the uncompressed cross_ticker baseline is the ONLY arm
        # whose val Sharpe (1.146) exceeds its train Sharpe (1.032);
        # every compressed arm shows train > val by 0.02-0.44
        # Sharpe. See WALKFORWARD.md for the full per-arm table.
        # Operational verdict: keep this checkpoint at full-resolution
        # fingerprints. compress_levels intentionally absent.
        RelationalCheckpoint(
            **_common('analog'),
            universe=universe,
            **{k: PHASE2_HPARAMS[k] for k in ('lookback', 'top_n')},
            train_start=PHASE2_TRAIN_START, train_end=PHASE2_TRAIN_END,
            val_start=PHASE2_VAL_START, val_end=PHASE2_VAL_END,
            train_sharpe=1.032, val_sharpe=1.146,
            strategy_kwargs=dict(
                fp_window=PHASE2_HPARAMS['fp_window'],
                k_neighbors=50,
                forward_horizon=20,
                min_sep_days=21,
                pool_mode='cross_ticker',
            ),
        ),
        # idea-C — centroid distance scoring
        RelationalCheckpoint(
            **_common('farthest'),
            universe=universe,
            **{k: PHASE2_HPARAMS[k] for k in ('lookback', 'top_n')},
            train_start=PHASE2_TRAIN_START, train_end=PHASE2_TRAIN_END,
            val_start=PHASE2_VAL_START, val_end=PHASE2_VAL_END,
            train_sharpe=1.07, val_sharpe=1.07,
            strategy_kwargs=dict(
                fp_window=PHASE2_HPARAMS['fp_window'],
            ),
        ),
        # idea-D — greedy farthest-first thinning
        RelationalCheckpoint(
            **_common('diversified'),
            universe=universe,
            **{k: PHASE2_HPARAMS[k] for k in ('lookback', 'top_n')},
            train_start=PHASE2_TRAIN_START, train_end=PHASE2_TRAIN_END,
            val_start=PHASE2_VAL_START, val_end=PHASE2_VAL_END,
            train_sharpe=1.07, val_sharpe=1.07,
            strategy_kwargs=dict(
                n_tail=PHASE2_HPARAMS['n_tail'],
                divergence=PHASE2_HPARAMS['divergence'],
                fp_window=PHASE2_HPARAMS['fp_window'],
                top_pool=20,  # 2x top_n, per idea_d_diversified.py default
            ),
        ),
    ]


def build_velocity_checkpoint(stooq_dir: Path) -> RelationalCheckpoint:
    """Wide-universe (~312 tickers) regime-velocity scorer (Phase-11)."""
    from ss_loaders import load_stooq_matrix
    # Match diagnostic_velocity.py:245 — min_history high enough for the
    # full lookback + tail + train window so the script's universe pin
    # mirrors the one tested.
    min_history = 120 + 20 + 252 + 10  # lookback + n_tail + train_days + slack
    prices, _, _, _ = load_stooq_matrix(
        str(stooq_dir),
        min_history=min_history,
        start_date=PHASE11_TRAIN_START,
        end_date=PHASE11_VAL_END,
        tickers=None,
    )
    universe = list(prices.columns)
    print(f'velocity universe resolved: {len(universe)} tickers from {stooq_dir}')

    return RelationalCheckpoint(
        **_common('velocity'),
        universe=universe,
        lookback=120,
        top_n=20,  # diagnostic_velocity.py default
        train_start=PHASE11_TRAIN_START, train_end=PHASE11_TRAIN_END,
        val_start=PHASE11_VAL_START, val_end=PHASE11_VAL_END,
        train_sharpe=0.60, val_sharpe=0.60,  # NO_OPTIONS phase-11 long-only
        strategy_kwargs=dict(
            fp_window=21,
            w_delta=20,
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description='Build canonical RelationalCheckpoint JSONs.')
    p.add_argument('--output-dir', default='Output',
                   help='Directory for the *.json output files.')
    p.add_argument('--stooq-us-long-dir',
                   default='apps/notebook/data/stooq_us_long',
                   help='Stooq archive path for the wide-universe '
                        'velocity checkpoint. Skip with --skip-velocity '
                        'if not on disk.')
    p.add_argument('--skip-velocity', action='store_true',
                   help="Don't try to load the wide stooq universe "
                        '(useful when the Stooq archive is not present).')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for cp in build_phase2_checkpoints():
        path = out / f'relational-{cp.strategy}.json'
        save_checkpoint(path, cp)
        written.append(path)
        print(f'  wrote {path}  '
              f'(strategy={cp.strategy}, universe={len(cp.universe)} names, '
              f'train_sharpe={cp.train_sharpe:+.2f})')

    if not args.skip_velocity:
        stooq_dir = Path(args.stooq_us_long_dir)
        if not stooq_dir.exists():
            print(f'WARN: {stooq_dir} not present — skipping velocity. '
                  f'Re-run with --stooq-us-long-dir <path> when ready.')
        else:
            cp = build_velocity_checkpoint(stooq_dir)
            path = out / f'relational-{cp.strategy}.json'
            save_checkpoint(path, cp)
            written.append(path)
            print(f'  wrote {path}  '
                  f'(strategy={cp.strategy}, universe={len(cp.universe)} names, '
                  f'train_sharpe={cp.train_sharpe:+.2f})')

    print(f'\nDone. {len(written)} checkpoints written.')
    print('Verify dry-run before live: '
          'uv run ss-relational live --params <path> --dry-run')


if __name__ == '__main__':
    main()
