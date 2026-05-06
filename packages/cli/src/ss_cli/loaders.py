"""Argparse helpers for the data-loader and save flag groups."""
from __future__ import annotations

import argparse


def add_single_ticker_loader_args(parser: argparse.ArgumentParser) -> None:
    """Add `--stooq-dir`, `--kaggle-dir`, `--start`, `--end`.

    For tools that load one ticker at a time and pick a backing source
    per invocation (e.g. the scalogram CLIs in apps/notebook). Defaults
    are left as `None` so callers can fall back to module-level
    constants like `ss_features.DEFAULT_STOOQ_DIR`.
    """
    parser.add_argument(
        '--stooq-dir', default=None,
        help='Stooq archive root (contains daily/). '
             'Default: ss_features.DEFAULT_STOOQ_DIR.')
    parser.add_argument(
        '--kaggle-dir', default=None,
        help='Slice one column from a Nasdaq3347-style CSV matrix '
             'instead of using Stooq.')
    parser.add_argument('--start', default=None, help='YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='YYYY-MM-DD')


def add_universe_loader_args(
    parser: argparse.ArgumentParser,
    *,
    default_start: str | None = None,
    default_end:   str | None = None,
    data_dir_help: str = 'Path to the daily price data directory.',
) -> None:
    """Add `--data-dir` (required), `--start`, `--end`.

    For tools that operate on a whole universe loaded from a single
    directory (regime trainer + every `regime/research/*` script).
    `default_start` / `default_end` let callers preserve their previous
    historical defaults (e.g. `'2010-01-01'` / `'2025-12-31'`) without
    redefining the flags.
    """
    parser.add_argument('--data-dir', required=True, help=data_dir_help)
    parser.add_argument('--start', default=default_start, help='YYYY-MM-DD')
    parser.add_argument('--end',   default=default_end,   help='YYYY-MM-DD')


def add_save_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str = 'Output',
) -> None:
    """Add `--save` (boolean flag) and `--output-dir`.

    Convention: scripts default to displaying interactively (matplotlib
    `plt.show()`); pass `--save` to persist artifacts under
    `--output-dir` instead. Universe-research scripts that always save
    can override `default_output_dir` and ignore the boolean flag.
    """
    parser.add_argument(
        '--save', action='store_true',
        help=f'Save artifacts to {default_output_dir}/ instead of '
             'displaying interactively.')
    parser.add_argument(
        '--output-dir', default=default_output_dir,
        help=f'Directory for saved artifacts (default: {default_output_dir}).')
