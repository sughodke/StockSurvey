"""Shared argparse helpers for StockSurvey CLI scripts.

Two distinct loader-arg groupings exist in the workspace and they don't
overlap:

  * Single-ticker scripts (notebook scalogram tools) accept *either*
    `--stooq-dir` or `--kaggle-dir` — pick one source per invocation.
    Use `add_single_ticker_loader_args(parser)`.

  * Universe scripts (regime trainer + research) want a single
    required `--data-dir`. Use `add_universe_loader_args(parser)`.

Both groupings get `--start` / `--end` for date-range bounds.
`add_save_args(parser)` adds the universal `--save` flag plus
`--output-dir` (default `Output`); use it on top of either loader
grouping.

The script's argparse usage stays the same — these helpers just append
the standardized flag block. No `parse_args` or read-back side effects.
"""
from ss_cli.loaders import (
    add_save_args,
    add_single_ticker_loader_args,
    add_universe_loader_args,
)

__all__ = [
    'add_save_args',
    'add_single_ticker_loader_args',
    'add_universe_loader_args',
]
