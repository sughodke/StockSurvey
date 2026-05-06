"""Walk-forward windowing primitives.

Two protocols, one module:

  * `block_windows(n_blocks, train_w, val_w, step_w)` — integer-slice
    windowing over rebalance-block indices. Used by the factor app's
    walk-forward IC scorer where blocks are pre-aggregated rebal events.

  * `calendar_windows(index, train_years, val_years, step_years)` —
    `pd.DateOffset`-based windowing over a `DatetimeIndex`. Used by the
    regime trainer's Optuna search where each window is a real
    calendar slice of the daily price frame.

Both are pure generators of (train, val) pairs — no model state, no
pandas / numpy on the hot path beyond the slice/loc lookup. Callers
do whatever they need inside each window.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def block_windows(
    n_blocks: int, train_w: int, val_w: int, step_w: int,
) -> list[tuple[slice, slice]]:
    """Roll a `(train_w, val_w)` pair forward by `step_w` blocks at a time.

    Returns one `(train_slice, val_slice)` per window that fits entirely
    inside `n_blocks`. The last window may not align to the end exactly;
    we don't pad — better to drop a partial window than evaluate on too
    few val blocks. Slices are over a 0..n_blocks-1 integer axis (e.g.
    pre-aggregated rebal-event indices).
    """
    if train_w < 2 or val_w < 2:
        raise ValueError(
            f'train_w={train_w} and val_w={val_w} must each be >= 2 '
            'for a meaningful walk-forward evaluation')
    if step_w < 1:
        raise ValueError(f'step_w={step_w} must be >= 1')

    out: list[tuple[slice, slice]] = []
    cursor = 0
    needed = train_w + val_w
    while cursor + needed <= n_blocks:
        out.append((
            slice(cursor, cursor + train_w),
            slice(cursor + train_w, cursor + needed),
        ))
        cursor += step_w
    return out


@dataclass(frozen=True)
class CalendarWindow:
    """A single rolling-calendar walk-forward window.

    `train` covers `[window_start, train_end]`, `val` covers
    `[train_end, val_end]` (the boundary day belongs to both — `pd.loc`
    is inclusive on both ends, which matches the regime-trainer's
    historical behaviour).
    """
    window_start: pd.Timestamp
    train_end:    pd.Timestamp
    val_end:      pd.Timestamp


def calendar_windows(
    index: pd.DatetimeIndex,
    *,
    train_years: int,
    val_years:   int,
    step_years:  int,
) -> list[CalendarWindow]:
    """Roll a `(train_years, val_years)` calendar window forward by
    `step_years` years at a time.

    Returns one `CalendarWindow` per window whose `val_end` fits inside
    `index[-1]`. Windows that don't fit (i.e. the rolling cursor would
    push `val_end` past the end of the data) are dropped.

    Note: this generator returns *only* the date boundaries; the caller
    slices its own DataFrame(s) via `df.loc[w.window_start:w.train_end]`
    etc. That keeps this module free of the consumer's data conventions
    (price frames, spread frames, multi-frame alignment).
    """
    if train_years < 1 or val_years < 1:
        raise ValueError(
            f'train_years={train_years} and val_years={val_years} '
            'must each be >= 1')
    if step_years < 1:
        raise ValueError(f'step_years={step_years} must be >= 1')

    start = index[0]
    end = index[-1]
    out: list[CalendarWindow] = []
    window_start = start
    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        if val_end > end:
            break
        out.append(CalendarWindow(
            window_start=window_start,
            train_end=train_end,
            val_end=val_end,
        ))
        window_start += pd.DateOffset(years=step_years)
    return out
