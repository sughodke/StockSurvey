"""Cluster-membership transition detection over a stabilized cluster_ids
matrix.

Companion to `relational.cluster_tracking`. Given a `(n_dates,
n_tickers)` int matrix of stable cluster IDs (Hungarian-matched across
refit boundaries), detect dates where a ticker *just changed clusters
and the new cluster has persisted for at least `persistence` days*. The
persistence filter kills boundary jitter (a ticker bouncing between
two centroids day-to-day) without losing real, sustained moves.

Transition events are then aggregated into trigger dates suitable for
`bt.algos.RunOnDate(*dates)`: a transition trigger fires on any date
where any ticker (or any top-N ticker) just transitioned. The
`transition-or-20d` variant unions these dates with the standard 20-day
rebalance grid.

Edge handling
-------------
* `cluster_ids[t, i] < 0` (pre-lookback or non-finite fingerprint) is
  not a transition source or target — transitions across `-1` are
  ignored.
* `t < persistence` from the panel start: not enough forward window to
  confirm persistence — never triggers a transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_transition_dates(
    cluster_ids: np.ndarray,
    *,
    persistence: int = 5,
) -> np.ndarray:
    """Per-(date, ticker) bool mask of confirmed cluster transitions.

    Parameters
    ----------
    cluster_ids : np.ndarray, shape `(n_eval, n_tickers)`, int
        Stabilized cluster IDs (use
        `relational.cluster_tracking.stabilize_cluster_ids`). Negative
        entries are treated as missing.
    persistence : int
        A change at `t` is a confirmed transition only if
        `cluster_ids[t:t+persistence, i]` is constant (and equals
        `cluster_ids[t, i]`). Default 5 trading days ≈ 1 week.

    Returns
    -------
    transitions : np.ndarray, shape `(n_eval, n_tickers)`, bool
        True where ticker `i` transitioned into a new cluster at date
        `t` and held it for at least `persistence` consecutive days
        starting at `t` (inclusive). Date 0 is always False (no prior
        cluster to compare against).
    """
    n_eval, n_tickers = cluster_ids.shape
    out = np.zeros((n_eval, n_tickers), dtype=bool)
    if n_eval < 2 or persistence < 1:
        return out

    prev = cluster_ids[:-1]                      # (n_eval-1, n_tickers)
    curr = cluster_ids[1:]
    changed = (curr != prev) & (curr >= 0) & (prev >= 0)  # (n_eval-1, n_tickers)

    # Persistence check: for each candidate (t, i), require that
    # cluster_ids[t:t+persistence, i] all equal cluster_ids[t, i] AND
    # are >= 0. Implement via a sliding equality check.
    for offset, t in enumerate(range(1, n_eval)):
        if t + persistence > n_eval:
            break
        window = cluster_ids[t:t + persistence]   # (persistence, n_tickers)
        target = cluster_ids[t]                   # (n_tickers,)
        persistent = np.all(window == target[None, :], axis=0) & (target >= 0)
        out[t] = changed[offset] & persistent

    return out


def trigger_dates_from_transitions(
    transition_mask: np.ndarray,
    prices_index: pd.DatetimeIndex,
    lookback: int,
    *,
    selected_columns: np.ndarray | None = None,
) -> list[pd.Timestamp]:
    """Sorted list of dates where any (selected) transition fires.

    Parameters
    ----------
    transition_mask : np.ndarray, shape `(n_eval, n_tickers)`, bool
        Output of `detect_transition_dates`.
    prices_index : pd.DatetimeIndex
        Full price index. The transition mask is offset by `lookback`
        (matches `cluster_ids[lookback:]` from
        `empirical_excess_divergence_scores(..., return_clusters=True)`).
    lookback : int
        Offset into `prices_index` for the first row of `transition_mask`.
    selected_columns : np.ndarray | None
        Optional `(n_eval, n_tickers)` bool mask restricting which
        ticker transitions count (e.g. only top-N picks). If None,
        any ticker's transition fires the trigger.

    Returns
    -------
    dates : list[pd.Timestamp]
        Sorted unique trigger dates, mapped back into `prices_index`.
    """
    if selected_columns is not None:
        active = transition_mask & selected_columns
    else:
        active = transition_mask
    rows_with_event = np.where(active.any(axis=1))[0]
    eval_index = prices_index[lookback:]
    dates = [eval_index[r] for r in rows_with_event if r < len(eval_index)]
    return sorted(set(dates))
