"""State + action feature builders.

State features are derived from public macro/market data at the
walk-forward `val_start` date — no use of `val_end` or any post-val info,
to ensure point-in-time correctness when Φ is invoked at deployment.

Action features are one-hot encodings of the action_key per app. Cross-app
training shares hidden weights but uses an app_id one-hot to let Φ learn
app-specific offsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from critic.dataset import Triple


REPO_ROOT = Path(__file__).resolve().parents[4]
MACRO_CACHE = REPO_ROOT / ".macro-cache"


def _load_vix_series() -> pd.Series:
    """Return FRED VIXCLS daily series."""
    import ss_macro

    s = ss_macro.load_fred_series("VIXCLS", cache_dir=str(MACRO_CACHE))
    s = pd.Series(s.values, index=pd.to_datetime(s.index), name="vix").sort_index()
    s = s.dropna()
    return s


def _load_spy_proxy() -> pd.Series:
    """Use SP500 (FRED) as the SPY trailing-return proxy.

    FRED's SP500 series is free and goes back to ~1993; SPY itself would
    require a different loader. For our purposes, SPY trailing-252d return
    is well-approximated by SP500 trailing-252d return.
    """
    import ss_macro

    s = ss_macro.load_fred_series("SP500", cache_dir=str(MACRO_CACHE))
    s = pd.Series(s.values, index=pd.to_datetime(s.index), name="sp500").sort_index()
    s = s.dropna()
    return s


def _as_of(series: pd.Series, date: pd.Timestamp) -> float | None:
    """Return last available value at-or-before `date`. None if no prior data."""
    sub = series.loc[:date]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1])


def _trailing_change(
    series: pd.Series, date: pd.Timestamp, days: int
) -> float | None:
    """Return value_at(date) − value_at(date − `days`)."""
    end = _as_of(series, date)
    start = _as_of(series, date - pd.Timedelta(days=days))
    if end is None or start is None:
        return None
    return end - start


def _trailing_log_return(
    series: pd.Series, date: pd.Timestamp, days: int
) -> float | None:
    """Return log(value_at(date) / value_at(date − `days`))."""
    end = _as_of(series, date)
    start = _as_of(series, date - pd.Timedelta(days=days))
    if end is None or start is None or end <= 0 or start <= 0:
        return None
    return float(np.log(end / start))


def build_state_features(triples: Sequence[Triple]) -> tuple[np.ndarray, list[str]]:
    """Build state-feature matrix `(N, n_state_features)`.

    Features (all point-in-time at `val_start`):
    - vix_level
    - vix_6m_change_pts (VIX − VIX-6mo-ago)
    - vix_1m_change_pts (VIX − VIX-1mo-ago)
    - spy_trailing_252d_log_ret
    - spy_trailing_63d_log_ret
    - app_id one-hot (factor / gate / pairs / vol)
    """
    vix = _load_vix_series()
    spy = _load_spy_proxy()

    apps = ["factor", "gate", "pairs", "vol"]

    rows = []
    names = [
        "vix_level",
        "vix_6m_change_pts",
        "vix_1m_change_pts",
        "spy_trailing_252d_log_ret",
        "spy_trailing_63d_log_ret",
    ] + [f"app_{a}" for a in apps]

    for t in triples:
        if not t.val_start:
            # Fall back to a per-app default if val_start was missing in the
            # source JSON. Factor's mixture-sweep entries are the main case.
            row = [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ] + [float(a == t.app) for a in apps]
            rows.append(row)
            continue

        date = pd.Timestamp(t.val_start)
        vix_level = _as_of(vix, date)
        vix_6m = _trailing_change(vix, date, 180)
        vix_1m = _trailing_change(vix, date, 30)
        spy_252 = _trailing_log_return(spy, date, 252)
        spy_63 = _trailing_log_return(spy, date, 63)

        row = [
            vix_level if vix_level is not None else np.nan,
            vix_6m if vix_6m is not None else np.nan,
            vix_1m if vix_1m is not None else np.nan,
            spy_252 if spy_252 is not None else np.nan,
            spy_63 if spy_63 is not None else np.nan,
        ] + [float(a == t.app) for a in apps]
        rows.append(row)

    X = np.asarray(rows, dtype=np.float64)

    # Median-impute any NaNs feature-by-feature (small-data — drop is too costly).
    for col in range(X.shape[1]):
        nan_mask = ~np.isfinite(X[:, col])
        if nan_mask.any():
            med = np.nanmedian(X[~nan_mask, col]) if (~nan_mask).any() else 0.0
            X[nan_mask, col] = med

    return X, names


def build_action_features(
    triples: Sequence[Triple], action_vocab: list[str]
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode the global action vocabulary.

    The vocab is a flat list across all apps (e.g. ['factor:fixed:h5',
    'factor:mixture:a0.0', 'gate:gated', ...]); we deliberately do NOT
    refactor it per-app because Φ's hidden layer can learn app-conditional
    action effects via the app_id state one-hot.
    """
    idx_of = {k: i for i, k in enumerate(action_vocab)}
    A = np.zeros((len(triples), len(action_vocab)), dtype=np.float64)
    for i, t in enumerate(triples):
        A[i, idx_of[t.action_key]] = 1.0
    return A, action_vocab


def standardize_continuous(
    X: np.ndarray, n_continuous: int, ref: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Z-score the first `n_continuous` columns; leave one-hots alone.

    If `ref` provided, use its statistics (for held-out fold normalization).
    """
    if ref is None:
        mu = X[:, :n_continuous].mean(axis=0)
        sd = X[:, :n_continuous].std(axis=0) + 1e-8
    else:
        mu = ref[:, :n_continuous].mean(axis=0)
        sd = ref[:, :n_continuous].std(axis=0) + 1e-8

    out = X.copy()
    out[:, :n_continuous] = (out[:, :n_continuous] - mu) / sd
    return out, {"mu": mu, "sd": sd, "n_continuous": n_continuous}
