"""Action menu — discrete (mode, gross) actions that resolve to portfolio weights.

The CFR-over-existing-scorers framing: each *action* is a (strategy
mode, gross level) tuple. At each rebalance the meta-allocator picks
one (or a mixture). The continuous portfolio-construction work is
delegated to the modes themselves; CFR is the meta-layer.

Phase 1 ships with universe-agnostic modes (cash, EW, top-K
momentum / reversal / low-vol / high-vol) that work on any price
panel without requiring a saved checkpoint. This lets the algorithm
be validated independently of any single scorer's quality. Modes
that wrap existing relational / factor checkpoints come in Phase 2+
once the imitation-pretrain step needs concrete expert-policy
priors.

Modes are precomputed across the full price panel so the
walk-forward training loop's hot path is a `(t, action) → weights`
lookup rather than a re-scoring per bar — at ~6,000 rebal bars ×
~10 actions × 100s of bars of trailing history per call, the
amortization is the difference between a 30-second walk-forward
and a 30-minute one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


def _safe_div(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(num, default, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _liquid_mask(prices: pd.DataFrame, *, min_lookback: int = 21) -> np.ndarray:
    """Per-bar boolean mask of tickers with enough history to compute a
    `min_lookback`-bar trailing window without leading-NaN poisoning.

    `prices` is `(T, N)`. Returns `(T, N)` bool. A ticker is considered
    *liquid at bar t* iff every bar in `[t - min_lookback + 1, t]` has a
    non-NaN close. This is the strict version — softer alternatives
    (allow up to k NaNs, ffill internally) live behind the caller's
    panel preparation.
    """
    if prices.empty:
        return np.zeros((0, 0), dtype=bool)
    p = prices.values
    valid = ~np.isnan(p)
    if min_lookback <= 1:
        return valid
    cum = np.cumsum(valid.astype(np.int64), axis=0)
    cum_pad = np.concatenate([np.zeros((1, cum.shape[1]), dtype=cum.dtype), cum], axis=0)
    trailing_valid = cum - np.roll(cum_pad, min_lookback, axis=0)[:-1]
    trailing_valid[:min_lookback - 1] = 0
    return trailing_valid >= min_lookback


def _normalize_top_k(scores: np.ndarray, mask: np.ndarray, top_k: int) -> np.ndarray:
    """Equal-weight portfolio over the top-K names by score per row.

    `scores` and `mask` are `(T, N)`. NaN / masked-out positions don't
    participate. Output is `(T, N)` weights summing to 1 per row when
    at least one liquid name exists, else zeros.
    """
    T, N = scores.shape
    weights = np.zeros((T, N), dtype=np.float64)
    s = scores.copy()
    s[~mask] = -np.inf
    s[np.isnan(s)] = -np.inf
    if top_k >= N:
        # No selection — equal-weight every liquid name.
        liquid_count = mask.sum(axis=1)
        for t in range(T):
            if liquid_count[t] > 0:
                weights[t, mask[t]] = 1.0 / liquid_count[t]
        return weights
    order = np.argsort(-s, axis=1)
    for t in range(T):
        liquid_t = int(mask[t].sum())
        if liquid_t == 0:
            continue
        k = min(top_k, liquid_t)
        picks = order[t, :k]
        # Filter out -inf positions (they were either NaN or masked
        # and `argsort` happily ranks them — drop them).
        picks = picks[np.isfinite(s[t, picks])]
        if len(picks) == 0:
            continue
        weights[t, picks] = 1.0 / len(picks)
    return weights


class BaseMode(Protocol):
    """Strategy mode contract.

    Implementations precompute per-bar portfolio weights over the full
    price panel. Output shape is `(T, N)`; row `t` is the *intended*
    weight vector at bar `t` (summing to 1 over the chosen names, or
    to 0 if no name passes liquidity).

    Optional `availability(prices) -> (T,) bool` declares which bars
    the mode has *real signal* on (vs phantom-cash). Modes that don't
    implement it are treated as always-available (the default for
    deterministic price-derived modes). Modes backed by external data
    that doesn't span the full panel (e.g., 13F panel starting in
    2013, options surfaces with limited coverage) should report
    `availability` so the CFR walk-forward can mask them out of
    sampling/regret/mixing in unavailable bars — preventing the
    Phase 2b cash-equivalent contamination bug.
    """
    name: str

    def precompute(self, prices: pd.DataFrame) -> np.ndarray: ...

    # Optional — implementers MAY define `availability(prices) -> ndarray`.
    # ActionMenu.precompute checks via getattr.


@dataclass(frozen=True)
class CashMode:
    """All-zero weights — exposure off."""
    name: str = 'cash'

    def precompute(self, prices: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(prices), prices.shape[1]), dtype=np.float64)


@dataclass(frozen=True)
class EqualWeightMode:
    """Equal-weight over liquid universe at each bar."""
    name: str = 'ew'
    min_lookback: int = 21

    def precompute(self, prices: pd.DataFrame) -> np.ndarray:
        mask = _liquid_mask(prices, min_lookback=self.min_lookback)
        counts = mask.sum(axis=1).astype(np.float64)
        T, N = mask.shape
        weights = np.zeros((T, N), dtype=np.float64)
        nonempty = counts > 0
        weights[nonempty] = mask[nonempty] / counts[nonempty, None]
        return weights


def _trailing_log_return(prices: pd.DataFrame, window: int) -> np.ndarray:
    """Per-bar trailing log-return over `window` bars. `(T, N)`."""
    p = prices.values
    out = np.full_like(p, np.nan, dtype=np.float64)
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    if window <= 0:
        return out
    out[window:] = log_p[window:] - log_p[:-window]
    return out


def _trailing_log_return_skip(
    prices: pd.DataFrame, window: int, skip: int,
) -> np.ndarray:
    """`log(p_{t-skip} / p_{t-window-skip})` per bar — trailing return
    over `window` bars ending `skip` bars ago.

    Used for **12-1 momentum** (Jegadeesh & Titman 1993): the trailing
    12-month return *excluding* the most recent 1 month, which avoids
    short-term reversal contamination of the medium-term momentum
    signal. With `window=231, skip=21` this is the canonical 12-1
    formulation.
    """
    p = prices.values
    out = np.full_like(p, np.nan, dtype=np.float64)
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    if window <= 0 or skip < 0:
        return out
    end = log_p.shape[0]
    if window + skip >= end:
        return out
    out[window + skip:] = log_p[skip:end - window] - log_p[: end - window - skip]
    # ^ at row t (>= window+skip), value = log_p[t-skip] - log_p[t-window-skip]
    return out


def _trailing_sharpe(prices: pd.DataFrame, window: int) -> np.ndarray:
    """Per-bar trailing daily-return Sharpe over `window` bars (no
    annualization — pure mean / std). `(T, N)`. NaN where < `window+1`
    history.

    Looped because `window` is small (~252) and the panel is
    `(T, N)` ≈ `(6500, 312)`; total cost ~6 seconds. Acceptable for
    one-shot precompute.
    """
    p = prices.values
    T, N = p.shape
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    log_ret = np.diff(log_p, axis=0, prepend=np.nan)
    out = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(window, T):
        sample = log_ret[t - window + 1:t + 1]
        valid_count = (~np.isnan(sample)).sum(axis=0)
        valid_cols = valid_count >= max(20, window // 4)
        if not valid_cols.any():
            continue
        mu = np.full(N, np.nan, dtype=np.float64)
        sd = np.full(N, np.nan, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            mu[valid_cols] = np.nanmean(sample[:, valid_cols], axis=0)
            sd[valid_cols] = np.nanstd(sample[:, valid_cols], axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[t] = np.where(sd > 1e-10, mu / sd, np.nan)
    return out


def _trailing_trend_strength(prices: pd.DataFrame, window: int) -> np.ndarray:
    """Per-bar trailing `(cumulative log return) / (max drawdown)` over
    `window` bars. `(T, N)`. NaN for non-positive denominators (any
    name that didn't drop in the trailing window has effectively
    infinite trend strength — we mask those to NaN so they don't
    dominate ranking on a single-bar artifact).

    Captures **smooth uptrends** — names with consistent gains and
    small drawdowns score higher than equally-returning names with
    larger intra-period dips. The classic "calmar ratio at window
    granularity."
    """
    p = prices.values
    T, N = p.shape
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    out = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(window, T):
        sample = log_p[t - window + 1:t + 1]
        valid_cols = (~np.isnan(sample)).sum(axis=0) >= max(20, window // 4)
        if not valid_cols.any():
            continue
        # Cumulative log return = sample[-1] - sample[0] (per column)
        cum_ret = sample[-1] - sample[0]
        # Max drawdown in log space = max running peak - current.
        # Restrict to valid columns to avoid `nanmax` on all-NaN slices.
        sample_valid = sample[:, valid_cols]
        running_peak = np.maximum.accumulate(
            np.where(np.isnan(sample_valid), -np.inf, sample_valid), axis=0)
        dd = running_peak - sample_valid
        with np.errstate(invalid='ignore'):
            max_dd_valid = np.nanmax(dd, axis=0)
        max_dd = np.full(N, np.nan, dtype=np.float64)
        max_dd[valid_cols] = max_dd_valid
        score = np.full(N, np.nan, dtype=np.float64)
        # Mask: needs valid history AND a positive max_dd (otherwise undefined)
        ok = valid_cols & (max_dd > 1e-6) & np.isfinite(cum_ret)
        score[ok] = cum_ret[ok] / max_dd[ok]
        out[t] = score
    return out


def _trailing_vol(prices: pd.DataFrame, window: int) -> np.ndarray:
    """Per-bar trailing realized vol over `window` bars (stdev of log
    returns). `(T, N)`. NaN where history < `window + 1`.

    Loop kept simple — `window` is small (~21) and the panel is
    `(T, N)` ≈ (6000, 300), so the inner cost is 1.8M scalar ops per
    column, well under a second.
    """
    p = prices.values
    T, N = p.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    log_p = np.log(p, where=(p > 0), out=np.full_like(p, np.nan, dtype=np.float64))
    log_ret = np.diff(log_p, axis=0, prepend=np.nan)
    for t in range(window, T):
        sample = log_ret[t - window + 1:t + 1]
        # Require at least 2 non-NaN values per column before computing
        # stdev so nanstd doesn't trigger a "Degrees of freedom <= 0"
        # warning on all-NaN slices.
        valid_count = (~np.isnan(sample)).sum(axis=0)
        valid_cols = valid_count >= max(2, window // 2)
        if not valid_cols.any():
            continue
        sd = np.full(N, np.nan, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            sd[valid_cols] = np.nanstd(sample[:, valid_cols], axis=0, ddof=1)
        out[t] = sd
    return out


@dataclass(frozen=True)
class TopKMode:
    """Equal-weight top-K names by a trailing scoring function.

    `score_kind`:
        - `'momentum'`     : higher trailing log return = higher score
        - `'reversal'`     : lower trailing log return = higher score
        - `'low_vol'`      : lower trailing vol = higher score
        - `'high_vol'`     : higher trailing vol = higher score
        - `'mom_12_1'`     : Jegadeesh-Titman 12-1 momentum (12mo return,
                             excluding most recent 1mo); `score_window` ignored
                             (hard-coded 231 + skip 21)
        - `'sharpe_top'`   : trailing daily-return Sharpe (mean/std);
                             `score_window` is the trailing window (default 252)
        - `'trend_str'`    : trailing (cumulative log return) / (max DD);
                             `score_window` is the trailing window (default 252)

    `top_k` is capped at the count of liquid names if it exceeds.
    For long-window scorers (`mom_12_1`, `sharpe_top`, `trend_str`),
    `min_lookback` should be set to at least the score's window so
    the liquidity filter doesn't admit names that are NaN on the
    score axis.
    """
    name: str
    score_kind: str
    score_window: int = 21
    top_k: int = 20
    min_lookback: int = 21

    def precompute(self, prices: pd.DataFrame) -> np.ndarray:
        if self.score_kind in ('momentum', 'reversal'):
            ret = _trailing_log_return(prices, self.score_window)
            if self.score_kind == 'reversal':
                ret = -ret
            scores = ret
        elif self.score_kind in ('low_vol', 'high_vol'):
            vol = _trailing_vol(prices, self.score_window)
            scores = -vol if self.score_kind == 'low_vol' else vol
        elif self.score_kind == 'mom_12_1':
            scores = _trailing_log_return_skip(prices, window=231, skip=21)
        elif self.score_kind == 'sharpe_top':
            scores = _trailing_sharpe(prices, self.score_window)
        elif self.score_kind == 'trend_str':
            scores = _trailing_trend_strength(prices, self.score_window)
        else:
            raise ValueError(f'unknown score_kind {self.score_kind!r}')
        mask = _liquid_mask(prices, min_lookback=self.min_lookback)
        return _normalize_top_k(scores, mask, self.top_k)


@dataclass(frozen=True)
class Action:
    """A `(mode, gross)` action.

    Gross 0 collapses every mode to cash (zero weights). The menu
    dedups those to a single canonical cash action so the regret
    table doesn't have N redundant entries.
    """
    mode_name: str
    gross: float

    def key(self) -> str:
        return 'cash' if self.gross == 0.0 else f'{self.mode_name}@g{self.gross:g}'


@dataclass
class ActionMenu:
    """Discrete action set + (T, n_actions, N) precomputed weight panel.

    Construction takes a list of `BaseMode` instances and a list of
    `gross` levels; the Cartesian product is the raw action set, but
    gross=0 entries collapse to a single canonical `cash` action.
    The order is stable across calls — `actions[i]` is the action
    indexed by integer `i` everywhere downstream (regret table,
    policy vector, etc).
    """
    modes: list[BaseMode]
    gross_levels: tuple[float, ...] = (0.0, 1.0)
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen_keys: set[str] = set()
        actions: list[Action] = []
        for mode in self.modes:
            for g in self.gross_levels:
                a = Action(mode_name=mode.name, gross=float(g))
                if a.key() in seen_keys:
                    continue
                seen_keys.add(a.key())
                actions.append(a)
        # Ensure a canonical cash action exists exactly once.
        if not any(a.gross == 0.0 for a in actions):
            actions.append(Action(mode_name='cash', gross=0.0))
        # Stable order: cash first, then by (mode insertion order, gross).
        cash_actions = [a for a in actions if a.gross == 0.0]
        nonzero = [a for a in actions if a.gross != 0.0]
        nonzero.sort(key=lambda a: ([m.name for m in self.modes].index(a.mode_name)
                                    if a.mode_name in [m.name for m in self.modes] else 999,
                                    a.gross))
        object.__setattr__(self, 'actions', cash_actions + nonzero)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def action_keys(self) -> list[str]:
        return [a.key() for a in self.actions]

    def precompute(self, prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build the `(T, n_actions, N)` weight tensor + `(T, n_actions)`
        availability mask.

        Each mode is precomputed once over the full panel; the gross
        scalar then scales each entry. Cash actions short-circuit to
        zeros and are always-available.

        Availability of action `(mode, gross>0)` at bar t is the
        underlying mode's availability at bar t (defaults to True if
        the mode doesn't define `availability`). Cash is always
        available. The walk-forward driver uses availability to mask
        unavailable actions out of sampling, regret update, and
        average-policy mixing — so a mode with limited temporal
        coverage (13F since 2013, IV since 2019, etc.) doesn't
        contaminate the regret table for bars where it's all-cash.
        """
        T, N = prices.shape
        mode_weights: dict[str, np.ndarray] = {}
        mode_avail: dict[str, np.ndarray] = {}
        for m in self.modes:
            mode_weights[m.name] = m.precompute(prices)
            avail_fn = getattr(m, 'availability', None)
            if avail_fn is None:
                mode_avail[m.name] = np.ones(T, dtype=bool)
            else:
                a = avail_fn(prices)
                mode_avail[m.name] = (np.ones(T, dtype=bool) if a is None
                                       else np.asarray(a, dtype=bool))

        out = np.zeros((T, self.n_actions, N), dtype=np.float64)
        avail_out = np.ones((T, self.n_actions), dtype=bool)
        for i, a in enumerate(self.actions):
            if a.gross == 0.0:
                continue  # cash: zero weights, always available (default True)
            base = mode_weights[a.mode_name]
            out[:, i, :] = base * a.gross
            avail_out[:, i] = mode_avail[a.mode_name]
        return out, avail_out


def default_phase1_menu(*, top_k: int = 20) -> ActionMenu:
    """The minimal universe-agnostic Phase 1 menu (16 actions).

    5 modes × {0, 0.5, 1.0, 2.0} gross, deduped to a single canonical
    cash → 16 actions: cash + ew/mom/rev/lowv/highv × {0.5, 1.0, 2.0}.

    The Phase 1 finding ([cfr-phase1](../findings/cfr-phase1.md))
    showed CFR over this menu ties naive uniform mix because the
    short-window factor exposures (5-21d) are individually too close
    to alpha-zero to reward concentration. Phase 2a swaps in
    documented-alpha modes (12-1 momentum, 12-month low-vol, etc.)
    via `default_phase2a_menu`.
    """
    # No explicit CashMode — gross=0.0 across any of the other modes
    # already produces canonical cash, and `ActionMenu.__post_init__`
    # dedups them to a single `cash` entry.
    modes = [
        EqualWeightMode(name='ew'),
        TopKMode(name='mom', score_kind='momentum', score_window=21, top_k=top_k),
        TopKMode(name='rev', score_kind='reversal', score_window=5, top_k=top_k),
        TopKMode(name='lowv', score_kind='low_vol', score_window=21, top_k=top_k),
        TopKMode(name='highv', score_kind='high_vol', score_window=21, top_k=top_k),
    ]
    return ActionMenu(
        modes=modes,
        gross_levels=(0.0, 0.5, 1.0, 2.0),
    )


def default_phase2a_menu(*, top_k: int = 20) -> ActionMenu:
    """Phase 2a menu — Phase 1 + documented-alpha modes.

    Adds four medium-horizon modes with academically-documented alpha
    on US equities:

    - **`mom121`** — Jegadeesh-Titman 12-1 momentum (top-K by 12-month
      return excluding most recent month). Multi-decade documented
      alpha; the canonical "medium-term momentum" effect.
    - **`lowv252`** — 12-month low-vol (top-K by *lowest* trailing
      252d realized vol). The "low-vol anomaly" — multi-decade
      documented alpha.
    - **`shtop`** — top-K by trailing 252d Sharpe (mean / std of
      daily log returns). Combines momentum and risk-aware ranking.
    - **`trend`** — top-K by trailing 252d (cumulative log return) /
      (max drawdown). Smooth-uptrend filter; the classical
      window-Calmar ranking.

    Total: 9 modes × 4 gross levels deduped = **28 actions** (cash +
    9 modes × 3 nonzero gross). Tabular CFR over 9 infosets × 28
    actions = 252 table entries; still well within converging on
    ~6,000 train rebals.

    These modes are *individually expected to be alpha-positive* in
    some regimes (otherwise the academic literature on them would not
    exist). The Phase 1 finding's binding-constraint hypothesis says
    CFR's regret matching should be able to concentrate on whichever
    of these is regime-active, lifting it materially over the
    Phase 1 menu's uniform-mix limit.

    The pre-registered Phase 2a cut: **CFR with this menu ≥ Phase 1
    CFR + 0.10 mean Sharpe** AND **CFR > naive uniform mix of this
    menu by ≥ +0.10 mean Sharpe** → confirms menu enrichment is the
    right lever; proceed to Phase 2b (real 13F integration).
    """
    modes = [
        EqualWeightMode(name='ew'),
        TopKMode(name='mom', score_kind='momentum', score_window=21, top_k=top_k),
        TopKMode(name='rev', score_kind='reversal', score_window=5, top_k=top_k),
        TopKMode(name='lowv', score_kind='low_vol', score_window=21, top_k=top_k),
        TopKMode(name='highv', score_kind='high_vol', score_window=21, top_k=top_k),
        # New documented-alpha modes — long-window scorers, so the
        # liquidity filter has to admit names with ≥252 history (the
        # Phase 1 default min_lookback=21 would let names with < 252d
        # of history score NaN and be ranked at the bottom).
        TopKMode(name='mom121', score_kind='mom_12_1',
                 score_window=252, top_k=top_k, min_lookback=252),
        TopKMode(name='lowv252', score_kind='low_vol',
                 score_window=252, top_k=top_k, min_lookback=252),
        TopKMode(name='shtop', score_kind='sharpe_top',
                 score_window=252, top_k=top_k, min_lookback=252),
        TopKMode(name='trend', score_kind='trend_str',
                 score_window=252, top_k=top_k, min_lookback=252),
    ]
    return ActionMenu(
        modes=modes,
        gross_levels=(0.0, 0.5, 1.0, 2.0),
    )


__all__ = [
    'Action', 'ActionMenu', 'BaseMode', 'CashMode', 'EqualWeightMode',
    'TopKMode', 'default_phase1_menu', 'default_phase2a_menu',
]
