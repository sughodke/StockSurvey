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
    """
    name: str

    def precompute(self, prices: pd.DataFrame) -> np.ndarray: ...


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
        - `'momentum'`  : higher trailing log return = higher score
        - `'reversal'`  : lower trailing log return = higher score
        - `'low_vol'`   : lower trailing vol = higher score
        - `'high_vol'`  : higher trailing vol = higher score

    `top_k` is capped at the count of liquid names if it exceeds.
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

    def precompute(self, prices: pd.DataFrame) -> np.ndarray:
        """Build the `(T, n_actions, N)` weight tensor.

        Each mode is precomputed once over the full panel; the gross
        scalar then scales each entry. Cash actions short-circuit to
        zeros. The result is dense and float64.
        """
        T, N = prices.shape
        mode_weights = {m.name: m.precompute(prices) for m in self.modes}
        out = np.zeros((T, self.n_actions, N), dtype=np.float64)
        for i, a in enumerate(self.actions):
            if a.gross == 0.0:
                continue  # already zeros
            base = mode_weights[a.mode_name]
            out[:, i, :] = base * a.gross
        return out


def default_phase1_menu(*, top_k: int = 20) -> ActionMenu:
    """The minimal universe-agnostic Phase 1 menu.

    Six modes × two non-zero gross levels + one canonical cash → 13
    actions:

      cash, ew@g0.5, ew@g1, ew@g2,
      mom@g0.5, mom@g1, mom@g2,
      rev@g0.5, rev@g1,
      low_vol@g1, high_vol@g1

    (Where `mom` is top-`top_k` 21d momentum, `rev` is top-`top_k`
    5d reversal, etc.)

    Gross levels are asymmetric across modes intentionally — extreme
    gross 2.0 only makes sense for the diversified EW / momentum
    arms; piling 2× into a high-vol bet is rarely something a sane
    meta-allocator should pick.
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


__all__ = [
    'Action', 'ActionMenu', 'BaseMode', 'CashMode', 'EqualWeightMode',
    'TopKMode', 'default_phase1_menu',
]
