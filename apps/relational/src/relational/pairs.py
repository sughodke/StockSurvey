"""Pair / spread construction transforms over existing top-N weight
DataFrames.

Two market-neutralizing wrappers, both producing weight matrices whose
rows sum to ~0 (gross = 2, net = 0):

  * `market_neutral_weights(long_weights)` — long the existing top-N
    basket, short the entire active universe equal-weight. The natural
    centroid hedge for idea C (`farthest`) — the universe-equal short
    *is* the cross-sectional centroid this scorer measures distance
    from. Useful for any scorer as a generic market-beta hedge.
  * `rank_spread_weights(top_weights, bot_weights)` — long top-N,
    short bot-N. Pure cross-sectional rank portfolio. Stronger than
    market-neutral if the scorer is informative on *both* tails;
    weaker / noisier if the scorer's tail isn't.

Both leave the original `top_weights` index and column structure
intact, so they drop into the existing `bt.algos.WeighTarget +
Rebalance` pipeline without further plumbing. `integer_positions=False`
in the bt.Backtest call is required for the negative weights to flow
through the rebalance solver — same caveat as the long-only path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_neutral_weights(
    long_weights: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long top-N (passed in) plus a short hedge of -1/M on every
    active ticker, where `M` = count of finite-price tickers that day.

    If `prices` is None, treats every column as always-active and uses
    `M = n_tickers`. Phase-2's 21 mega-caps trade throughout the panel
    so this is fine; broader universes with delisted/late-listed names
    should pass `prices` for proper masking.
    """
    n_dates, n_tickers = long_weights.shape
    long_arr = long_weights.fillna(0).values
    short_arr = np.zeros_like(long_arr, dtype=np.float64)
    if prices is None:
        short_arr[:] = -1.0 / n_tickers
    else:
        valid = np.isfinite(
            prices.reindex(index=long_weights.index,
                           columns=long_weights.columns).values)
        m = valid.sum(axis=1, keepdims=True).astype(np.float64)
        m_safe = np.where(m > 0, m, 1.0)
        short_arr = np.where(valid, -1.0 / m_safe, 0.0)
    combined = long_arr + short_arr
    return pd.DataFrame(combined,
                        index=long_weights.index,
                        columns=long_weights.columns)


def rank_spread_weights(
    top_weights: pd.DataFrame,
    bot_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Long-short combination: `top_weights - bot_weights`.

    Both inputs are expected to be standard top-N matrices summing to
    ~1 per row. Output sums to ~0 per row with gross ≈ 2.
    """
    return top_weights.fillna(0) - bot_weights.fillna(0)


def cluster_pair_weights(
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    prices: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Idea-A natural pair: per empirical cluster, long the highest-
    excess-divergence stock and short the cluster aggregate (= equal-
    weight cluster constituents). Each active cluster contributes
    equally to the basket.

    Faithful to the scorer's definition:
        score[i] = stock_divergence[i] - cluster_aggregate_divergence[c(i)]
    The high-score names are bets *against the cluster aggregate*, so
    hedging with the cluster aggregate isolates the idiosyncratic move
    rather than universe-wide market beta.

    `scores` and `cluster_ids` are both `(n_eval, n_tickers)`; output is
    a `(n_eval, n_tickers)` DataFrame with `prices.index[lookback:]` as
    its index. Per row: long ≈ +1, short ≈ −1, gross ≈ 2, net ≈ 0.

    Single-member clusters (no peers to hedge against) are skipped that
    day; if all clusters are degenerate the row is all zeros.
    """
    n_eval, n_tickers = scores.shape
    out = np.zeros((n_eval, n_tickers), dtype=np.float64)
    for t in range(n_eval):
        score_t = scores[t]
        cl_t = cluster_ids[t]
        active: list[tuple[np.ndarray, np.ndarray]] = []
        for c in np.unique(cl_t):
            if c < 0:
                continue
            members = np.where(cl_t == c)[0]
            valid_score = members[np.isfinite(score_t[members])]
            if len(valid_score) < 2:
                continue
            active.append((members, valid_score))
        if not active:
            continue
        w_each = 1.0 / len(active)
        for members, valid_score in active:
            winner = valid_score[np.argmax(score_t[valid_score])]
            cluster_size = len(members)
            out[t, winner] += w_each
            out[t, members] -= w_each / cluster_size
    return pd.DataFrame(
        out,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
