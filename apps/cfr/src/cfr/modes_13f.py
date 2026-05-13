"""13F-consensus action mode.

Wraps a `(period, ticker) → in_consensus` panel from `ss_edgar` into
a CFR `BaseMode`. At each price-panel bar, looks up the most recent
13F quarter (using a 45-day filing lag — filings dribble in for 45
days after quarter-end, so we don't see Q1 holdings until ~mid-May)
and EW-portfolios over the flagged consensus names.

Bars before the first 13F quarter (with lag applied) are flat
(`weight=0` over all tickers — the canonical cash entry per the
ActionMenu dedup rule).

Universe handling: the price panel typically has more tickers than
the consensus panel. We restrict the consensus picks to tickers
that appear in BOTH (the panel's column set), and renormalize. If
no consensus tickers are present in the price panel, the bar is
cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Top13FConsensusMode:
    """EW portfolio over the top-K most-broadly-held names by 13F filers.

    `consensus_panel` is a DataFrame indexed by quarter-end
    (`pd.Timestamp`), columns are tickers, values are 1.0 if the
    ticker is in that quarter's top-K consensus, else 0.0. Built by
    `ss_edgar.build_consensus_top_k(count_panel, top_k=K)`.

    `filing_lag_days` (default 45) accounts for the SEC's 45-day
    filing window — at any bar `t`, only holdings reported for
    `period <= (t - 45 days)` are visible. Without this lag,
    walk-forward leaks future information.
    """
    name: str
    consensus_panel: pd.DataFrame
    filing_lag_days: int = 45

    def precompute(self, prices: pd.DataFrame) -> np.ndarray:
        T, N = prices.shape
        out = np.zeros((T, N), dtype=np.float64)
        if self.consensus_panel.empty:
            return out

        # Restrict consensus columns to tickers also in the price panel.
        common_cols = [c for c in self.consensus_panel.columns
                       if c in prices.columns]
        if not common_cols:
            return out
        panel = self.consensus_panel[common_cols].copy()
        # Map ticker -> column index in price panel
        ticker_to_idx = {c: i for i, c in enumerate(prices.columns)}
        col_indices = np.array([ticker_to_idx[c] for c in common_cols])

        # For each bar t, find the most recent panel period <= t - lag.
        bar_dates = prices.index
        lagged_dates = bar_dates - pd.Timedelta(days=self.filing_lag_days)
        # searchsorted on the panel index
        panel_periods = panel.index
        # Last panel period <= lagged_date is searchsorted-right - 1
        idx = np.searchsorted(panel_periods.values, lagged_dates.values, side='right') - 1
        for t in range(T):
            i = int(idx[t])
            if i < 0:
                continue   # before first lagged 13F quarter
            row = panel.iloc[i].values   # (n_common,)
            picks = row > 0
            n_picks = int(picks.sum())
            if n_picks == 0:
                continue
            w_per_pick = 1.0 / n_picks
            out[t, col_indices[picks]] = w_per_pick
        return out


__all__ = ['Top13FConsensusMode']
