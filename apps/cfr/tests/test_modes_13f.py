"""Top13FConsensusMode tests — availability mask + EW portfolio over consensus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfr.modes_13f import Top13FConsensusMode


def _consensus_panel():
    """3-quarter consensus panel: each quarter, one ticker is in top-K."""
    return pd.DataFrame(
        {'AAPL': [1.0, 0.0, 1.0],
         'MSFT': [1.0, 1.0, 0.0],
         'GOOG': [0.0, 1.0, 1.0]},
        index=pd.to_datetime(['2020-03-31', '2020-06-30', '2020-09-30']),
    )


def _prices():
    """6 months of bars covering pre-, during-, and post-consensus periods."""
    return pd.DataFrame(
        np.full((150, 4), 100.0, dtype=np.float64),
        index=pd.date_range('2020-01-01', periods=150, freq='B'),
        columns=['AAPL', 'MSFT', 'GOOG', 'OTHER'],
    )


def test_availability_pre_first_quarter_is_false():
    panel = _consensus_panel()
    p = _prices()
    mode = Top13FConsensusMode(name='top13f', consensus_panel=panel,
                               filing_lag_days=45)
    avail = mode.availability(p)
    # First quarter = 2020-03-31, +45d lag = 2020-05-15 → bars before that unavailable
    cutoff = pd.Timestamp('2020-05-15')
    pre = p.index < cutoff
    post = p.index >= cutoff
    assert not avail[pre].any()
    # post-cutoff bars should be available
    assert avail[post].all()


def test_precompute_pre_first_quarter_is_zeros():
    """Bars before the first lagged 13F quarter should have zero weight."""
    panel = _consensus_panel()   # Q1 = 2020-03-31
    p = _prices()
    mode = Top13FConsensusMode(name='top13f', consensus_panel=panel,
                               filing_lag_days=45)
    w = mode.precompute(p)
    avail = mode.availability(p)
    # All unavailable bars should be all-zero weights
    pre_idx = np.where(~avail)[0]
    assert len(pre_idx) > 0   # there should be SOME pre-coverage bars
    assert np.allclose(w[pre_idx], 0.0)


def test_precompute_uses_most_recent_quarter():
    """After Q1's filing lag elapses but before Q2 lands, weights should
    reflect Q1's consensus (AAPL + MSFT)."""
    panel = _consensus_panel()
    p = _prices()
    mode = Top13FConsensusMode(name='top13f', consensus_panel=panel,
                               filing_lag_days=45)
    w = mode.precompute(p)
    # Pick a bar that's certainly between Q1+45d (2020-05-15) and Q2+45d (2020-08-14):
    # Take the first available bar at or after 2020-06-01.
    target = pd.Timestamp('2020-06-15')
    diffs = np.abs((p.index - target).total_seconds())
    bar_idx = int(np.argmin(diffs))
    bar_date = p.index[bar_idx]
    assert pd.Timestamp('2020-05-15') <= bar_date <= pd.Timestamp('2020-08-14')
    row = w[bar_idx]
    aapl_w = row[p.columns.get_loc('AAPL')]
    msft_w = row[p.columns.get_loc('MSFT')]
    other_w = row[p.columns.get_loc('OTHER')]
    np.testing.assert_allclose(aapl_w, 0.5)
    np.testing.assert_allclose(msft_w, 0.5)
    assert other_w == 0.0


def test_universe_filter_drops_off_universe_tickers():
    """Consensus tickers not in the price panel are silently dropped."""
    panel = pd.DataFrame(
        {'AAPL': [1.0], 'OFFUNIVERSE': [1.0], 'MSFT': [1.0]},
        index=pd.to_datetime(['2020-03-31']),
    )
    p = pd.DataFrame(
        np.full((100, 3), 100.0),
        index=pd.date_range('2020-01-01', periods=100, freq='B'),
        columns=['AAPL', 'MSFT', 'OTHER'],
    )
    mode = Top13FConsensusMode(name='top13f', consensus_panel=panel,
                               filing_lag_days=45)
    w = mode.precompute(p)
    # First bar that's available (after Q1+45d=2020-05-15)
    avail = mode.availability(p)
    avail_idxs = np.where(avail)[0]
    assert len(avail_idxs) > 0
    bar_idx = int(avail_idxs[0])
    row = w[bar_idx]
    np.testing.assert_allclose(row[p.columns.get_loc('AAPL')], 0.5)
    np.testing.assert_allclose(row[p.columns.get_loc('MSFT')], 0.5)
    assert row[p.columns.get_loc('OTHER')] == 0.0


def test_empty_consensus_returns_zeros_and_unavailable():
    panel = pd.DataFrame()
    p = _prices()
    mode = Top13FConsensusMode(name='top13f', consensus_panel=panel,
                               filing_lag_days=45)
    w = mode.precompute(p)
    avail = mode.availability(p)
    assert w.shape == (len(p), p.shape[1])
    assert np.allclose(w, 0.0)
    assert not avail.any()
