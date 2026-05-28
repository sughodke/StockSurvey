"""Walk-forward backtest for the leadership-disclosure follower.

Strategy:
    * Each trading day, scan disclosures with ``filed + 1 trading day``
      ≤ today (the disclosure-lag-honest entry rule). Among the surviving
      disclosures, take the top-K most-recent (or top-K by frequency over
      the trailing 90 trading days; see `--filter`) PURCHASES.
    * Equal-weight long basket; hold each name for ``hold_days``
      trading days; carry to delisting if NaN appears.
    * 10 bps round-trip friction charged on every position open + close.

The backtest is fully causal: the position vector at time-t uses only
disclosures with ``filed + 1 ≤ t``, and the close used to compute the
next return is ``closes.iloc[t+1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from follow.data import DisclosurePanel


@dataclass
class BacktestResult:
    daily_returns: pd.Series      # net of friction
    gross_returns: pd.Series      # before friction
    weights: pd.DataFrame         # date × ticker, sum-of-row in [0,1]
    n_holdings: pd.Series         # how many names long each day
    turnover: pd.Series           # sum |Δw| per day
    config: dict
    drop_stats: dict


def _entry_date(filed: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    """filed + 1 trading day. None if past index end."""
    pos = int(np.searchsorted(idx.values, np.datetime64(filed), side='right'))
    if pos >= len(idx):
        return None
    return idx[pos]


def build_position_history(
    panel: DisclosurePanel,
    *,
    hold_days: int = 60,
    top_k: int = 25,
    filter_mode: str = 'recency',  # 'recency' | 'frequency'
    consensus_lookback: int = 1,  # ≥1 leadership member buying same ticker in window
) -> pd.DataFrame:
    """Compute per-day boolean position roster (date × ticker).

    Each "buy event" opens a position on `filed + 1 trading day` and
    closes after `hold_days`. Multiple overlapping events on the same
    ticker simply extend the close date to max(closes). Top-K is
    applied per-day on the currently-open candidates ranked by the
    most recent open-event date (recency) or by count of open events
    in the trailing `consensus_lookback` 90d window (frequency).
    """
    closes = panel.closes
    idx = pd.DatetimeIndex(sorted(closes.index))
    tickers = list(closes.columns)
    ticker_pos = {t: i for i, t in enumerate(tickers)}

    # Per (date, ticker) buy event matrix: 1 where a disclosure-driven
    # open occurs on that trading day.
    events = np.zeros((len(idx), len(tickers)), dtype=np.int32)
    for _, row in panel.disclosures.iterrows():
        if row['ticker'] not in ticker_pos:
            continue
        entry = _entry_date(row['filed'], idx)
        if entry is None:
            continue
        ti = ticker_pos[row['ticker']]
        di = int(idx.get_loc(entry))
        events[di, ti] += 1

    # Open-bar mask: position open if there is at least one buy event
    # in [t - hold_days + 1, t].
    open_mask = np.zeros_like(events, dtype=bool)
    # Cumulative sum over time per ticker; rolling window sum =
    # cum[t] - cum[t - hold_days].
    cum = np.cumsum(events, axis=0)
    pad = np.zeros((1, events.shape[1]), dtype=cum.dtype)
    cum_p = np.vstack([pad, cum])  # cum_p[i] = sum(events[:i])
    for i in range(len(idx)):
        lo = max(0, i - hold_days + 1)
        window_sum = cum_p[i + 1] - cum_p[lo]
        open_mask[i] = window_sum > 0

    # Optional consensus filter (≥N distinct leadership members on the
    # same ticker in trailing window). Skip for v0 — consensus_lookback
    # is reserved for follow-up; documented for future-arc honesty.
    _ = consensus_lookback

    # Score: most-recent-open-date (recency) or rolling 90d count
    # (frequency).
    if filter_mode == 'recency':
        # last open-event index for each (date, ticker) <= date
        last_open = np.full(events.shape, -1, dtype=np.int64)
        cur = np.full(events.shape[1], -1, dtype=np.int64)
        for i in range(len(idx)):
            mask = events[i] > 0
            cur[mask] = i
            last_open[i] = cur
        score = last_open  # higher = more recent
    elif filter_mode == 'frequency':
        # rolling 90 trading-day count
        win = 90
        pad = np.zeros((1, events.shape[1]), dtype=cum.dtype)
        cum_p2 = np.vstack([pad, cum])
        score = np.zeros_like(events, dtype=np.int64)
        for i in range(len(idx)):
            lo = max(0, i - win + 1)
            score[i] = cum_p2[i + 1] - cum_p2[lo]
    else:
        raise ValueError(f'unknown filter_mode={filter_mode!r}')

    # Top-K mask: among currently-open positions, keep top-K by score.
    pos = np.zeros_like(open_mask, dtype=bool)
    for i in range(len(idx)):
        cands = np.where(open_mask[i])[0]
        if len(cands) == 0:
            continue
        s = score[i, cands]
        if len(cands) <= top_k:
            pos[i, cands] = True
        else:
            keep = cands[np.argpartition(-s, top_k)[:top_k]]
            pos[i, keep] = True

    pos_df = pd.DataFrame(pos, index=idx, columns=tickers)
    return pos_df


def run_backtest(
    panel: DisclosurePanel,
    *,
    hold_days: int = 60,
    top_k: int = 25,
    filter_mode: str = 'recency',
    commission_bps: float = 10.0,
) -> BacktestResult:
    """Equal-weight long-only follower backtest with friction."""
    closes = panel.closes.sort_index()
    pos_df = build_position_history(
        panel, hold_days=hold_days, top_k=top_k, filter_mode=filter_mode)
    # Align
    closes = closes.loc[pos_df.index]
    # Equal-weight among positions, NaN-aware (drop a ticker from the
    # day's basket if its close is NaN — delisted / not yet listed).
    valid = ~closes.isna()
    in_basket = pos_df.values & valid.values
    n_holdings = in_basket.sum(axis=1)
    # Weights: 1/n among in-basket names; 0 elsewhere.
    w = np.where(in_basket, 1.0 / np.maximum(n_holdings[:, None], 1), 0.0)
    weights = pd.DataFrame(w, index=closes.index, columns=closes.columns)

    # Daily simple return per ticker (ffill across NaN gaps inside the
    # panel; leading/trailing NaN remains).
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    rets = rets.where(valid, 0.0)  # NaN → 0 only where in-basket
    # Gross portfolio return = sum_i w_{i,t-1} * r_{i,t}. We hold w_{t-1}
    # into day t and earn r_t — shift weights forward by 1.
    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag.values * rets.values).sum(axis=1)
    gross = pd.Series(gross, index=closes.index, name='gross_ret').fillna(0.0)

    # Turnover & friction. Charge on |Δw| (round-trip ~bps).
    dw = weights.diff().abs().fillna(weights.abs()).values
    turnover = pd.Series(dw.sum(axis=1), index=closes.index, name='turnover')
    friction = turnover * (commission_bps / 1e4)
    net = (gross - friction).rename('net_ret')

    return BacktestResult(
        daily_returns=net,
        gross_returns=gross,
        weights=weights,
        n_holdings=pd.Series(n_holdings, index=closes.index, name='n_holdings'),
        turnover=turnover,
        config={
            'hold_days': hold_days,
            'top_k': top_k,
            'filter_mode': filter_mode,
            'commission_bps': commission_bps,
        },
        drop_stats=panel.drop_stats,
    )
