"""Point-in-time universe queries over an ss_stream event store.

The store is two parquet files (see `ingest.py`); this module is the
read-side. The two operations a backtest actually needs are:

  * `active_at(date)` — set of tickers whose listing_date <= date <=
    last_seen_date. The frame the trainer should iterate at each
    rebalance.
  * `bars_between(start, end, tickers)` — long-form OHLCV slice.
    Cheap to pivot to a wide panel via `panel(...)` when downstream
    code expects the matrix shape.

`Universe` lazily loads bars on first access; small workloads that
only need `instruments` (e.g. for liquidity filtering or universe
sizing) skip the bar read entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class Universe:
    """Read-side wrapper over an `ingest_stooq` output directory."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        ins_path = self.root / 'instruments.parquet'
        if not ins_path.exists():
            raise FileNotFoundError(
                f'no instruments.parquet at {ins_path}; run `ss-stream ingest` first')
        self._instruments = pd.read_parquet(ins_path)
        # parquet round-trips datetime64[ns] cleanly but be defensive.
        for col in ('listing_date', 'last_seen_date'):
            self._instruments[col] = pd.to_datetime(self._instruments[col])
        self._bars: pd.DataFrame | None = None

    @property
    def instruments(self) -> pd.DataFrame:
        return self._instruments

    def _load_bars(self) -> pd.DataFrame:
        if self._bars is None:
            bars_path = self.root / 'bars' / 'data.parquet'
            if not bars_path.exists():
                raise FileNotFoundError(
                    f'no bars at {bars_path}; ingest produced metadata only')
            df = pd.read_parquet(bars_path)
            df['date'] = pd.to_datetime(df['date'])
            self._bars = df
        return self._bars

    def active_at(self, when) -> set[str]:
        """Tickers whose `listing_date <= when <= last_seen_date`.

        `when` accepts anything `pd.Timestamp` accepts (ISO string,
        date, datetime, Timestamp). The bound is inclusive on both
        sides — a ticker is "active" on its listing date and on its
        last-seen date.
        """
        when = pd.Timestamp(when)
        ins = self._instruments
        m = (ins['listing_date'] <= when) & (ins['last_seen_date'] >= when)
        return set(ins.loc[m, 'ticker'])

    def bars_between(
        self,
        start,
        end,
        tickers: list[str] | set[str] | None = None,
    ) -> pd.DataFrame:
        """Long-form OHLCV slice on `[start, end]` (inclusive).

        Optional `tickers` filter; when None, all tickers in the
        store are returned for the date window.
        """
        bars = self._load_bars()
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        m = (bars['date'] >= s) & (bars['date'] <= e)
        if tickers is not None:
            m &= bars['ticker'].isin(tickers)
        return bars.loc[m].copy()

    def panel(
        self,
        start,
        end,
        field: str = 'close',
        tickers: list[str] | set[str] | None = None,
    ) -> pd.DataFrame:
        """Wide panel: rows=date, columns=ticker, values=`field`.

        Only the tickers active in `[start, end]` will have non-NaN
        values; the rest of the panel is NaN-padded by the pivot.
        """
        b = self.bars_between(start, end, tickers)
        return b.pivot(index='date', columns='ticker', values=field).sort_index()

    def __len__(self) -> int:
        return len(self._instruments)

    def __repr__(self) -> str:
        ins = self._instruments
        return (
            f'<Universe root={self.root} instruments={len(ins):,} '
            f'range={ins.listing_date.min().date()}'
            f'->{ins.last_seen_date.max().date()}>')
