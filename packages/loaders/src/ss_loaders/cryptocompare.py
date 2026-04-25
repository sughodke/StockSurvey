"""CryptoCompare REST client for historical crypto OHLCV."""

from __future__ import annotations

import datetime

import pandas as pd
import requests

SECONDS_IN_HOUR = 60 * 60
SECONDS_IN_DAY = 60 * 60 * 24


def load_cryptocompare(
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    identifier: str,
    period: str = 'day',
) -> pd.DataFrame:
    """Fetch historical OHLCV for one crypto ticker against USD.

    `period` is 'day' (histoday) or anything else (histohour). Returns a
    DataFrame indexed by datetime with columns including open, high, low,
    close, adj_close (= close), volume (= volumeto + volumefrom).
    """
    payload = {
        'fsym': identifier,
        'tsym': 'USD',
        'toTs': int(enddate.timestamp()),
        'limit': (enddate - startdate).total_seconds(),
    }

    if period == 'day':
        payload['limit'] //= SECONDS_IN_DAY
        r = requests.get('https://min-api.cryptocompare.com/data/histoday',
                         params=payload)
    else:
        payload['limit'] //= SECONDS_IN_HOUR
        r = requests.get('https://min-api.cryptocompare.com/data/histohour',
                         params=payload)

    df = pd.DataFrame(r.json()['Data'])
    df.index = pd.to_datetime(df.time, unit='s')
    df.drop('time', axis=1, inplace=True)

    df['adj_close'] = df['close']
    df['volume'] = df['volumeto'] + df['volumefrom']
    return df
