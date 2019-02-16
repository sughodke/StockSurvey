import datetime
import logging

import pandas as pd
import pandas_datareader as pdr
import requests

SECONDS_IN_HOUR = 60 * 60
SECONDS_IN_DAY = 60 * 60 * 24


def load_data(startdate, enddate, ticker):
    """
    :return: NDFrame with columns: date, open, high, low, close, volume, adj_close
    """

    df = pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
    df.rename(str.lower, axis='columns', inplace=True)
    df.rename(index=str, columns={'adj close': 'adj_close'}, inplace=True)
    df.set_index(pd.to_datetime(df.index), inplace=True)
    return df


def load_crypto_data(startdate, enddate, identifier, period='day'):
    payload = {
        'fsym': identifier,
        'tsym': 'USD',
        'toTs': int(enddate.timestamp()),
        'limit': (enddate - startdate).total_seconds()
    }

    if period == 'day':
        payload['limit'] //= SECONDS_IN_DAY
        r = requests.get('https://min-api.cryptocompare.com/data/histoday', params=payload)
    else:
        payload['limit'] //= SECONDS_IN_HOUR
        r = requests.get('https://min-api.cryptocompare.com/data/histohour', params=payload)

    df = pd.DataFrame(r.json()['Data'])
    df.index = pd.to_datetime(df.time, unit='s')
    df.drop('time', axis=1, inplace=True)

    df['adj_close'] = df['close']
    df['volume'] = df['volumeto'] + df['volumefrom']
    return df


def load_crypto_data_v1(startdate, enddate, identifier, granularity=str(60*60*2)):
    """
    DEPRECATED

    Get historical crypto-coin prices

     - str(60*60*2) = 2h granularity
     - str(60*60*8) = 8h granularity

    Returns:
        Dataset as a Pandas DataFrame
    """
    import gdax
    quotes = gdax.PublicClient().get_product_historic_rates(identifier, start=startdate, end=enddate,
                                                            granularity=granularity)
    df = pd.DataFrame(quotes, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.index = pd.to_datetime(df.time, unit='s')
    df.drop('time', axis=1, inplace=True)

    df['adj_close'] = df['close']
    return df


def load_forex_data(startdate, enddate, currency):
    """
    **Partially complete**

    Get historic prices for currency to USD from startdate to enddate

    Returns:
        Dataset as a Pandas DataFrame
    """
    df = pd.DataFrame.from_csv('/Users/SidGhodke/Documents/Code/Sid/StockSurvey/DEXINUS.csv')

    for k in ['low', 'high', 'open', 'close', 'adj_close', 'volume']:
        df[k] = pd.to_numeric(df['DEXINUS'], errors='coerce')

    df.drop('DEXINUS', axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)

    return df[startdate:enddate]
