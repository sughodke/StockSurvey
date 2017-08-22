import datetime
import logging

import intrinio

import gdax
import pandas as pd
import requests

intrinio.client.username = "5aa358835739a7a4cf76b63193451dd3"
intrinio.client.password = "f4e816313de6afff9f9a0ddb923b8827"

# intrinio.client.username, intrinio.client.password = \
#     "2851b6532120373e9b859785f05f3d79:c55ea5a0d6e448eb46d0a78740f82dc3".split(':')

intrinio.client.api_base_url = 'http://172.93.55.89:8081'


def load_data(startdate, enddate, ticker):
    """
    :return: numpy.recarray with fields: date, open, high, low, close, volume, adj_close
    """

    return intrinio.prices(ticker, start_date=startdate, end_date=enddate)


def load_crypto_data(startdate, enddate, identifier):
    payload = {
        'fsym': identifier,
        'tsym': 'USD',
        'toTs': int(enddate.timestamp()),
        'limit': (enddate - startdate).days
    }
    r = requests.get('https://min-api.cryptocompare.com/data/histoday', params=payload)

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