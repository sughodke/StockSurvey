import datetime
import logging

from matplotlib import finance
from matplotlib import mlab


def retrieve_stock_data(symbol, d1 = datetime.datetime(2003, 1, 1), d2 = datetime.datetime(2008, 1, 1)):
    """
    DEPRECATED

    Choose a time period reasonably calm (not too long ago so that we get
    high-tech firms, and before the 2008 crash)
    """
    import urllib2
    try:
        from matplotlib.finance import quotes_historical_yahoo
    except ImportError:
        from matplotlib.finance import quotes_historical_yahoo_ohlc as quotes_historical_yahoo

    try:
        return quotes_historical_yahoo(symbol, d1, d2, asobject=True)
    except urllib2.HTTPError as e:
        logging.info(
            '{} was not able to be retrieved.  Err: {}'.format(symbol, e)
        )
        return None


def correct_size(q):
    try:
        return q.shape == (1258,)
    except Exception as e:
        logging.info(
            '{} did not have enough data.  Err: {}'.format(q, e)
        )
        return False


def load_quotes(symbols):
    return filter(correct_size, map(retrieve_stock_data, symbols))


def load_data(startdate, enddate, ticker):
    """
    :return: numpy.recarray with fields: date, open, high, low, close, volume, adj_close
    """
    fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)

    r = mlab.csv2rec(fh)
    fh.close()
    r.sort()

    return r
