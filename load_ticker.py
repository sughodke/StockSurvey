import datetime
try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
import urllib2
import logging

logging.basicConfig(level=logging.DEBUG)

# Choose a time period reasonably calm (not too long ago so that we get
# high-tech firms, and before the 2008 crash)
d1 = datetime.datetime(2003, 1, 1)
d2 = datetime.datetime(2008, 1, 1)


def mapper(symbol):
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
    return filter(correct_size, map(mapper, symbols))
