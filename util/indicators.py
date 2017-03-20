import numpy as np
import pandas as pd

interesting_fib = [0., 14.6, 23.6, 38.2, 50., 61.8, 100.]  # , 161.8


def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi


def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow


def fibonacci_retracement(prices, n=90):
    """
    compute the support and resistance lines for a given n-day period

    http://www.swing-trade-stocks.com/support-and-resistance.html
    http://www.investopedia.com/terms/f/fibonacciextensions.asp
    https://www.tradestation.com/education/labs/analysis-concepts/introducing-the-fibonacci-retracement-channel
    """
    p = prices[-n:]
    min_idx, max_idx = np.argmin(p), np.argmax(p)

    # chronological times
    t1, t2 = min(min_idx, max_idx), max(min_idx, max_idx)
    p1, p2 = p[t1], p[t2]
    diff = p2 - p1

    prices_fib = [p1 + diff * i / 100. for i in interesting_fib]

    offset = len(prices) - 90
    return offset + t1, offset + t2, prices_fib


def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    df = pd.DataFrame(price)
    r = df.rolling(length)
    ave = r.mean()
    sd = r.std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)
