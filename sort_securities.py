import logging
import numpy as np
from scipy import signal
from sklearn.decomposition import TruncatedSVD
from models.security import Security

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

tickers = """
AAPL,ADBE,ADI,ADP,ADSK,AKAM,ALXN,AMAT,AMGN,AMZN,ANET,ATVI,AVGO,BIDU,BIIB,CA,CELG,CHKP,CHTR,COST,CSCO,
CSX,CTAS,CTRP,CTSH,DATA,DISCA,DISCK,DISH,DLTR,EA,EBAY,EXPE,FB,FISV,GLD,GOOG,GOOGL,HAS,HOLX,HSIC,ILMN,INCY,INTC,
INTU,ISRG,JD,KHC,LBTYK,LRCX,LUV,LVNTA,MCHP,MOBL,MSFT,MU,MXIM,MYL,NCLH,NFLX,NTES,NVDA,NXPI,PAYX,PCLN,PYPL,RACE,
REGN,SBAC,SBUX,SHPG,SIRI,SNAP,STX,SWKS,SYMC,TEAM,TSLA,TWLO,TWTR,TXN,VOD,VRSK,VRTX,VSAT,WBA,WDC,WMT,XLNX,XRAY
"""
tickers = tickers.replace('\n', '').split(',')


def transform(inp):
    wavelet = signal.ricker
    widths = np.arange(1, 5, 0.05)

    inp = np.nan_to_num(inp)
    X = signal.cwt(inp, wavelet, widths).T

    svd = TruncatedSVD(n_components=1, n_iter=7)
    transformed = svd.fit_transform(X)

    return transformed.reshape((len(X),))


def value_security(ticker):
    s = Security.load(ticker)

    # Create a view of the data for the timespan we are interested in
    with s.span('daily') as so:
        # r = transform(so.calc.rsi)
        rsi, ma10, _ = so.calc.rsi_values
        r = (rsi + ma10) / 2

    s.save()

    return r[-1]


def sortby_relevance(only_names=False):
    val = map(value_security, tickers)
    d = dict(zip(tickers, val))

    if only_names:
        return sorted(d, key=d.get)

    return [(k, d[k]) for k in sorted(d, key=d.get)]


if __name__ == '__main__':
    from pprint import pprint
    pprint(sortby_relevance(only_names=False))
