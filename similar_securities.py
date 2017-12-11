"""
similar_securities : Use kMeans on wavelet to group similar securities
"""
import logging
import numpy as np
import seaborn as sns

from pprint import pprint
from joblib import Parallel, delayed

from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from finance_ndx import NDX_constituents
from models.security import Security

sns.set(style="ticks")
# opts, args = evaluate_securities.opts, evaluate_securities.args
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def compute_wavelet(ticker):
    s = Security.load(ticker, force_fetch=False, crypto=False)
    with s.span('daily') as so:
        r = so.dataset

        wavelet = signal.ricker
        widths = np.arange(1, 5, 0.1)

        # metric = r.close
        # metric = np.gradient(r.close)
        # metric = (r.close - r.open)/r.open
        metric = so.calc.rsi

        X = signal.cwt(metric, wavelet, widths).T

        svd = TruncatedSVD(n_components=1, n_iter=7)
        transformed = svd.fit_transform(X)

        return transformed.reshape((len(X),))


if __name__ == '__main__':
    NDX_constituents = list(NDX_constituents)
    # wavelets = list(map(compute_wavelet, NDX_constituents))
    wavelets = Parallel(n_jobs=2)(delayed(compute_wavelet)(ticker)
                                  for ticker in NDX_constituents)
    clf = KMeans(n_jobs=2, n_clusters=16)
    cl_id = clf.fit(wavelets).predict(wavelets)

    for ticker, id in zip(NDX_constituents, cl_id.tolist()):
        print('{} - {}'.format(ticker, id))
