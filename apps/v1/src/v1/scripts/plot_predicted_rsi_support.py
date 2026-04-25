import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, accuracy_score

from indicators import moving_average, relative_strength, bbands
from load_ticker import load_data

fillcolor = "darkgoldenrod"


def go(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', replaydate=datetime.date.today()):
    r = load_data(startdate, enddate, ticker)
    X, y = prepare_data(r)
    delta = enddate - replaydate

    X_train, y_train = X[:-delta.days], y[:-delta.days]
    X_replay, y_replay = X[-delta.days:], y[-delta.days:]

    clf = train(X_train, y_train, ticker)
    replay(clf, X_replay, y_replay, replaydate)


def replay(clf, X, y, replaydate):
    pred_y = clf.predict(X)
    # prob_y = clf.predict_proba(X)

    for idx, truth, prediction in zip(range(len(X)), y, pred_y):
        print('On {date} we expected {desc}; {met}'.format(
            date=replaydate + datetime.timedelta(days=idx),
            desc=describe_quadrant(prediction),
            met='it did' if prediction == truth else 'instead ' + describe_quadrant(truth),
        ))


def rad_to_quadrant(a):
    """
    for i, r in enumerate(np.arange(0, 1, 0.5) + 0.5):
        if a < np.pi * r:
            return i

    # from math import fabs
    # a = fabs(a)
    """
    if a > 0:
        if a < np.pi / 2:
            return 0
        if a < np.pi:
            return 1
    else:
        if a > -np.pi:
            return 2
        if a > -np.pi / 2:
            return 3
    return -1


def describe_quadrant(q):
    try:
        return {
            0: 'RSI to go up, moving toward B-Resistance',
            1: '(unlikely) RSI to go down, moving toward B-Resistance',
            2: 'RSI to go down, moving toward B-Support',
            3: '(unlikely) RSI to go up, moving toward B-Support',
        }[q]
    except KeyError:
        return 'Whoops'


def prepare_data(r):
    prices = r.adj_close
    ma14 = moving_average(prices, 14, type='simple')
    # ma20 = moving_average(prices, 20, type='simple')

    avgBB, upperBB, lowerBB = bbands(prices, 21, 2)
    pct_b = (prices - lowerBB) / (upperBB - lowerBB)
    support = pct_b * 100.

    rsi = relative_strength(prices, 7)
    rsi_prime = np.gradient(rsi)
    # rsi_prime_zeros = np.diff(np.sign(rsi_prime))

    d_rsi = np.gradient(rsi)
    d_support = np.gradient(support)
    arctan = np.arctan2(d_support, d_rsi)

    X = np.vstack([
        avgBB,
        rsi_prime,
        ma14,
        (prices - ma14),
        support,
        rsi,
        relative_strength(prices, 14),
        relative_strength(prices, 21)
    ]).T

    X = pd.DataFrame(X)
    X = X.dropna()

    arctan = arctan[-len(X):]
    y = np.flipud(moving_average(np.flipud(arctan), 2, 'exponential'))
    y = np.array(map(rad_to_quadrant, y))

    print('X: %r  y: %r' % (X.shape, y.shape))

    return X, y


def train(X, y, ticker):
    decision_tree = False

    print('{} has {} samples with {} features'.format(ticker, *X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if decision_tree:
        clf = DecisionTreeClassifier(random_state=0)
    else:
        # X_train = PCA().fit_transform(X_train, y_train)
        # clf = KMeans(init='k-means++')
        # clf = AgglomerativeClustering(n_clusters=6, linkage='ward')
        # clf = MiniBatchKMeans(init='k-means++')
        # clf = label_propagation.LabelSpreading()
        clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Our classifier has scored with %.3f accuracy' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if decision_tree:
        feat = dict(zip(['avgBB', 'rsi_prime', 'ma14', 'prices - ma14', 'bol_pct', 'rsi_7', 'rsi_14', 'rsi_21'],
                        clf.feature_importances_))

        print('With the following contributors')
        for k in sorted(feat, key=feat.get, reverse=True):
            v = feat[k]
            print('%.2f %s' % (v, k))

    return clf


if __name__ == '__main__':
    go(ticker='NVDA', replaydate=datetime.date(2017, 1, 21))

