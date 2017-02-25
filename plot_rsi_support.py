import datetime

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from indicators import moving_average, relative_strength, bbands
from load_ticker import load_data

fillcolor = "darkgoldenrod"


def go(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', save=True):
    r = load_data(startdate, enddate, ticker)
    clf = train(r, ticker, save)
    # replay(enddate, datetime.date.today(), ticker, clf)


# def replay(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', clf):
#     r = load_data(startdate, enddate, ticker)

def train(r, ticker, save=False):
    prices = r.adj_close
    ma14 = moving_average(prices, 14, type='simple')
    # ma20 = moving_average(prices, 20, type='simple')

    avgBB, upperBB, lowerBB = bbands(prices, 21, 2)
    pct_b = (prices - lowerBB) / (upperBB - lowerBB)
    support = pct_b * 100  #
    support = support[20:]  # remove empties

    rsi = relative_strength(prices, 7)
    rsi_prime = np.gradient(rsi)
    # rsi_prime_zeros = np.diff(np.sign(rsi_prime))

    rsi = rsi[20:]  # remove empties
    # TODO Use gradient instead of diff?
    d_rsi = np.diff(rsi)
    d_support = np.diff(support)
    # d_rsi = np.gradient(rsi)
    # d_support = np.gradient(support)
    arctan = np.arctan2(d_support, d_rsi)

    X = np.vstack([
        avgBB[-len(rsi):],
        rsi_prime[-len(rsi):],
        ma14[-len(rsi):],
        (prices - ma14)[-len(rsi):],
        support,
        rsi,
        relative_strength(prices, 14)[-len(rsi):],
        relative_strength(prices, 21)[-len(rsi):],
    ]).T

    y = np.flipud(moving_average(np.flipud(arctan), 2, 'exponential'))

    # average of RSI, Support not diff
    # y = np.flipud(moving_average(np.flipud(np.roll(rsi, -1)), 4, 'exponential'))

    def rad_to_quadrant(a):
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
    y = np.array([-1] + map(rad_to_quadrant, y))
    # y = np.array(map(rad_to_quadrant, y))

    print('{} has {} samples with {} features'.format(ticker, *X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    print('Our classifier has scored with %.3f accuracy' % clf.score(X_test, y_test))

    feat = dict(zip(['avgBB', 'rsi_prime', 'ma14', 'prices - ma14', 'bol_pct', 'rsi_7', 'rsi_14', 'rsi_21'],
                    clf.feature_importances_))

    print('With the following contributors')
    for k in sorted(feat, key=feat.get, reverse=True):
        v = feat[k]
        print('%.2f %s' % (v, k))

    return clf


if __name__ == '__main__':
    go(ticker='NVDA', enddate=datetime.date(2017, 2, 21), save=False)

