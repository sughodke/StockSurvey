import datetime
import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from indicators import relative_strength, bbands
from load_ticker import load_data


r = load_data(startdate=datetime.date(2016, 6, 1),
              enddate=datetime.date.today(),
              ticker='GLD')
rsi = relative_strength(r.close)
mid, top, bot = bbands(r.close, 7)

fig = plt.figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)

plt.plot(r.date, r.close)
plt.plot(r.date, top)
plt.plot(r.date, mid)
plt.plot(r.date, bot)
plt.plot(r.date, rsi)

plot_prediction = True
if plot_prediction:
    """Predict RSI from BBands"""
    length = len(rsi)

    X = np.hstack([mid, top, bot])[-length:]
    mask = np.all(np.isnan(X), axis=1)
    X = X[~mask]

    y = np.roll(rsi, -1)[-X.shape[0]:]

    print(X.shape)
    print(y.shape)

    c = 0.75 * len(X)
    X_train, X_test = X[:c], X[c:]
    y_train, y_test = y[:c], y[c:]

    clf = SVR(kernel='rbf')
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    plt.plot(r.date[-len(y_test):], clf.predict(X_test), color="darkslategrey")
plt.show()
