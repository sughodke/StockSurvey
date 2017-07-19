"""
rsi : Predict RSI using wavelets on the security's price
"""
import datetime
import logging
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.svm import SVR

import evaluate_securities
from models.security import Security
from util.indicators import relative_strength, bbands, moving_average

sns.set(style="ticks")
opts, args = evaluate_securities.opts, evaluate_securities.args
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class PredictRSI(object):

    def go(self, ticker='VSAT'):
        # startdate=datetime.date(2015, 6, 1),
        r = Security.load(ticker, force_fetch=False, crypto=False).daily

        rsi = relative_strength(r.close, 21)
        ema_rsi = moving_average(rsi, 14, 'exponential')
        mid, top, bot = bbands(r.close, 7)

        draw_price = False
        if draw_price:
            plt.plot(r.index, r.close, label='Price')
            plt.plot(r.index, top, label='BBand T')
            plt.plot(r.index, mid, label='BBand M')
            plt.plot(r.index, bot, label='BBand L')

        plt.plot(r.index, rsi, label='RSI(21)', alpha=0.2)
        plt.plot(r.index - datetime.timedelta(days=14), ema_rsi, label='EMA(RSI)')

        plt.axhline(70, color='darkgoldenrod')
        plt.axhline(30, color='darkgoldenrod')

        self.rsi = rsi
        self.r = r

        self.plot_prediction(1, 'pink')
        self.plot_prediction(3, draw_deviations=True)
        self.plot_prediction(7, 'teal')

        # plt.ylim([0, 100])
        plt.title('{} {}'.format(ticker, 'daily'))
        plt.legend()

        if opts.save_plot:
            plt.savefig('PredictOsc/Predicted RSI {} {}.png'.format(ticker, 'Daily'), dpi=100)
        else:
            plt.show()
        plt.close()

    def plot_prediction(self, forward_days=3, color="darkslategrey", draw_deviations=False):
        """Predict RSI from Price wavelets"""
        rsi = self.rsi
        r = self.r

        length = len(rsi)

        if False:
            X = np.hstack([mid, top, bot])[-length:]

            mask = np.all(np.isnan(X), axis=1)
            X = X[~mask]
            y = np.roll(rsi[~mask], -1)
        else:
            wavelet = signal.ricker
            widths = np.arange(1, 5, 0.1)  # 0.05)

            metric = r.close
            # metric = np.gradient(r.close)
            # metric = (r.close - r.open)/r.open
            X = signal.cwt(metric[-length:], wavelet, widths).T

        y = np.roll(rsi[-length:], -forward_days)
        # y = np.roll(ema_rsi[-length:], -forward_days - 14)

        logging.info('Inputs {} Outputs {}'.format(X.shape, y.shape))

        c = int(0.75 * len(X))
        logging.info('training with {} samples'.format(c))
        X_train, X_test = X[:c], X[c:]
        y_train, y_test = y[:c], y[c:]

        clf = SVR(kernel='rbf')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print('{}-Day Model score: {}'.format(forward_days, score))

        y_pred = clf.predict(X)
        plt.plot(r.index[-length:] + datetime.timedelta(days=forward_days), y_pred,
                 color=color, label='{}D Pred ({:0.2f})'.format(forward_days, score))

        test_train_marker = r.index[-len(X_test)]
        plt.axvline(test_train_marker)
        plt.annotate('  Prediction', (test_train_marker, 30),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     )
        plt.annotate('Train  ', (test_train_marker, 30),
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     )

        if draw_deviations:
            y_pred = y_pred[:c]
            dev = np.array([-1, 0, 1]) * y_pred.std()
            for d in dev:
                plt.axhline(d + y_pred.mean(), color='goldenrod', linestyle='--')


if __name__ == '__main__':
    for t in args:
        PredictRSI().go(t)
