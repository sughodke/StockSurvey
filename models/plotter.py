import datetime
import logging
import os
import io
import numpy as np
from math import log10, fabs

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pytrends.request import TrendReq

from util import cwd
from util.indicators import moving_average, fibonacci_retracement, interesting_fib

MARKER_SIZE = 30

fillcolor = 'darkgoldenrod'
textsize = 9
IMGDIR = os.path.join(cwd, 'Output/')


class PlotBaseMixin(object):
    def __init__(self, d, ticker, calc, decide, eval, cadence='daily'):
        self.dataset = d
        self.ticker = ticker
        self.calc = calc
        self.decide = decide
        self.eval = eval
        self.cadence = cadence

    def plot_data(self, save=False, web=False):
        r = self.dataset
        ticker = self.ticker

        plt.rc('axes', grid=True)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
        # plt.rc('marker', edgecolor='black')
        plt.rc('lines', linewidth=1)

        left, width = 0.1, 0.8
        rect1 = [left, 0.6, width, 0.3]
        rect2 = [left, 0.2, width, 0.4]
        rect3 = [left, 0.1, width, 0.1]

        fig = plt.figure(facecolor='white')
        axescolor = '#f6f6f6'  # the axes background color

        ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
        ax2 = fig.add_axes(rect2, facecolor=axescolor, sharex=ax1)
        ax3 = fig.add_axes(rect3, facecolor=axescolor, sharex=ax1)

        ax1.set_title('%s %s' % (ticker, self.cadence))

        # plot the relative strength indicator
        self.plot_indicator(ax1, ax2, r)

        # plot the buysell points
        clean_buy, clean_sell, _ = self.decide.clean_buysellvol
        self.plot_buysell(ax2, clean_buy, clean_sell, r)

        # plot the price and volume data
        prices = r.adj_close.values
        self.plot_price(ax2, r)
        self.plot_price_ma(ax2, r.index, prices)

        self.plot_retracement(ax2, r.index, prices)

        last = r.tail(1)
        s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
            datetime.date.today().strftime('%d-%b-%Y'),
            last.open, last.high,
            last.low, last.close,
            0.,  # TODO: last.volume*1e-6,
            last.close - last.open)
        ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

        self.plot_purse(ax3, r, clean_sell)

        # turn off upper axis tick labels, rotate the lower ones, etc
        for ax in ax1, ax2, ax3:
            if ax != ax3:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
            else:
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')

            ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        class MyLocator(mticker.MaxNLocator):
            def __init__(self, *args, **kwargs):
                mticker.MaxNLocator.__init__(self, *args, **kwargs)

            def __call__(self, *args, **kwargs):
                return mticker.MaxNLocator.__call__(self, *args, **kwargs)

        ax2.yaxis.set_major_locator(MyLocator(5, prune='both'))
        ax3.yaxis.set_major_locator(MyLocator(5, prune='both'))

        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        if save:
            important_events = {
                'buy': last.index[0] - r.index[clean_buy][-1],
                'sell': last.index[0] - r.index[clean_sell][-1]
            }
            important_events = {k: i.days for k, i in important_events.items()}
            closest_event = min(important_events, key=important_events.get)

            fig.set_size_inches(18.5, 10.5)
            filename = IMGDIR + '%s-%s%d-%s%.0f.png' % (
                ticker, closest_event, important_events[closest_event],
                '' if self.eval.performance > 0 else 'n',
                fabs(log10(fabs(self.eval.performance))))
            fig.savefig(filename, dpi=100)
            logging.info('Plot saved {}'.format(filename))
        elif web:
            # TODO move this to a new plotter class
            fig.set_dpi(100)
            fig.set_size_inches(7, 6.5)  # (10.5, 5.5)
            fig.set_tight_layout(True)

            output = io.StringIO()
            fig.savefig(output, format='svg')
            plt.close()
            return output
        else:
            plt.show(block=True)
        plt.close()

    def plot_indicator(self, ax1, ax2, r):
        raise NotImplemented()

    def plot_purse(self, ax3, r, clean_sell):
        val = self.eval.val
        cumval = np.cumsum(val)
        ax3.plot(r.index[clean_sell], 100. * cumval / r.open[-1],
                 color='darkslategrey', label='cumulative', lw=2)
        ax3.bar(r.index[clean_sell], 100. * val / r.open[-1],
                width=4,  # TODO: width is absolute px size, cannot set as pct of chart
                color=cm.jet(-np.sign(val)),
                alpha=0.7, label='instantaneous')
        ax3.axhline()
        ax3.text(0.025, 0.95, 'Purse (pct of today value)', va='top',
                 transform=ax3.transAxes, fontsize=textsize)

    def plot_retracement(self, ax2, date, prices):
        fib_start, fib_end, fib_retracements = fibonacci_retracement(prices)
        for label, fib in zip(interesting_fib, fib_retracements):
            ax2.step([date[fib_start], date[-1]], [fib] * 2, alpha=0.6)
            ax2.text(s='%.1f' % label, x=date[-1], y=fib, alpha=0.6,
                     fontsize=8, horizontalalignment='right')

    def plot_price_ma(self, ax2, date, prices):
        ma20 = moving_average(prices, 20, type='simple')
        ax2.plot(date, ma20, color='blue', lw=2, label='MA (20)')

    def plot_price(self, ax2, r):
        dx = r.adj_close - r.close
        high = r.high + dx
        low = r.low + dx

        # Find bear/bull days
        prices = r.adj_close.values
        deltas = np.zeros_like(prices)
        deltas[1:] = np.diff(prices)
        up = deltas > 0

        plotargs = {'color': '#69B85D', 'linewidths': 1.}
        ax2.vlines(r.index[up], low[up], high[up], label='_nolegend_', **plotargs)
        ax2.scatter(r.index[up], r.open[up], marker=0, s=20, **plotargs)
        ax2.scatter(r.index[up], r.close[up], marker=1, s=20, **plotargs)
        plotargs = {'color': '#B84D4F', 'linewidths': 1.}
        ax2.vlines(r.index[~up], low[~up], high[~up], label='_nolegend_', **plotargs)
        ax2.scatter(r.index[~up], r.open[~up], marker=0, s=20, **plotargs)
        ax2.scatter(r.index[~up], r.close[~up], marker=1, s=20, **plotargs)

    def plot_buysell(self, ax2, clean_buy, clean_sell, r):
        ax2.scatter(r.index[clean_buy], r.low[clean_buy], c='lightgreen', marker='^', edgecolors='black', s=20)
        ax2.scatter(r.index[clean_sell], r.high[clean_sell], c='lightpink', marker='v', edgecolors='black', s=20)


class PlotMixin(PlotBaseMixin):

    def plot_indicator(self, ax1, ax2, r):
        rsi, rsi_ma10, rsi_prime = self.calc.rsi_values
        rsi_prime_zeros = self.calc.rsi_prime_zeros
        rsi_ma_cross = self.calc.rsi_ma_cross

        self.plot_gtrends(ax2, r.index)
        self.plot_rsi(ax1, r.index, rsi)
        self.plot_rsi_direction_change(ax1, r.index, rsi_prime_zeros, rsi[rsi_prime_zeros])
        self.plot_rsi_ma(ax1, r.index, rsi, rsi_ma10, rsi_ma_cross)

    def plot_gtrends(self, ax2, date):
        # TODO y-value on the cursor is the twin axis, not the orig
        ax2t = ax2.twinx()
        start, end = min(date), max(date)

        pytrend = TrendReq()

        # TODO Add hour?
        timeframe = ' '.join((start.isoformat()[:10], end.isoformat()[:10]))

        pytrend.build_payload(
            kw_list=[
                '{ticker} price'.format(ticker=self.ticker),
                '{ticker} to USD'.format(ticker=self.ticker),
                '{ticker} USD'.format(ticker=self.ticker),
                '{ticker} BTC'.format(ticker=self.ticker),
                '{ticker} to BTC'.format(ticker=self.ticker),
            ],
            timeframe=timeframe,
        )

        interest_over_time_df = pytrend.interest_over_time()
        volume = interest_over_time_df.sum(numeric_only=True, axis=1)

        vmax = volume.max()
        ax2t.fill_between(interest_over_time_df.index, volume, 0, label='Search Volume',
                          facecolor=fillcolor, edgecolor=fillcolor)
        ax2t.set_ylim(0, 5 * vmax)
        ax2t.set_yticks([])

    def plot_rsi_ma(self, ax1, date, rsi, rsi_ma10, rsi_ma_cross):
        ax1.plot(date, rsi_ma10, color='blue', lw=2)
        ax1.scatter(date[rsi_ma_cross], rsi[rsi_ma_cross], c='teal', marker='s', s=MARKER_SIZE,
                    edgecolors='black')

    def plot_rsi_direction_change(self, ax1, date, rsi_prime_zeros, zeros_y):
        ax1.scatter(
            date[rsi_prime_zeros],
            zeros_y,
            c=zeros_y / 100.,
            s=MARKER_SIZE,
            edgecolors='black'
        )

    def plot_rsi(self, ax1, date, rsi):
        ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
        ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([30, 70])
        ax1.text(0.025, 0.95, 'RSI (7)', va='top', transform=ax1.transAxes, fontsize=textsize)

        ax1.plot(date, rsi, color=fillcolor, linewidth=1)
        ax1.axhline(70, color=fillcolor)
        ax1.axhline(50, color=fillcolor, linestyle='--')
        ax1.axhline(30, color=fillcolor)
        ax1.fill_between(date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
        ax1.fill_between(date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)


class MACDPlotMixin(PlotBaseMixin):

    def plot_indicator(self, ax1, ax2, r):
        slow, fast, macd = self.calc.macd_values
        signal = self.calc.macd_signal

        macd_zero_cross = self.calc.macd_zero_cross
        macd_signal_cross = self.calc.macd_signal_cross

        self.plot_macd(ax1, r.index, macd, signal)
        self.plot_signal_cross(ax1, r.index[macd_signal_cross], macd[macd_signal_cross])

    def plot_macd(self, ax1, date, macd, signal):
        ax1.bar(date, macd - signal, color='darkslategrey', alpha=0.7, label='instantaneous')

        ax1.plot(date, macd, color=fillcolor, linewidth=1)
        ax1.plot(date, signal, color='blue', linewidth=1)

        ax1.text(0.025, 0.95, 'MACD (26, 12, 10)', va='top', transform=ax1.transAxes, fontsize=textsize)

    def plot_signal_cross(self, ax1, date, y_val):
        ax1.scatter(date, y_val, c=y_val / 100., s=MARKER_SIZE, edgecolors='black')
