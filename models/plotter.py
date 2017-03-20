import datetime

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from util.indicators import moving_average, relative_strength, fibonacci_retracement, interesting_fib
from math import log10, fabs

fillcolor = 'darkgoldenrod'
textsize = 9


class PlotMixin(object):
    def plot_data(self, save=False):
        r = self.dataset
        ticker = self.ticker

        plt.rc('axes', grid=True)
        plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

        left, width = 0.1, 0.8
        rect1 = [left, 0.6, width, 0.3]
        rect2 = [left, 0.2, width, 0.4]
        rect3 = [left, 0.1, width, 0.1]

        fig = plt.figure(facecolor='white')
        axescolor = '#f6f6f6'  # the axes background color

        ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
        ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
        ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)

        ax1.set_title('%s daily' % ticker)

        # plot the relative strength indicator
        prices = r.adj_close
        rsi, rsi_ma10, rsi_prime = self.rsi_values
        rsi_prime_zeros = self.rsi_prime_zeros
        rsi_ma_cross = self.rsi_ma_cross

        self.plot_rsi(ax1, r.date, rsi)
        self.plot_rsi_direction_change(ax1, r.date, rsi_prime_zeros, rsi[rsi_prime_zeros])
        self.plot_rsi_ma(ax1, r.date, rsi, rsi_ma10, rsi_ma_cross)

        clean_buy, clean_sell, vol_buy = self.clean_buysellvol
        self.plot_buysell(ax2, clean_buy, clean_sell, r)

        # plot the price and volume data
        dx = r.adj_close - r.close
        self.plot_price(ax2, r.high + dx, r.low + dx, prices, r)
        self.plot_price_ma(ax2, r.date, prices)

        self.plot_retracement(ax2, r.date, prices)

        last = r[-1]
        s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
            datetime.date.today().strftime('%d-%b-%Y'),
            last.open, last.high,
            last.low, last.close,
            last.volume*1e-6,
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

        important_events = {
            'buy': datetime.date.today() - r.date[clean_buy][-1].astype('O'),
            'sell': datetime.date.today() - r.date[clean_sell][-1].astype('O')
        }
        important_events = {k: i.days for k, i in important_events.iteritems()}
        closest_event = min(important_events, key=important_events.get)

        if save:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig('%s-%s%d-%s%.0f.png' %
                        (ticker,
                         closest_event,
                         important_events[closest_event],
                         '' if performance > 0 else 'n',
                         fabs(log10(fabs(performance)))),
                        dpi=100)
        else:
            # plt.savefig('%s-6m.svg' % ticker, format='svg')
            plt.show()

    def plot_purse(self, ax3, r, clean_sell):
        cumval = np.cumsum(self.val)
        ax3.plot(r.date[clean_sell], 100. * cumval / r.open[-1], color='darkslategrey', label='cumulative', lw=2)
        ax3.bar(r.date[clean_sell], 100. * self.val / r.open[-1],
                width=4, color=cm.jet(-np.sign(self.val)),
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

    def plot_price(self, ax2, high, low, prices, r):
        deltas = np.zeros_like(prices)
        deltas[1:] = np.diff(prices)
        up = deltas > 0
        plotargs = {'color': '#69B85D', 'linewidths': 1.}
        ax2.vlines(r.date[up], low[up], high[up], label='_nolegend_', **plotargs)
        ax2.scatter(r.date[up], r.open[up], marker=0, **plotargs)
        ax2.scatter(r.date[up], r.close[up], marker=1, **plotargs)
        plotargs = {'color': '#B84D4F', 'linewidths': 1.}
        ax2.vlines(r.date[~up], low[~up], high[~up], label='_nolegend_', **plotargs)
        ax2.scatter(r.date[~up], r.open[~up], marker=0, **plotargs)
        ax2.scatter(r.date[~up], r.close[~up], marker=1, **plotargs)

    def plot_rsi_ma(self, ax1, date, rsi, rsi_ma10, rsi_ma_cross):
        ax1.plot(date, rsi_ma10, color='blue', lw=2)
        ax1.scatter(date[rsi_ma_cross], rsi[rsi_ma_cross], c='teal', marker='s', s=50)

    def plot_buysell(self, ax2, clean_buy, clean_sell, r):
        ax2.scatter(r.date[clean_buy], r.low[clean_buy], c='lightgreen', marker='^')
        ax2.scatter(r.date[clean_sell], r.high[clean_sell], c='lightpink', marker='v')

    def plot_rsi_direction_change(self, ax1, date, rsi_prime_zeros, zeros_y):
        ax1.scatter(
            date[rsi_prime_zeros],
            zeros_y,
            c=zeros_y / 100.,
            s=50)

    def plot_rsi(self, ax1, date, rsi):
        ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
        ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([30, 70])
        ax1.text(0.025, 0.95, 'RSI (7)', va='top', transform=ax1.transAxes, fontsize=textsize)

        ax1.plot(date, rsi, color=fillcolor)
        ax1.axhline(70, color=fillcolor)
        ax1.axhline(30, color=fillcolor)
        ax1.fill_between(date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
        ax1.fill_between(date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)
