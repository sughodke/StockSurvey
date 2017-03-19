import datetime

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from util.indicators import moving_average, relative_strength, fibonacci_retracement, interesting_fib
from util.load_ticker import load_data
from math import log10, fabs

today = datetime.date.today()


def go(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', save=True):
    # today = enddate
    r = load_data(startdate, enddate, ticker)
    plot_data(r, ticker, save)


def plot_data(r, ticker, save):
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

    textsize = 9
    left, width = 0.1, 0.8
    rect1 = [left, 0.6, width, 0.3]
    rect2 = [left, 0.2, width, 0.4]
    rect3 = [left, 0.1, width, 0.1]

    fig = plt.figure(facecolor='white')
    axescolor = '#f6f6f6'  # the axes background color

    ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
    ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
    ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)

    # plot the relative strength indicator
    prices = r.adj_close
    rsi = relative_strength(prices, 7)
    rsi_prime = np.gradient(rsi)

    fillcolor = 'darkgoldenrod'

    ax1.plot(r.date, rsi, color=fillcolor)
    ax1.axhline(70, color=fillcolor)
    ax1.axhline(30, color=fillcolor)
    ax1.fill_between(r.date, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
    ax1.fill_between(r.date, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)

    # http://stackoverflow.com/a/3843124
    rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]

    zeros_y = rsi[rsi_prime_zeros]
    ax1.scatter(
        r.date[rsi_prime_zeros],
        zeros_y,
        c=zeros_y/100.,
        s=50)

    rsi_ma10 = moving_average(rsi, 10, type='exponential')
    ax1.plot(r.date, rsi_ma10, color='blue', lw=2)

    rsi_ma_cross = np.where(np.diff(np.sign(rsi - rsi_ma10)))[0]
    ax1.scatter(r.date[rsi_ma_cross], rsi[rsi_ma_cross], c='teal', marker='s', s=50)

    buy_idx, sell_idx = [], []
    k = 0
    for i in rsi_ma_cross:
        j = rsi_prime_zeros[k]
        while r.date[j] < r.date[i]:
            k += 1
            try:
                j = rsi_prime_zeros[k]
            except IndexError:
                j = rsi_prime_zeros[-1]
                break

        print('%s %s %s' % (r.date[j], 'buy' if rsi[j] < rsi[i] else 'sell', r.open[j]))

        if rsi[j] < rsi[i]:
            buy_idx.append(j)
        else:
            sell_idx.append(j)

    clean_buy, clean_sell = [], []
    j = 0
    for i in xrange(len(buy_idx)):
        idx_i = buy_idx[i]
        try:
            idx_j = sell_idx[j]
        except IndexError:
            j = -1
            idx_j = sell_idx[j]
        di, dj = r.date[idx_i], r.date[idx_j]
        if di < dj:
            continue
        clean_buy.append(idx_i)
        while di > dj:
            j += 1
            try:
                idx_j = sell_idx[j]
                dj = r.date[idx_j]
            except IndexError:
                idx_j = sell_idx[j-1]
                break
        clean_sell.append(idx_j)

    ax2.scatter(r.date[clean_buy], r.low[clean_buy], c='lightgreen', marker='^')
    ax2.scatter(r.date[clean_sell], r.high[clean_sell], c='lightpink', marker='v')

    val_buy = r.open[clean_buy]
    val_sell = r.open[clean_sell]
    vol_buy = 10 - rsi[clean_buy] // 10  # confidence that we are buying at the correct time
    val = vol_buy * (val_sell - val_buy)
    sumval = np.sum(val)
    performance = 100.*sumval/r.open[-1]
    print('With %d trades we stand to make %f (%f%%).' % (len(val_buy), sumval, performance))

    ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
    ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([30, 70])
    ax1.text(0.025, 0.95, 'RSI (7)', va='top', transform=ax1.transAxes, fontsize=textsize)
    ax1.set_title('%s daily' % ticker)

    # plot the price and volume data
    dx = r.adj_close - r.close
    low = r.low + dx
    high = r.high + dx

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

    ma20 = moving_average(prices, 20, type='simple')
    linema20, = ax2.plot(r.date, ma20, color='blue', lw=2, label='MA (20)')

    fib_start, fib_end, fib_retracements = fibonacci_retracement(prices)
    for label, fib in zip(interesting_fib, fib_retracements):
        ax2.step([r.date[fib_start], r.date[-1]], [fib] * 2, alpha=0.6)
        ax2.text(s='%.1f' % label, x=r.date[-1], y=fib, alpha=0.6,
                 fontsize=8, horizontalalignment='right')

    last = r[-1]
    s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
        today.strftime('%d-%b-%Y'),
        last.open, last.high,
        last.low, last.close,
        last.volume*1e-6,
        last.close - last.open)
    t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

    cumval = np.cumsum(val)
    ax3.plot(r.date[clean_sell], 100.*cumval/r.open[-1], color='darkslategrey', label='cumulative', lw=2)
    ax3.bar(r.date[clean_sell], 100.*val/r.open[-1],
            width=4, color=cm.jet(-np.sign(val)),
            alpha=0.7, label='instantaneous')

    ax3.axhline()
    ax3.text(0.025, 0.95, 'Purse (pct of today value)', va='top',
             transform=ax3.transAxes, fontsize=textsize)

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

    # at most 5 ticks, pruning the upper and lower so they don't overlap
    # with other ticks
    #ax2.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
    #ax3.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))

    ax2.yaxis.set_major_locator(MyLocator(5, prune='both'))
    ax3.yaxis.set_major_locator(MyLocator(5, prune='both'))

    important_events = {
        'buy': datetime.date.today() - r.date[clean_buy][-1],
        'sell': datetime.date.today() - r.date[clean_sell][-1]
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


if __name__ == '__main__':
    go(ticker='GLD', save=False)
