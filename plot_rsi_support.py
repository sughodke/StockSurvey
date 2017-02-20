import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.finance as finance

from indicators import moving_average, relative_strength, bbands

fillcolor = "darkgoldenrod"


def go(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', save=True):
    r = load_data(startdate, enddate, ticker)
    rsi, support = transform_data(r)
    plot_data(rsi, support, ticker, save)


def load_data(startdate, enddate, ticker):
    fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
    # a numpy record array with fields: date, open, high, low, close, volume, adj_close)

    r = mlab.csv2rec(fh)
    fh.close()
    r.sort()

    return r


def transform_data(r):
    prices = r.adj_close
    # ma14 = moving_average(prices, 14, type='simple')
    # ma20 = moving_average(prices, 20, type='simple')

    avgBB, upperBB, lowerBB = bbands(prices, 21, 2)
    pct_b = (prices - lowerBB) / (upperBB - lowerBB)
    support = pct_b * 100  # prices - ma14

    rsi = relative_strength(prices, 7)

    # return rsi[20:], support[20:]
    return rsi, support


def annotate(ax, s, x, align):
    ax.text(x, 0.8, s, fontsize=9, color=fillcolor, transform=ax.transAxes,
            horizontalalignment=align)


def plot_data(rsi, support, ticker, save=False):
    g = sns.jointplot(rsi, support, color="k", xlim=(0, 100))

    # g.plot_joint(plt.quiver)
    g.plot_joint(sns.kdeplot, zorder=0, n_levels=6)

    g.ax_joint.scatter(rsi[rsi < 30], support[rsi < 30], c=fillcolor)
    g.ax_joint.scatter(rsi[rsi > 70], support[rsi > 70], c=fillcolor)

    g.ax_marg_x.axvline(30, color=fillcolor)
    g.ax_marg_x.axvline(70, color=fillcolor)
    # annotate(g.ax_marg_x, "%.0f%% oversold" % (1.0,), 0.29, 'right')
    # annotate(g.ax_marg_x, "%.0f%% overbought" % (1.0,), 0.71, 'left')

    g.ax_marg_y.set_title("%s daily" % ticker)
    g.set_axis_labels("RSI", "Support")

    if save:
        g.savefig('%s-RSI-Support.png' % ticker)
    else:
        plt.show()


if __name__ == '__main__':
    go(ticker='BIIB', save=False)

