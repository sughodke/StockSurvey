import datetime

import numpy as np
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from indicators import moving_average, relative_strength, bbands
from load_ticker import load_data

fillcolor = "darkgoldenrod"


def go(startdate=datetime.date(2016, 6, 1), enddate=datetime.date.today(), ticker='GLD', save=True):
    r = load_data(startdate, enddate, ticker)
    plot_data(r, ticker, save)


def plot_data(r, ticker, save=False):
    mark_zeros = False
    quiver = True
    heatmap = False

    prices = r.adj_close
    # ma14 = moving_average(prices, 14, type='simple')
    # ma20 = moving_average(prices, 20, type='simple')

    avgBB, upperBB, lowerBB = bbands(prices, 21, 2)
    pct_b = (prices - lowerBB) / (upperBB - lowerBB)
    support = pct_b * 100  # prices - ma14
    support = support[20:]  # remove empties

    rsi = relative_strength(prices, 7)
    rsi = rsi[20:]  # remove empties
    rsi_prime = np.gradient(rsi)
    rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]

    g = sns.jointplot(rsi, support, color="k", xlim=(0, 100))

    g.plot_joint(sns.kdeplot, zorder=0, n_levels=6)

    plt.sca(g.ax_joint)
    plt.scatter(rsi[rsi < 30], support[rsi < 30], c=fillcolor)
    plt.scatter(rsi[rsi > 70], support[rsi > 70], c=fillcolor)

    if mark_zeros:
        plt.scatter(rsi[rsi_prime_zeros], support[rsi_prime_zeros],
                    color='w', marker='x')

    d_rsi = np.diff(rsi)
    d_support = np.diff(support)
    arctan = np.arctan2(d_support, d_rsi)
    arctan = np.flipud(moving_average(np.flipud(arctan), 2, 'exponential'))

    if quiver:
        plt.quiver(rsi[:-1], support[:-1],
                   np.cos(arctan), np.sin(arctan),
                   color=cm.inferno(arctan), pivot='mid',
                   alpha=0.3)

    if heatmap:
        arctan = (arctan + np.pi) / (2 * np.pi)
        plt.scatter(rsi[:-1], support[:-1], c=cm.inferno(arctan),
                    marker='s', s=200, alpha=0.2)

    g.ax_marg_x.axvline(30, color=fillcolor)
    g.ax_marg_x.axvline(70, color=fillcolor)

    g.ax_marg_y.set_title("%s daily" % ticker)
    g.set_axis_labels("RSI (n=7)", "Bol % (n=21, sd=2)")

    if save:
        g.savefig('%s-RSI-Support.png' % ticker)
    else:
        plt.show()


if __name__ == '__main__':
    go(ticker='LUV', save=False)

