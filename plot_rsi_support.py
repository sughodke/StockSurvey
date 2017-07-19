import datetime

import numpy as np
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from models.plotter import PlotBaseMixin
from util.indicators import moving_average
from models.security import Security

MARKER_SIZE = 12
fillcolor = "darkgoldenrod"


def go(ticker='GLD', save=True):
    s = Security.load(ticker, force_fetch=False, crypto=False)
    with s.span('daily', 'rsi') as so_rsi, s.span('daily', 'bbands') as so_bbands:

        # Use a specialized plotter that can process two indicators
        OscillatorSupportPlot(
            so_rsi.dataset,
            ticker,
            {'rsi': so_rsi.calc, 'bbands': so_bbands.calc},
            None,
            None,
            'daily'
        ).plot_data(save)


class OscillatorSupportPlot(PlotBaseMixin):

    def plot_data(self, save=False):
        mark_zeros = False
        quiver = False
        heatmap = False

        """
        # Legacy
        prices = self.dataset
        # ma14 = moving_average(prices, 14, type='simple')
        # ma20 = moving_average(prices, 20, type='simple')
    
        support = support[20:]  # TODO auto remove empties
        rsi = rsi[20:]  # remove empties
        """

        rsi, rsi_ma10, rsi_prime = self.calc['rsi'].rsi_values
        rsi_prime_zeros = self.calc['rsi'].rsi_prime_zeros
        support = self.calc['bbands'].support

        g = sns.jointplot(rsi, support, color="k", xlim=(0, 100), s=MARKER_SIZE)

        g.plot_joint(sns.kdeplot, zorder=0, n_levels=6)

        plt.sca(g.ax_joint)
        plt.scatter(rsi[rsi < 30], support[rsi < 30], c=fillcolor, s=MARKER_SIZE)
        plt.scatter(rsi[rsi > 70], support[rsi > 70], c=fillcolor, s=MARKER_SIZE)

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

        g.ax_marg_y.set_title("%s daily" % self.ticker)
        g.set_axis_labels("RSI (n=7)", "Bol % (n=21, sd=2)")

        if save:
            g.savefig('%s-RSI-Support.png' % self.ticker)
        else:
            plt.show()


if __name__ == '__main__':
    go(ticker='XLNX', save=False)

