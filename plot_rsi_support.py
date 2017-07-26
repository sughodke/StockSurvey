"""
plot_rsi_support : graph an oscillator against support to find a pattern
"""

import numpy as np
import seaborn as sns

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import animation
from scipy import signal
from sklearn.decomposition import TruncatedSVD

import evaluate_securities
from util.indicators import moving_average
from models.plotter import PlotBaseMixin
from models.security import Security


MARKER_SIZE = 12
fillcolor = "darkgoldenrod"

opts, args = evaluate_securities.opts, evaluate_securities.args


def go(ticker='GLD'):
    s = Security.load(ticker, force_fetch=opts.force, crypto=opts.crypto)
    with s.span(opts.span, 'rsi') as so_rsi, s.span(opts.span, 'bbands') as so_bbands:

        # Use a specialized plotter that can process two indicators
        OscillatorSupportPlot(
            so_rsi.dataset,
            ticker,
            {'rsi': so_rsi.calc, 'bbands': so_bbands.calc},
            None,
            None,
            opts.span
        ).plot_data(opts.save_plot)


class OscillatorSupportPlot(PlotBaseMixin):

    def plot_data(self, save=False):
        mark_zeros = False
        quiver = True
        heatmap = False
        anim = True  # False if save else True

        rsi, rsi_ma10, rsi_prime = self.calc['rsi'].rsi_values
        rsi_prime_zeros = self.calc['rsi'].rsi_prime_zeros
        support = self.calc['bbands'].support

        rsi = self.transform(rsi)
        support = self.transform(support)

        g = sns.jointplot(rsi, support, color="k", s=MARKER_SIZE)

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

        g.ax_marg_y.set_title('%s %s' % (self.ticker, self.cadence))
        g.set_axis_labels("RSI (n=7)", "Bol % (n=21, sd=2)")

        if anim:
            scat = g.ax_joint.scatter([], [], s=100)
            ani = animation.FuncAnimation(g.fig, self.update_plot, frames=range(len(self.dataset)),
                                          blit=False, interval=25,
                                          fargs=(scat, g.ax_marg_y.set_title,
                                                 rsi, support, self.dataset.index))

        if save:
            # g.savefig('OscSupport/%s-RSI-Support.png' % self.ticker, dpi=100)
            ani.save('OscSupport/%s-RSI-Support.mp4' % self.ticker, fps=15)
        else:
            plt.show()
        plt.close()

    def update_plot(self, i, scat, update_title, rsi, support, index):
        scat.set_offsets(np.array([rsi[i], support[i]]))
        update_title('%s %s\n%s' % (self.ticker, self.cadence, index[i].date()))
        return scat,

    @staticmethod
    def transform(inp):
        wavelet = signal.ricker
        widths = np.arange(1, 5, 0.05)

        inp = np.nan_to_num(inp)
        X = signal.cwt(inp, wavelet, widths).T

        svd = TruncatedSVD(n_components=1, n_iter=7)
        transformed = svd.fit_transform(X)

        return transformed.reshape((len(X),))


if __name__ == '__main__':
    # go(args[0])
    Parallel(n_jobs=4)(delayed(go)(ticker) for ticker in args)
