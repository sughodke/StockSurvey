"""
graph rsi for multiple time spans
"""

import numpy as np
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import animation

import evaluate_securities
from models.indicators import RSIMixin
from util.indicators import moving_average
from models.plotter import PlotBaseMixin
from models.security import Security


MARKER_SIZE = 12
fillcolor = "darkgoldenrod"

opts, args = evaluate_securities.opts, evaluate_securities.args


def go(ticker='BAC'):
    s = Security.load(ticker, force_fetch=opts.force, crypto=opts.crypto)
    s.daily['rsi'] = RSIMixin(s.daily).rsi
    s.weekly['rsi'] = RSIMixin(s.weekly).rsi

    rsi_weekly_resampled = s.weekly['rsi'].resample('B').bfill()

    rsi_joined = s.daily.join(rsi_weekly_resampled, rsuffix='_weekly')
    rsi_joined.dropna(inplace=True)

    rsi_weekly = rsi_joined['rsi_weekly']
    rsi_daily = rsi_joined['rsi']

    # rsi_prime_zeros = self.calc['rsi_weekly'].rsi_prime_zeros

    g = sns.jointplot(rsi_weekly, rsi_daily, color="k", s=MARKER_SIZE)

    g.plot_joint(sns.kdeplot, zorder=0, n_levels=6)

    plt.sca(g.ax_joint)
    plt.scatter(rsi_weekly[rsi_weekly < 30], rsi_daily[rsi_weekly < 30], c=fillcolor, s=MARKER_SIZE)
    plt.scatter(rsi_weekly[rsi_weekly > 70], rsi_daily[rsi_weekly > 70], c=fillcolor, s=MARKER_SIZE)

    g.ax_marg_x.axvline(30, color=fillcolor)
    g.ax_marg_x.axvline(70, color=fillcolor)

    g.ax_marg_y.set_title('%s' % (ticker,))
    g.set_axis_labels("RSI Weekly (7)", "RSI Daily (7)")

    time_plot = rsi_joined[['close', 'rsi', 'rsi_weekly']].plot()
    time_plot.set_title(ticker)

    plt.show()

    def update_plot(self, i, scat, update_title, rsi, support, index):
        scat.set_offsets(np.array([rsi[i], support[i]]))
        update_title('%s %s\n%s' % (self.ticker, self.cadence, index[i].date()))
        return scat,


if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(go)(ticker) for ticker in args)
