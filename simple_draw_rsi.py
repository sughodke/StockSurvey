import datetime
import numpy as np
import seaborn as sns
from scipy import signal
import matplotlib.pyplot as plt

from indicators import relative_strength, moving_average, bbands
from load_ticker import load_quotes, load_data


r = load_data(datetime.date(2016, 6, 1), datetime.date.today(), 'GLD')

prices = r.adj_close
rsi = relative_strength(prices)
ma20 = moving_average(prices, 20)

mid, upp, bot = bbands(prices)
bandwidth = upp - bot
pct_b = (prices - bot) / bandwidth

# widths = np.arange(1, 41)
# cwtmatr = signal.cwt(rsi, signal.ricker, widths)
#
# plt.imshow(cwtmatr, extent=[-1, 1, 1, widths.max()], cmap='viridis',  # PRGn
#            aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


left, width = 0.1, 0.8
rect1 = [left, 0.7, width, 0.2]
rect2 = [left, 0.3, width, 0.4]
rect3 = [left, 0.1, width, 0.2]

fig = plt.figure(facecolor='white')
axescolor = '#f6f6f6'  # the axes background color

ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)

dx = r.adj_close - r.close
low = r.low + dx
high = r.high + dx

deltas = np.zeros_like(prices)
deltas[1:] = np.diff(prices)
up = deltas > 0

ax1.vlines(r.date[up], low[up], high[up],
           color='green', label='_nolegend_')
ax1.vlines(r.date[~up], low[~up], high[~up],
           color='red', label='_nolegend_')

ax1.plot(r.date, bot)
ax1.plot(r.date, mid)
ax1.plot(r.date, upp)

ax2.plot(r.date, pct_b)
ax3.plot(r.date, bandwidth)

plt.show()
