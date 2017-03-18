import numpy as np
from util.indicators import relative_strength, moving_average


class RSIMixin(object):
    def computed_indicators(self):
        prices = self.daily.adj_close
        rsi = relative_strength(prices, 7)

        rsi_ma10 = moving_average(rsi, 10, type='exponential')
        rsi_prime = np.gradient(rsi)

        rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]
        rsi_ma_cross = np.where(np.diff(np.sign(rsi - rsi_ma10)))[0]

        self.computed = np.vstack(
            (self.daily.date.astype('O'),
             rsi, rsi_ma10, rsi_prime)
        ).transpose()

        # self.computed.view([('date', '<M8[D]'), ('rsi', '<f8'),
        # ('rsi_ma10', '<f8'), ('rsi_prime', '<f8')])

        self.date_rsi_dir_change = self.daily.date[rsi_prime_zeros]
        self.date_rsi_ma_cross = self.daily.date[rsi_ma_cross]

