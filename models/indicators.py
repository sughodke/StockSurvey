from util.indicators import relative_strength


class RSIMixin(object):
    def computed_indicators(self):
        prices = self.daily.adj_close
        rsi = relative_strength(prices, 7)
        rsi_prime = np.gradient(rsi)
        rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]
        rsi_ma10 = moving_average(rsi, 10, type='exponential')
        rsi_ma_cross = np.where(np.diff(np.sign(rsi - rsi_ma10)))[0]