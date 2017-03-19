import logging
import numpy as np
from util.indicators import relative_strength, moving_average


class Event(object):
    TYPES = {
        'RSI Direction Change': 0,
        'RSI MA Cross': 1
    }
    RSI_direction_change = 'RSI Direction Change'
    RSI_MA_cross = 'RSI MA Cross'

    def __init__(self, event_type, date):
        self.event_name = event_type
        self.event_type = self.TYPES[event_type]
        self.date = date

    def flatten_span(self, span):
        pass

    def __repr__(self):
        return '{} on {}'.format(self.event_name, self.date)


class RSIMixin(object):
    def rsi(self):
        prices = self.dataset.adj_close
        rsi = relative_strength(prices, 7)

        rsi_ma10 = moving_average(rsi, 10, type='exponential')
        rsi_prime = np.gradient(rsi)

        rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]
        rsi_ma_cross = np.where(np.diff(np.sign(rsi - rsi_ma10)))[0]

        self.rsi_values = np.vstack(
            (self.dataset.date.astype('O'),
             rsi, rsi_ma10, rsi_prime)
        ).transpose()

        logging.info('Computed RSI {0}, {1}'.format(*self.rsi_values.shape))

        # self.computed.view([('date', '<M8[D]'), ('rsi', '<f8'),
        # ('rsi_ma10', '<f8'), ('rsi_prime', '<f8')])

        for d in self.dataset.date[rsi_prime_zeros]:
            self.events.append(Event(Event.RSI_direction_change, d))

        for d in self.dataset.date[rsi_ma_cross]:
            self.events.append(Event(Event.RSI_MA_cross, d))

        self.rsi = rsi
        self.rsi_prime_zeros = rsi_prime_zeros
        self.rsi_ma_cross = rsi_ma_cross

        return self


class TheEvaluator(object):
    def evaluate(self, orders):
        buy, sell, vol_buy = orders
        val_buy, val_sell = self.dataset.open[list(buy)], self.dataset.open[list(sell)]

        val = np.array(vol_buy) * (val_sell - val_buy)
        sumval = np.sum(val)
        performance = 100. * sumval / self.dataset.open[-1]
        logging.info('With %d trades we stand to make %f (%f%%).' % (len(val_buy), sumval, performance))
