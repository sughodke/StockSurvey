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
    def __init__(self, d):
        self.dataset = d
        prices = self.dataset.adj_close.values
        rsi = relative_strength(prices, 7)

        rsi_ma10 = moving_average(rsi, 10, type='exponential')
        rsi_prime = np.gradient(rsi)

        self.rsi_prime_zeros = np.where(np.diff(np.sign(rsi_prime)))[0]
        self.rsi_ma_cross = np.where(np.diff(np.sign(rsi - rsi_ma10)))[0]

        self.rsi_values = (rsi, rsi_ma10, rsi_prime)

        logging.info('Computed RSI {}, {}, {}'.format(*map(len, self.rsi_values)))

        self.rsi = rsi

    def events(self):
        r = []
        for d in self.dataset.date[rsi_prime_zeros]:
            r.append(Event(Event.RSI_direction_change, d))

        for d in self.dataset.date[rsi_ma_cross]:
            r.append(Event(Event.RSI_MA_cross, d))

        return r


class TheEvaluator(object):
    def __init__(self, d):
        self.dataset = d
        self.val = None
        self.performance = None

    def evaluate(self, orders):
        buy, sell, vol_buy = orders
        val_buy = self.dataset.open[buy].values
        val_sell = self.dataset.open[sell].values

        self.val = val = vol_buy * (val_sell - val_buy)
        sumval = np.sum(val)
        self.performance = 100. * sumval / self.dataset.open[-1]
        logging.info('With %d trades we stand to make %f (%f%%).' % (len(val_buy), sumval, self.performance))
