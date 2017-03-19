import logging
import numpy as np


class TheDecider(object):
    """Decides Buy/Sell"""
    def compute_orders(self):
        logging.info('RSI crossed %d times' %
                     self.rsi_ma_cross.shape)
        logging.info('RSI changed direction %d times' %
                     self.rsi_prime_zeros.shape)

        buy_index, sell_index = self.compute_possible_buysell()
        return self.filter_buysell(buy_index, sell_index)

    def compute_possible_buysell(self):
        buy_idx, sell_idx = [], []
        k = 0
        for i in self.rsi_ma_cross:

            # get the next time the RSI changes directions
            j = self.rsi_prime_zeros[k]
            while self.dataset.date[j] < self.dataset.date[i]:
                k += 1
                try:
                    j = self.rsi_prime_zeros[k]
                except IndexError:
                    j = self.rsi_prime_zeros[-1]
                    break

            logging.info('%s %s %s' % (self.dataset.date[j],
                                       'buy' if self.rsi[j] < self.rsi[i] else 'sell',
                                       self.dataset.open[j]))

            # we know this direction change is important
            # but should we buy or sell?
            # if the ma is ^ shaped, buy
            # if the ma is u shaped, sell
            if self.rsi[j] < self.rsi[i]:
                buy_idx.append(j)
            else:
                sell_idx.append(j)

        return buy_idx, sell_idx

    def filter_buysell(self, buy_idx, sell_idx):
        clean_buy, clean_sell = [], []
        j = 0
        for i in range(len(buy_idx)):

            # walk through the buy-events
            idx_i = buy_idx[i]

            # look for a matching next sell-event
            try:
                idx_j = sell_idx[j]
            except IndexError:
                j = -1
                idx_j = sell_idx[j]

            di, dj = self.dataset.date[idx_i], self.dataset.date[idx_j]
            if di < dj:
                continue
            clean_buy.append(idx_i)
            while di > dj:
                j += 1
                try:
                    idx_j = sell_idx[j]
                    dj = self.dataset.date[idx_j]
                except IndexError:
                    idx_j = sell_idx[j - 1]
                    break
            clean_sell.append(idx_j)

        # confidence that we are buying at the correct time
        vol_buy = 10 - self.rsi[clean_buy] // 10

        return clean_buy, clean_sell, vol_buy


class NumpyDecider(TheDecider):
    def compute_possible_buysell(self):
        buy_idx, sell_idx = [], []

        date_ma_cross = self.dataset.date[self.rsi_ma_cross]
        for rsi_dir_change in self.rsi_prime_zeros:
            date_dir_change = self.dataset.date[rsi_dir_change]

            # look at what the MA is doing
            earliest_cross = np.argmax(date_ma_cross >= date_dir_change) - 1
            if earliest_cross < 0:
                continue

            # we know this direction change is important,
            # but should we buy or sell?
            # if the ma is ^ shaped, buy
            # if the ma is u shaped, sell
            if self.rsi[earliest_cross] > self.rsi[rsi_dir_change]:
                action = 'buy'
            else:
                action = 'sell'

            {
                'buy': buy_idx,
                'sell': sell_idx
            }[action].append(rsi_dir_change)
            logging.info('%s %s %s' % (date_dir_change, action,
                                       self.dataset.open[rsi_dir_change]))

        logging.info('Buy/Sell events {}/{}'.format(len(buy_idx), len(sell_idx)))

        return buy_idx, sell_idx


class NumpyDeciderOrig(object):
    """Decides Buy/Sell"""
    def compute_orders(self):
        buy_idx, sell_idx = [], []

        logging.info('RSI crossed %d times' %
                     self.rsi_ma_cross.shape)
        logging.info('RSI changed direction %d times' %
                     self.rsi_prime_zeros.shape)

        date_ma_cross = self.dataset.date[self.rsi_ma_cross]
        for rsi_dir_change in self.rsi_prime_zeros:
            date_dir_change = self.dataset.date[rsi_dir_change]

            # look at what the MA is doing
            earliest_cross = np.argmax(date_ma_cross >= date_dir_change) - 1
            if earliest_cross < 0:
                continue

            # we know this direction change is important,
            # but should we buy or sell?
            # if the ma is ^ shaped, buy
            # if the ma is u shaped, sell
            if self.rsi[earliest_cross] > self.rsi[rsi_dir_change]:
                action = 'buy'
            else:
                action = 'sell'

            {
                'buy': buy_idx,
                'sell': sell_idx
            }[action].append(rsi_dir_change)
            logging.info('%s %s %s' % (date_dir_change, action,
                                       self.dataset.open[rsi_dir_change]))

        logging.info('Buy/Sell events {}/{}'.format(len(buy_idx), len(sell_idx)))

        clean_buy, clean_sell = [], []
        j = 0
        for i in range(len(buy_idx)):

            # walk through the buy-events
            idx_i = buy_idx[i]

            # look for a matching next sell-event
            try:
                idx_j = sell_idx[j]
            except IndexError:
                j = -1
                idx_j = sell_idx[j]

            di, dj = self.dataset.date[idx_i], self.dataset.date[idx_j]
            if di < dj:
                continue
            clean_buy.append(idx_i)
            while di > dj:
                j += 1
                try:
                    idx_j = sell_idx[j]
                    dj = self.dataset.date[idx_j]
                except IndexError:
                    idx_j = sell_idx[j - 1]
                    break
            clean_sell.append(idx_j)

        # confidence that we are buying at the correct time
        vol_buy = 10 - self.rsi[clean_buy] // 10

        return clean_buy, clean_sell, vol_buy
