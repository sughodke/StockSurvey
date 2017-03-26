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
        logging.info('Buy/Sell events {}/{}'.format(len(buy_index), len(sell_index)))

        clean_buy, clean_sell = self.filter_buysell(buy_index, sell_index)

        # confidence that we are buying at the correct time
        vol_buy = 10 - self.rsi[clean_buy] // 10

        self.clean_buysellvol = (clean_buy, clean_sell, vol_buy)
        return self.clean_buysellvol

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

        return clean_buy, clean_sell


class NumpyDecider(TheDecider):
    def compute_possible_buysell(self):
        """
        rsi_prime_zeros[k]  Index on self.rsi when RSI Prime is 0

        """
        dir_change_dates = self.dataset.date[self.rsi_prime_zeros]

        buy_idx, sell_idx = [], []
        for cross_index in self.rsi_ma_cross:
            cross_date = self.dataset.date[cross_index]

            # get the next time the RSI changes directions
            try:
                dir_change_index = np.where(cross_date < dir_change_dates)[0][0]
                dir_change_index = self.rsi_prime_zeros[dir_change_index]
            except IndexError:
                dir_change_index = -1

            logging.info('%s %s %s' % (self.dataset.date[dir_change_index],
                                       'buy' if self.rsi[dir_change_index] < self.rsi[cross_index] else 'sell',
                                       self.dataset.open[dir_change_index]))

            # we know this direction change is important
            # but should we buy or sell?
            # if the ma is ^ shaped, buy
            # if the ma is u shaped, sell
            if self.rsi[dir_change_index] < self.rsi[cross_index]:
                buy_idx.append(dir_change_index)
            else:
                sell_idx.append(dir_change_index)

        return buy_idx, sell_idx

    def filter_buysell(self, buy_idx, sell_idx):
        """Sell when price is higher than when I bought"""
        sell_dates = self.dataset.date[sell_idx]
        clean_buy, clean_sell = [], []

        # walk through the buy-events
        for buy, buy_date in zip(buy_idx, self.dataset.date[buy_idx]):
            rsi_at_buy = self.rsi[buy]
            price_at_buy = self.dataset.adj_close[buy]

            mask = np.where(sell_dates >= buy_date)
            future_sell_dates = sell_dates[mask]

            future_sell_index = [np.argmax(self.dataset.date == future_sell_date)
                                 for future_sell_date in future_sell_dates]
            future_sell_rsi = self.rsi[future_sell_index]
            future_sell_price = self.dataset.adj_close[future_sell_index]

            try:
                # TODO: refactor first_beat to be a virtual function
                # NOTE argmax returns 0 even on failure
                # first_beat = np.where(future_sell_rsi > rsi_at_buy)[0][0]
                first_beat = np.argmax(future_sell_price > price_at_buy)
                matching_sell = future_sell_index[first_beat]
            except (ValueError, IndexError):
                matching_sell = None

            # look for a matching next sell-event
            clean_buy.append(buy)
            clean_sell.append(matching_sell)

        return clean_buy, clean_sell


class DirectionChangeDecider(TheDecider):
    def compute_possible_buysell(self):
        buy_idx, sell_idx = [], []

        date_ma_cross = self.dataset.date[self.rsi_ma_cross]
        for rsi_dir_change in self.rsi_prime_zeros:
            date_dir_change = self.dataset.date[rsi_dir_change]

            # look at what the MA is doing
            # find the last time ma crossed rsi
            nearest_ma_cross = np.argmax(date_ma_cross >= date_dir_change) - 1
            if nearest_ma_cross < 0:
                continue

            # we know this direction change is important,
            # but should we buy or sell?
            # if the ma is ^ shaped, buy
            # if the ma is u shaped, sell

            # TODO: should look at RSI double prime, predict direction
            if self.rsi[nearest_ma_cross] > self.rsi[rsi_dir_change]:
                # RSI at Direction Change is more over-sold than at the MA cross
                action = 'buy'
            else:
                action = 'sell'

            {
                'buy': buy_idx,
                'sell': sell_idx
            }[action].append(rsi_dir_change)
            logging.info('%s %s %s' % (date_dir_change, action,
                                       self.dataset.open[rsi_dir_change]))

        return buy_idx, sell_idx

    def filter_buysell(self, buy_idx, sell_idx):
        sell_dates = self.dataset.date[sell_idx]
        clean_buy, clean_sell = [], []

        # walk through the buy-events
        for buy, buy_date in zip(buy_idx, self.dataset.date[buy_idx]):
            rsi_at_buy = self.rsi[buy]
            price_at_buy = self.dataset.adj_close[buy]

            mask = np.where(sell_dates >= buy_date)
            future_sell_dates = sell_dates[mask]

            future_sell_index = [np.argmax(self.dataset.date == future_sell_date)
                                 for future_sell_date in future_sell_dates]
            future_sell_rsi = self.rsi[future_sell_index]
            future_sell_price = self.dataset.adj_close[future_sell_index]

            try:
                first_beat = np.where(future_sell_rsi > rsi_at_buy)[0][0]  # argmax returns 0 even on failure
                # first_beat = np.argmax(future_sell_price > price_at_buy)
                matching_sell = future_sell_index[first_beat]
            except (ValueError, IndexError):
                matching_sell = None

            """
            # look for a matching next sell-event
            matching_sell = None
            for sell_date in future_sell_dates:
                sell = np.argmax(self.dataset.date == sell_date)
                rsi_at_sell = self.rsi[sell]

                if rsi_at_sell > rsi_at_buy:
                    matching_sell = sell
                    break  # don't get greedy
            """

            if matching_sell is not None:
                clean_buy.append(buy)
                clean_sell.append(matching_sell)

        return clean_buy, clean_sell

