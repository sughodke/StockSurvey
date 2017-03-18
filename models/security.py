import datetime
import os
import joblib
import numpy as np

from models.indicators import RSIMixin
from util.load_ticker import load_data

import logging


# TODO: find a place to store models
cwd = '/tmp/'
ds_path = 'DataStore/'


class Security(RSIMixin):
    STARTDATE = datetime.date(2016, 6, 1)
    store_dir = os.path.join(cwd, ds_path)

    def __init__(self, ticker='GLD'):
        self.ticker = ticker
        self.startdate = self.STARTDATE
        self.enddate = None
        self.daily = None

        self.sync()

    def sync(self):
        today = self._today
        if not self.enddate:
            self.enddate = self.STARTDATE

        if self.enddate < today:
            logging.info('Sync necessary, retrieving missing data')

            delta = load_data(self.enddate + datetime.timedelta(days=1), today, self.ticker)
            delta = self._fix_dtypes(delta)

            if self.daily is not None:
                self.daily = np.hstack((self.daily, delta)).view(np.recarray)
            else:
                self.daily = delta

            self.daily = self._fix_dtypes(self.daily)
            self.enddate = today

    @staticmethod
    def _fix_dtypes(recarr):
        """
        MLAB is a POS
        Overwriting the datatypes to ensure serialize works smoothly
        """
        return recarr.astype([('date', '<M8[D]'), ('open', '<f8'),
                              ('high', '<f8'), ('low', '<f8'),
                              ('close', '<f8'), ('volume', '<i8'),
                              ('adj_close', '<f8')])

    @classmethod
    def _filename(cls, ticker):
        return os.path.join(cls.store_dir, '{}'.format(ticker))

    @property
    def _today(self):
        return datetime.date.today()

    def save(self):
        joblib.dump(self, self._filename(self.ticker), compress=False)

    @classmethod
    def load(cls, ticker, sync=False):
        try:
            security = joblib.load(cls._filename(ticker))
            logging.info('Security loaded sucessfully')
            if sync:
                security.sync()
            return security
        except IOError as e:
            logging.info('Cache miss, creating new Security')
            return Security(ticker)

