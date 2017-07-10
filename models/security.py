import datetime
import logging
import os

import joblib
import pandas as pd

from models.span import Span
from models.timespan import AddTimeSpan
from util import load_data, cwd, load_crypto_data

ds_path = 'DataStore'
store_dir = os.path.join(cwd, ds_path)


class Security(AddTimeSpan):
    STARTDATE = datetime.date(2017, 1, 1)

    def __init__(self, ticker='GLD', crypto=False):
        self.ticker = ticker
        self.is_crypto = crypto
        self.startdate = self.STARTDATE
        self.enddate = None
        self.daily = None

        self.sync()
        self.weekly = self.add_week(self.daily)
        self.monthly = self.add_month(self.daily)

    def sync(self):
        today = self._today
        if not self.enddate:
            self.enddate = self.STARTDATE - datetime.timedelta(days=1)

        if self.enddate < today:
            logging.info('Sync necessary, retrieving missing data')

            if not self.is_crypto:
                delta = load_data(self.enddate + datetime.timedelta(days=1), today, self.ticker)
            else:
                delta = load_crypto_data(self.enddate + datetime.timedelta(days=1), today, self.ticker)

            if self.daily is not None:
                self.daily = pd.concat((self.daily, delta))
            else:
                self.daily = delta

            self.enddate = today

        self.daily.sort_index(inplace=True)

    @classmethod
    def _filename(cls, ticker):
        return os.path.join(store_dir, '{}'.format(ticker))

    @property
    def _today(self):
        return datetime.date.today()

    def save(self):
        joblib.dump(self, self._filename(self.ticker), compress=False)

    @classmethod
    def load(cls, ticker, sync=True, force_fetch=False, crypto=False):
        try:
            if force_fetch:
                raise IOError('Triggering Cache Miss')

            security = joblib.load(cls._filename(ticker))
            logging.info('Security loaded sucessfully')
            if sync:
                security.sync()
            return security
        except IOError as e:
            logging.info('Cache miss, creating new Security')
            return Security(ticker, crypto)

    def span(self, span):
        """acts as a context manager"""
        return Span(self, span)
