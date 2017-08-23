import datetime
import logging
import os

import joblib
import pandas as pd

from models.span import Span, MACDSpan, BBandsSpan
from models.timespan import AddTimeSpan
from util import load_data, cwd, load_crypto_data

ds_path = 'DataStore'
store_dir = os.path.join(cwd, ds_path)


class Security(AddTimeSpan):
    STARTDATE = datetime.datetime(2016, 6, 1)

    def __init__(self, ticker='GLD', crypto=False):
        self.ticker = ticker
        self.is_crypto = crypto
        self.enddate = None
        self.daily = None

        self.sync()

        # TODO: make this into a pandas view (or transform)
        self.weekly = self.add_week(self.daily)
        self.monthly = self.add_month(self.daily)

    def sync(self):
        today = self._today
        if not self.enddate:
            self.enddate = self.STARTDATE - datetime.timedelta(days=1)

        staleness = datetime.timedelta(days=1)
        if today - self.enddate >= staleness:
            logging.info('Sync necessary, retrieving missing data')

            if not getattr(self, 'is_crypto', False):
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
        if getattr(self, 'is_crypto', False):  # TODO: (versioning) migrate this into a property
            return datetime.datetime.utcnow()
        return datetime.datetime.now()  # TODO: tzdata to convert to EST (intrinio)

    def save(self):
        joblib.dump(self, self._filename(self.ticker), compress=False)

    @classmethod
    def load(cls, ticker, sync=True, force_fetch=False, crypto=False):
        try:
            if force_fetch:
                raise IOError('Triggering Cache Miss')

            security = joblib.load(cls._filename(ticker))
            logging.info('Security {} loaded successfully'.format(ticker))
            if sync:
                security.sync()
            return security
        except IOError as e:
            logging.info('Cache miss, creating new Security {}'.format(ticker))
            return Security(ticker, crypto)

    def span(self, freq, klass='rsi', **kwargs):
        """return a new Span workflow as a context manager for the freq time-window"""
        klass_lookup = {
            'rsi': Span,
            'macd': MACDSpan,
            'bbands': BBandsSpan
        }.get(klass, Span)

        return klass_lookup(self, freq, **kwargs)
