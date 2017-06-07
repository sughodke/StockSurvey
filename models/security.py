import datetime
import os
import joblib
import pandas as pd

from models.plotter import PlotMixin
from models.indicators import RSIMixin, TheEvaluator
from models.directors import TheDecider, NumpyDecider
from models.timespan import AddTimeSpan
from util import load_data, cwd

import logging

ds_path = 'DataStore'
store_dir = os.path.join(cwd, ds_path)


class Security(AddTimeSpan):
    STARTDATE = datetime.date(2016, 6, 1)

    def __init__(self, ticker='GLD'):
        self.ticker = ticker
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

            delta = load_data(self.enddate + datetime.timedelta(days=1), today, self.ticker)

            if self.daily is not None:
                self.daily = pd.concat((self.daily, delta))
            else:
                self.daily = delta

            self.enddate = today

    @classmethod
    def _filename(cls, ticker):
        return os.path.join(store_dir, '{}'.format(ticker))

    @property
    def _today(self):
        return datetime.date.today()

    def save(self):
        joblib.dump(self, self._filename(self.ticker), compress=False)

    @classmethod
    def load(cls, ticker, sync=True, force_fetch=False):
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
            return Security(ticker)

    def span(self, span):
        """acts as a context manager"""
        return Span(self, span)


class Span(object):
    def __init__(self, security, span=None):
        # TODO: Weekly Span should increase the look back to 1.5 years
        self.dataset = getattr(security, span, security.daily)
        self.events = []

        self.ticker = security.ticker
        self.span = span or 'daily'

        self.calc = RSIMixin(self.dataset)
        self.decide = NumpyDecider(self.dataset, self.calc)
        self.eval = TheEvaluator(self.dataset)
        self.plot = PlotMixin(self.dataset, self.ticker,
                              self.calc, self.decide, self.eval)

    def recent_events(self, last_n):
        return [str(event) for event in self.events][:last_n]
