import logging
from contextlib import ContextDecorator

from models.directors import NumpyDecider, MACDDecider
from models.indicators import RSIMixin, TheEvaluator, MACDMixin, BBandsMixin
from models.plotter import PlotMixin, MACDPlotMixin


class BaseSpan(ContextDecorator):
    def __init__(self, security, span=None):
        # TODO: Weekly Span should increase the look back to 1.5 years

        self.dataset = getattr(security, span, security.daily)
        self.events = []

        self.ticker = security.ticker
        self.span = span or 'daily'

        self.calc = self.decide = self.eval = self.plot = None

    def __enter__(self):
        logging.info('Setting span for {} to {}'.format(self.ticker, self.span))
        self.workflow()
        return self

    def __exit__(self, *args):
        pass

    def recent_events(self, last_n):
        return [str(event) for event in self.events][:last_n]

    def workflow(self):
        raise NotImplemented()


class Span(BaseSpan):

    def workflow(self):
        self.calc = RSIMixin(self.dataset)
        self.decide = NumpyDecider(self.dataset, self.calc)
        self.eval = TheEvaluator(self.dataset)
        self.plot = PlotMixin(self.dataset, self.ticker,
                              self.calc, self.decide, self.eval)


class MACDSpan(BaseSpan):

    def workflow(self):
        self.calc = MACDMixin(self.dataset)
        self.decide = MACDDecider(self.dataset, self.calc)
        self.eval = TheEvaluator(self.dataset)
        self.plot = MACDPlotMixin(self.dataset, self.ticker,
                                  self.calc, self.decide, self.eval)


class BBandsSpan(BaseSpan):

    def workflow(self):
        self.calc = BBandsMixin(self.dataset)
        self.decide = self.eval = self.plot = None
