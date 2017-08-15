import logging
from contextlib import ContextDecorator

from models.directors import NumpyDecider, MACDDecider
from models.indicators import RSIMixin, TheEvaluator, MACDMixin, BBandsMixin
from models.plotter import PlotMixin, MACDPlotMixin


class BaseSpan(ContextDecorator):
    def __init__(self, security, span=None):
        # TODO: Weekly Span should increase the look back to 1.5 years

        self.dataset = getattr(security, span, security.daily)

        self.ticker = security.ticker
        self.span = span or 'daily'

        self.calc = self.decide = self.eval = self.plot = None

    def __enter__(self):
        logging.info('Setting span for {} to {}'.format(self.ticker, self.span))
        self.workflow()
        return self

    def __exit__(self, *args):
        # TODO: save the span to disk
        pass

    @property
    def events(self):
        l = [['buy {}'.format(self.dataset.index[b]), 'sell {}'.format(self.dataset.index[s])]
             for b, s in zip(self.decide.clean_buysellvol[0:2])]
        return [item for sublist in l for item in sublist]

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
                              self.calc, self.decide, self.eval,
                              self.span)


class MACDSpan(BaseSpan):

    def workflow(self):
        self.calc = MACDMixin(self.dataset)
        self.decide = MACDDecider(self.dataset, self.calc)
        self.eval = TheEvaluator(self.dataset)
        self.plot = MACDPlotMixin(self.dataset, self.ticker,
                                  self.calc, self.decide, self.eval,
                                  self.span)


class BBandsSpan(BaseSpan):

    def workflow(self):
        self.calc = BBandsMixin(self.dataset)
        self.decide = self.eval = self.plot = None
