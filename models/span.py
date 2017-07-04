import logging

from models.directors import NumpyDecider
from models.indicators import RSIMixin, TheEvaluator
from models.plotter import PlotMixin


class Span(object):
    def __init__(self, security, span=None):
        # TODO: Weekly Span should increase the look back to 1.5 years
        logging.info('Setting span for {} to {}'.format(security.ticker, span))
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