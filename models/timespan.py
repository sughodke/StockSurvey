import numpy as np

from pandas import DatetimeIndex
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())


class AddTimeSpan(object):
    def compute_calendar(self):
        self.cal = DatetimeIndex(freq=us_bd,
                                 start=self.daily.date[0],
                                 end=self.daily.date[-1])

    def add_week(self):
        self.compute_calendar()
        self.weekly = self.add_span('week')

    def add_month(self):
        self.compute_calendar()
        self.monthly = self.add_span('month')

    def add_span(self, span='week'):
        span_changed = np.diff(getattr(self.cal, span))
        span_changed += [0]

        r = []
        high, low, open, close, adj_close = (0., float("inf"), 0., 0., 0.)
        for h, l, o, c, ac, wk, date in zip(self.daily.high, self.daily.low,
                                            self.daily.open, self.daily.close,
                                            self.daily.adj_close,
                                            span_changed, self.daily.date):
            if wk == 1:
                close = c
                adj_close = ac

                r.append((date, high, low, open, close, adj_close))
                high, low, open, close, adj_close = (0., float('inf'), 0., 0., 0.)

            if open == 0.:
                open = o

            high = max(high, h)
            low = min(low, l)

        r = np.array(r, dtype=[('date', '<M8[D]'), ('high', '<f8'), ('low', '<f8'),
                               ('open', '<f8'), ('close', '<f8'), ('adj_close', '<f8')])
        return r.view(np.recarray)

