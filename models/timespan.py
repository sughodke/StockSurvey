import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())


class AddTimeSpan(object):

    def add_week(self, f):
        open = f.open.resample('W-MON').last()
        close = f.close.resample('W-FRI').last().resample('W-MON').last()
        adj_close = f.adj_close.resample('W-FRI').last().resample('W-MON').last()
        high = f.high.resample('W-MON').max()
        low = f.low.resample('W-MON').min()
        vol = f.volume.resample('W-MON').sum()

        return pd.concat([open, close, high, low, vol, adj_close], axis=1)

    def add_month(self, f):
        open = f.open.resample('MS').last()
        close = f.close.resample('M').last().resample('MS').last()
        adj_close = f.adj_close.resample('M').last().resample('MS').last()
        high = f.high.resample('MS').max()
        low = f.low.resample('MS').min()
        vol = f.volume.resample('MS').sum()

        return pd.concat([open, close, high, low, vol, adj_close], axis=1)

