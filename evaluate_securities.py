"""
evalute_securities : load and evaluate performance of a security

Ideally, the confidence of the buy/sell should translate to a Stop/Limit order (OrderManager?)

When we set the span to week, we need to pull YTD data. While daily should be limited to 6m.

"""

from pprint import pprint

from models.security import Security

import logging
from optparse import OptionParser


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = OptionParser()
parser.add_option("--span", dest="span", default='daily',
                  help="Select a timespan to process on (daily, weekly, monthly)")
parser.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print additional information to the console")
parser.add_option("--save-plot",
                  action="store_true", dest="save_plot", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--force-fetch",
                  action="store_true", dest="force", default=False,
                  help="Invalidate cache and perform a full fetch")

(opts, args) = parser.parse_args()


# TODO: iterate over args
ticker = args[0] if len(args) > 0 else 'GLD'
s = Security.load(ticker, force_fetch=opts.force)

# Create a view of the data for the timespan we are interested in
so = s.span(opts.span)

so.rsi()

if opts.verbose:
    print('Events for {} strategy'.format(so.span))
    pprint(so.recent_events(last_n=5))
    print('')

# Use our strategy to figure out when to buy and sell
orders = so.compute_orders()
if opts.verbose:
    print('List of Buy/Sell')
    pprint(zip(*orders))
    print('')

# Evaluate our strategy
so.evaluate(orders)

# Save a plot of our work
so.plot_data(save=opts.save_plot)

s.save()


