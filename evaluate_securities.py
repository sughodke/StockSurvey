"""
evaluate_securities : load and evaluate performance of a security

Ideally, the confidence of the buy/sell should translate to a Stop/Limit order (OrderManager?)

When we set the span to week, we need to pull YTD data. While daily should be limited to 6m.

"""

import logging
from optparse import OptionParser
from pprint import pprint
from sklearn.externals.joblib import Parallel, delayed

from finance_ndx import NDX_constituents, my_faves
from models.security import Security

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = OptionParser()
parser.add_option("--span", dest="span", default='daily',
                  help="Select a timespan to process on (daily, weekly, monthly)")
parser.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print additional information to the console")
parser.add_option("--save", "--save-plot",
                  action="store_true", dest="save_plot", default=False,
                  help="don't print status messages to stdout")
parser.add_option("--force", "--force-fetch",
                  action="store_true", dest="force", default=False,
                  help="Invalidate cache and perform a full fetch")
parser.add_option("--ndx",
                  action="store_true", dest="ndx", default=False,
                  help="Execute for all stocks on NDX100 and myfaves."
                  " Forces save plot.")
parser.add_option("--coin", "--crypto",
                  action="store_true", dest="crypto", default=False,
                  help="Indicate that the arguments are crypto-currencies")
parser.add_option("--macd",
                  action="store_const", dest="indicator",
                  default='rsi', const='macd',
                  help="Use moving average convergence divergence indicator")
parser.add_option("--start-date", dest="start_date",
                  default=None,
                  help="Truncate the plot, starting from given start date")

# TODO span can be defined as the first argument instead of flag

(opts, args) = parser.parse_args()

if opts.ndx:
    opts.save_plot = True
    args = list(NDX_constituents) + my_faves
elif len(args) == 0:
    args = ['GLD']

raise_exception = True


def run_one(ticker):
    try:
        s = Security.load(ticker, force_fetch=opts.force, crypto=opts.crypto)

        # Create a view of the data for the timespan we are interested in
        with s.span(opts.span, opts.indicator, start_date=opts.start_date) as so:

            if opts.verbose:
                print('Events for {} strategy'.format(so.span))
                pprint(so.recent_events(last_n=5))
                print('')

            # Use our strategy to figure out when to buy and sell
            orders = so.decide.compute_orders()
            if opts.verbose:
                print('List of Buy/Sell')
                pprint(zip(*orders))
                print('')

            # Evaluate our strategy
            so.eval.evaluate(orders)

            # Save a plot of our work
            so.plot.plot_data(save=opts.save_plot)

        s.save()
    except Exception as e:
        if not raise_exception:
            logging.error('{} blew up with {}'.format(ticker, e))
        else:
            raise e


if __name__ == '__main__':
    # Parallel(n_jobs=4)(delayed(run_one)(ticker) for ticker in args)
    for ticker in args:
        run_one(ticker)
