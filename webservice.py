import logging


from aiohttp import web
from aiohttp_swagger import *

from sort_securities import sortby_relevance
from models.security import Security
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


async def evaluate(request):
    """
    Evaluate a security
    ---
    tags:
    - Evaluate
    summary: Evaluate and plot a security
    description:
    produces:
    - image/svg+xml
    parameters:
    - in: query
      name: ticker
      description: Security to evaluate
      required: true
      type: string
    - in: query
      name: force
      description: Force reload of data
      required: false
      type: boolean
    - in: query
      name: span
      description: Specify the span range (daily, weekly)
      required: false
      type: string

    responses:
      "200":
        description: successful operation
    """

    ticker = request.query['ticker']
    force = request.query.get('force', False)
    crypto = request.query.get('crypto', False)
    span = request.query.get('span', 'daily')

    s = Security.load(ticker, force_fetch=force, crypto=crypto)

    # Create a view of the data for the timespan we are interested in
    with s.span(span) as so:
        # Use our strategy to figure out when to buy and sell
        orders = so.decide.compute_orders()

        # Evaluate our strategy
        so.eval.evaluate(orders)

        # Save a plot of our work
        plot = so.plot.plot_data(web=True)

    s.save()

    return web.Response(text=plot.getvalue(), content_type='image/svg+xml')

async def relevance(request):
    """
    Sort interesting securities by relevance
    ---
    tags:
    - Relevance
    summary: Sort a bunch of securities
    description:
    produces:
    - application/json
    responses:
      "200":
        description: successful operation
    """

    return web.json_response(data=sortby_relevance(only_names=True),
                             headers={'Access-Control-Allow-Origin': '*'})


app = web.Application()
app.router.add_route('GET', '/evaluate', evaluate)
app.router.add_route('GET', '/relevance', relevance)
app.router.add_static('/', 'Frontend/')

setup_swagger(app)  # "/api/doc"
web.run_app(app, host=None)
