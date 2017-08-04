import logging


from aiohttp import web
from aiohttp_swagger import *

from models.security import Security
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

async def index(request):
    with open('Frontend/index.html') as f:
        r = f.read()
    return web.Response(body=r, content_type='text/html')

async def handle(request):
    """
    Description end-point
    parameters:
    - in: ticker
        name: ticker
        description: Security to evaluate
        required: true
        schema:
            type: string
    ---
    tags:
    - Evaluate
    summary: Evaluate and plot a security
    description: This can only be done by the logged in user.
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


app = web.Application()
app.router.add_route('GET', '/', index)
app.router.add_route('GET', '/evaluate', handle)

setup_swagger(app)
web.run_app(app, host=None)
