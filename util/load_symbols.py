import csv
import json
import random
import pandas as pd

FNAME = './s-and-p-500-companies/data/constituents.csv'


def nasdaq_gen():
    with open('./s-and-p-500-companies/nasdaqlisted.txt') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            symbol = row[0]
            if symbol.upper() == symbol:
                yield row[0]


def nasdaq():
    return list(nasdaq_gen())


def coin100():
    df = coin100_dataframe()
    return df['symbol'].tolist()


def coin100_dataframe():
    df = pd.read_json('./s-and-p-500-companies/coin100.json')
    return df


def old_coins():
    """Deprecated"""
    with open('./s-and-p-500-companies/topvolumeusd.json') as f:
        d = json.load(f)

    d = d['Data']

    # def _split_cast(x):
    #     cells = x.split('~')
    #     ret = []
    #     for cell in cells:
    #         try:
    #             ret.append(float(cell))
    #         except ValueError:
    #             ret.append(cell)
    #     return ret
    # d = list(map(_split_cast, d))
    d = list(map(lambda x: x.split('~'), d))

    r = pd.DataFrame(d, dtype=float,
                  columns=['Id', 'Sh1', 'Coin', 'BaseCurrency', 'Sh2', 'Price',
                           'Timestamp', 'PercentChange', 'BaseChange',
                           'CoinVolume', '24HCoinVolume', '24HBaseVolume',
                           'Low', 'High', 'Open', 'Exchange', 'Sh7'])

    # dtype=[int, str, str, str, str, float,
    #        int, float, float,
    #        float, float, float,
    #        float, float, float, str, str])

    r['MarketCap'] = r['Price'] * r['CoinVolume']
    return r


def snp_500(count=500):
    ret = {}

    f = open(FNAME, 'rb')
    reader = csv.DictReader(f)

    for line in reader:
        pretty_name = '{}-{}'.format(line['Name'], line['Sector'][0:4])
        ret[line['Symbol']] = pretty_name

    return {k: ret[k] for k in random.sample(ret.keys(), count)}


def static_symbols(count=61):
    ret = {
        'TOT': 'Total',
        'XOM': 'Exxon',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'VLO': 'Valero Energy',
        'MSFT': 'Microsoft',
        'IBM': 'IBM',
        'TWX': 'Time Warner',
        'CMCSA': 'Comcast',
        'CVC': 'Cablevision',
        'YHOO': 'Yahoo',
        'DELL': 'Dell',
        'HPQ': 'HP',
        'AMZN': 'Amazon',
        'TM': 'Toyota',
        'CAJ': 'Canon',
        'MTU': 'Mitsubishi',
        'SNE': 'Sony',
        'F': 'Ford',
        'HMC': 'Honda',
        'NAV': 'Navistar',
        'NOC': 'Northrop Grumman',
        'BA': 'Boeing',
        'KO': 'Coca Cola',
        'MMM': '3M',
        'MCD': 'Mc Donalds',
        'PEP': 'Pepsi',
        'MDLZ': 'Kraft Foods',
        'K': 'Kellogg',
        'UN': 'Unilever',
        'MAR': 'Marriott',
        'PG': 'Procter Gamble',
        'CL': 'Colgate-Palmolive',
        'GE': 'General Electrics',
        'WFC': 'Wells Fargo',
        'JPM': 'JPMorgan Chase',
        'AIG': 'AIG',
        'AXP': 'American express',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'AAPL': 'Apple',
        'SAP': 'SAP',
        'CSCO': 'Cisco',
        'TXN': 'Texas instruments',
        'XRX': 'Xerox',
        'LMT': 'Lookheed Martin',
        'WMT': 'Wal-Mart',
        'WBA': 'Walgreen',
        'HD': 'Home Depot',
        'GSK': 'GlaxoSmithKline',
        'PFE': 'Pfizer',
        'SNY': 'Sanofi-Aventis',
        'NVS': 'Novartis',
        'KMB': 'Kimberly-Clark',
        'R': 'Ryder',
        'GD': 'General Dynamics',
        'RTN': 'Raytheon',
        'CVS': 'CVS',
        'CAT': 'Caterpillar',
        'DD': 'DuPont de Nemours'
    }

    return {k: ret[k] for k in random.sample(ret.keys(), count)}
