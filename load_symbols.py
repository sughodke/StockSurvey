import csv
import random

FNAME = './s-and-p-500-companies/data/constituents.csv'


def snp_500(count=500):
    ret = {}

    f = open(FNAME, 'rb')
    reader = csv.DictReader(f)

    for line in reader:
        pretty_name = '{}-{}'.format(line['Name'], line['Sector'][0:4])
        ret[line['Symbol']] = pretty_name

    return {k: ret[k] for k in random.sample(ret.keys(), count)}


def static_symbols():
    return {
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
