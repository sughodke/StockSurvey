"""Static symbol lists: NASDAQ-100, NASDAQ all, top-100 crypto, S&P 500.

These are convenience accessors over the `s-and-p-500-companies/` git
submodule (NASDAQ + crypto) and an embedded NDX-100 list. File-based
accessors (`nasdaq`, `coin100`, `snp_500`) require the submodule to be
present at the repo root.
"""

from __future__ import annotations

import csv
import json
import random
from textwrap import dedent
from typing import Iterator

import pandas as pd

SYMBOL_DATA_DIR: str = './s-and-p-500-companies'

_NDX_RAW = """
ATVI ADBE AKAM ALXN GOOG GOOGL AMZN AAL AMGN ADI AAPL AMAT ADSK ADP
BIDU BIIB BMRN AVGO CELG CERN CHTR CHKP CTAS CSCO CTXS CTSH CMCSA COST
CSX CTRP XRAY DISCA DISCK DISH DLTR EBAY EA EXPE ESRX FB FAST FISV
GILD HAS HSIC HOLX ILMN INCY INTC INTU ISRG JD KLAC LRCX LBTYA LBTYK
LILA LILAK MAR MAT MXIM MCHP MU MSFT MDLZ MNST MYL NTES NFLX NCLH NVDA
NXPI ORLY PCAR PAYX PYPL QCOM REGN ROST SBAC STX SHPG SIRI SWKS SBUX
SYMC TMUS TSLA TXN KHC TSCO TRIP FOX FOXA ULTA VRSK VRTX VIAB VOD WBA
WDC XLNX
"""

NDX_CONSTITUENTS: list[str] = list(filter(None, dedent(_NDX_RAW).split()))

MY_FAVES: list[str] = [
    'TWTR', 'MOBL', 'GLD', 'LUV', 'T', 'SNAP', 'RACE', 'VSAT',
    'DATA', 'YELP', 'TWLO', 'TEAM', 'WMT', 'SHAK', 'ANET',
    'DXCM', 'MDT', 'TNDM',
]


def nasdaq_gen(data_dir: str = SYMBOL_DATA_DIR) -> Iterator[str]:
    """Yield each fully-uppercase NASDAQ symbol from `nasdaqlisted.txt`."""
    with open(f'{data_dir}/nasdaqlisted.txt') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            symbol = row[0]
            if symbol.upper() == symbol:
                yield symbol


def nasdaq(data_dir: str = SYMBOL_DATA_DIR) -> list[str]:
    return list(nasdaq_gen(data_dir))


def coin100(data_dir: str = SYMBOL_DATA_DIR) -> list[str]:
    return coin100_dataframe(data_dir)['symbol'].tolist()


def coin100_dataframe(data_dir: str = SYMBOL_DATA_DIR) -> pd.DataFrame:
    return pd.read_json(f'{data_dir}/coin100.json')


def snp_500(count: int = 500, data_dir: str = SYMBOL_DATA_DIR) -> dict[str, str]:
    """Random sample of `count` S&P 500 names from the constituents CSV."""
    ret: dict[str, str] = {}
    with open(f'{data_dir}/data/constituents.csv', 'rb') as f:
        reader = csv.DictReader(f)
        for line in reader:
            pretty = f"{line['Name']}-{line['Sector'][0:4]}"
            ret[line['Symbol']] = pretty
    return {k: ret[k] for k in random.sample(list(ret.keys()), count)}


def old_coins(data_dir: str = SYMBOL_DATA_DIR) -> pd.DataFrame:
    """Deprecated: parses the legacy `topvolumeusd.json` snapshot."""
    with open(f'{data_dir}/topvolumeusd.json') as f:
        d = json.load(f)
    d = list(map(lambda x: x.split('~'), d['Data']))
    df = pd.DataFrame(
        d, dtype=float,
        columns=['Id', 'Sh1', 'Coin', 'BaseCurrency', 'Sh2', 'Price',
                 'Timestamp', 'PercentChange', 'BaseChange', 'CoinVolume',
                 '24HCoinVolume', '24HBaseVolume', 'Low', 'High', 'Open',
                 'Exchange', 'Sh7'])
    df['MarketCap'] = df['Price'] * df['CoinVolume']
    return df
