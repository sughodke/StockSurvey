from time import sleep
from textwrap import dedent
import finance_work2 as fw

"""
Run backtrading algorithm against all NASDAQ-100 equities
"""

NDX_constituents = """
ATVI
ADBE
AKAM
ALXN
GOOG
GOOGL
AMZN
AAL
AMGN
ADI
AAPL
AMAT
ADSK
ADP
BIDU
BIIB
BMRN
AVGO
CA
CELG
CERN
CHTR
CHKP
CTAS
CSCO
CTXS
CTSH
CMCSA
COST
CSX
CTRP
XRAY
DISCA
DISCK
DISH
DLTR
EBAY
EA
EXPE
ESRX
FB
FAST
FISV
GILD
HAS
HSIC
HOLX
ILMN
INCY
INTC
INTU
ISRG
JD
KLAC
LRCX
LBTYA
LBTYK
LILA
LILAK
LVNTA
QVCA
MAR
MAT
MXIM
MCHP
MU
MSFT
MDLZ
MNST
MYL
NTES
NFLX
NCLH
NVDA
NXPI
ORLY
PCAR
PAYX
PYPL
QCOM
REGN
ROST
SBAC
STX
SHPG
SIRI
SWKS
SBUX
SYMC
TMUS
TSLA
TXN
KHC
PCLN
TSCO
TRIP
FOX
FOXA
ULTA
VRSK
VRTX
VIAB
VOD
WBA
WDC
XLNX
YHOO
"""

NDX_constituents = dedent(NDX_constituents).split('\n')
NDX_constituents = filter(None, NDX_constituents)

for c in NDX_constituents:
    print('\n## Plotting {}'.format(c))
    try:
        fw.go(ticker=c)
    except Exception as e:
        print('skipping... %s' % e)
    sleep(1)
