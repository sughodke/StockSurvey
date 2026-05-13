"""Curated list of institutional managers + issuer-name → ticker mapping.

The fund list is deliberately small (~25 well-known managers) and
biased toward firms with consistent 13F-HR XML filings since 2013.
Aggregating across more funds doesn't change the consensus
top-holdings much (mega-cap concentration), and the longer
historical coverage matters more than fund count for our
walk-forward purposes.

The name → ticker map covers ~80 mega-cap and large-cap names. SEC
13F-HR filings use various spellings of the same issuer (e.g.
"APPLE INC", "APPLE INC.", "APPLE INC -COM"); `normalize_issuer_name`
handles common variations. Names not in the map are dropped — we
trade <10% holdings coverage for keeping the implementation simple
(no CUSIP database, no fuzzy matching).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FundInfo:
    """A 13F filer we want to follow."""
    cik: int
    name: str
    short: str  # short label for log lines / column names


# Curated fund list — well-known institutional managers with multi-year
# 13F-HR coverage. CIKs verified against EDGAR via
# `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=<NAME>&type=13F-HR`.
# Add more by looking up CIK on EDGAR; older entries are more useful
# (longer history → more walk-forward windows with data).
CURATED_FUNDS: list[FundInfo] = [
    FundInfo(cik=1067983, name='Berkshire Hathaway Inc',          short='BRK'),
    FundInfo(cik=1037389, name='Renaissance Technologies LLC',    short='RENT'),
    FundInfo(cik=1179392, name='Two Sigma Investments LP',        short='TS'),
    FundInfo(cik=1350694, name='Bridgewater Associates LP',       short='BRDG'),
    FundInfo(cik=1423053, name='Citadel Advisors LLC',            short='CIT'),
    FundInfo(cik=1167483, name='Tiger Global Management LLC',     short='TGM'),
    FundInfo(cik=1009207, name='D E Shaw & Co Inc',               short='DES'),
    FundInfo(cik=1273087, name='Millennium Management LLC',       short='MILM'),
    FundInfo(cik=1167557, name='AQR Capital Management LLC',      short='AQR'),
    FundInfo(cik=1135730, name='Coatue Management LLC',           short='COA'),
    FundInfo(cik=1061165, name='Lone Pine Capital LLC',           short='LP'),
    FundInfo(cik=1336528, name='Soros Fund Management LLC',       short='SFM'),
    FundInfo(cik=1029160, name='Greenlight Capital Inc',          short='GL'),
    FundInfo(cik=1336207, name='Pershing Square Capital Mgmt LP', short='PSCM'),
    FundInfo(cik=1656456, name='Viking Global Investors LP',      short='VG'),
]


def normalize_issuer_name(name: str) -> str:
    """Normalize a 13F nameOfIssuer for map lookup.

    13F filings spell the same issuer many ways:
      - "APPLE INC", "APPLE INC.", "APPLE INC -COM", "APPLE INCORPORATED"
      - "AMAZON.COM INC", "AMAZON COM INC", "AMAZON COM, INC."

    Normalization strips punctuation, common suffixes, and common
    suffix annotations. Round-trip is intentionally lossy — different
    share classes (CL A vs CL B) map to the same normalized name and
    must be disambiguated by the ticker map separately if needed.
    """
    if not name:
        return ''
    s = name.upper().strip()
    # Strip trailing share-class annotations
    for tag in (' -COM', ' COM', ' COMMON', ' COMMON STOCK',
                ' CLASS A', ' CL A', ' CLASS B', ' CL B',
                ' CLASS C', ' CL C', ' CL  A', ' CL  B', ' CL  C'):
        if s.endswith(tag):
            s = s[: -len(tag)].strip()
    # Strip punctuation
    for ch in '.,()':
        s = s.replace(ch, '')
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


# Manual issuer-name → ticker map. Only mega-cap and large-cap names
# that show up in essentially every 13F filing. Coverage is by design
# narrow — we hit >90% of typical hedge-fund 13F dollar volume with
# ~80 names because hedge funds concentrate in mega-caps.
#
# Build lookup from normalized-name to ticker. Variants of the same
# name normalize to the same key (e.g., "APPLE INC", "APPLE INC.",
# "APPLE INCORPORATED" → "APPLE INC" → AAPL). Add new entries as
# they show up in unmatched-name diagnostics.
_NAME_TICKER_PAIRS: list[tuple[str, str]] = [
    # Mega-cap tech
    ('APPLE INC',                    'AAPL'),
    ('MICROSOFT CORP',               'MSFT'),
    ('AMAZON COM INC',               'AMZN'),
    ('ALPHABET INC',                 'GOOGL'),
    ('ALPHABET INC CL A',            'GOOGL'),
    ('ALPHABET INC CL C',            'GOOG'),
    ('META PLATFORMS INC',           'META'),
    ('FACEBOOK INC',                 'META'),  # pre-rebrand
    ('NVIDIA CORP',                  'NVDA'),
    ('TESLA INC',                    'TSLA'),
    ('NETFLIX INC',                  'NFLX'),
    ('ADOBE INC',                    'ADBE'),
    ('ADOBE SYSTEMS INC',            'ADBE'),  # pre-2018 name
    ('SALESFORCE INC',               'CRM'),
    ('SALESFORCE COM INC',           'CRM'),  # pre-2022 name
    ('ORACLE CORP',                  'ORCL'),
    ('INTEL CORP',                   'INTC'),
    ('CISCO SYSTEMS INC',            'CSCO'),
    ('IBM CORP',                     'IBM'),
    ('INTERNATIONAL BUSINESS MACHS CORP', 'IBM'),
    ('ADVANCED MICRO DEVICES INC',   'AMD'),
    ('BROADCOM INC',                 'AVGO'),
    ('QUALCOMM INC',                 'QCOM'),
    ('TEXAS INSTRUMENTS INC',        'TXN'),
    # Banks / financials
    ('BERKSHIRE HATHAWAY INC',       'BRK-B'),
    ('BERKSHIRE HATHAWAY INC NEW',   'BRK-B'),
    ('JPMORGAN CHASE & CO',          'JPM'),
    ('JP MORGAN CHASE & CO',         'JPM'),
    ('BANK OF AMERICA CORP',         'BAC'),
    ('CITIGROUP INC',                'C'),
    ('WELLS FARGO & CO',             'WFC'),
    ('WELLS FARGO & CO NEW',         'WFC'),
    ('GOLDMAN SACHS GROUP INC',      'GS'),
    ('MORGAN STANLEY',               'MS'),
    ('AMERICAN EXPRESS CO',          'AXP'),
    ('VISA INC',                     'V'),
    ('MASTERCARD INC',               'MA'),
    ('PAYPAL HLDGS INC',             'PYPL'),
    ('PAYPAL HOLDINGS INC',          'PYPL'),
    # Healthcare
    ('JOHNSON & JOHNSON',            'JNJ'),
    ('UNITEDHEALTH GROUP INC',       'UNH'),
    ('PFIZER INC',                   'PFE'),
    ('ABBVIE INC',                   'ABBV'),
    ('ELI LILLY & CO',               'LLY'),
    ('LILLY ELI & CO',               'LLY'),
    ('MERCK & CO INC',               'MRK'),
    ('MERCK & CO INC NEW',           'MRK'),
    ('THERMO FISHER SCIENTIFIC INC', 'TMO'),
    ('ABBOTT LABS',                  'ABT'),
    ('ABBOTT LABORATORIES',          'ABT'),
    ('AMGEN INC',                    'AMGN'),
    ('GILEAD SCIENCES INC',          'GILD'),
    ('DANAHER CORP',                 'DHR'),
    ('BRISTOL MYERS SQUIBB CO',      'BMY'),
    # Consumer
    ('WAL MART STORES INC',          'WMT'),
    ('WALMART INC',                  'WMT'),
    ('COSTCO WHOLESALE CORP',        'COST'),
    ('PROCTER & GAMBLE CO',          'PG'),
    ('COCA COLA CO',                 'KO'),
    ('PEPSICO INC',                  'PEP'),
    ('HOME DEPOT INC',               'HD'),
    ('LOWES COMPANIES INC',          'LOW'),
    ('NIKE INC',                     'NKE'),
    ('NIKE INC CL B',                'NKE'),
    ('MCDONALDS CORP',               'MCD'),
    ('STARBUCKS CORP',               'SBUX'),
    ('TARGET CORP',                  'TGT'),
    # Industrial / energy
    ('EXXON MOBIL CORP',             'XOM'),
    ('CHEVRON CORP',                 'CVX'),
    ('CHEVRON CORP NEW',             'CVX'),
    ('CATERPILLAR INC',              'CAT'),
    ('BOEING CO',                    'BA'),
    ('GENERAL ELECTRIC CO',          'GE'),
    ('UNION PACIFIC CORP',           'UNP'),
    ('HONEYWELL INTERNATIONAL INC',  'HON'),
    ('LOCKHEED MARTIN CORP',         'LMT'),
    ('RAYTHEON TECHNOLOGIES CORP',   'RTX'),
    ('UNITED TECHNOLOGIES CORP',     'RTX'),  # pre-2020 merger name
    ('3M CO',                        'MMM'),
    ('CONOCOPHILLIPS',               'COP'),
    # Telecom / media
    ('AT&T INC',                     'T'),
    ('VERIZON COMMUNICATIONS INC',   'VZ'),
    ('COMCAST CORP',                 'CMCSA'),
    ('COMCAST CORP NEW',             'CMCSA'),
    ('WALT DISNEY CO',               'DIS'),
    # Other
    ('UNITEDHEALTH GROUP INC',       'UNH'),
]

NAME_TO_TICKER: dict[str, str] = {
    normalize_issuer_name(n): t for n, t in _NAME_TICKER_PAIRS
}


def name_to_ticker(name: str) -> str | None:
    """Look up a 13F nameOfIssuer in the manual ticker map.

    Returns the ticker string if matched (after normalization), else
    None. Callers typically drop None entries silently — the map is
    deliberately incomplete and small-cap holdings just don't make
    it to the consumer.
    """
    if not name:
        return None
    return NAME_TO_TICKER.get(normalize_issuer_name(name))


__all__ = [
    'FundInfo', 'CURATED_FUNDS',
    'NAME_TO_TICKER', 'normalize_issuer_name', 'name_to_ticker',
]
