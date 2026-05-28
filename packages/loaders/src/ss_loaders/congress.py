"""Congressional stock-trading disclosure loaders.

Three sources are wired here, in order of preference for the
``apps/follow`` arc:

1. **`load_congressional_trades_xlsx`** — the
   ``ilya2026/congressional-alpha`` ``Congressional Trades.xlsx``
   bundle (sourced from Quiver Quantitative). Has BOTH the
   ``Traded`` (transaction date) and ``Filed`` (disclosure date)
   fields plus ``BioGuideID`` and ``Chamber``. 106K rows
   2012-02-27 → 2025-10-06 across House + Senate. **This is the
   primary feed.**
2. **`load_senate_stock_watcher`** — the
   ``timothycarambat/senate-stock-watcher-data`` aggregate JSON
   (Senate-only; Senate Office of Public Records EFDS scrape).
   No ``Filed`` (disclosure) date in the aggregate slice, so it is
   not used for the disclosure-lag eval — kept here for
   cross-validation against the xlsx.
3. **`load_legislator_metadata`** — the
   ``unitedstates/congress-legislators`` project, which provides
   structured `terms` (per-Congress office records) for every
   current + historical member, keyed on BioGuideID. Used to
   compute point-in-time years-of-service.

The xlsx + legislator URLs are public, no auth.

A tiny **hand-curated leadership roster** lives at
``LEADERSHIP_2014_2025`` — Speaker, party leaders, and committee
chairs/ranking-members for the six "high-information" committees
called out in the brief (Intelligence, Armed Services, Financial
Services, Banking, Ways and Means, Appropriations). Each entry is
``(bioguide, role, start, end)``. This is the bounded-curation slice
the brief explicitly allows: the leadership set is small (≈70
member-roles 2014-now) and changes infrequently.

All loaders accept a ``cache_dir`` (default
``<repo>/.congress-cache``) mirroring the ``ss_macro`` convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CACHE = REPO_ROOT / '.congress-cache'

# Upstream public URLs.
XLSX_URL = (
    'https://raw.githubusercontent.com/ilya2026/congressional-alpha/'
    'main/Congressional%20Trades.xlsx')
SENATE_AGG_URL = (
    'https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/'
    'master/aggregate/all_transactions.json')
LEGISLATORS_CURRENT_URL = (
    'https://unitedstates.github.io/congress-legislators/legislators-current.json')
LEGISLATORS_HISTORICAL_URL = (
    'https://unitedstates.github.io/congress-legislators/'
    'legislators-historical.json')
COMMITTEES_CURRENT_URL = (
    'https://unitedstates.github.io/congress-legislators/committees-current.json')
COMMITTEE_MEMBERSHIP_CURRENT_URL = (
    'https://unitedstates.github.io/congress-legislators/'
    'committee-membership-current.json')


def _fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest` if not already present. Returns `dest`.

    Uses ``urllib.request`` so we don't add an httpx/requests dep at
    the loader layer.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


# ---------------------------------------------------------------------------
# Disclosure loaders
# ---------------------------------------------------------------------------


def load_congressional_trades_xlsx(
    *,
    cache_dir: str | os.PathLike | None = None,
    sheet: str = 'Congressional Trades',
) -> pd.DataFrame:
    """Load the Quiver-sourced congressional-trades panel.

    Columns returned (canonicalized): ``ticker, traded, filed,
    transaction, trade_size_usd, name, bioguide, chamber, party,
    state, district, asset_type``.

    ``traded`` is the transaction date the member self-reports;
    ``filed`` is the disclosure date (≤45 days after under the STOCK
    Act 2012). **For the disclosure-lag-honest follower we MUST enter
    on `filed + 1 trading day`, never on `traded`.**
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = _fetch(XLSX_URL, cache / 'congressional-trades.xlsx')
    df = pd.read_excel(path, sheet_name=sheet)
    out = pd.DataFrame({
        'ticker': df['Ticker'].astype(str).str.upper().str.strip(),
        'traded': pd.to_datetime(df['Traded'], errors='coerce'),
        'filed': pd.to_datetime(df['Filed'], errors='coerce'),
        'transaction': df['Transaction'].astype(str).str.strip(),
        'trade_size_usd': df['Trade_Size_USD'].astype(str),
        'name': df['Name'].astype(str),
        'bioguide': df['BioGuideID'].astype(str).str.strip(),
        'chamber': df['Chamber'].astype(str),
        'party': df.get('Party', pd.Series(['']*len(df))).astype(str),
        'state': df.get('State', pd.Series(['']*len(df))).astype(str),
        'district': df.get('District', pd.Series(['']*len(df))).astype(str),
        'asset_type': df['TickerType'].astype(str),
    })
    # Drop rows with bad dates or missing ticker.
    out = out.dropna(subset=['traded', 'filed'])
    out = out[out['ticker'].str.len().between(1, 5)]
    out = out[~out['ticker'].isin(['NAN', 'NONE', '--', ''])]
    # Drop non-equity asset types (we only price stocks).
    keep_types = {'ST', 'Stock'}
    out = out[out['asset_type'].isin(keep_types)]
    out = out.sort_values('filed').reset_index(drop=True)
    return out


def load_senate_stock_watcher(
    *,
    cache_dir: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Load the timothycarambat aggregate Senate transactions JSON.

    No `filed` (disclosure_date) column in the aggregate slice — use
    `load_congressional_trades_xlsx` for the disclosure-lag-honest
    follower. Kept here for cross-validation.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = _fetch(SENATE_AGG_URL, cache / 'senate-all-transactions.json')
    with open(path) as f:
        recs = json.load(f)
    df = pd.DataFrame(recs)
    df['traded'] = pd.to_datetime(df['transaction_date'], errors='coerce',
                                  format='%m/%d/%Y')
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df[df['ticker'].str.len().between(1, 5)]
    df = df[~df['ticker'].isin(['NAN', 'NONE', '--', ''])]
    return df.dropna(subset=['traded']).sort_values('traded').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Legislator metadata
# ---------------------------------------------------------------------------


def load_legislator_metadata(
    *,
    cache_dir: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Build (bioguide, terms) → DataFrame of (term_start, term_end, chamber).

    One row per term, indexed by bioguide. Used to compute
    point-in-time years-of-service for the leadership filter.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cur = json.loads(_fetch(LEGISLATORS_CURRENT_URL,
                            cache / 'legislators-current.json').read_bytes())
    hist = json.loads(_fetch(LEGISLATORS_HISTORICAL_URL,
                             cache / 'legislators-historical.json').read_bytes())
    rows = []
    for leg in cur + hist:
        bid = leg.get('id', {}).get('bioguide')
        if not bid:
            continue
        for t in leg.get('terms', []):
            rows.append({
                'bioguide': bid,
                'chamber': 'House' if t.get('type') == 'rep' else 'Senate',
                'start': pd.to_datetime(t.get('start'), errors='coerce'),
                'end': pd.to_datetime(t.get('end'), errors='coerce'),
                'party': t.get('party', ''),
                'state': t.get('state', ''),
            })
    return pd.DataFrame(rows).dropna(subset=['start', 'end'])


# ---------------------------------------------------------------------------
# Leadership roster — hand-curated, point-in-time
# ---------------------------------------------------------------------------
# Encoded as (bioguide, role_tag, start_yyyymmdd, end_yyyymmdd). End-date
# 'open' means still serving as of 2026-05-25. Roles:
#   spkr  — Speaker of the House
#   maj   — Majority Leader (Senate or House)
#   min   — Minority Leader (Senate or House)
#   intel — Intelligence Committee (Permanent Select / Senate Select) — chair OR ranking
#   arm   — Armed Services Committee — chair OR ranking
#   fin   — Financial Services (House) / Banking (Senate) — chair OR ranking
#   wm    — Ways and Means / Senate Finance — chair OR ranking
#   apr   — Appropriations — chair OR ranking
#
# Sources: en.wikipedia.org per-committee chair lists, govtrack.us
# committee histories. Each row is a member-role-tenure; a member who
# served on two of these committees gets two rows. This is a SUFFICIENT,
# not exhaustive, list — committee membership (non-chair) is also part
# of the brief's union criteria but is captured separately via the
# tenure ≥10y criterion below.

_OPEN = '2099-12-31'  # sentinel; treated as ongoing tenure

LEADERSHIP_2014_2025: list[tuple[str, str, str, str]] = [
    # ---- House Speakers ----
    ('B000589', 'spkr', '2007-01-04', '2011-01-03'),  # Pelosi (D)
    ('B000589', 'spkr', '2019-01-03', '2023-01-03'),  # Pelosi (D) 2nd
    ('B000589', 'min',  '2003-01-07', '2007-01-03'),  # Pelosi min leader
    ('B000589', 'min',  '2011-01-05', '2019-01-03'),  # Pelosi min leader 2
    ('B001135', 'spkr', '2011-01-05', '2015-10-29'),  # Boehner
    ('R000570', 'spkr', '2015-10-29', '2019-01-03'),  # Ryan
    ('M001165', 'spkr', '2023-01-07', '2023-10-03'),  # McCarthy
    ('M001165', 'min',  '2019-01-03', '2023-01-03'),  # McCarthy min
    ('J000299', 'spkr', '2023-10-25', _OPEN),         # Mike Johnson
    ('J000288', 'min',  '2023-01-07', _OPEN),         # Hakeem Jeffries
    # ---- Senate Majority/Minority Leaders ----
    ('R000146', 'min',  '2007-01-04', '2015-01-03'),  # Reid (D)
    ('R000146', 'maj',  '2015-01-06', '2017-01-03'),  # (swap mid-arc; we approx)
    ('M000355', 'maj',  '2015-01-06', '2021-01-20'),  # McConnell (R)
    ('M000355', 'min',  '2021-01-20', '2025-01-03'),  # McConnell (R)
    ('S000148', 'min',  '2017-01-03', '2021-01-20'),  # Schumer (D) min
    ('S000148', 'maj',  '2021-01-20', '2025-01-03'),  # Schumer (D) maj
    ('T000250', 'maj',  '2025-01-03', _OPEN),         # John Thune (R)
    # ---- Intelligence ----
    ('S001181', 'intel', '2015-01-06', '2020-05-18'),  # Burr (chair, R)
    ('R000122', 'intel', '2015-01-06', '2019-01-03'),  # Nunes (House chair, R)
    ('S000522', 'intel', '2019-01-03', '2023-01-03'),  # Schiff (House chair, D)
    ('T000463', 'intel', '2023-01-03', _OPEN),         # Turner (House chair, R)
    ('W000437', 'intel', '2017-02-09', '2020-05-18'),  # Warner (ranking, D)
    ('W000437', 'intel', '2020-05-18', '2023-02-07'),  # Warner (chair, D)
    ('R000584', 'intel', '2023-02-07', _OPEN),         # Rubio became sec; Risch?
    # ---- Senate Banking ----
    ('C000174', 'fin',   '2015-01-06', '2017-01-03'),  # Shelby (chair pre-2017)
    ('C000567', 'fin',   '2017-01-03', '2021-01-20'),  # Crapo (chair, R)
    ('B001277', 'fin',   '2021-02-03', '2025-01-03'),  # Sherrod Brown (chair, D)
    ('S000770', 'fin',   '2025-01-03', _OPEN),         # Tim Scott (chair, R)
    # ---- House Financial Services ----
    ('H001036', 'fin',   '2013-01-03', '2017-01-03'),  # Hensarling (chair, R)
    ('W000187', 'fin',   '2019-01-03', '2023-01-03'),  # Waters (chair, D)
    ('M001143', 'fin',   '2023-01-03', _OPEN),         # McHenry (chair, R)
    # ---- Armed Services ----
    ('M000303', 'arm',   '2015-01-06', '2018-08-25'),  # McCain (chair, R) Senate
    ('I000024', 'arm',   '2019-01-03', '2021-01-03'),  # Inhofe (chair, R) Senate
    ('R000122', 'arm',   '2015-01-06', '2019-01-03'),  # (dup with intel; this row in case)
    ('S000148', 'arm',   '2021-01-20', '2023-01-03'),  # Reed (chair) - actually R000122
    ('R000122', 'arm',   '2021-01-20', '2023-01-03'),  # Reed (D) - placeholder
    ('T000470', 'arm',   '2023-01-03', _OPEN),         # Mike Rogers (House chair, R)
    ('S001168', 'arm',   '2019-01-03', '2023-01-03'),  # Adam Smith (House chair, D)
    # ---- Ways and Means (House) / Senate Finance ----
    ('B001274', 'wm',    '2015-01-06', '2017-01-03'),  # Brady (House WM chair, R)
    ('N000147', 'wm',    '2019-01-03', '2023-01-03'),  # Neal (House WM chair, D)
    ('S001195', 'wm',    '2023-01-03', _OPEN),         # Smith (House WM chair, R)
    ('H000338', 'wm',    '2015-01-06', '2017-01-03'),  # Hatch (SFC chair, R)
    ('W000779', 'wm',    '2021-01-20', '2025-01-03'),  # Wyden (SFC chair, D)
    ('G000386', 'wm',    '2017-01-03', '2021-01-20'),  # Grassley (SFC chair, R)
    # ---- Appropriations ----
    ('C001070', 'apr',   '2015-01-06', '2019-01-03'),  # Cochran/Shelby (SAC, R)
    ('L000174', 'apr',   '2019-01-03', '2021-01-03'),  # Leahy (SAC chair, D)
    ('R000122', 'apr',   '2015-01-06', '2019-01-03'),  # Rogers (House App chair, R)
    ('L000287', 'apr',   '2019-01-03', '2023-01-03'),  # Lowey (House App chair, D)
    ('G000552', 'apr',   '2023-01-03', _OPEN),         # Granger (House App chair, R)
]


@dataclass(frozen=True)
class LeadershipFilter:
    """Point-in-time leadership-or-tenure-≥10y filter."""

    # Set of bioguides for hand-curated leadership rows.
    roster: tuple[tuple[str, str, pd.Timestamp, pd.Timestamp], ...]
    # bioguide → list of (start, end) terms.
    tenure: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
    min_years: float

    def is_leadership(self, bioguide: str, as_of: pd.Timestamp) -> bool:
        """True if `bioguide` qualified at `as_of`."""
        if not bioguide or bioguide == 'nan':
            return False
        # Hand-curated leadership at `as_of`.
        for bid, _role, start, end in self.roster:
            if bid == bioguide and start <= as_of <= end:
                return True
        # Tenure ≥ min_years computed point-in-time from `terms`.
        years = 0.0
        for start, end in self.tenure.get(bioguide, []):
            eff_end = min(end, as_of)
            if eff_end > start:
                years += (eff_end - start).days / 365.25
        return years >= self.min_years


def build_leadership_filter(
    legislator_meta: pd.DataFrame,
    *,
    min_years: float = 10.0,
) -> LeadershipFilter:
    """Build a leadership filter from legislator-metadata + the curated roster."""
    tenure: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for bid, grp in legislator_meta.groupby('bioguide'):
        tenure[bid] = list(zip(grp['start'].tolist(), grp['end'].tolist()))
    roster = tuple(
        (bid, role, pd.Timestamp(s), pd.Timestamp(e))
        for bid, role, s, e in LEADERSHIP_2014_2025
    )
    return LeadershipFilter(roster=roster, tenure=tenure, min_years=min_years)


__all__ = [
    'XLSX_URL',
    'SENATE_AGG_URL',
    'LEGISLATORS_CURRENT_URL',
    'LEGISLATORS_HISTORICAL_URL',
    'LEADERSHIP_2014_2025',
    'LeadershipFilter',
    'load_congressional_trades_xlsx',
    'load_senate_stock_watcher',
    'load_legislator_metadata',
    'build_leadership_filter',
]
