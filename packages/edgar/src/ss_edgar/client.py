"""Rate-limited HTTP client for SEC EDGAR + on-disk cache.

SEC etiquette (https://www.sec.gov/os/accessing-edgar-data):
  - User-Agent must include name + email contact
  - 10 requests/second hard limit
  - 600 requests / 10 minute soft limit (we don't hit this in practice)

Cache layout under `.edgar-cache/`:
  submissions/{cik:010d}.json     — submissions list per CIK
  filings/{cik:010d}/{accession_no_no_dashes}/{filename}  — raw 13F-HR XML

Cache is content-addressed by CIK + accession; once written, never
re-fetched. No TTL — SEC filings are immutable. Manual deletion
forces refresh.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests


SEC_BASE = 'https://www.sec.gov'
SEC_DATA = 'https://data.sec.gov'
DEFAULT_USER_AGENT = 'StockSurvey research bot research@example.com'
DEFAULT_RATE_LIMIT = 10.0   # requests per second; SEC's hard limit


@dataclass(frozen=True)
class EdgarFiling:
    """One 13F-HR filing entry from a CIK's submissions list."""
    cik: int
    accession_no: str        # e.g. '0001067983-23-000027'
    filing_date: str         # YYYY-MM-DD
    period_of_report: str    # YYYY-MM-DD (the quarter-end date the filing reports on)
    primary_document: str    # filename of the main filing doc

    @property
    def accession_no_clean(self) -> str:
        return self.accession_no.replace('-', '')

    @property
    def info_table_url(self) -> str:
        """URL of the holdings information table XML.

        13F-HR filings have an `infotable.xml` (formal informationTable
        XML) inside the filing directory. The `primary_document` is
        usually a coverpage XML or HTML, not the holdings — we glob
        for `infotable.xml` separately.
        """
        return (f'{SEC_BASE}/Archives/edgar/data/{self.cik}/'
                f'{self.accession_no_clean}/infotable.xml')

    @property
    def filing_index_url(self) -> str:
        """URL of the filing's index.json — lists all files in the filing."""
        return (f'{SEC_BASE}/Archives/edgar/data/{self.cik}/'
                f'{self.accession_no_clean}/index.json')


class EdgarClient:
    """Thin wrapper around requests.Session with rate limiting + cache.

    Usage:
        client = EdgarClient(cache_dir=Path('.edgar-cache'),
                             user_agent='MyName myemail@example.com')
        filings = client.list_13f_hr_filings(cik=1067983)
        for f in filings:
            xml = client.fetch_info_table(f)
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        user_agent: str = DEFAULT_USER_AGENT,
        rate_limit_qps: float = DEFAULT_RATE_LIMIT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / 'submissions').mkdir(exist_ok=True)
        (self.cache_dir / 'filings').mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov',  # overridden per-request
        })
        self.rate_limit_qps = rate_limit_qps
        self._last_request_t = 0.0

    def _rate_limit(self) -> None:
        """Sleep just enough to honor the SEC's 10 req/s hard limit."""
        min_interval = 1.0 / self.rate_limit_qps
        elapsed = time.monotonic() - self._last_request_t
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_t = time.monotonic()

    def _request(self, url: str) -> bytes:
        self._rate_limit()
        host = url.split('/')[2]
        headers = dict(self.session.headers)
        headers['Host'] = host
        r = self.session.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.content

    def list_13f_hr_filings(
        self, cik: int, *, force_refresh: bool = False,
    ) -> list[EdgarFiling]:
        """Fetch (or read cached) submissions list for `cik`, return all
        13F-HR filing entries sorted by `period_of_report` ascending.

        Submissions list is cached at
        `.edgar-cache/submissions/{cik:010d}.json`.
        """
        cache_path = self.cache_dir / 'submissions' / f'{cik:010d}.json'
        if cache_path.exists() and not force_refresh:
            data = json.loads(cache_path.read_text())
        else:
            url = f'{SEC_DATA}/submissions/CIK{cik:010d}.json'
            raw = self._request(url)
            data = json.loads(raw)
            cache_path.write_bytes(raw)

        # Walk the recent filings table.
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        filing_dates = recent.get('filingDate', [])
        periods = recent.get('reportDate', [])
        primary_docs = recent.get('primaryDocument', [])
        out: list[EdgarFiling] = []
        for form, acc, fd, per, doc in zip(
                forms, accessions, filing_dates, periods, primary_docs):
            if form != '13F-HR':
                continue
            out.append(EdgarFiling(
                cik=cik, accession_no=acc, filing_date=fd,
                period_of_report=per, primary_document=doc,
            ))

        # Older filings live in the `files` array — each entry is a
        # path to a separate JSON file with more historical filings.
        for older_block in data.get('filings', {}).get('files', []):
            block_name = older_block.get('name')
            if not block_name:
                continue
            block_path = self.cache_dir / 'submissions' / block_name
            if not block_path.exists() or force_refresh:
                url = f'{SEC_DATA}/submissions/{block_name}'
                raw = self._request(url)
                block_path.write_bytes(raw)
            block_data = json.loads(block_path.read_text())
            forms = block_data.get('form', [])
            accessions = block_data.get('accessionNumber', [])
            filing_dates = block_data.get('filingDate', [])
            periods = block_data.get('reportDate', [])
            primary_docs = block_data.get('primaryDocument', [])
            for form, acc, fd, per, doc in zip(
                    forms, accessions, filing_dates, periods, primary_docs):
                if form != '13F-HR':
                    continue
                out.append(EdgarFiling(
                    cik=cik, accession_no=acc, filing_date=fd,
                    period_of_report=per, primary_document=doc,
                ))

        out.sort(key=lambda f: (f.period_of_report, f.filing_date))
        return out

    def fetch_info_table(
        self,
        filing: EdgarFiling,
        *,
        force_refresh: bool = False,
    ) -> bytes | None:
        """Download (or read cached) the holdings infotable XML for one filing.

        Returns the XML bytes, or None if no infotable is present (some
        13F-NT/legacy filings lack one). Newer 13F-HR filings always
        have `infotable.xml`; older filings may use a different name —
        fall back to the filing-index lookup if the canonical path is
        missing.
        """
        cache_path = (self.cache_dir / 'filings' /
                      f'{filing.cik:010d}' /
                      filing.accession_no_clean /
                      'infotable.xml')
        if cache_path.exists() and not force_refresh:
            data = cache_path.read_bytes()
            return data if data else None

        # Try the canonical infotable.xml path first.
        try:
            data = self._request(filing.info_table_url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return data
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code != 404:
                raise

        # Fallback: walk the filing index for a likely XML. Newer 13F-
        # HR filings (post-2022) use a numeric filename like
        # `50240.xml` for the holdings infotable; older ones use
        # `infotable.xml` or similar. Strategy: take any .xml that is
        # NOT `primary_doc.xml` (which is the cover page) and try
        # parsing it. We try each candidate until one yields >0
        # holdings.
        try:
            idx = json.loads(self._request(filing.filing_index_url))
        except requests.HTTPError:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b'')
            return None

        candidates: list[str] = []
        for item in idx.get('directory', {}).get('item', []):
            name = item.get('name', '')
            if not name.lower().endswith('.xml'):
                continue
            if name.lower() == 'primary_doc.xml':
                continue   # this is the form-cover XML, not holdings
            candidates.append(name)
        # Prefer files with 'info' in the name first (legacy), then
        # numeric names (newer schema).
        candidates.sort(key=lambda n: (
            0 if 'info' in n.lower() else 1,
            len(n),
            n,
        ))

        for candidate in candidates:
            url = (f'{SEC_BASE}/Archives/edgar/data/{filing.cik}/'
                   f'{filing.accession_no_clean}/{candidate}')
            try:
                data = self._request(url)
            except requests.HTTPError:
                continue
            # Sanity: the holdings XML contains '<infoTable' or
            # 'informationTable'. The cover-page XML doesn't.
            if b'infoTable' in data or b'informationTable' in data:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
                return data
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b'')
        return None


__all__ = ['EdgarClient', 'EdgarFiling', 'SEC_BASE', 'SEC_DATA']
