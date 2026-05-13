"""13F-HR informationTable XML parser.

Schema reference:
https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F-HR

Each `<informationTable>` contains zero or more `<infoTable>` rows
with these fields (we extract a subset):

    nameOfIssuer            text — the issuer's company name
    titleOfClass            text — share class (COM, COM CL A, ...)
    cusip                   9-digit CUSIP
    value                   integer dollars (×1000 in older filings;
                            unscaled in post-2022 filings — the
                            `lessThan5pct` schema migration)
    shrsOrPrnAmt
        sshPrnamt           integer share count (or principal $)
        sshPrnamtType       'SH' for shares, 'PRN' for principal
    investmentDiscretion    text — 'SOLE', 'DEFINED', 'OTR'
    votingAuthority         (optional: sole / shared / none)

The XML is namespaced (`http://www.sec.gov/edgar/document/thirteenf/informationtable`)
but the parser handles both with and without namespace prefixes —
older filings vary in their use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class HoldingRow:
    """One holding from one 13F filing."""
    name_of_issuer: str
    title_of_class: str
    cusip: str
    value_dollars: float       # already scaled — see parse_13f_xml docstring
    shares: int
    share_type: str            # 'SH' or 'PRN'

    @property
    def is_shares(self) -> bool:
        return self.share_type == 'SH'


def _strip_ns(tag: str) -> str:
    """Remove `{namespace}` prefix from an ElementTree tag if present."""
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _find_text(elem: ET.Element, name: str) -> str:
    """Find the first descendant whose local-name matches `name`."""
    for child in elem.iter():
        if _strip_ns(child.tag) == name:
            return (child.text or '').strip()
    return ''


def _find_int(elem: ET.Element, name: str, default: int = 0) -> int:
    s = _find_text(elem, name)
    if not s:
        return default
    try:
        return int(s.replace(',', ''))
    except ValueError:
        return default


def parse_13f_xml(
    xml_bytes: bytes, *,
    period_of_report: str | None = None,
) -> list[HoldingRow]:
    """Parse a 13F-HR informationTable XML into `HoldingRow` records.

    SEC scaling note: prior to FY2022, `<value>` was reported in
    *thousands of dollars* (per the legacy 13F instructions). After
    the `lessThan5pct` schema migration in 2022, `<value>` is the
    full dollar amount. The migration date varies slightly by filing
    type — pre-2023 filings under the old schema all reported in
    thousands. We use **`period_of_report < '2023-01-01'` → ×1000**
    to upscale, then everything is in raw dollars.

    Pass `period_of_report` (YYYY-MM-DD) to enable the upscaling.
    Without it, no scaling is applied and the caller must be aware
    that pre-2023 values are in thousands.
    """
    if not xml_bytes:
        return []
    # Strip the XML declaration if present and any BOM.
    text = xml_bytes.decode('utf-8', errors='replace').lstrip('﻿')
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # Some older filings have malformed namespace declarations;
        # try a permissive recovery by stripping namespace prefixes.
        text_clean = re.sub(r'xmlns(:\w+)?="[^"]*"', '', text)
        text_clean = re.sub(r'<(\w+):', r'<', text_clean)
        text_clean = re.sub(r'</(\w+):', r'</', text_clean)
        try:
            root = ET.fromstring(text_clean)
        except ET.ParseError:
            return []

    # Pre-2023 filings reported `<value>` in thousands of dollars.
    scale = 1000.0
    if period_of_report:
        try:
            year = int(period_of_report.split('-')[0])
            if year >= 2023:
                scale = 1.0
        except (ValueError, IndexError):
            pass

    out: list[HoldingRow] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) != 'infoTable':
            continue
        name = _find_text(elem, 'nameOfIssuer')
        title = _find_text(elem, 'titleOfClass')
        cusip = _find_text(elem, 'cusip')
        value = _find_int(elem, 'value')
        # shrsOrPrnAmt block has sshPrnamt + sshPrnamtType
        shares = 0
        share_type = ''
        for child in elem.iter():
            if _strip_ns(child.tag) == 'shrsOrPrnAmt':
                shares = _find_int(child, 'sshPrnamt')
                share_type = _find_text(child, 'sshPrnamtType')
                break
        out.append(HoldingRow(
            name_of_issuer=name,
            title_of_class=title,
            cusip=cusip,
            value_dollars=float(value) * scale,
            shares=shares,
            share_type=share_type,
        ))
    return out


__all__ = ['HoldingRow', 'parse_13f_xml']
