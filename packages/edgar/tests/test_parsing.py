"""13F-HR XML parser tests + name normalization."""
from __future__ import annotations

from ss_edgar.funds import (
    NAME_TO_TICKER, name_to_ticker, normalize_issuer_name,
)
from ss_edgar.parsing import HoldingRow, parse_13f_xml


# Inline mini 13F-HR XML fixture (Berkshire-style holdings, ×1000 scaling
# era — period 2022-12-31 → scale ×1000).
FIXTURE_PRE2023 = b"""<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>119000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>915560382</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>BANK AMER CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>060505104</cusip>
    <value>33000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>1010100606</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
  </infoTable>
</informationTable>
"""


# Post-2023 era — value in raw dollars (no ×1000)
FIXTURE_POST2023 = b"""<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>2500000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>6500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""


def test_parse_pre2023_scales_value_x1000():
    """Pre-2023 reportPeriod → value column is in thousands of dollars."""
    rows = parse_13f_xml(FIXTURE_PRE2023, period_of_report='2022-12-31')
    assert len(rows) == 2
    apple = rows[0]
    assert apple.name_of_issuer == 'APPLE INC'
    assert apple.cusip == '037833100'
    assert apple.shares == 915_560_382
    # 119_000_000 thousand dollars = 119 billion dollars
    assert apple.value_dollars == 119_000_000_000.0
    assert apple.is_shares


def test_parse_post2023_no_scaling():
    """Post-2023 reportPeriod → value column is in raw dollars."""
    rows = parse_13f_xml(FIXTURE_POST2023, period_of_report='2023-12-31')
    assert len(rows) == 1
    msft = rows[0]
    assert msft.name_of_issuer == 'MICROSOFT CORP'
    assert msft.value_dollars == 2_500_000_000.0


def test_parse_empty_returns_empty_list():
    assert parse_13f_xml(b'') == []
    assert parse_13f_xml(b'<not-xml>') == []


def test_normalize_strips_punctuation_and_class_suffix():
    assert normalize_issuer_name('APPLE INC.') == 'APPLE INC'
    assert normalize_issuer_name('AMAZON.COM INC') == 'AMAZONCOM INC'
    assert normalize_issuer_name('ALPHABET INC CL A') == 'ALPHABET INC'
    assert normalize_issuer_name('ALPHABET INC -COM') == 'ALPHABET INC'
    assert normalize_issuer_name('  Berkshire Hathaway Inc.  ') == 'BERKSHIRE HATHAWAY INC'


def test_name_to_ticker_known_names():
    assert name_to_ticker('APPLE INC') == 'AAPL'
    assert name_to_ticker('Apple Inc.') == 'AAPL'
    assert name_to_ticker('MICROSOFT CORP') == 'MSFT'
    assert name_to_ticker('AMAZON COM INC') == 'AMZN'
    assert name_to_ticker('BANK AMER CORP') is None  # not in map (would need to add)
    assert name_to_ticker('Bank of America Corp') == 'BAC'  # this normalizes to BANK OF AMERICA CORP


def test_name_to_ticker_unknown_returns_none():
    assert name_to_ticker('TINY CAP NOBODY HOLDS INC') is None
    assert name_to_ticker('') is None
