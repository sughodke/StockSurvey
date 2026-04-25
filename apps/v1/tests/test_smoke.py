"""Smoke tests for the parked v1 workflow.

Verifies imports resolve under the new v1.* prefix. The legacy network-
dependent tests in `v1/src/v1/models/security_test.py` are intentionally
not collected (they require a live Yahoo Finance fetch); they're kept
in-place for reference but excluded by the root pytest config.
"""

from __future__ import annotations


def test_models_import():
    from v1.models import (
        directors,
        indicators,
        plotter,
        security,
        span,
        timespan,
    )
    # Key symbols are present
    assert hasattr(security, 'Security')
    assert hasattr(span, 'Span')
    assert hasattr(span, 'MACDSpan')
    assert hasattr(span, 'BBandsSpan')
    assert hasattr(directors, 'NumpyDecider')
    assert hasattr(directors, 'MACDDecider')
    assert hasattr(plotter, 'PlotMixin')
    assert hasattr(plotter, 'MACDPlotMixin')
    assert hasattr(timespan, 'AddTimeSpan')
    assert hasattr(indicators, 'RSIMixin')
    assert hasattr(indicators, 'MACDMixin')
    assert hasattr(indicators, 'BBandsMixin')
    assert hasattr(indicators, 'TheEvaluator')


def test_util_import():
    from v1.util import indicators, load_symbols, load_ticker
    # Legacy 1D indicator API
    assert callable(indicators.relative_strength)
    assert callable(indicators.moving_average)
    assert callable(indicators.moving_average_convergence)
    assert callable(indicators.bbands)
    assert callable(indicators.fibonacci_retracement)
    # Loaders
    assert callable(load_ticker.load_data)
    assert callable(load_ticker.load_crypto_data)
    # Symbol helpers
    assert callable(load_symbols.nasdaq)
    assert callable(load_symbols.coin100)


def test_scripts_import():
    from v1.scripts import finance_ndx
    constituents = list(finance_ndx.NDX_constituents)
    assert len(constituents) > 50  # NDX-100 has ~100 names
    assert 'AAPL' in constituents
    assert isinstance(finance_ndx.my_faves, list)
    assert 'GLD' in finance_ndx.my_faves
