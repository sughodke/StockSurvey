"""Tests for the vol-v3 live scaffold.

Covers the pieces that have actual implementations:
  - persist:    JSON round-trip + validate()
  - inference:  predict_iv_rv_gap, select_top_k, gate_fires
  - iv_compute: Black-Scholes inversion round-trips, ATM-IV synth
  - strangle:   end-to-end build on a fixture chain
  - live:       risk rails (kill-switch, gate-closed abort) with mocks

The chain-query layer and the Alpaca multi-leg submission are
deliberately NotImplementedError in this scaffold; the tests confirm
the orchestration calls them and surfaces the error cleanly rather
than silently no-op'ing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol.inference import gate_fires, predict_iv_rv_gap, select_top_k
from vol.iv_compute import (
    ChainQuote, atm_iv_from_chain, bs_call_price, bs_delta_call, bs_delta_put,
    bs_put_price, bs_vega, build_feature_row, implied_vol_call,
    implied_vol_put, realized_vol_from_bars,
)
from vol.persist import (
    LIVE_FEATURE_NAMES, StranglesConfig, VolCheckpoint, load_checkpoint,
    save_checkpoint, validate,
)
from vol.strangle import build_short_strangle


# --------------------------------------------------------------- persist

def _sample_checkpoint() -> VolCheckpoint:
    return VolCheckpoint(
        feature_names=list(LIVE_FEATURE_NAMES),
        coefs=[+0.0118, +0.0376, -0.0073, -0.0112, +0.0040],
        feat_mean=[1.05, 0.0, 0.0, 0.0],
        feat_std=[0.4, 1.0, 0.05, 0.05],
        universe=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META'],
        gate_fred_series='VIXCLS',
        gate_lookback_trading_days=126,
        top_k=3,
        strangle=StranglesConfig(),
        train_period='2019-10-14 → 2023-07-28',
        val_period='2023-08-01 → 2026-04-30',
        val_pearson_r=0.165,
        n_obs_oos=33,
        oos_ann_sharpe=2.822,
        oos_deflated_t=5.549,
        notes='unit-test fixture',
    )


def test_persist_round_trip(tmp_path: Path):
    cp = _sample_checkpoint()
    path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, path)
    loaded = load_checkpoint(path)
    assert loaded.coefs == cp.coefs
    assert loaded.universe == cp.universe
    assert loaded.gate_lookback_trading_days == 126
    assert loaded.strangle.target_delta_call == 0.20


def test_persist_drops_unknown_keys(tmp_path: Path):
    """Forward-compatible: a future version can add keys without
    breaking the loader."""
    cp = _sample_checkpoint()
    path = tmp_path / 'vol.json'
    save_checkpoint(cp, path)
    raw = json.loads(path.read_text())
    raw['unknown_future_field'] = 'should-be-dropped'
    path.write_text(json.dumps(raw))
    loaded = load_checkpoint(path)
    assert loaded.coefs == cp.coefs


def test_validate_catches_bad_features():
    cp = _sample_checkpoint()
    bad = VolCheckpoint(
        **{**cp.__dict__, 'feature_names': ['iv', 'hv']})
    with pytest.raises(ValueError, match='feature_names mismatch'):
        validate(bad)


def test_validate_catches_top_k_oob():
    cp = _sample_checkpoint()
    bad = VolCheckpoint(**{**cp.__dict__, 'top_k': 999})
    with pytest.raises(ValueError, match='top_k'):
        validate(bad)


# ------------------------------------------------------------ inference

def test_predict_iv_rv_gap_basic():
    cp = _sample_checkpoint()
    features = pd.DataFrame({
        'iv_over_hv':   [1.20, 0.95, 1.05, np.nan, 1.30],
        'iv_z':         [+1.0, -0.5,  0.0,  +0.2, +2.0],
        'iv_change_4w': [+0.02, -0.01, 0.0, +0.01, +0.03],
        'hv_change_4w': [-0.01, +0.01, 0.0, +0.01, +0.02],
    }, index=['A', 'B', 'C', 'D', 'E'])
    pred = predict_iv_rv_gap(features, cp)
    assert 'D' not in pred.index, 'NaN-feature row must be dropped'
    assert pred.shape[0] == 4
    assert np.isfinite(pred.values).all()


def test_predict_iv_rv_gap_missing_column():
    cp = _sample_checkpoint()
    features = pd.DataFrame({
        'iv_over_hv': [1.0], 'iv_z': [0.0], 'iv_change_4w': [0.0]
    }, index=['A'])
    with pytest.raises(ValueError, match='missing columns'):
        predict_iv_rv_gap(features, cp)


def test_select_top_k_with_eligible_filter():
    pred = pd.Series({'A': 0.1, 'B': 0.5, 'C': 0.3, 'D': 0.2})
    picks = select_top_k(pred, top_k=2, eligible=['A', 'B', 'D'])
    assert list(picks.index) == ['B', 'D']  # C filtered out


def test_gate_fires_simple_above_median():
    idx = pd.date_range('2020-01-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 15.0), index=idx)
    vix.iloc[-1] = 25.0  # spike above median
    fires, vix_now, med = gate_fires(vix, lookback_trading_days=126)
    assert fires
    assert vix_now == 25.0
    assert med == 15.0


def test_gate_does_not_fire_below_median():
    idx = pd.date_range('2020-01-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 15.0), index=idx)
    vix.iloc[-1] = 10.0
    fires, _, _ = gate_fires(vix, lookback_trading_days=126)
    assert not fires


def test_gate_insufficient_history():
    """Less than lookback/2 history -> abstain (do not fire)."""
    idx = pd.date_range('2020-01-01', periods=10, freq='B')
    vix = pd.Series(np.linspace(10, 30, 10), index=idx)
    fires, vix_now, med = gate_fires(vix, lookback_trading_days=126)
    assert not fires
    assert math.isnan(vix_now) and math.isnan(med)


# ----------------------------------------------------------- iv_compute

def test_bs_call_intrinsic_at_zero_vol():
    # At T>0 sigma=0, the call should be max(S - K*exp(-rT), 0).
    p = bs_call_price(S=100, K=90, T=0.0, r=0.05, sigma=0.0)
    assert p == pytest.approx(10.0)


def test_bs_iv_round_trip_call():
    S, K, T, r, sigma = 100.0, 100.0, 30/365, 0.04, 0.30
    price = bs_call_price(S, K, T, r, sigma)
    iv = implied_vol_call(price, S, K, T, r)
    assert iv == pytest.approx(sigma, abs=1e-3)


def test_bs_iv_round_trip_put():
    S, K, T, r, sigma = 100.0, 110.0, 30/365, 0.04, 0.40
    price = bs_put_price(S, K, T, r, sigma)
    iv = implied_vol_put(price, S, K, T, r)
    assert iv == pytest.approx(sigma, abs=1e-3)


def test_bs_iv_arbitrage_returns_nan():
    """A price below intrinsic is arbitrage; IV inverter should return NaN."""
    iv = implied_vol_call(price=0.5, S=200, K=100, T=30/365, r=0.04)
    assert math.isnan(iv)


def test_bs_delta_atm_calls_near_half():
    S, K, T, r, sigma = 100, 100, 30/365, 0.04, 0.30
    d = bs_delta_call(S, K, T, r, sigma)
    assert 0.45 < d < 0.65  # ATM with drift ~ in this band


def test_bs_vega_positive_for_atm():
    v = bs_vega(S=100, K=100, T=30/365, r=0.04, sigma=0.30)
    assert v > 0


def test_atm_iv_from_chain_synthesizes_30d():
    today = pd.Timestamp('2026-05-23')
    exp = today + pd.Timedelta(days=30)
    T = 30/365
    # Synthesize fair-priced call + put at strike=100, sigma=0.25
    sigma_true = 0.25
    c_price = bs_call_price(100, 100, T, 0.04, sigma_true)
    p_price = bs_put_price(100, 100, T, 0.04, sigma_true)
    chain = [
        ChainQuote(expiration=exp, strike=100.0, option_type='call',
                   bid=c_price-0.05, ask=c_price+0.05, bid_size=50,
                   ask_size=50, open_interest=500),
        ChainQuote(expiration=exp, strike=100.0, option_type='put',
                   bid=p_price-0.05, ask=p_price+0.05, bid_size=50,
                   ask_size=50, open_interest=500),
    ]
    iv = atm_iv_from_chain(chain, underlying_price=100.0, today=today)
    assert iv == pytest.approx(sigma_true, abs=5e-3)


def test_realized_vol_from_bars_matches_known():
    rng = np.random.default_rng(42)
    log_r = rng.normal(0, 0.02, 252)  # 2% daily std
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_r)))
    rv = realized_vol_from_bars(prices, window=20)
    # Annualized 2% daily std ≈ 31.7%, but only 20 obs → wide tolerance.
    assert 0.15 < rv < 0.50


def test_build_feature_row_shapes():
    iv_now = pd.Series([0.30, 0.40, 0.20], index=['A', 'B', 'C'])
    hv_now = pd.Series([0.25, 0.35, 0.22], index=['A', 'B', 'C'])
    iv_hist = pd.DataFrame(
        {'A': np.full(25, 0.28), 'B': np.full(25, 0.38), 'C': np.full(25, 0.19)},
        index=pd.date_range('2026-04-01', periods=25, freq='B'))
    hv_hist = iv_hist * 0.9
    feat = build_feature_row(iv_now, hv_now, iv_hist, hv_hist)
    assert list(feat.columns) == ['iv_over_hv', 'iv_z', 'iv_change_4w', 'hv_change_4w']
    assert feat.shape == (3, 4)
    # iv_change_4w should be roughly iv_now - iv_4w_ago (positive in this fixture)
    assert (feat['iv_change_4w'] > 0).all()


# --------------------------------------------------------------- strangle

def _synth_chain(S: float = 100.0, today=None) -> tuple[list[ChainQuote], pd.Timestamp]:
    if today is None:
        today = pd.Timestamp('2026-05-23')
    exp = today + pd.Timedelta(days=30)
    T = 30/365
    sigma = 0.25
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    chain = []
    for K in strikes:
        c = bs_call_price(S, K, T, 0.04, sigma)
        p = bs_put_price(S, K, T, 0.04, sigma)
        chain.append(ChainQuote(
            expiration=exp, strike=float(K), option_type='call',
            bid=c-0.10, ask=c+0.10, bid_size=50, ask_size=50, open_interest=500))
        chain.append(ChainQuote(
            expiration=exp, strike=float(K), option_type='put',
            bid=p-0.10, ask=p+0.10, bid_size=50, ask_size=50, open_interest=500))
    return chain, today


def test_strangle_builds_at_target_delta():
    chain, today = _synth_chain(S=100.0)
    cfg = StranglesConfig(target_delta_call=0.20, target_delta_put=0.20,
                          vega_budget_per_name_usd=200.0,
                          max_bid_ask_spread_pct=0.50)
    s = build_short_strangle('AAPL', 100.0, chain, cfg, today=today)
    assert s is not None
    # Calls Δ should be ~ +0.20, puts Δ should be ~ -0.20
    assert 0.10 < s.call.delta_at_construction < 0.35
    assert -0.35 < s.put.delta_at_construction < -0.10
    # Net delta near 0 (strikes chosen at matched |Δ|)
    assert abs(s.net_delta) < 0.5 * abs(s.net_vega)
    # Short strangle -> net_vega negative
    assert s.net_vega < 0
    # OCC symbols look right
    assert s.call.contract_symbol.startswith('AAPL')
    assert 'C' in s.call.contract_symbol
    assert 'P' in s.put.contract_symbol


def test_strangle_skips_illiquid_contracts():
    """OI below the floor should disqualify the chain."""
    chain, today = _synth_chain(S=100.0)
    chain_low_oi = [
        ChainQuote(**{**q.__dict__, 'open_interest': 5}) for q in chain]
    cfg = StranglesConfig(min_open_interest=100)
    s = build_short_strangle('AAPL', 100.0, chain_low_oi, cfg, today=today)
    assert s is None, 'must reject when OI below floor'


def test_strangle_vega_budget_sets_qty():
    chain, today = _synth_chain(S=100.0)
    cfg_small = StranglesConfig(vega_budget_per_name_usd=100.0,
                                max_bid_ask_spread_pct=0.50)
    cfg_large = StranglesConfig(vega_budget_per_name_usd=1000.0,
                                max_bid_ask_spread_pct=0.50)
    s_small = build_short_strangle('AAPL', 100.0, chain, cfg_small, today=today)
    s_large = build_short_strangle('AAPL', 100.0, chain, cfg_large, today=today)
    assert s_small is not None and s_large is not None
    assert s_large.call.qty > s_small.call.qty


# --------------------------------------------------------------- live

def test_live_kill_switch_aborts(tmp_path: Path):
    """Rail 1: kill-switch file present -> abort, no Alpaca calls."""
    from vol.live import run_live
    cp = _sample_checkpoint()
    cp_path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, cp_path)
    ks = tmp_path / '.vol-killswitch'
    ks.touch()

    # Pass a broker mock that would fail if called -- it must not be called.
    class BoomBroker:
        def get_account(self):
            raise RuntimeError('rail bypassed: broker called despite kill-switch')

    result = run_live(
        cp_path, broker=BoomBroker(), options_data=object(),
        bars_data=object(), vix_loader=lambda: pd.Series(dtype=float),
        dry_run=True, killswitch_path=ks)
    assert result.aborted_reason is not None
    assert 'kill-switch' in result.aborted_reason


def test_live_gate_closed_aborts_cleanly(tmp_path: Path):
    """Rail 3: VIX gate closed -> abort, no chain queries."""
    from vol.live import run_live
    cp = _sample_checkpoint()
    cp_path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, cp_path)

    class FlatAccount:
        equity = 100_000.0
        cash = 100_000.0
    class StubBroker:
        _paper = True
        def get_account(self):
            return FlatAccount()

    # VIX = 15, last 126d median = 20 (gate CLOSED).
    idx = pd.date_range('2025-01-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 20.0), index=idx)
    vix.iloc[-1] = 15.0

    result = run_live(
        cp_path, broker=StubBroker(),
        options_data=object(), bars_data=object(),
        vix_loader=lambda: vix, dry_run=True,
        killswitch_path=tmp_path / 'does-not-exist')
    assert result.aborted_reason is not None
    assert 'VIX gate closed' in result.aborted_reason
    assert result.gate.fired is False
    assert len(result.strangles) == 0


def test_live_chain_query_layer_not_yet_wired(tmp_path: Path):
    """Documents the staged state of the MVP: when the gate fires and
    we'd actually need the chain-query layer, the scaffold raises
    NotImplementedError. The CLI surfaces this as exit code 2.
    """
    from vol.live import run_live
    cp = _sample_checkpoint()
    cp_path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, cp_path)

    class FlatAccount:
        equity = 100_000.0
        cash = 100_000.0
    class StubBroker:
        _paper = True
        def get_account(self):
            return FlatAccount()

    # VIX = 25, last 126d median = 15 (gate FIRES).
    idx = pd.date_range('2025-01-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 15.0), index=idx)
    vix.iloc[-1] = 25.0

    with pytest.raises(NotImplementedError, match='_build_feature_panel'):
        run_live(
            cp_path, broker=StubBroker(),
            options_data=object(), bars_data=object(),
            vix_loader=lambda: vix, dry_run=True,
            killswitch_path=tmp_path / 'does-not-exist')
