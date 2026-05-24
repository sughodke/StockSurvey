"""Phase-B integration test for vol.live — exercises the full
chain-query → strangle-build → multi-leg-submit path with mocked
Alpaca clients. No network calls; the test makes sure the wiring is
right and dry-run yields synthetic order ids without contacting the
broker.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol.iv_compute import bs_call_price, bs_put_price
from vol.persist import (
    LIVE_FEATURE_NAMES, StranglesConfig, VolCheckpoint, save_checkpoint,
)


def _sample_checkpoint() -> VolCheckpoint:
    return VolCheckpoint(
        feature_names=list(LIVE_FEATURE_NAMES),
        coefs=[+0.0118, +0.0376, -0.0073, -0.0112, +0.005],
        feat_mean=[1.05, 0.0, 0.0, 0.0],
        feat_std=[0.4, 1.0, 0.05, 0.05],
        universe=['AAPL', 'MSFT'],
        gate_fred_series='VIXCLS',
        gate_lookback_trading_days=126,
        top_k=2,
        strangle=StranglesConfig(max_bid_ask_spread_pct=0.50,
                                 min_open_interest=100,
                                 vega_budget_per_name_usd=200.0),
        train_period='2019-10-14 → 2023-07-28',
        val_period='2023-08-01 → 2026-04-30',
        val_pearson_r=0.165, n_obs_oos=33, oos_ann_sharpe=2.822,
        oos_deflated_t=5.549,
    )


# --- Mock Alpaca clients ---------------------------------------------------

class _FakeAccount:
    equity = 100_000.0
    cash = 100_000.0


class _MockBroker:
    _paper = True
    submitted = []

    def get_account(self):
        return _FakeAccount()

    def submit_order(self, order_req):
        # Real Alpaca returns an Order with .id; replicate.
        class _Resp:
            id = f'ord-{len(_MockBroker.submitted)}'
        _MockBroker.submitted.append(order_req)
        return _Resp()


class _MockStocksClient:
    """Fake Alpaca StockHistoricalDataClient.get_stock_bars."""

    def __init__(self, universe: list[str], start_price: float = 100.0):
        self.universe = universe
        # Build a 40-day daily price panel; geometric random walk per name.
        rng = np.random.default_rng(0)
        n = 40
        idx = pd.date_range('2026-04-15', periods=n, freq='B').normalize()
        # Wide DataFrame mimicking the post-unstack alpaca-py format.
        frames = []
        for sym in universe:
            log_r = rng.normal(0, 0.015, n)
            close = start_price * np.exp(np.cumsum(log_r))
            frames.append(pd.DataFrame({'close': close}, index=idx).assign(symbol=sym))
        df = pd.concat(frames)
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.set_index(['symbol', 'timestamp'])
        self._df = df

    def get_stock_bars(self, _req):
        class _Resp:
            df = self._df
        return _Resp()


class _MockContract:
    def __init__(self, sym, underlier, exp, strike, opt_type, oi=500):
        self.symbol = sym
        self.underlying_symbol = underlier
        self.expiration_date = exp
        self.strike_price = strike
        self.type = opt_type   # 'call' / 'put' string OK for our str(...) test
        self.open_interest = oi


class _MockQuote:
    def __init__(self, bid, ask, bid_size=50, ask_size=50):
        self.bid_price = bid
        self.ask_price = ask
        self.bid_size = bid_size
        self.ask_size = ask_size


class _MockSnapshot:
    def __init__(self, quote):
        self.latest_quote = quote


class _MockOptionsClient:
    """Fake Alpaca client supporting both get_option_contracts and
    get_option_snapshot. Returns a synthetic BS-priced chain."""

    def __init__(self, underliers: list[str], S: float = 100.0, sigma: float = 0.25):
        self.underliers = underliers
        self.S = S
        self.sigma = sigma
        self.today = pd.Timestamp('2026-05-23').normalize()
        self.exp = (self.today + pd.Timedelta(days=30)).date()
        self.r = 0.04
        self.T = 30 / 365.0

    def _chain_for(self, underlier):
        contracts = []
        snapshots = {}
        for K in [80, 90, 95, 100, 105, 110, 120]:
            c_price = bs_call_price(self.S, K, self.T, self.r, self.sigma)
            p_price = bs_put_price(self.S, K, self.T, self.r, self.sigma)
            c_sym = f'{underlier}{self.exp.strftime("%y%m%d")}C{int(K*1000):08d}'
            p_sym = f'{underlier}{self.exp.strftime("%y%m%d")}P{int(K*1000):08d}'
            contracts.append(_MockContract(c_sym, underlier, self.exp, K, 'call'))
            contracts.append(_MockContract(p_sym, underlier, self.exp, K, 'put'))
            snapshots[c_sym] = _MockSnapshot(_MockQuote(c_price-0.05, c_price+0.05))
            snapshots[p_sym] = _MockSnapshot(_MockQuote(p_price-0.05, p_price+0.05))
        return contracts, snapshots

    def get_option_contracts(self, req):
        underliers = list(getattr(req, 'underlying_symbols', []))
        all_contracts = []
        for u in underliers:
            if u not in self.underliers:
                continue
            contracts, snapshots = self._chain_for(u)
            all_contracts.extend(contracts)
            # Stash snapshots for the subsequent get_option_snapshot call.
            self._stashed_snapshots = getattr(self, '_stashed_snapshots', {})
            self._stashed_snapshots.update(snapshots)
        class _Resp:
            option_contracts = all_contracts
        return _Resp()

    def get_option_snapshot(self, req):
        return {s: self._stashed_snapshots[s]
                for s in req.symbol_or_symbols
                if s in getattr(self, '_stashed_snapshots', {})}


# --- The end-to-end happy-path test ---------------------------------------

def test_phase_b_end_to_end_dry_run(tmp_path: Path):
    """Gate fires; pipeline pulls (mock) Alpaca bars + chain, computes
    features, scores top-K, builds strangles, and emits DRY_RUN_*
    order ids without touching real Alpaca."""
    from vol.live import run_live
    _MockBroker.submitted.clear()

    cp = _sample_checkpoint()
    cp_path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, cp_path)

    # Seed the IV-history cache via the production append_snapshot API
    # so dtype/schema contract is identical to what the runtime writes
    # — sidesteps a pyarrow type-inference issue when seeding via
    # to_parquet directly.
    cache = tmp_path / 'iv-history.parquet'
    import vol.iv_history as ih
    orig = ih.DEFAULT_CACHE_PATH
    ih.DEFAULT_CACHE_PATH = str(cache)
    seed_iv = pd.Series({s: 0.28 for s in cp.universe}, dtype=float)
    seed_hv = pd.Series({s: 0.25 for s in cp.universe}, dtype=float)
    ih.append_snapshot(seed_iv, seed_hv,
                       as_of=pd.Timestamp('2026-04-25'),
                       cache_path=cache)

    # VIX series with last bar > rolling-median (gate FIRES).
    idx = pd.date_range('2025-09-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 12.0), index=idx)
    vix.iloc[-1] = 22.0

    try:
        result = run_live(
            cp_path,
            broker=_MockBroker(),
            options_data=_MockOptionsClient(cp.universe),
            bars_data=_MockStocksClient(cp.universe),
            vix_loader=lambda: vix,
            dry_run=True,
            killswitch_path=tmp_path / 'does-not-exist',
            max_total_vega_usd=5000.0,
        )
    finally:
        ih.DEFAULT_CACHE_PATH = orig

    # The gate fired
    assert result.gate.fired
    # We had bars
    assert result.last_bar_date
    # We constructed strangles (both AAPL and MSFT in this fixture)
    assert len(result.strangles) >= 1, f'no strangles built: {result.notes}'
    # Each strangle is short (net_vega negative)
    for s in result.strangles:
        assert s.net_vega < 0
        assert s.call.qty == s.put.qty
    # dry_run path emitted DRY_RUN_* order ids
    assert all(oid.startswith('DRY_RUN_') for oid in result.submitted_order_ids)
    # Nothing went to the broker
    assert _MockBroker.submitted == []
    # And no rejections
    assert result.rejected_orders == []
    # Total |vega| within budget
    total_vega = sum(abs(s.net_vega) for s in result.strangles)
    assert total_vega <= 5000.0 + 1e-6


def test_phase_b_live_path_submits_to_broker(tmp_path: Path):
    """Same fixtures, dry_run=False — confirms submit_short_strangle
    actually invokes the (mock) broker and returns real-looking order ids."""
    from vol.live import run_live
    _MockBroker.submitted.clear()

    cp = _sample_checkpoint()
    cp_path = tmp_path / 'vol-v3.json'
    save_checkpoint(cp, cp_path)

    cache = tmp_path / 'iv-history-live.parquet'
    import vol.iv_history as ih
    orig = ih.DEFAULT_CACHE_PATH
    ih.DEFAULT_CACHE_PATH = str(cache)
    seed_iv = pd.Series({s: 0.28 for s in cp.universe}, dtype=float)
    seed_hv = pd.Series({s: 0.25 for s in cp.universe}, dtype=float)
    ih.append_snapshot(seed_iv, seed_hv,
                       as_of=pd.Timestamp('2026-04-25'),
                       cache_path=cache)

    idx = pd.date_range('2025-09-01', periods=200, freq='B')
    vix = pd.Series(np.full(200, 12.0), index=idx)
    vix.iloc[-1] = 22.0

    try:
        result = run_live(
            cp_path,
            broker=_MockBroker(),
            options_data=_MockOptionsClient(cp.universe),
            bars_data=_MockStocksClient(cp.universe),
            vix_loader=lambda: vix,
            dry_run=False,
            killswitch_path=tmp_path / 'does-not-exist',
            max_total_vega_usd=5000.0,
        )
    finally:
        ih.DEFAULT_CACHE_PATH = orig

    assert len(result.strangles) >= 1
    # Broker was called once per strangle
    assert len(_MockBroker.submitted) == len(result.strangles)
    # Each submitted order has the MLEG class shape
    for order_req in _MockBroker.submitted:
        # legs is a list of OptionLegRequest; one call + one put
        assert len(order_req.legs) == 2
        # The strategy is to sell both legs (short strangle)
        from alpaca.trading.enums import OrderSide
        assert all(leg.side == OrderSide.SELL for leg in order_req.legs)
    # Order ids look real, not synthetic
    assert all(oid.startswith('ord-') for oid in result.submitted_order_ids)
