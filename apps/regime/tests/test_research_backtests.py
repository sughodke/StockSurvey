"""End-to-end tests for regime.research.backtest_bt and backtest_ranking.

These exercise the weight builders and rankers on a tiny synthetic universe
to catch wiring regressions when the underlying ss_* primitives or the
local glue shifts. Numerical correctness of the primitives themselves is
covered in packages/*/tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-01', periods=300)
    tickers = [f'T{i:02d}' for i in range(20)]
    increments = rng.normal(0.0005, 0.01, size=(300, 20))
    prices = 100 * np.exp(np.cumsum(increments, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def synthetic_prices_dict(synthetic_prices) -> dict[str, pd.DataFrame]:
    return {
        t: pd.DataFrame({'adj_close': synthetic_prices[t].values},
                        index=synthetic_prices.index)
        for t in synthetic_prices.columns
    }


# --- backtest_bt: weight builders ---

def test_weights_rsi_shape_and_normalization(synthetic_prices):
    from regime.research.backtest_bt import weights_rsi
    w = weights_rsi(synthetic_prices, lookback=60, n_tail=5, top_n=5)
    assert w.shape == (300 - 60, 20)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-9)
    # Every nonzero cell is exactly 1/top_n
    np.testing.assert_allclose(w.values[w.values > 0], 0.2, rtol=1e-9)


def test_weights_scalogram_shape_and_normalization(synthetic_prices):
    from regime.research.backtest_bt import weights_scalogram
    w = weights_scalogram(synthetic_prices, lookback=120, n_tail=10, top_n=5)
    assert w.shape == (300 - 120, 20)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-9)


def test_weights_regime_matches_direct_kl_call(synthetic_prices):
    """weights_regime delegates to ss_indicators.symmetric_kl_divergence
    + ss_portfolio.select_top_n_matrix. Recompute via those primitives
    directly and check the output DataFrames are identical — this is the
    regression test that protects the dedup we just did."""
    import jax.numpy as jnp
    from ss_indicators import symmetric_kl_divergence
    from ss_portfolio import select_top_n_matrix
    from ss_wavelets import causal_cwt
    from regime.research.backtest_bt import weights_regime

    lookback, n_tail, top_n = 120, 20, 5
    scales = [5, 7, 10, 12, 21, 26, 50, 90]

    w_built = weights_regime(synthetic_prices, lookback=lookback,
                             n_tail=n_tail, top_n=top_n)

    # Recompute the same scores directly via the package primitives
    coeffs = causal_cwt(synthetic_prices.values, scales, lookback)
    power = coeffs ** 2
    n_dates, n_tickers = synthetic_prices.shape
    scores = np.full((n_dates - lookback, n_tickers), np.nan)
    log_w = jnp.zeros(len(scales))
    for i in range(lookback, n_dates):
        recent = np.mean(power[:, i - n_tail + 1:i + 1, :], axis=1)
        historical = np.mean(power[:, i - lookback:i - n_tail + 1, :], axis=1)
        kl = symmetric_kl_divergence(jnp.asarray(recent),
                                     jnp.asarray(historical), log_w)
        scores[i - lookback] = np.asarray(kl)
    w_direct = select_top_n_matrix(scores, top_n, ascending=False)

    np.testing.assert_allclose(w_built.values, w_direct, atol=1e-9)


def test_weights_equal_allocates_largest_top_n(synthetic_prices):
    from regime.research.backtest_bt import weights_equal
    w = weights_equal(synthetic_prices, top_n=5)
    assert w.shape == (300, 20)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-9)
    # Selection is by last-bar price; row content is constant across dates
    np.testing.assert_allclose(w.iloc[0].values, w.iloc[-1].values)


def test_weights_regime_uses_imported_screening(synthetic_prices):
    """spread_df screening must NaN-mask high-spread names so they're
    excluded from the basket. Confirms the rename _apply_spread_mask →
    apply_spread_mask kept the wiring intact."""
    from regime.research.backtest_bt import weights_regime

    # Pin one ticker to permanently-wide spread; it must never be picked
    spread = pd.DataFrame(0.001,
                          index=synthetic_prices.index,
                          columns=synthetic_prices.columns)
    blocked = synthetic_prices.columns[0]
    spread[blocked] = 0.5  # well above default max_spread=0.02

    w = weights_regime(synthetic_prices, lookback=120, n_tail=20, top_n=5,
                       spread_df=spread)
    assert (w[blocked] == 0).all()


# --- backtest_ranking: rankers ---

def test_rank_regime_change_returns_negative_kl(synthetic_prices_dict,
                                                synthetic_prices):
    """Docstring contract: returns -KL so lower = more divergence. Package
    KL is non-negative, therefore every score must be ≤ 0."""
    from regime.research.backtest_ranking import rank_regime_change
    scores = rank_regime_change(synthetic_prices_dict,
                                synthetic_prices.index[-1],
                                lookback=120, n_tail=20)
    assert len(scores) == 20
    assert all(np.isfinite(v) for v in scores.values())
    assert all(v <= 1e-7 for v in scores.values())


def test_rank_regime_change_matches_direct_kl_call(synthetic_prices_dict,
                                                   synthetic_prices):
    """Same regression test as for weights_regime: recompute directly via
    ss_indicators.symmetric_kl_divergence and confirm the per-ticker
    scores agree to float32 precision."""
    import jax.numpy as jnp
    from ss_indicators import symmetric_kl_divergence
    from regime.research.backtest_ranking import rank_regime_change, _ricker_cwt_1d

    date = synthetic_prices.index[-1]
    lookback, n_tail = 120, 20
    scales = np.array([5, 7, 10, 12, 21, 26, 50, 90])
    log_w = jnp.zeros(len(scales))

    expected: dict[str, float] = {}
    for ticker, df in synthetic_prices_dict.items():
        chunk = df.loc[:date].tail(lookback)
        prices = chunk.adj_close.values
        x = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)
        coeffs = _ricker_cwt_1d(x, scales)
        power = coeffs ** 2
        recent = np.mean(power[:, -n_tail:], axis=1)
        historical = np.mean(power[:, :-n_tail], axis=1)
        kl = float(symmetric_kl_divergence(
            jnp.asarray(recent), jnp.asarray(historical), log_w))
        expected[ticker] = -kl

    scores = rank_regime_change(synthetic_prices_dict, date,
                                lookback=lookback, n_tail=n_tail)
    assert scores.keys() == expected.keys()
    for ticker, v in scores.items():
        assert v == pytest.approx(expected[ticker], rel=1e-5, abs=1e-7)


def test_rank_rsi_returns_one_score_per_ticker(synthetic_prices_dict,
                                                synthetic_prices):
    from regime.research.backtest_ranking import rank_rsi
    scores = rank_rsi(synthetic_prices_dict, synthetic_prices.index[-1],
                      lookback=60, n_tail=5)
    assert len(scores) == 20
    assert all(0 <= v <= 100 for v in scores.values())  # RSI bounds


def test_rank_scalogram_returns_one_score_per_ticker(synthetic_prices_dict,
                                                     synthetic_prices):
    from regime.research.backtest_ranking import rank_scalogram
    scores = rank_scalogram(synthetic_prices_dict,
                            synthetic_prices.index[-1],
                            lookback=120, n_tail=10)
    assert len(scores) == 20
    assert all(np.isfinite(v) for v in scores.values())
