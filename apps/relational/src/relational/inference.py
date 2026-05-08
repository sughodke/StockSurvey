"""Inference forward pass: prices in, target weights out.

Loads a `RelationalCheckpoint` and returns target weights for the
*latest* date in the supplied OHLC frames. Dispatches on
`checkpoint.strategy` to the matching `weights_*` builder, takes the
last row of its `(n_eval_dates, n_tickers)` output, applies a
Corwin-Schultz spread gate (names with `spread > checkpoint.max_spread`
are zeroed and the survivors renormalized), and returns a Series.

The relational `weights_*` builders compute the full panel sweep —
that's seconds at the universe sizes we care about (≤ 312 tickers,
lookback ~120), and re-using the production builder verbatim
guarantees the live signal matches what was backtested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relational.persist import RelationalCheckpoint
from ss_features import Compression
from ss_indicators import corwin_schultz_spread
from ss_wavelets import KERNEL_HALF_EXTENT


def _maybe_compression(kw: dict) -> Compression | None:
    """Build a `Compression` from `strategy_kwargs` keys when
    `compress_levels > 0`, else return None (full-resolution
    fingerprints, the legacy behaviour). Defaults match the
    `Compression` defaults so older checkpoints without these keys
    keep the same uncompressed behaviour."""
    levels = int(kw.get('compress_levels', 0))
    if levels < 1:
        return None
    return Compression(
        kind=str(kw.get('compress_kind', 'dwt')),
        levels=levels,
        wavelet=str(kw.get('compress_wavelet', 'haar')),
        pad_mode=str(kw.get('compress_pad_mode', 'periodization')))


def target_weights(
    prices: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    checkpoint: RelationalCheckpoint,
) -> pd.Series:
    """Compute the strategy's target portfolio weights for the latest bar.

    Parameters
    ----------
    prices, highs, lows :
        Wide DataFrames indexed by date, columns are tickers. Must share
        an index and column set, and contain at least `lookback + 1` rows.
        The most recent row is the rebalance date.
    checkpoint :
        Configured `RelationalCheckpoint` produced by
        `persist.load_checkpoint`. `strategy` selects the dispatch;
        `strategy_kwargs` carries the per-strategy knobs.

    Returns
    -------
    pd.Series
        Target portfolio weights indexed by ticker, summing to 1 over
        liquid survivors of the spread gate (or all-zero if no name
        passes — the caller should treat this as a flat day).
    """
    _validate_inputs(prices, highs, lows, checkpoint)
    weights_df = _build_weights_panel(prices, checkpoint)
    last = weights_df.iloc[-1].reindex(prices.columns).fillna(0.0)
    last = _apply_spread_gate(last, highs, lows, checkpoint.max_spread)
    return last.rename(prices.index[-1])


def _validate_inputs(prices, highs, lows, checkpoint):
    if not (prices.columns.equals(highs.columns) and prices.columns.equals(lows.columns)):
        raise ValueError('prices/highs/lows must share columns')
    if not (prices.index.equals(highs.index) and prices.index.equals(lows.index)):
        raise ValueError('prices/highs/lows must share index')
    # All relational strategies use the CWT, so the latest bar's
    # wavelet needs KERNEL_HALF_EXTENT*max_scale + lookback bars to
    # have full kernel support — otherwise it's silently zero-padded.
    max_scale = max(checkpoint.scales) if checkpoint.scales else 0
    min_bars = KERNEL_HALF_EXTENT * max_scale + checkpoint.lookback + 1
    if len(prices) < min_bars:
        raise ValueError(
            f'need at least {min_bars} bars (KERNEL_HALF_EXTENT*max_scale + '
            f'lookback + 1 = {KERNEL_HALF_EXTENT}*{max_scale} + '
            f'{checkpoint.lookback} + 1), got {len(prices)}')


def _build_weights_panel(
    prices: pd.DataFrame,
    cp: RelationalCheckpoint,
) -> pd.DataFrame:
    """Dispatch to the appropriate `weights_*` builder. Each branch
    forwards `cp.lookback`, `cp.top_n`, `cp.scales` plus whatever extra
    knobs that strategy uses out of `cp.strategy_kwargs`. Builders take
    *only* their declared kwargs, so silently passing extras through
    would TypeError — we filter."""
    kw = dict(cp.strategy_kwargs)

    if cp.strategy == 'empirical':
        from relational.empirical_sectors import weights_excess_regime_empirical
        return weights_excess_regime_empirical(
            prices, lookback=cp.lookback, top_n=cp.top_n, scales=cp.scales,
            n_tail=int(kw.get('n_tail', 20)),
            divergence=str(kw.get('divergence', 'kl')),
            k_clusters=int(kw.get('k_clusters', 11)),
            fp_window=int(kw.get('fp_window', 21)),
            refit_days=int(kw.get('refit_days', 252)),
        )

    if cp.strategy == 'gmm':
        from relational.empirical_sectors_gmm import weights_excess_regime_gmm
        return weights_excess_regime_gmm(
            prices, lookback=cp.lookback, top_n=cp.top_n, scales=cp.scales,
            n_tail=int(kw.get('n_tail', 20)),
            divergence=str(kw.get('divergence', 'kl')),
            n_components=int(kw.get('n_components', 11)),
            fp_window=int(kw.get('fp_window', 21)),
            refit_days=int(kw.get('refit_days', 252)),
        )

    if cp.strategy == 'analog':
        from relational.analog_knn import weights_regime_analog
        return weights_regime_analog(
            prices, lookback=cp.lookback, top_n=cp.top_n, scales=cp.scales,
            fp_window=int(kw.get('fp_window', 21)),
            k_neighbors=int(kw.get('k_neighbors', 50)),
            forward_horizon=int(kw.get('forward_horizon', 20)),
            min_sep_days=int(kw.get('min_sep_days', 21)),
            pool_mode=str(kw.get('pool_mode', 'cross_ticker')),
            compression=_maybe_compression(kw),
        )

    if cp.strategy == 'farthest':
        from relational.farthest import weights_regime_farthest
        return weights_regime_farthest(
            prices, lookback=cp.lookback, top_n=cp.top_n, scales=cp.scales,
            fp_window=int(kw.get('fp_window', 21)),
            compression=_maybe_compression(kw),
        )

    if cp.strategy == 'diversified':
        from relational.diversify import weights_regime_diversified
        return weights_regime_diversified(
            prices, lookback=cp.lookback, scales=cp.scales,
            n_tail=int(kw.get('n_tail', 20)),
            k_keep=cp.top_n,
            top_pool=int(kw.get('top_pool', max(2 * cp.top_n, 20))),
            divergence=str(kw.get('divergence', 'kl')),
            fp_window=int(kw.get('fp_window', 21)),
            compression=_maybe_compression(kw),
        )

    if cp.strategy == 'velocity':
        from relational.regime_velocity import weights_velocity_magnitude
        return weights_velocity_magnitude(
            prices, lookback=cp.lookback, top_n=cp.top_n, scales=cp.scales,
            fp_window=int(kw.get('fp_window', 21)),
            w_delta=int(kw.get('w_delta', 20)),
        )

    raise ValueError(f'unknown strategy {cp.strategy!r}')


def _apply_spread_gate(
    weights: pd.Series,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    max_spread: float,
) -> pd.Series:
    """Zero out names whose Corwin-Schultz spread on the rebalance bar
    exceeds `max_spread`, then renormalize. Mirrors what the regime
    inference path does inside its score function — relational
    `weights_*` builders don't gate on spread internally, so we apply
    the same rail here."""
    spread_df = corwin_schultz_spread(highs, lows)
    spread_last = spread_df.iloc[-1].reindex(weights.index)
    illiquid = ~(spread_last <= max_spread)
    gated = weights.where(~illiquid, 0.0)
    total = float(gated.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(weights)), index=weights.index, name=weights.name)
    return gated / total


__all__ = ['target_weights']
