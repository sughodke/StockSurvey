"""Inference forward pass: prices in, target weights out.

Mirrors `relational.inference.target_weights` so a future `live.py` can share
the `apps/regime` / `apps/relational` orchestration pattern verbatim.

Pipeline:

1. Validate input shapes and bar count vs `checkpoint.lookback`.
2. Dispatch on `checkpoint.strategy` to the matching `weights_*` builder
   (v1: just `hrp`).
3. Apply the Corwin-Schultz spread gate (zero out names with `spread >
   max_spread` on the rebalance bar; renormalize the survivors to sum 1).
4. If `checkpoint.use_symmetry_modulator` is set, multiply the (post-gate)
   weight vector by the gross-exposure scalar from `effective_rank`. This is
   applied LAST so the spread-gate's renormalization doesn't undo the
   symmetry-driven gross scaling.
5. Optional top-N truncation.

Inputs are pandas DataFrames -- matching `relational` -- so the eventual
`live.py` can call this with the panel it already has from the broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lie.hrp import weights_hrp
from lie.persist import LieCheckpoint
from lie.symmetry_rank import gross_exposure_modulator, trailing_effective_rank
from ss_indicators import corwin_schultz_spread


def target_weights(
    prices: pd.DataFrame,
    highs: pd.DataFrame | None,
    lows: pd.DataFrame | None,
    checkpoint: LieCheckpoint,
) -> pd.Series:
    """Compute target portfolio weights for the latest bar.

    Parameters
    ----------
    prices :
        Wide DataFrame indexed by date, columns = tickers in
        `checkpoint.universe` order. Must contain at least
        `lookback + 1` rows.
    highs, lows :
        Same shape as `prices`. Used for the Corwin-Schultz spread gate.
        Pass `None` to both to skip the gate (research only).
    checkpoint :
        Loaded `LieCheckpoint`.

    Returns
    -------
    pd.Series
        Target weights indexed by ticker. Sums to <= 1 (== 1 if no
        symmetry modulator and there are spread-gate survivors).
    """
    _validate_inputs(prices, highs, lows, checkpoint)

    panel = prices.to_numpy()
    if checkpoint.strategy == 'hrp':
        kw = dict(checkpoint.strategy_kwargs)
        w = weights_hrp(
            panel,
            lookback=checkpoint.lookback,
            linkage_method=str(kw.get('linkage_method', 'single')))
    else:
        raise ValueError(f'unknown strategy {checkpoint.strategy!r}')

    weights = pd.Series(w, index=prices.columns, name=prices.index[-1])

    if highs is not None and lows is not None and checkpoint.max_spread > 0:
        weights = _apply_spread_gate(weights, highs, lows, checkpoint.max_spread)

    if checkpoint.use_symmetry_modulator:
        eff = trailing_effective_rank(panel, lookback=checkpoint.lookback)
        n_active = int((weights > 0).sum())
        scalar = gross_exposure_modulator(
            eff, n_assets=n_active, floor=checkpoint.symmetry_floor)
        weights = weights * scalar

    if checkpoint.top_n and checkpoint.top_n > 0:
        weights = _truncate_top_n(weights, checkpoint.top_n)

    return weights


def _validate_inputs(
    prices: pd.DataFrame,
    highs: pd.DataFrame | None,
    lows: pd.DataFrame | None,
    checkpoint: LieCheckpoint,
) -> None:
    if list(prices.columns) != list(checkpoint.universe):
        raise ValueError(
            f'prices columns do not match checkpoint universe '
            f'({len(prices.columns)} vs {len(checkpoint.universe)} names, '
            f'and/or different ordering)')
    if highs is not None and lows is not None:
        if not (prices.columns.equals(highs.columns)
                and prices.columns.equals(lows.columns)):
            raise ValueError('prices/highs/lows must share columns')
        if not (prices.index.equals(highs.index)
                and prices.index.equals(lows.index)):
            raise ValueError('prices/highs/lows must share index')
    min_bars = checkpoint.lookback + 1
    if len(prices) < min_bars:
        raise ValueError(
            f'need at least {min_bars} bars (lookback + 1); '
            f'got {len(prices)}')


def _apply_spread_gate(
    weights: pd.Series,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    max_spread: float,
) -> pd.Series:
    """Zero out names whose Corwin-Schultz spread on the rebalance bar
    exceeds `max_spread`, then renormalize over the survivors."""
    spread_df = corwin_schultz_spread(highs, lows)
    spread_last = spread_df.iloc[-1].reindex(weights.index)
    illiquid = ~(spread_last <= max_spread)
    gated = weights.where(~illiquid, 0.0)
    total = float(gated.sum())
    if total <= 0:
        return pd.Series(
            np.zeros(len(weights)), index=weights.index, name=weights.name)
    return gated / total


def _truncate_top_n(weights: pd.Series, top_n: int) -> pd.Series:
    """Keep the top-N names by weight; renormalize survivors to sum 1."""
    if top_n >= int((weights > 0).sum()):
        return weights
    keep_idx = weights.nlargest(top_n).index
    out = pd.Series(0.0, index=weights.index, name=weights.name)
    out.loc[keep_idx] = weights.loc[keep_idx]
    s = float(out.sum())
    return out / s if s > 0 else out


__all__ = ['target_weights']
