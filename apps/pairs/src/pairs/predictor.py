"""Trade-signal generators — classical baseline only for v1.

`trade_signals(z, entry, exit)` returns a position state machine
output: `+1` (long-spread = long-A / short-B), `-1` (short-spread =
short-A / long-B), `0` (flat). State transitions:

    flat  →  long  if z <  -entry      (spread cheap, expect revert up)
    flat  →  short if z >  +entry      (spread rich,  expect revert down)
    long  →  flat  if z >  -exit       (revert hit close-target)
    short →  flat  if z <  +exit       (revert hit close-target)

Stop-out (`stop`) is symmetric — exit if z moves further in the
adverse direction past `±stop` (default disabled, set
`stop=float('inf')` to skip).

This is the classical Gatev-Goetzmann-Rouwenhorst (2006) trading
rule, the simplest deployable spec for a pair-trading test. ML
heads come later if classical shows promise.
"""
from __future__ import annotations

import numpy as np


def trade_signals(
    z: np.ndarray, entry: float = 2.0, exit_z: float = 0.5,
    stop: float = float('inf'),
) -> np.ndarray:
    """State-machine position over time.

    `z[t]` is the spread z-score at bar t. Position decision at
    bar t is based on z[t-1] (callers should align downstream PnL
    accordingly). For simplicity here we return position[t] = the
    state after seeing z[t]; the backtest layer is responsible for
    lagging by 1 bar to avoid look-ahead.
    """
    n = len(z)
    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for t in range(n):
        zt = z[t]
        if state == 0:
            if zt < -entry:
                state = +1
            elif zt > +entry:
                state = -1
        elif state == +1:
            if zt > -exit_z or zt < -stop:
                state = 0
        elif state == -1:
            if zt < +exit_z or zt > +stop:
                state = 0
        pos[t] = state
    return pos


__all__ = ['trade_signals']
