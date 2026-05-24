"""Tinygrad <-> numpy parity test for the differentiable
Sharpe-difference loss.

The tinygrad `block_studentized_sharpe_diff` must agree with the
numpy `studentized_sharpe_diff` to high precision so that:
  - It can be used as a training loss without numerical surprises
  - The downstream eval (Ledoit-Wolf CI) computes against the same
    statistic the gradient was optimizing
"""
from __future__ import annotations

import numpy as np
import pytest
from tinygrad.tensor import Tensor

from factor.objectives import (
    block_studentized_sharpe_diff, soft_excludes_zero_tensor,
)
from ss_portfolio import studentized_sharpe_diff, soft_excludes_zero


def test_tinygrad_matches_numpy_plain():
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.012, size=500)
    b = rng.normal(0.0,   0.011, size=500)

    np_t = studentized_sharpe_diff(a, b, with_moments=False)
    tg_t = float(block_studentized_sharpe_diff(
        Tensor(a.astype(np.float32)),
        Tensor(b.astype(np.float32)),
        with_moments=False,
    ).numpy())
    # float32 in tinygrad gives ~1e-3 relative agreement vs numpy float64
    assert tg_t == pytest.approx(np_t, rel=1e-2, abs=1e-3)


def test_tinygrad_matches_numpy_with_moments():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0008, 0.012, size=800)
    b = rng.normal(0.0,    0.011, size=800)

    np_t = studentized_sharpe_diff(a, b, with_moments=True)
    tg_t = float(block_studentized_sharpe_diff(
        Tensor(a.astype(np.float32)),
        Tensor(b.astype(np.float32)),
        with_moments=True,
    ).numpy())
    assert tg_t == pytest.approx(np_t, rel=1e-2, abs=1e-2)


def test_tinygrad_soft_excludes_matches_numpy():
    """At a fixed t-stat, the smooth indicator should agree between
    numpy and tinygrad to high precision."""
    for t in [-3.0, -1.0, 0.0, 1.96, 2.5, 3.5]:
        np_v = soft_excludes_zero(t, temperature=0.5)
        tg_v = float(soft_excludes_zero_tensor(
            Tensor([t], dtype='float32'), temperature=0.5
        ).numpy()[0])
        assert tg_v == pytest.approx(np_v, rel=5e-3, abs=1e-3), (
            f't={t}: np={np_v} vs tg={tg_v}')


def test_tinygrad_t_stat_sign_matches_numpy_at_short_n():
    """At workspace-relevant short sample sizes (vol-v3 n=33), the
    sign and order of magnitude must agree."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.02, 0.04, size=33)
    b = rng.normal(0.005, 0.025, size=33)
    np_t = studentized_sharpe_diff(a, b)
    tg_t = float(block_studentized_sharpe_diff(
        Tensor(a.astype(np.float32)),
        Tensor(b.astype(np.float32)),
    ).numpy())
    assert np.sign(np_t) == np.sign(tg_t)
    # Order-of-magnitude agreement at short n with f32 — relax tolerance
    assert tg_t == pytest.approx(np_t, rel=5e-2, abs=0.1)
