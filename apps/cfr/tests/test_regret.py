"""Counterfactual regret math invariants."""
from __future__ import annotations

import numpy as np

from cfr.regret import (
    compute_block_log_returns, compute_block_regrets, regret_matching,
    sample_action,
)


def test_block_log_return_cash_is_zero():
    """Cash (all-zero weights) → zero log return regardless of forward."""
    ret = np.array([0.05, -0.02, 0.10])
    w = np.zeros((1, 3))
    r = compute_block_log_returns(ret, w)
    assert np.allclose(r, 0.0)


def test_block_log_return_full_invest():
    """Single-name long → that name's log return."""
    ret = np.array([0.05, -0.02, 0.10])
    w = np.eye(3)
    r = compute_block_log_returns(ret, w)
    # log(1 * exp(0.05)) = 0.05 exactly
    np.testing.assert_allclose(r, ret, atol=1e-12)


def test_block_log_return_small_return_linear():
    """For small returns the answer is approximately w · r."""
    ret = np.array([0.001, -0.0005, 0.0008])
    w = np.array([[0.4, 0.3, 0.3]])
    r = compute_block_log_returns(ret, w)
    linear = w @ ret
    np.testing.assert_allclose(r, linear, atol=1e-6)


def test_regret_at_played_is_zero():
    """Regret of the played action against itself is identically zero."""
    ret = np.array([0.03, -0.01, 0.02])
    w = np.array([
        [1.0, 0.0, 0.0],  # action 0: hold ticker 0
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    regrets = compute_block_regrets(ret, w, played_action=1)
    assert regrets[1] == 0.0


def test_regret_sign_consistency():
    """The best-returning action has the highest regret-vs-played."""
    ret = np.array([0.10, 0.00, -0.05])
    w = np.eye(3)
    regrets = compute_block_regrets(ret, w, played_action=2)
    assert np.argmax(regrets) == 0   # action 0 was best
    assert regrets[0] > 0
    assert regrets[2] == 0


def test_regret_matching_positive_only():
    """Negative cumulative regret entries don't contribute."""
    R = np.array([1.0, -2.0, 0.5])
    pi = regret_matching(R)
    np.testing.assert_allclose(pi, np.array([2/3, 0.0, 1/3]))


def test_regret_matching_all_nonpositive_uniform():
    R = np.array([-1.0, -2.0, 0.0])
    pi = regret_matching(R, uniform_fallback=True)
    np.testing.assert_allclose(pi, np.array([1/3, 1/3, 1/3]))


def test_regret_matching_no_fallback_zero():
    R = np.array([-1.0, -2.0])
    pi = regret_matching(R, uniform_fallback=False)
    np.testing.assert_allclose(pi, np.array([0.0, 0.0]))


def test_sample_action_respects_policy():
    """Single-action policy always picks that action."""
    pi = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_action(pi, rng) == 1
