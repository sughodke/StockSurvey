"""DeepCFR core tests — RegretNet trains, buffer samples, regret matching masks."""
from __future__ import annotations

import numpy as np

import pytest


pytest.importorskip('tinygrad')   # skip whole module if tinygrad isn't available

from cfr.deep import (
    RegretNet, DeepCFRBuffer, policy_from_predicted_regret,
)


def test_regret_net_trains_to_lower_loss():
    """A few SGD steps should reduce MSE on a fixed batch."""
    net = RegretNet(n_features=10, n_actions=8, hidden=32, lr=1e-3)
    rng = np.random.default_rng(0)
    states = rng.normal(0, 1, size=(64, 10)).astype(np.float32)
    targets = rng.normal(0, 0.05, size=(64, 8)).astype(np.float32)
    avail = np.ones((64, 8), dtype=bool)
    losses = []
    for _ in range(20):
        loss = net.train_step(states, targets, avail)
        losses.append(loss)
    # Final loss should be at least ~half of initial
    assert np.isfinite(losses[0])
    assert np.isfinite(losses[-1])
    assert losses[-1] < losses[0] * 0.9, f'loss did not decrease: {losses[0]} → {losses[-1]}'


def test_regret_net_predict_shapes():
    net = RegretNet(n_features=6, n_actions=10, hidden=16)
    state = np.random.default_rng(0).normal(0, 1, size=6).astype(np.float32)
    pred_single = net.predict(state)
    assert pred_single.shape == (10,)
    states = np.random.default_rng(0).normal(0, 1, size=(5, 6)).astype(np.float32)
    pred_batch = net.predict(states)
    assert pred_batch.shape == (5, 10)


def test_buffer_append_and_sample():
    buf = DeepCFRBuffer(capacity=100)
    rng = np.random.default_rng(0)
    for _ in range(50):
        buf.append(
            rng.normal(0, 1, size=8).astype(np.float32),
            rng.normal(0, 0.05, size=4).astype(np.float32),
            np.ones(4, dtype=bool),
        )
    assert len(buf) == 50
    s, t, a = buf.sample_batch(16, rng)
    assert s.shape == (16, 8)
    assert t.shape == (16, 4)
    assert a.shape == (16, 4)


def test_buffer_capacity_evicts_oldest():
    buf = DeepCFRBuffer(capacity=10)
    rng = np.random.default_rng(0)
    for _ in range(15):
        buf.append(
            rng.normal(0, 1, size=4).astype(np.float32),
            rng.normal(0, 0.05, size=3).astype(np.float32),
            np.ones(3, dtype=bool),
        )
    assert len(buf) == 10


def test_policy_from_predicted_regret_masks_unavailable():
    """Unavailable actions get zero policy mass."""
    R_pred = np.array([0.5, -0.2, 0.3, 0.1, -0.4])
    avail = np.array([True, False, True, True, False])
    pi = policy_from_predicted_regret(R_pred, avail)
    assert pi[1] == 0.0
    assert pi[4] == 0.0
    assert np.isclose(pi.sum(), 1.0)
    # Positive entries are 0.5, 0.3, 0.1; sum 0.9; pi = [5/9, 0, 3/9, 1/9, 0]
    np.testing.assert_allclose(pi, [5/9, 0, 3/9, 1/9, 0])


def test_policy_falls_back_to_uniform_over_available():
    """Negative cumulative regret on all available → uniform over available."""
    R_pred = np.array([-0.5, -0.2, -0.3, -0.1, 0.0])
    avail = np.array([True, True, False, True, True])
    pi = policy_from_predicted_regret(R_pred, avail)
    assert pi[2] == 0.0
    # 4 available actions → uniform 1/4 each
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.0, 0.25, 0.25])


def test_policy_all_unavailable_returns_cash_at_index_0():
    """No available actions → all-cash (index 0)."""
    R_pred = np.array([0.3, 0.1, -0.2])
    avail = np.array([False, False, False])
    pi = policy_from_predicted_regret(R_pred, avail)
    assert pi[0] == 1.0
    assert pi[1] == 0.0
    assert pi[2] == 0.0
