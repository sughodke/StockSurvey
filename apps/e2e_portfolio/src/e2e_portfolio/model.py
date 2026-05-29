"""Tinygrad model: per-asset 1D conv encoder + macro MLP + cross-asset head.

Architecture (Zhang-Zohren-Roberts style):
  Per-asset encoder (shared across N assets):
    Conv1d(F_asset -> 32, k=5) -> ReLU
    Conv1d(32 -> 32, k=5) -> ReLU
    GlobalAvgPool over T
  Macro encoder:
    Linear(F_macro -> 32) -> ReLU -> Linear(32 -> 32) -> ReLU
    mean-pool over T
  Head:
    Concat (per-asset embedding, broadcast macro context) -> (B, N, 64)
    Linear(64 -> 32) -> ReLU -> Linear(32 -> 1) per asset
    Append a 1-logit cash position -> (B, N+1)
    Softmax over the N+1 dim
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, nn


@dataclass
class Hparams:
    n_assets: int = 13
    t_lookback: int = 60
    f_asset: int = 6
    f_macro: int = 4
    hidden: int = 32
    head_hidden: int = 32


class Allocator:
    """Direct-Sharpe portfolio allocator.

    Params are held as a flat list for the tinygrad optimizer.
    """

    def __init__(self, hp: Hparams, seed: int = 0):
        self.hp = hp
        Tensor.manual_seed(seed)
        # Per-asset 1D convs (shared weights).
        self.conv1 = nn.Conv1d(hp.f_asset, hp.hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hp.hidden, hp.hidden, kernel_size=5, padding=2)
        # Macro MLP.
        self.macro_w1 = Tensor.kaiming_uniform(hp.hidden, hp.f_macro)
        self.macro_b1 = Tensor.zeros(hp.hidden)
        self.macro_w2 = Tensor.kaiming_uniform(hp.hidden, hp.hidden)
        self.macro_b2 = Tensor.zeros(hp.hidden)
        # Head.
        head_in = hp.hidden * 2
        self.head_w1 = Tensor.kaiming_uniform(hp.head_hidden, head_in)
        self.head_b1 = Tensor.zeros(hp.head_hidden)
        self.head_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.head_b2 = Tensor.zeros(1)
        # Learned cash logit (single scalar broadcast across batch).
        self.cash_logit = Tensor.zeros(1)
        # Mark params trainable.
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self) -> list[Tensor]:
        return [
            self.conv1.weight, self.conv1.bias,
            self.conv2.weight, self.conv2.bias,
            self.macro_w1, self.macro_b1,
            self.macro_w2, self.macro_b2,
            self.head_w1, self.head_b1,
            self.head_w2, self.head_b2,
            self.cash_logit,
        ]

    def encode_assets(self, x: Tensor) -> Tensor:
        """x: (B, N, T, F_asset) -> (B, N, hidden)."""
        B, N, T, F = x.shape
        # Reshape to (B*N, F, T) for conv1d.
        x = x.reshape(B * N, T, F).transpose(1, 2)  # (B*N, F, T)
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        # Global average pool over T.
        h = h.mean(axis=-1)  # (B*N, hidden)
        return h.reshape(B, N, self.hp.hidden)

    def encode_macro(self, m: Tensor) -> Tensor:
        """m: (B, T, F_macro) -> (B, hidden)."""
        h = (m @ self.macro_w1.transpose() + self.macro_b1).relu()
        h = (h @ self.macro_w2.transpose() + self.macro_b2).relu()
        # mean-pool over T
        return h.mean(axis=1)

    def __call__(self, x_assets: Tensor, x_macro: Tensor) -> Tensor:
        """Returns weights (B, N+1) summing to 1 along last dim."""
        z_assets = self.encode_assets(x_assets)  # (B, N, H)
        z_macro = self.encode_macro(x_macro)     # (B, H)
        B, N, H = z_assets.shape
        # Broadcast macro to per-asset.
        z_macro_bn = z_macro.reshape(B, 1, H).expand(B, N, H)
        z = z_assets.cat(z_macro_bn, dim=-1)  # (B, N, 2H)
        # Head MLP (applied per asset).
        h = (z @ self.head_w1.transpose() + self.head_b1).relu()
        logits = (h @ self.head_w2.transpose() + self.head_b2).squeeze(-1)  # (B, N)
        # Append cash logit.
        cash = self.cash_logit.reshape(1, 1).expand(B, 1)
        full_logits = logits.cat(cash, dim=-1)  # (B, N+1)
        return full_logits.softmax(axis=-1)


def sharpe_loss(weights: Tensor, fwd_ret_plus_cash: Tensor) -> Tensor:
    """Negative annualized-ish Sharpe over batch.

    weights: (B, N+1)
    fwd_ret_plus_cash: (B, N+1) with cash leg already appended as zeros.
    """
    pnl = (weights * fwd_ret_plus_cash).sum(axis=-1)  # (B,)
    mean = pnl.mean()
    # Population std for stable backprop on small batches.
    var = ((pnl - mean) ** 2).mean()
    std = (var + 1e-6).sqrt()
    return -mean / std


def save_npz(model: Allocator, path: str) -> None:
    arrays = {}
    names = [
        'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b',
        'macro_w1', 'macro_b1', 'macro_w2', 'macro_b2',
        'head_w1', 'head_b1', 'head_w2', 'head_b2',
        'cash_logit',
    ]
    for n, p in zip(names, model.parameters()):
        arrays[n] = p.numpy()
    np.savez(path, **arrays)


def load_npz(model: Allocator, path: str) -> None:
    d = np.load(path)
    names = [
        'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b',
        'macro_w1', 'macro_b1', 'macro_w2', 'macro_b2',
        'head_w1', 'head_b1', 'head_w2', 'head_b2',
        'cash_logit',
    ]
    params = model.parameters()
    for n, p in zip(names, params):
        arr = d[n]
        p.assign(Tensor(arr))
