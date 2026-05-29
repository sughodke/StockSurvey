"""v2 model: v1 architecture + IV-aware features + continuous vol_position head.

Per-asset encoder accepts F_ASSET=12 instead of 6. After the pooled
(N, hidden) cross-asset embedding, a tiny pooled-state MLP emits a
single scalar `z_vol` -> `vol_position = 2.5 * sigmoid(z_vol)` that
multiplies a synthetic short-vol return computed from raw IV/HV.

Total daily return:
    r_total = w_etf_softmax @ asset_returns + vol_position * r_short_vol_synth
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, nn


@dataclass
class HparamsV2:
    n_assets: int = 13
    t_lookback: int = 60
    f_asset: int = 12
    f_macro: int = 4
    hidden: int = 32
    head_hidden: int = 32
    vol_pos_max: float = 2.5


class AllocatorV2:
    def __init__(self, hp: HparamsV2, seed: int = 0):
        self.hp = hp
        Tensor.manual_seed(seed)
        self.conv1 = nn.Conv1d(hp.f_asset, hp.hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hp.hidden, hp.hidden, kernel_size=5, padding=2)
        self.macro_w1 = Tensor.kaiming_uniform(hp.hidden, hp.f_macro)
        self.macro_b1 = Tensor.zeros(hp.hidden)
        self.macro_w2 = Tensor.kaiming_uniform(hp.hidden, hp.hidden)
        self.macro_b2 = Tensor.zeros(hp.hidden)
        head_in = hp.hidden * 2
        self.head_w1 = Tensor.kaiming_uniform(hp.head_hidden, head_in)
        self.head_b1 = Tensor.zeros(hp.head_hidden)
        self.head_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.head_b2 = Tensor.zeros(1)
        self.cash_logit = Tensor.zeros(1)
        # vol_position head: pooled state -> scalar.
        # Pooled state = mean(per-asset embedding) cat macro embedding.
        self.vol_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.vol_b1 = Tensor.zeros(hp.head_hidden)
        self.vol_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vol_b2 = Tensor.zeros(1)
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
            self.vol_w1, self.vol_b1,
            self.vol_w2, self.vol_b2,
        ]

    def encode_assets(self, x: Tensor) -> Tensor:
        B, N, T, F = x.shape
        x = x.reshape(B * N, T, F).transpose(1, 2)
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        h = h.mean(axis=-1)
        return h.reshape(B, N, self.hp.hidden)

    def encode_macro(self, m: Tensor) -> Tensor:
        h = (m @ self.macro_w1.transpose() + self.macro_b1).relu()
        h = (h @ self.macro_w2.transpose() + self.macro_b2).relu()
        return h.mean(axis=1)

    def __call__(self, x_assets: Tensor, x_macro: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (weights (B, N+1), vol_position (B,))."""
        z_assets = self.encode_assets(x_assets)
        z_macro = self.encode_macro(x_macro)
        B, N, H = z_assets.shape
        z_macro_bn = z_macro.reshape(B, 1, H).expand(B, N, H)
        z = z_assets.cat(z_macro_bn, dim=-1)
        h = (z @ self.head_w1.transpose() + self.head_b1).relu()
        logits = (h @ self.head_w2.transpose() + self.head_b2).squeeze(-1)
        cash = self.cash_logit.reshape(1, 1).expand(B, 1)
        full_logits = logits.cat(cash, dim=-1)
        weights = full_logits.softmax(axis=-1)
        # vol_position head: pool per-asset embedding then cat macro.
        pooled = z_assets.mean(axis=1)  # (B, H)
        pooled_z = pooled.cat(z_macro, dim=-1)  # (B, 2H)
        vh = (pooled_z @ self.vol_w1.transpose() + self.vol_b1).relu()
        z_vol = (vh @ self.vol_w2.transpose() + self.vol_b2).squeeze(-1)  # (B,)
        vol_position = z_vol.sigmoid() * self.hp.vol_pos_max
        return weights, vol_position


def sharpe_loss_v2(weights: Tensor, fwd_ret_plus_cash: Tensor,
                   vol_position: Tensor, fwd_vol_pnl: Tensor) -> Tensor:
    """Direct Sharpe on total = etf + vol_position * synthetic_short_vol_pnl."""
    pnl_etf = (weights * fwd_ret_plus_cash).sum(axis=-1)  # (B,)
    pnl = pnl_etf + vol_position * fwd_vol_pnl
    mean = pnl.mean()
    var = ((pnl - mean) ** 2).mean()
    std = (var + 1e-6).sqrt()
    return -mean / std


_PARAM_NAMES = [
    'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b',
    'macro_w1', 'macro_b1', 'macro_w2', 'macro_b2',
    'head_w1', 'head_b1', 'head_w2', 'head_b2',
    'cash_logit',
    'vol_w1', 'vol_b1', 'vol_w2', 'vol_b2',
]


def save_npz(model: AllocatorV2, path: str) -> None:
    arrays = {n: p.numpy() for n, p in zip(_PARAM_NAMES, model.parameters())}
    np.savez(path, **arrays)


def load_npz(model: AllocatorV2, path: str) -> None:
    d = np.load(path)
    for n, p in zip(_PARAM_NAMES, model.parameters()):
        p.assign(Tensor(d[n]))
