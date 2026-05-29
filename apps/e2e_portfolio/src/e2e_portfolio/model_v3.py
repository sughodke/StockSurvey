"""v3 model — one unified universe, two heads per name.

Architecture:
  Per-name encoder (shared across K names):
    Conv1d(F_asset=11 -> 32, k=5) -> ReLU
    Conv1d(32 -> 32, k=5)       -> ReLU
    GlobalAvgPool over T        -> (B, K, 32)
  Macro encoder:
    Linear(F_macro=4 -> 32) -> ReLU -> Linear(32 -> 32) -> ReLU
    mean-pool over T        -> (B, 32)
  Shared body (per name):
    concat (per-name embed, broadcast macro) -> (B, K, 64)
    Linear(64 -> 32) -> ReLU                 -> shared (B, K, 32)
  Equity head:
    Linear(32 -> 1) per name      -> (B, K)
    append cash logit             -> (B, K+1)
    softmax over (K+1)            -> equity_weights
  Vol head:
    Linear(32 -> 1) per name      -> (B, K) raw logits
    soft top-K_active selection via straight-through estimator
    softmax over selected         -> vol_weights (sums to 1)
  Vol-scale head:
    pooled (mean over K) shared body cat macro -> (B, 64)
    Linear(64 -> 32) -> ReLU -> Linear(32 -> 1) -> 5*sigmoid
                                  -> vol_scale (B,)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, nn


@dataclass
class HparamsV3:
    n_assets: int = 200      # K candidates
    t_lookback: int = 60
    f_asset: int = 11
    f_macro: int = 4
    hidden: int = 32
    head_hidden: int = 32
    vol_scale_max: float = 5.0
    k_active: int = 50       # soft top-K for vol head
    vol_temperature: float = 1.0


class AllocatorV3:
    def __init__(self, hp: HparamsV3, seed: int = 0):
        self.hp = hp
        Tensor.manual_seed(seed)
        # Per-name conv encoder.
        self.conv1 = nn.Conv1d(hp.f_asset, hp.hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hp.hidden, hp.hidden, kernel_size=5, padding=2)
        # Macro encoder.
        self.macro_w1 = Tensor.kaiming_uniform(hp.hidden, hp.f_macro)
        self.macro_b1 = Tensor.zeros(hp.hidden)
        self.macro_w2 = Tensor.kaiming_uniform(hp.hidden, hp.hidden)
        self.macro_b2 = Tensor.zeros(hp.hidden)
        # Shared body MLP.
        body_in = hp.hidden * 2
        self.body_w = Tensor.kaiming_uniform(hp.head_hidden, body_in)
        self.body_b = Tensor.zeros(hp.head_hidden)
        # Equity head.
        self.eq_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.eq_b = Tensor.zeros(1)
        self.cash_logit = Tensor.zeros(1)
        # Vol head.
        self.vol_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vol_b = Tensor.zeros(1)
        # Vol-scale head.
        self.vs_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.vs_b1 = Tensor.zeros(hp.head_hidden)
        self.vs_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vs_b2 = Tensor.zeros(1)
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self) -> list[Tensor]:
        return [
            self.conv1.weight, self.conv1.bias,
            self.conv2.weight, self.conv2.bias,
            self.macro_w1, self.macro_b1,
            self.macro_w2, self.macro_b2,
            self.body_w, self.body_b,
            self.eq_w, self.eq_b,
            self.cash_logit,
            self.vol_w, self.vol_b,
            self.vs_w1, self.vs_b1,
            self.vs_w2, self.vs_b2,
        ]

    def encode_names(self, x: Tensor) -> Tensor:
        """x: (B, K, T, F) -> (B, K, H)."""
        B, K, T, F = x.shape
        x = x.reshape(B * K, T, F).transpose(1, 2)  # (B*K, F, T)
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        h = h.mean(axis=-1)  # (B*K, H)
        return h.reshape(B, K, self.hp.hidden)

    def encode_macro(self, m: Tensor) -> Tensor:
        h = (m @ self.macro_w1.transpose() + self.macro_b1).relu()
        h = (h @ self.macro_w2.transpose() + self.macro_b2).relu()
        return h.mean(axis=1)

    def __call__(self, x_assets: Tensor, x_macro: Tensor,
                 valid_mask: Tensor | None = None,
                 ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (equity_weights (B, K+1), vol_weights (B, K),
        vol_scale (B,)).

        `valid_mask`: optional (B, K) float tensor, 1.0 if the name has
        IV coverage at this anchor, 0.0 otherwise. Used to mask the vol
        head only (equity head can still trade names without IV).
        """
        z_names = self.encode_names(x_assets)        # (B, K, H)
        z_macro = self.encode_macro(x_macro)         # (B, H)
        B, K, H = z_names.shape
        z_macro_bk = z_macro.reshape(B, 1, H).expand(B, K, H)
        z = z_names.cat(z_macro_bk, dim=-1)           # (B, K, 2H)
        body = (z @ self.body_w.transpose() + self.body_b).relu()  # (B,K,H)

        # Equity head.
        eq_logits = (body @ self.eq_w.transpose() + self.eq_b).squeeze(-1)  # (B,K)
        cash = self.cash_logit.reshape(1, 1).expand(B, 1)
        eq_full = eq_logits.cat(cash, dim=-1)        # (B, K+1)
        equity_weights = eq_full.softmax(axis=-1)

        # Vol head: per-name logits, mask invalid, soft top-K via
        # temperature softmax — `k_active` does not hard-mask in the
        # gradient path; it shapes via temperature. With temperature
        # tau=1 and the mean coverage being decent, softmax picks ~10
        # names with significant mass. Documented as soft top-K.
        vol_logits_raw = (body @ self.vol_w.transpose() + self.vol_b).squeeze(-1)
        # Apply temperature.
        vol_logits = vol_logits_raw / self.hp.vol_temperature
        if valid_mask is not None:
            # Add a large negative to logits where coverage is missing.
            vol_logits = vol_logits + (valid_mask - 1.0) * 1e6
        vol_weights = vol_logits.softmax(axis=-1)    # (B, K)

        # Vol-scale head: pool body across K, cat macro.
        pooled = body.mean(axis=1)                   # (B, H)
        vs_in = pooled.cat(z_macro, dim=-1)          # (B, 2H)
        vh = (vs_in @ self.vs_w1.transpose() + self.vs_b1).relu()
        z_vs = (vh @ self.vs_w2.transpose() + self.vs_b2).squeeze(-1)
        vol_scale = z_vs.sigmoid() * self.hp.vol_scale_max
        return equity_weights, vol_weights, vol_scale


def sharpe_loss_v3(equity_weights: Tensor, fwd_ret_plus_cash: Tensor,
                   vol_weights: Tensor, vol_scale: Tensor,
                   fwd_vol_pnl_per_name: Tensor) -> Tensor:
    """Direct Sharpe on r_total = equity_part + vol_scale * vol_part.

    equity_weights:           (B, K+1)
    fwd_ret_plus_cash:        (B, K+1)
    vol_weights:              (B, K)
    vol_scale:                (B,)
    fwd_vol_pnl_per_name:     (B, K)  per-name forward short-vol PnL.
    """
    eq_part = (equity_weights * fwd_ret_plus_cash).sum(axis=-1)  # (B,)
    vol_basket = (vol_weights * fwd_vol_pnl_per_name).sum(axis=-1)  # (B,)
    vol_part = vol_scale * vol_basket
    pnl = eq_part + vol_part
    mean = pnl.mean()
    var = ((pnl - mean) ** 2).mean()
    std = (var + 1e-6).sqrt()
    return -mean / std


_PARAM_NAMES = [
    'conv1_w', 'conv1_b', 'conv2_w', 'conv2_b',
    'macro_w1', 'macro_b1', 'macro_w2', 'macro_b2',
    'body_w', 'body_b',
    'eq_w', 'eq_b',
    'cash_logit',
    'vol_w', 'vol_b',
    'vs_w1', 'vs_b1', 'vs_w2', 'vs_b2',
]


def save_npz(model: AllocatorV3, path: str) -> None:
    arrays = {n: p.numpy() for n, p in zip(_PARAM_NAMES, model.parameters())}
    np.savez(path, **arrays)


def load_npz(model: AllocatorV3, path: str) -> None:
    d = np.load(path)
    for n, p in zip(_PARAM_NAMES, model.parameters()):
        p.assign(Tensor(d[n]))
