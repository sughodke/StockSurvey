"""v3.5 model — extends v3 with a single long-vol output head.

Identical to v3 in every respect except for ONE additional scalar
output head:

  long_vol_position = long_vol_max * sigmoid(z_long)

The long_vol_position multiplies a synthetic long-VIX daily return
(from VIXY daily close-to-close return, computed in eval). This is
the Zhang-Zohren-Roberts 2020 analog: their model survived COVID
because the action menu included VIXY; v3 lacked this and could only
turn short-vol OFF, not hedge.

All other architecture choices preserved verbatim from v3: per-name
1D conv encoder, macro MLP, shared body MLP, equity-head softmax,
short-vol-head softmax, scalar short-vol_scale head, hparams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, nn


@dataclass
class HparamsV3p5:
    n_assets: int = 200
    t_lookback: int = 60
    f_asset: int = 11
    f_macro: int = 4
    hidden: int = 32
    head_hidden: int = 32
    vol_scale_max: float = 5.0
    long_vol_max: float = 5.0     # NEW: matching short-vol scale envelope
    k_active: int = 50
    vol_temperature: float = 1.0


class AllocatorV3p5:
    def __init__(self, hp: HparamsV3p5, seed: int = 0):
        self.hp = hp
        Tensor.manual_seed(seed)
        self.conv1 = nn.Conv1d(hp.f_asset, hp.hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hp.hidden, hp.hidden, kernel_size=5, padding=2)
        self.macro_w1 = Tensor.kaiming_uniform(hp.hidden, hp.f_macro)
        self.macro_b1 = Tensor.zeros(hp.hidden)
        self.macro_w2 = Tensor.kaiming_uniform(hp.hidden, hp.hidden)
        self.macro_b2 = Tensor.zeros(hp.hidden)
        body_in = hp.hidden * 2
        self.body_w = Tensor.kaiming_uniform(hp.head_hidden, body_in)
        self.body_b = Tensor.zeros(hp.head_hidden)
        self.eq_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.eq_b = Tensor.zeros(1)
        self.cash_logit = Tensor.zeros(1)
        self.vol_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vol_b = Tensor.zeros(1)
        # Short-vol scale head.
        self.vs_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.vs_b1 = Tensor.zeros(hp.head_hidden)
        self.vs_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vs_b2 = Tensor.zeros(1)
        # NEW: long-vol scale head — same shape as short-vol scale head.
        self.lv_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.lv_b1 = Tensor.zeros(hp.head_hidden)
        self.lv_w2 = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.lv_b2 = Tensor.zeros(1)
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
            self.lv_w1, self.lv_b1,
            self.lv_w2, self.lv_b2,
        ]

    def encode_names(self, x: Tensor) -> Tensor:
        B, K, T, F = x.shape
        x = x.reshape(B * K, T, F).transpose(1, 2)
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        h = h.mean(axis=-1)
        return h.reshape(B, K, self.hp.hidden)

    def encode_macro(self, m: Tensor) -> Tensor:
        h = (m @ self.macro_w1.transpose() + self.macro_b1).relu()
        h = (h @ self.macro_w2.transpose() + self.macro_b2).relu()
        return h.mean(axis=1)

    def __call__(self, x_assets: Tensor, x_macro: Tensor,
                 valid_mask: Tensor | None = None,
                 ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (equity_weights (B, K+1), vol_weights (B, K),
        short_vol_scale (B,), long_vol_position (B,))."""
        z_names = self.encode_names(x_assets)
        z_macro = self.encode_macro(x_macro)
        B, K, H = z_names.shape
        z_macro_bk = z_macro.reshape(B, 1, H).expand(B, K, H)
        z = z_names.cat(z_macro_bk, dim=-1)
        body = (z @ self.body_w.transpose() + self.body_b).relu()

        # Equity head.
        eq_logits = (body @ self.eq_w.transpose() + self.eq_b).squeeze(-1)
        cash = self.cash_logit.reshape(1, 1).expand(B, 1)
        eq_full = eq_logits.cat(cash, dim=-1)
        equity_weights = eq_full.softmax(axis=-1)

        # Short-vol head (per-name).
        vol_logits_raw = (body @ self.vol_w.transpose() + self.vol_b).squeeze(-1)
        vol_logits = vol_logits_raw / self.hp.vol_temperature
        if valid_mask is not None:
            vol_logits = vol_logits + (valid_mask - 1.0) * 1e6
        vol_weights = vol_logits.softmax(axis=-1)

        # Short-vol scale head.
        pooled = body.mean(axis=1)
        vs_in = pooled.cat(z_macro, dim=-1)
        vh = (vs_in @ self.vs_w1.transpose() + self.vs_b1).relu()
        z_vs = (vh @ self.vs_w2.transpose() + self.vs_b2).squeeze(-1)
        short_vol_scale = z_vs.sigmoid() * self.hp.vol_scale_max

        # NEW: long-vol position head — independent scalar from same
        # pooled body + macro context. Sigmoid output * long_vol_max.
        lh = (vs_in @ self.lv_w1.transpose() + self.lv_b1).relu()
        z_lv = (lh @ self.lv_w2.transpose() + self.lv_b2).squeeze(-1)
        long_vol_position = z_lv.sigmoid() * self.hp.long_vol_max

        return equity_weights, vol_weights, short_vol_scale, long_vol_position


def sharpe_loss_v3p5(equity_weights: Tensor, fwd_ret_plus_cash: Tensor,
                     vol_weights: Tensor, short_vol_scale: Tensor,
                     fwd_vol_pnl_per_name: Tensor,
                     long_vol_position: Tensor,
                     fwd_long_vol_ret: Tensor) -> Tensor:
    """Direct Sharpe on r_total = equity + short_vol + long_vol.

    equity_weights:          (B, K+1)
    fwd_ret_plus_cash:       (B, K+1)
    vol_weights:             (B, K)
    short_vol_scale:         (B,)
    fwd_vol_pnl_per_name:    (B, K)
    long_vol_position:       (B,)
    fwd_long_vol_ret:        (B,)  — forward VIXY daily-equivalent return.
    """
    eq_part = (equity_weights * fwd_ret_plus_cash).sum(axis=-1)
    vol_basket = (vol_weights * fwd_vol_pnl_per_name).sum(axis=-1)
    short_vol_part = short_vol_scale * vol_basket
    long_vol_part = long_vol_position * fwd_long_vol_ret
    pnl = eq_part + short_vol_part + long_vol_part
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
    'lv_w1', 'lv_b1', 'lv_w2', 'lv_b2',
]


def save_npz(model: AllocatorV3p5, path: str) -> None:
    arrays = {n: p.numpy() for n, p in zip(_PARAM_NAMES, model.parameters())}
    np.savez(path, **arrays)


def load_npz(model: AllocatorV3p5, path: str) -> None:
    d = np.load(path)
    for n, p in zip(_PARAM_NAMES, model.parameters()):
        p.assign(Tensor(d[n]))
