"""v3.7 model — v3.5 architecture + imitation-prior bias init + structural
floor on long-vol head.

Three load-bearing changes vs v3.5:

1. **Structural floor on `long_vol_position`.** The output is now
   `floor + (max - floor) * sigmoid(z_long)` rather than
   `max * sigmoid(z_long)`. The optimizer can never drive
   `long_vol_position` below `floor` — this mirrors ZZR's actual
   COVID-survival mechanism (their 4-asset softmax has no zero-floor,
   so VIXY is always present at some baseline weight regardless of
   what the policy learns about VIXY during calm periods).

2. **Bias initialization toward the deterministic recipe.**
   - Equity-head biases set so softmax(equity_logits) at init
     ≈ uniform over the 13 Phase 4d ETFs present in the universe,
     ~0 on all other K-13 candidates, ~0 on cash. This is the DCA
     basket as the starting allocation.
   - `short_vol_scale` bias set so sigmoid output ≈ 0.4 at init
     (→ short_vol_scale ≈ 2.0 — the canonical deterministic recipe).
   - `long_vol_position` bias set so sigmoid output ≈ 0 at init
     (→ long_vol_position ≈ floor). Deterministic recipe doesn't use
     long-vol; v3.7 starts there.

3. **The pre-floor `z_long` logit can still adapt to features** —
   gradient signal during stress regimes (high VIX percentile)
   should lift sigmoid output above 0, adding to floor. Floor sets
   a *minimum*, not a maximum.

Net effect: at step 0 the model emits ≈ deterministic recipe; at
step N the model is deterministic + a Sharpe-gradient-driven tilt
where tilts are justified. The optimizer is constrained from
discovering "turn off long-vol entirely" — exactly the failure mode
that killed v3.5 on fold-2 COVID.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tinygrad import Tensor, nn, dtypes

from ss_tinygrad import maybe_bf16, cast_back_fp32


# The 13-ETF Phase 4d basket — the canonical DCA universe.
PHASE4D_TICKERS = (
    'DBC', 'GLD', 'IEF', 'TLT',
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY',
)


@dataclass
class HparamsV3p7:
    n_assets: int = 200
    t_lookback: int = 60
    f_asset: int = 11
    f_macro: int = 4
    hidden: int = 32
    head_hidden: int = 32
    vol_scale_max: float = 5.0
    long_vol_max: float = 5.0
    long_vol_floor: float = 0.3              # NEW: structural floor — ZZR analog
    k_active: int = 50
    vol_temperature: float = 1.0
    use_bf16: bool = False  # opt-in; T4 CUDA path doesn't compile bf16 ops
    # Imitation-prior strengths (logit magnitudes at init):
    eq_bias_dca: float = 5.0                 # logit lift for the 13 Phase 4d ETFs
    eq_bias_other: float = -3.0              # logit suppression for other K-13 names
    eq_bias_cash: float = -3.0               # logit suppression for cash
    vs_bias_init: float = -0.4               # sigmoid(-0.4) ~ 0.4 -> svs ~ 2.0
    lv_bias_init: float = -6.0               # sigmoid(-6) ~ 0.0025 -> lvp -> floor


class AllocatorV3p7:
    def __init__(self, hp: HparamsV3p7, tickers: list[str], seed: int = 0):
        self.hp = hp
        self.tickers = list(tickers)
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

        # Equity head — output bias is PER-NAME so we can imitation-bias each
        # of the 13 Phase 4d ETFs.
        # Linear layer output shape (B, K, 1) — biases are scalar shared, so
        # we add a separate per-name bias vector after the linear.
        self.eq_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.eq_b_scalar = Tensor.zeros(1)
        self.eq_b_pername = self._build_equity_bias().contiguous().realize()

        # Cash logit — single learnable scalar, biased downward at init.
        self.cash_logit = (Tensor.zeros(1) + hp.eq_bias_cash).contiguous().realize()

        # Short-vol head.
        self.vol_w = Tensor.kaiming_uniform(1, hp.head_hidden)
        self.vol_b = Tensor.zeros(1)

        # Short-vol scale head — bias init for sigmoid out ≈ 0.4 → svs ≈ 2.0.
        # `vs_w2` zero-init so at step 0 the output depends only on `vs_b2`;
        # gradient still flows once optimization starts.
        self.vs_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.vs_b1 = Tensor.zeros(hp.head_hidden)
        self.vs_w2 = Tensor.zeros(1, hp.head_hidden)
        self.vs_b2 = (Tensor.zeros(1) + hp.vs_bias_init).contiguous().realize()

        # Long-vol position head — bias init for sigmoid out ≈ 0 → lvp ≈ floor.
        # `lv_w2` zero-init for same reason.
        self.lv_w1 = Tensor.kaiming_uniform(hp.head_hidden, hp.hidden * 2)
        self.lv_b1 = Tensor.zeros(hp.head_hidden)
        self.lv_w2 = Tensor.zeros(1, hp.head_hidden)
        self.lv_b2 = (Tensor.zeros(1) + hp.lv_bias_init).contiguous().realize()

        for p in self.parameters():
            p.requires_grad = True

    def _build_equity_bias(self) -> Tensor:
        """Per-name bias vector — large positive at Phase 4d ETFs in our
        universe, large negative elsewhere. This anchors the equity head
        at init to the DCA basket."""
        bias = np.full(self.hp.n_assets, self.hp.eq_bias_other, dtype=np.float32)
        present = 0
        for i, t in enumerate(self.tickers):
            if t in PHASE4D_TICKERS:
                bias[i] = self.hp.eq_bias_dca
                present += 1
        # `present` will be <=13 (the panel only contains DoltHub-covered names;
        # typically 9 of 13). Documented in the finding.
        return Tensor(bias)

    def parameters(self) -> list[Tensor]:
        return [
            self.conv1.weight, self.conv1.bias,
            self.conv2.weight, self.conv2.bias,
            self.macro_w1, self.macro_b1,
            self.macro_w2, self.macro_b2,
            self.body_w, self.body_b,
            self.eq_w, self.eq_b_scalar, self.eq_b_pername,
            self.cash_logit,
            self.vol_w, self.vol_b,
            self.vs_w1, self.vs_b1,
            self.vs_w2, self.vs_b2,
            self.lv_w1, self.lv_b1,
            self.lv_w2, self.lv_b2,
        ]

    def encode_names(self, x: Tensor) -> Tensor:
        B, K, T, F = x.shape
        x = maybe_bf16(x.reshape(B * K, T, F).transpose(1, 2),
                       self.hp.use_bf16)
        h = self.conv1(x).relu()
        h = self.conv2(h).relu()
        h = cast_back_fp32(h.mean(axis=-1))
        return h.reshape(B, K, self.hp.hidden)

    def encode_macro(self, m: Tensor) -> Tensor:
        mc = maybe_bf16(m, self.hp.use_bf16)
        h = (mc @ self.macro_w1.transpose() + self.macro_b1).relu()
        h = (h @ self.macro_w2.transpose() + self.macro_b2).relu()
        return cast_back_fp32(h.mean(axis=1))

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

        # Equity head — scalar bias from Linear + per-name imitation bias.
        eq_linear = (body @ self.eq_w.transpose() + self.eq_b_scalar).squeeze(-1)
        eq_logits = eq_linear + self.eq_b_pername.reshape(1, K)
        cash = self.cash_logit.reshape(1, 1).expand(B, 1)
        eq_full = eq_logits.cat(cash, dim=-1)
        equity_weights = eq_full.softmax(axis=-1)

        # Short-vol head.
        vol_logits_raw = (body @ self.vol_w.transpose() + self.vol_b).squeeze(-1)
        vol_logits = vol_logits_raw / self.hp.vol_temperature
        if valid_mask is not None:
            vol_logits = vol_logits + (valid_mask - 1.0) * 1e6
        vol_weights = vol_logits.softmax(axis=-1)

        # Pool body + macro for scale heads.
        pooled = body.mean(axis=1)
        vs_in = pooled.cat(z_macro, dim=-1)

        # Short-vol scale: 5 * sigmoid (bias-init to ~2.0).
        vh = (vs_in @ self.vs_w1.transpose() + self.vs_b1).relu()
        z_vs = (vh @ self.vs_w2.transpose() + self.vs_b2).squeeze(-1)
        short_vol_scale = z_vs.sigmoid() * self.hp.vol_scale_max

        # Long-vol position with structural floor:
        #     lvp = floor + (max - floor) * sigmoid(z_lv)
        # At init z_lv ≈ -6 → sigmoid ~ 0.0025 → lvp ≈ floor. The Sharpe
        # gradient can lift sigmoid output (up to 1.0 → lvp = max) but
        # cannot drive lvp below floor — closing the v3.5 fold-2 failure
        # mode in which the optimizer pushed z_long to -8.5 and turned
        # off long-vol entirely.
        lh = (vs_in @ self.lv_w1.transpose() + self.lv_b1).relu()
        z_lv = (lh @ self.lv_w2.transpose() + self.lv_b2).squeeze(-1)
        long_vol_position = (
            self.hp.long_vol_floor
            + (self.hp.long_vol_max - self.hp.long_vol_floor) * z_lv.sigmoid()
        )

        return equity_weights, vol_weights, short_vol_scale, long_vol_position


def sharpe_loss_v3p7(equity_weights: Tensor, fwd_ret_plus_cash: Tensor,
                     vol_weights: Tensor, short_vol_scale: Tensor,
                     fwd_vol_pnl_per_name: Tensor,
                     long_vol_position: Tensor,
                     fwd_long_vol_ret: Tensor) -> Tensor:
    """Same Sharpe loss as v3.5 — only the policy head outputs change."""
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
    'eq_w', 'eq_b_scalar', 'eq_b_pername',
    'cash_logit',
    'vol_w', 'vol_b',
    'vs_w1', 'vs_b1', 'vs_w2', 'vs_b2',
    'lv_w1', 'lv_b1', 'lv_w2', 'lv_b2',
]


def save_npz(model: AllocatorV3p7, path: str) -> None:
    arrays = {n: p.numpy() for n, p in zip(_PARAM_NAMES, model.parameters())}
    np.savez(path, **arrays)
