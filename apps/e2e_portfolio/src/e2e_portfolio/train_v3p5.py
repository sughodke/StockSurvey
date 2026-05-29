"""v3.5 training: Sharpe loss on (equity + short-vol + long-vol).

JIT pattern: pre-allocate buffer Tensors of fixed shape outside the
training loop, copy random-batch contents into them with `.assign(...)`,
then call a `@TinyJit`-wrapped step function that takes the buffer
tensors as inputs. The JIT records the kernel schedule on the 3rd call
and replays it for every subsequent step — same pattern proven in
`apps/factor/cwt_gru_walkforward.py` (4x speedup vs un-jitted).
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from tinygrad import Tensor, TinyJit, nn

from e2e_portfolio.data_v3p5 import PanelV3p5
from e2e_portfolio.model_v3p5 import AllocatorV3p5, HparamsV3p5, sharpe_loss_v3p5


@dataclass
class TrainConfigV3p5:
    n_steps: int = 5000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    log_every: int = 200
    val_every: int = 500


def _attach_cash_zero(fwd_ret_np: np.ndarray) -> np.ndarray:
    B, K = fwd_ret_np.shape
    out = np.zeros((B, K + 1), dtype=np.float32)
    out[:, :K] = fwd_ret_np
    return out


def evaluate_sharpe_v3p5(model: AllocatorV3p5, panel: PanelV3p5,
                         batch_size: int = 128) -> float:
    n = len(panel.dates)
    if n == 0:
        return float('nan')
    prev = Tensor.training
    Tensor.training = False
    pnls = []
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xa = Tensor(panel.X_assets[i:j])
        xm = Tensor(panel.X_macro[i:j])
        vmask = Tensor(panel.valid_mask[i:j])
        fwd = _attach_cash_zero(panel.fwd_ret[i:j])
        ew, vw, svs, lvp = model(xa, xm, vmask)
        eq_pnl = (ew * Tensor(fwd)).sum(axis=-1).numpy()
        vol_basket = (vw.numpy() * panel.fwd_vol_pnl[i:j]).sum(axis=-1)
        short_vol_pnl = svs.numpy() * vol_basket
        long_vol_pnl = lvp.numpy() * panel.fwd_long_vol_ret[i:j]
        pnls.append(eq_pnl + short_vol_pnl + long_vol_pnl)
    Tensor.training = prev
    pnls = np.concatenate(pnls)
    if pnls.std() <= 1e-9:
        return 0.0
    return float(pnls.mean() / pnls.std())


def train_one_fold_v3p5(
    panel_train: PanelV3p5,
    panel_val: PanelV3p5 | None,
    cfg: TrainConfigV3p5,
    hp: HparamsV3p5,
) -> tuple[AllocatorV3p5, dict]:
    rng = np.random.default_rng(cfg.seed)
    model = AllocatorV3p5(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)

    n_train = len(panel_train.dates)
    if n_train < cfg.batch_size:
        raise ValueError(f'train fold too small: {n_train}')

    B = cfg.batch_size
    K = hp.n_assets
    T = hp.t_lookback
    F = hp.f_asset
    Fm = hp.f_macro

    # Pre-allocate buffer Tensors. Same Tensor objects reused every step;
    # only the underlying data changes via `.assign()`. This lets TinyJit
    # bind to stable Tensor identities and replay the kernel schedule.
    xa_buf = Tensor.zeros(B, K, T, F).contiguous().realize()
    xm_buf = Tensor.zeros(B, T, Fm).contiguous().realize()
    vmask_buf = Tensor.zeros(B, K).contiguous().realize()
    fwd_buf = Tensor.zeros(B, K + 1).contiguous().realize()
    vol_pnl_buf = Tensor.zeros(B, K).contiguous().realize()
    long_vol_buf = Tensor.zeros(B).contiguous().realize()

    @TinyJit
    def _train_step() -> Tensor:
        opt.zero_grad()
        ew, vw, svs, lvp = model(xa_buf, xm_buf, vmask_buf)
        loss = sharpe_loss_v3p5(ew, fwd_buf, vw, svs, vol_pnl_buf,
                                 lvp, long_vol_buf)
        loss.backward()
        opt.step()
        return loss.realize()

    best_val_sh = -np.inf
    best_params = [p.numpy().copy() for p in model.parameters()]
    history = {'step': [], 'train_loss': [], 'val_sharpe': []}

    Tensor.training = True
    t0 = time.time()
    for step in range(cfg.n_steps):
        idx = rng.integers(0, n_train, size=B)
        # Copy fresh per-step random batch into the persistent buffers.
        xa_buf.assign(Tensor(panel_train.X_assets[idx])).realize()
        xm_buf.assign(Tensor(panel_train.X_macro[idx])).realize()
        vmask_buf.assign(Tensor(panel_train.valid_mask[idx])).realize()
        fwd_buf.assign(Tensor(_attach_cash_zero(panel_train.fwd_ret[idx]))).realize()
        vol_pnl_buf.assign(Tensor(
            panel_train.fwd_vol_pnl[idx].astype(np.float32))).realize()
        long_vol_buf.assign(Tensor(
            panel_train.fwd_long_vol_ret[idx].astype(np.float32))).realize()

        loss = _train_step()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            history['step'].append(step)
            history['train_loss'].append(float(loss.numpy()))

        if panel_val is not None and (step % cfg.val_every == 0
                                       or step == cfg.n_steps - 1):
            Tensor.training = False
            val_sh = evaluate_sharpe_v3p5(model, panel_val)
            Tensor.training = True
            history['val_sharpe'].append((step, val_sh))
            if val_sh > best_val_sh + 1e-4:
                best_val_sh = val_sh
                best_params = [p.numpy().copy() for p in model.parameters()]
            print(f'  step {step:5d}  loss={float(loss.numpy()):+.4f}  '
                  f'val_sh={val_sh:+.3f}  best={best_val_sh:+.3f}  '
                  f't={time.time()-t0:.0f}s', flush=True)

    for p, arr in zip(model.parameters(), best_params):
        p.assign(Tensor(arr.astype(np.float32)))
    Tensor.training = False
    return model, history
