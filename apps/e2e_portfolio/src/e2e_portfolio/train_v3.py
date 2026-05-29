"""v3 training: Sharpe loss on (equity_part + vol_scale * vol_part)."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from tinygrad import Tensor, nn

from e2e_portfolio.data_v3 import PanelV3
from e2e_portfolio.model_v3 import AllocatorV3, HparamsV3, sharpe_loss_v3


@dataclass
class TrainConfigV3:
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


def evaluate_sharpe_v3(model: AllocatorV3, panel: PanelV3,
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
        ew, vw, vs = model(xa, xm, vmask)
        eq_pnl = (ew * Tensor(fwd)).sum(axis=-1).numpy()
        vol_basket = (vw.numpy() * panel.fwd_vol_pnl[i:j]).sum(axis=-1)
        vol_pnl = vs.numpy() * vol_basket
        pnls.append(eq_pnl + vol_pnl)
    Tensor.training = prev
    pnls = np.concatenate(pnls)
    if pnls.std() <= 1e-9:
        return 0.0
    return float(pnls.mean() / pnls.std())


def train_one_fold_v3(
    panel_train: PanelV3,
    panel_val: PanelV3 | None,
    cfg: TrainConfigV3,
    hp: HparamsV3,
) -> tuple[AllocatorV3, dict]:
    rng = np.random.default_rng(cfg.seed)
    model = AllocatorV3(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)

    n_train = len(panel_train.dates)
    if n_train < cfg.batch_size:
        raise ValueError(f'train fold too small: {n_train}')

    best_val_sh = -np.inf
    best_params = [p.numpy().copy() for p in model.parameters()]
    history = {'step': [], 'train_loss': [], 'val_sharpe': []}

    Tensor.training = True
    t0 = time.time()
    for step in range(cfg.n_steps):
        idx = rng.integers(0, n_train, size=cfg.batch_size)
        xa = Tensor(panel_train.X_assets[idx])
        xm = Tensor(panel_train.X_macro[idx])
        vmask = Tensor(panel_train.valid_mask[idx])
        fwd = _attach_cash_zero(panel_train.fwd_ret[idx])
        vol_pnl = panel_train.fwd_vol_pnl[idx].astype(np.float32)
        opt.zero_grad()
        ew, vw, vs = model(xa, xm, vmask)
        loss = sharpe_loss_v3(ew, Tensor(fwd), vw, vs, Tensor(vol_pnl))
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            history['step'].append(step)
            history['train_loss'].append(float(loss.numpy()))

        if panel_val is not None and (step % cfg.val_every == 0
                                       or step == cfg.n_steps - 1):
            Tensor.training = False
            val_sh = evaluate_sharpe_v3(model, panel_val)
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
