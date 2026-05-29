"""v2 training loop with dual-head Sharpe loss."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from tinygrad import Tensor, nn

from e2e_portfolio.data_v2 import PanelV2
from e2e_portfolio.model_v2 import AllocatorV2, HparamsV2, sharpe_loss_v2


@dataclass
class TrainConfigV2:
    n_steps: int = 5000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    log_every: int = 200
    val_every: int = 500
    early_stop_patience: int = 100  # effectively disabled; diamond hands


def _attach_cash_zero(fwd_ret_np: np.ndarray) -> np.ndarray:
    B, N = fwd_ret_np.shape
    out = np.zeros((B, N + 1), dtype=np.float32)
    out[:, :N] = fwd_ret_np
    return out


def evaluate_sharpe_v2(model: AllocatorV2, panel: PanelV2,
                       batch_size: int = 256) -> float:
    n = len(panel.dates)
    if n == 0:
        return float('nan')
    pnls = []
    prev = Tensor.training
    Tensor.training = False
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xa = Tensor(panel.X_assets[i:j])
        xm = Tensor(panel.X_macro[i:j])
        fwd = _attach_cash_zero(panel.fwd_ret[i:j])
        w, vp = model(xa, xm)
        pnl_etf = (w * Tensor(fwd)).sum(axis=-1).numpy()
        pnl_vol = (vp.numpy() * panel.fwd_vol_pnl[i:j])
        pnls.append(pnl_etf + pnl_vol)
    Tensor.training = prev
    pnls = np.concatenate(pnls)
    if pnls.std() <= 1e-9:
        return 0.0
    return float(pnls.mean() / pnls.std())


def train_one_fold_v2(
    panel_train: PanelV2,
    panel_val: PanelV2 | None,
    cfg: TrainConfigV2,
    hp: HparamsV2,
) -> tuple[AllocatorV2, dict]:
    rng = np.random.default_rng(cfg.seed)
    model = AllocatorV2(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)

    n_train = len(panel_train.dates)
    if n_train < cfg.batch_size:
        raise ValueError(f'train fold too small: {n_train}')

    best_val_sh = -np.inf
    best_params = [p.numpy().copy() for p in model.parameters()]
    no_improve = 0
    history = {'step': [], 'train_loss': [], 'val_sharpe': []}

    Tensor.training = True
    t0 = time.time()
    for step in range(cfg.n_steps):
        idx = rng.integers(0, n_train, size=cfg.batch_size)
        xa = Tensor(panel_train.X_assets[idx])
        xm = Tensor(panel_train.X_macro[idx])
        fwd = _attach_cash_zero(panel_train.fwd_ret[idx])
        vol_pnl = panel_train.fwd_vol_pnl[idx].astype(np.float32)
        opt.zero_grad()
        w, vp = model(xa, xm)
        loss = sharpe_loss_v2(w, Tensor(fwd), vp, Tensor(vol_pnl))
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            history['step'].append(step)
            history['train_loss'].append(float(loss.numpy()))

        if panel_val is not None and (step % cfg.val_every == 0
                                       or step == cfg.n_steps - 1):
            Tensor.training = False
            val_sh = evaluate_sharpe_v2(model, panel_val)
            Tensor.training = True
            history['val_sharpe'].append((step, val_sh))
            if val_sh > best_val_sh + 1e-4:
                best_val_sh = val_sh
                best_params = [p.numpy().copy() for p in model.parameters()]
                no_improve = 0
            else:
                no_improve += 1
            print(f'  step {step:5d}  loss={float(loss.numpy()):+.4f}  '
                  f'val_sh={val_sh:+.3f}  best={best_val_sh:+.3f}  '
                  f't={time.time()-t0:.0f}s', flush=True)

    for p, arr in zip(model.parameters(), best_params):
        p.assign(Tensor(arr.astype(np.float32)))
    Tensor.training = False
    return model, history
