"""Training loop with direct-Sharpe loss in tinygrad."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from tinygrad import Tensor, nn

from e2e_portfolio.data import Panel
from e2e_portfolio.model import Allocator, Hparams, sharpe_loss


@dataclass
class TrainConfig:
    n_steps: int = 5000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    log_every: int = 200
    val_every: int = 500
    early_stop_patience: int = 6


def _attach_cash_zero(fwd_ret_np: np.ndarray) -> np.ndarray:
    B, N = fwd_ret_np.shape
    out = np.zeros((B, N + 1), dtype=np.float32)
    out[:, :N] = fwd_ret_np
    return out


def evaluate_sharpe(model: Allocator, panel: Panel, batch_size: int = 256) -> float:
    """Compute mean-of-batch Sharpe of model on full panel (no SGD)."""
    n = len(panel.dates)
    if n == 0:
        return float('nan')
    pnls = []
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        xa = Tensor(panel.X_assets[i:j])
        xm = Tensor(panel.X_macro[i:j])
        fwd = _attach_cash_zero(panel.fwd_ret[i:j])
        prev = Tensor.training
        Tensor.training = False
        w = model(xa, xm)
        pnl = (w * Tensor(fwd)).sum(axis=-1).numpy()
        Tensor.training = prev
        pnls.append(pnl)
    pnls = np.concatenate(pnls)
    if pnls.std() <= 1e-9:
        return 0.0
    # Annualize: K-day forward returns, so ~252/K rebals/year.
    # But for ranking purposes (val Sharpe) use the raw per-sample Sharpe.
    return float(pnls.mean() / pnls.std())


def train_one_fold(
    panel_train: Panel,
    panel_val: Panel | None,
    cfg: TrainConfig,
    hp: Hparams,
) -> tuple[Allocator, dict]:
    rng = np.random.default_rng(cfg.seed)
    model = Allocator(hp, seed=cfg.seed)
    opt = nn.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_train = len(panel_train.dates)
    if n_train < cfg.batch_size:
        raise ValueError(f'train fold too small: {n_train} < batch_size {cfg.batch_size}')

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
        opt.zero_grad()
        w = model(xa, xm)
        loss = sharpe_loss(w, Tensor(fwd))
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            history['step'].append(step)
            history['train_loss'].append(float(loss.numpy()))

        if panel_val is not None and (step % cfg.val_every == 0 or step == cfg.n_steps - 1):
            Tensor.training = False
            val_sh = evaluate_sharpe(model, panel_val)
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
                  f'no_improve={no_improve}  t={time.time()-t0:.0f}s')
            if no_improve >= cfg.early_stop_patience:
                print(f'  early stop at step {step}')
                break

    # Restore best.
    for p, arr in zip(model.parameters(), best_params):
        p.assign(Tensor(arr.astype(np.float32)))

    Tensor.training = False
    return model, history
