"""Walk-forward driver for Deep CFR.

Same shape as `cfr.walkforward.CFRWalkForward` but replaces tabular
infoset + regret table with continuous state vector + regret_net.
Per-bar action availability handled the same way (Phase 2b bugfix
unchanged).

Pre-registered Phase 3 PASS cut:
    deep CFR mean Sharpe ≥ Phase 1 CFR + 0.15 (i.e., ≥ +0.74)
    AND deep CFR > naive uniform mix on Phase 2b menu by ≥ +0.10
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from cfr.menu import ActionMenu
from cfr.regret import compute_block_regrets
from cfr.state_vec import StateVecBuilder
from cfr.deep import RegretNet, DeepCFRBuffer, policy_from_predicted_regret
from cfr.baselines import (
    PassiveEW, TrailingBestGreedy, NaiveUniform, evaluate_baseline,
    _portfolio_simulate,
)


@dataclass
class DeepWindowResult:
    """Per-window record. Mirrors `cfr.walkforward.WindowResult` shape
    so leaderboard ingestion is uniform across tabular and deep."""
    window_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train_rebals: int
    n_val_rebals: int
    cfr_sharpe: float
    passive_ew_sharpe: float
    trailing_best_sharpe: float
    naive_uniform_sharpe: float
    cfr_alpha: float
    trailing_best_alpha: float
    naive_uniform_alpha: float
    cfr_avg_gross: float
    cfr_avg_turnover: float
    state_dim: int
    final_train_loss: float


@dataclass
class DeepCFRWalkForward:
    """Deep CFR walk-forward driver.

    Same windowing / friction as the tabular `CFRWalkForward`. Per-
    window pipeline:

      1. Fit `StateVecBuilder` on train (compute z-score stats)
      2. Initialize fresh `RegretNet` (so each window's net trains
         independently — no cross-window leak; the regret_net would
         otherwise carry pre-train regime statistics into the val
         test, which is the same trap macro v1a fell into).
      3. Walk train rebals chronologically:
           - state_t = state_vec[t]
           - R_pred = regret_net.predict(state_t)
           - pi = regret matching with avail mask
           - sample played; compute closed-form regrets
           - buffer.append((state, regrets, avail))
           - every `train_every` rebals: take a batch, do
             `n_sgd_per_batch` Adam steps
      4. Eval on val using regret_net + avail mask + portfolio sim
    """
    menu_builder: callable
    state_vec_builder_factory: callable

    train_window_days: int = 1260
    val_window_days:   int = 780
    step_window_days:  int = 780
    rebal_days:        int = 20
    commission_bps:    float = 10.0
    rng_seed:          int = 0

    # Deep CFR specific
    hidden:            int = 64
    learning_rate:     float = 5e-4
    weight_decay:      float = 1e-3
    batch_size:        int = 64
    train_every:       int = 5     # do SGD every N rebals
    n_sgd_per_batch:   int = 5
    buffer_capacity:   int = 50_000
    # If True, use the policy-mixed expected return as the regret
    # baseline (lower variance than sampling). The played action
    # is still drawn from the policy for the cumulative-strategy
    # tracking, but the regret estimator uses the expectation.
    expected_baseline: bool = True
    # Optional per-rebal gate function: `(bar_date: pd.Timestamp) -> bool`.
    # When False at a val rebal, the target portfolio is set to cash
    # (zeros). When None, no gating. Used by Phase 4a to suspend
    # deployment in low-VIX regimes per the macro v1b finding.
    pre_rebal_gate:    callable = None

    def windows(self, n: int) -> list[tuple[int, int, int]]:
        out = []
        start = 0
        while start + self.train_window_days + self.val_window_days <= n:
            out.append((start,
                        start + self.train_window_days,
                        start + self.train_window_days + self.val_window_days))
            start += self.step_window_days
        return out

    def _train_net(
        self,
        train_prices: pd.DataFrame,
        train_macro: Optional[pd.DataFrame],
        action_weights_train: np.ndarray,
        action_avail_train: np.ndarray,
        rebal_indices_train: np.ndarray,
        builder: StateVecBuilder,
        menu: ActionMenu,
        rng: np.random.Generator,
    ) -> tuple[RegretNet, float]:
        """Run one chronological pass of Deep CFR + interleaved SGD."""
        state_train = builder.transform(train_prices, train_macro)
        valid_train = builder.valid_mask(train_prices, train_macro)
        log_p = np.log(train_prices.values,
                       where=(train_prices.values > 0),
                       out=np.full_like(train_prices.values, np.nan, dtype=np.float64))

        net = RegretNet(
            n_features=builder.n_features,
            n_actions=menu.n_actions,
            hidden=self.hidden,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        buffer = DeepCFRBuffer(capacity=self.buffer_capacity)
        last_loss = float('nan')

        for k, t in enumerate(rebal_indices_train):
            t = int(t)
            t_end = t + self.rebal_days
            if t_end >= len(train_prices):
                break
            if not valid_train[t]:
                continue
            avail_t = action_avail_train[t]
            if not avail_t.any():
                continue

            state_t = state_train[t]                          # (F,)
            R_pred = net.predict(state_t)                     # (n_actions,)
            pi = policy_from_predicted_regret(R_pred, avail_t)
            played = int(rng.choice(menu.n_actions, p=pi))

            block_logret = log_p[t_end] - log_p[t]            # (N,)
            action_w_t = action_weights_train[t]              # (A, N)
            if self.expected_baseline:
                # Lower-variance regret: subtract policy-mixed expected
                # return rather than the sampled-action realized return.
                # Equivalent to "expected sampling" CFR — given closed-
                # form regret, the played action is only used for the
                # cumulative-strategy table tracking.
                from cfr.regret import compute_block_log_returns
                realized = compute_block_log_returns(block_logret, action_w_t)
                expected = float((pi * realized).sum())
                regrets = realized - expected
            else:
                regrets = compute_block_regrets(
                    block_logret, action_w_t, played)

            buffer.append(state_t, regrets, avail_t)

            if (k + 1) % self.train_every == 0 and len(buffer) >= self.batch_size:
                for _ in range(self.n_sgd_per_batch):
                    s_b, r_b, a_b = buffer.sample_batch(self.batch_size, rng)
                    last_loss = net.train_step(s_b, r_b, a_b)
        return net, last_loss

    def _eval_net(
        self,
        val_prices: pd.DataFrame,
        val_macro: Optional[pd.DataFrame],
        action_weights_val: np.ndarray,
        action_avail_val: np.ndarray,
        rebal_indices_val: np.ndarray,
        net: RegretNet,
        builder: StateVecBuilder,
        menu: ActionMenu,
    ) -> tuple[np.ndarray, float, float]:
        """Apply trained regret_net to val window and integrate portfolio."""
        T, A, N = action_weights_val.shape
        state_val = builder.transform(val_prices, val_macro)
        valid_val = builder.valid_mask(val_prices, val_macro)

        target_per_rebal = np.zeros((len(rebal_indices_val), N), dtype=np.float64)
        gross_per_rebal = np.zeros(len(rebal_indices_val), dtype=np.float64)
        for k, t in enumerate(rebal_indices_val):
            t = int(t)
            avail_t = action_avail_val[t]
            if not valid_val[t] or not avail_t.any():
                target_per_rebal[k] = 0.0
                gross_per_rebal[k] = 0.0
                continue
            # Optional pre-rebal gate (e.g., VIX-above-median). If gate
            # returns False, set target to cash and skip CFR.
            if self.pre_rebal_gate is not None:
                bar_date = val_prices.index[t]
                if not self.pre_rebal_gate(bar_date):
                    target_per_rebal[k] = 0.0
                    gross_per_rebal[k] = 0.0
                    continue
            R_pred = net.predict(state_val[t])
            pi = policy_from_predicted_regret(R_pred, avail_t)
            mixed_w = (pi[:, None] * action_weights_val[t]).sum(axis=0)
            target_per_rebal[k] = mixed_w
            gross_per_rebal[k] = float(mixed_w.sum())
        daily_ret = _portfolio_simulate(
            val_prices, target_per_rebal, rebal_indices_val,
            commission_bps=self.commission_bps,
        )
        if len(target_per_rebal) >= 2:
            turnover = float(np.mean(
                np.abs(target_per_rebal[1:] - target_per_rebal[:-1]).sum(axis=1)))
        else:
            turnover = 0.0
        return daily_ret, float(gross_per_rebal.mean()), turnover

    def run(
        self,
        prices: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
    ) -> tuple[list[DeepWindowResult], dict]:
        rng = np.random.default_rng(self.rng_seed)
        windows = self.windows(len(prices))
        if not windows:
            raise SystemExit(
                f'no windows fit: have {len(prices)} bars, need '
                f'train+val={self.train_window_days + self.val_window_days}')

        menu = self.menu_builder()
        action_weights, action_avail = menu.precompute(prices)

        per_window: list[DeepWindowResult] = []
        for w_idx, (lo, mid, hi) in enumerate(windows):
            train_prices = prices.iloc[lo:mid]
            val_prices   = prices.iloc[mid:hi]
            action_weights_train = action_weights[lo:mid]
            action_weights_val   = action_weights[mid:hi]
            action_avail_train   = action_avail[lo:mid]
            action_avail_val     = action_avail[mid:hi]

            train_macro = (macro.loc[train_prices.index[0]:train_prices.index[-1]]
                           if macro is not None else None)
            val_macro = (macro.loc[val_prices.index[0]:val_prices.index[-1]]
                         if macro is not None else None)

            builder = self.state_vec_builder_factory()
            builder.fit(train_prices, train_macro)

            min_warmup = 60
            rebal_indices_train = np.arange(
                min_warmup, mid - lo - self.rebal_days, self.rebal_days,
                dtype=np.int64,
            )
            rebal_indices_val = np.arange(
                min_warmup, hi - mid, self.rebal_days, dtype=np.int64,
            )

            t0 = time.perf_counter()
            net, final_loss = self._train_net(
                train_prices, train_macro,
                action_weights_train, action_avail_train,
                rebal_indices_train, builder, menu, rng,
            )
            train_wall = time.perf_counter() - t0

            cfr_daily, cfr_gross, cfr_turnover = self._eval_net(
                val_prices, val_macro,
                action_weights_val, action_avail_val,
                rebal_indices_val, net, builder, menu,
            )
            cfr_metrics = evaluate_baseline(cfr_daily)

            passive = PassiveEW(rebal_days=self.rebal_days,
                                commission_bps=self.commission_bps)
            passive_daily = passive.daily_returns(val_prices)
            passive_metrics = evaluate_baseline(passive_daily)

            trailing = TrailingBestGreedy(rebal_days=self.rebal_days,
                                          commission_bps=self.commission_bps)
            trailing_daily = trailing.daily_returns(
                val_prices, menu, action_weights_val)
            trailing_metrics = evaluate_baseline(trailing_daily)

            naive = NaiveUniform(rebal_days=self.rebal_days,
                                 commission_bps=self.commission_bps)
            naive_daily = naive.daily_returns(val_prices, action_weights_val)
            naive_metrics = evaluate_baseline(naive_daily)

            wr = DeepWindowResult(
                window_idx=w_idx,
                train_start=str(train_prices.index[0].date()),
                train_end=str(train_prices.index[-1].date()),
                val_start=str(val_prices.index[0].date()),
                val_end=str(val_prices.index[-1].date()),
                n_train_rebals=len(rebal_indices_train),
                n_val_rebals=len(rebal_indices_val),
                cfr_sharpe=cfr_metrics['sharpe'],
                passive_ew_sharpe=passive_metrics['sharpe'],
                trailing_best_sharpe=trailing_metrics['sharpe'],
                naive_uniform_sharpe=naive_metrics['sharpe'],
                cfr_alpha=cfr_metrics['sharpe'] - passive_metrics['sharpe'],
                trailing_best_alpha=(trailing_metrics['sharpe']
                                     - passive_metrics['sharpe']),
                naive_uniform_alpha=(naive_metrics['sharpe']
                                     - passive_metrics['sharpe']),
                cfr_avg_gross=cfr_gross,
                cfr_avg_turnover=cfr_turnover,
                state_dim=builder.n_features,
                final_train_loss=final_loss,
            )
            per_window.append(wr)
            print(f'  w{w_idx}: train wall {train_wall:.1f}s  '
                  f'cfr_sh {cfr_metrics["sharpe"]:+.3f}  '
                  f'pas {passive_metrics["sharpe"]:+.3f}  '
                  f'naive {naive_metrics["sharpe"]:+.3f}  '
                  f'final_loss {final_loss:.4e}', flush=True)

        summary = self._summarize(per_window, menu)
        return per_window, summary

    def _summarize(
        self, per_window: list[DeepWindowResult], menu: ActionMenu,
    ) -> dict:
        cfr_sharpe = np.array([w.cfr_sharpe for w in per_window])
        passive_sh = np.array([w.passive_ew_sharpe for w in per_window])
        trailing_sh = np.array([w.trailing_best_sharpe for w in per_window])
        naive_sh = np.array([w.naive_uniform_sharpe for w in per_window])
        cfr_alpha = cfr_sharpe - passive_sh
        mean_alpha = float(cfr_alpha.mean()) if len(cfr_alpha) else 0.0
        pos_frac = float((cfr_alpha > 0).mean()) if len(cfr_alpha) else 0.0
        cfr_minus_naive = float(np.mean(cfr_sharpe - naive_sh)) if len(cfr_sharpe) else 0.0
        trailing_lift = float(np.mean(cfr_sharpe - trailing_sh)) if len(cfr_sharpe) else 0.0

        # Pre-registered Phase 3 cut.
        # PHASE_1_CFR_MEAN = +0.593 from Phase 1 leaderboard row.
        n_windows = len(per_window)
        phase_1_floor = 0.593 + 0.15
        if n_windows == 0:
            verdict = 'INSUFFICIENT-DATA'
        elif n_windows < 4:
            verdict = (f'SMOKE-ONLY ({n_windows} windows < 4) — '
                       f'cfr vs phase1 lift: {float(cfr_sharpe.mean()) - 0.593:+.3f}, '
                       f'cfr vs naive: {cfr_minus_naive:+.3f}')
        else:
            cfr_mean = float(cfr_sharpe.mean())
            if cfr_mean >= phase_1_floor and cfr_minus_naive >= 0.10:
                verdict = (f'PASS — deep CFR {cfr_mean:+.3f} ≥ '
                           f'Phase 1 + 0.15 = {phase_1_floor:+.3f} '
                           f'AND beats naive uniform by {cfr_minus_naive:+.3f} '
                           f'≥ +0.10. Architecture validated.')
            elif cfr_mean < 0.593 - 0.10 or cfr_minus_naive < -0.10:
                verdict = (f'FAIL (confirmed-null) — deep CFR {cfr_mean:+.3f} '
                           f'< Phase 1 − 0.10 OR cfr vs naive '
                           f'{cfr_minus_naive:+.3f} < −0.10. Park.')
            else:
                verdict = (f'MARGINAL — deep CFR {cfr_mean:+.3f} between '
                           f'Phase 1 ±0.15 OR vs naive in ±0.10 band. '
                           f'Stratify per window before deciding.')

        return {
            'n_windows':                  n_windows,
            'menu_action_keys':           menu.action_keys,
            'mean_cfr_sharpe':            float(cfr_sharpe.mean()) if n_windows else 0.0,
            'mean_passive_sharpe':        float(passive_sh.mean()) if n_windows else 0.0,
            'mean_trailing_sharpe':       float(trailing_sh.mean()) if n_windows else 0.0,
            'mean_naive_sharpe':          float(naive_sh.mean()) if n_windows else 0.0,
            'mean_cfr_alpha':             mean_alpha,
            'positive_alpha_fraction':    pos_frac,
            'mean_cfr_minus_naive':       cfr_minus_naive,
            'mean_cfr_minus_trailing':    trailing_lift,
            'verdict':                    verdict,
        }


__all__ = ['DeepCFRWalkForward', 'DeepWindowResult']
