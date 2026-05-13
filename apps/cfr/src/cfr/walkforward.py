"""Walk-forward driver — train tabular CFR per window, eval vs passive EW.

This is the load-bearing artifact: walk-forward across `n_windows`
train/val splits, train the tabular CFR on train, evaluate on val,
report alpha vs passive EW per window. Pre-registered cuts per the
[`apps/cfr` TODO](../../docs/TODO/apps-cfr.md) Phase 1 spec
determine the arc-level verdict.

The driver also runs `TrailingBestGreedy` and `NaiveUniform` as
control baselines so the CFR row can be read in context — the
algorithm earns its keep only if it beats both.

Outputs a JSON summary with per-window stats + verdict. Same shape
as `apps/gate/scripts/run_walkforward.py`'s output so leaderboard
ingestion is uniform across apps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cfr.menu import ActionMenu
from cfr.state import InfosetBuilder
from cfr.tabular import TabularCFR
from cfr.regret import compute_block_regrets, regret_matching
from cfr.baselines import (
    PassiveEW, TrailingBestGreedy, NaiveUniform, evaluate_baseline,
    _portfolio_simulate,
)


def _mask_and_renormalize(policy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out unavailable actions in `policy` and renormalize.

    Falls back to uniform over the available subset if the original
    policy puts zero mass on any available action (e.g., regret
    matching said "concentrate on action X" but X is unavailable
    here). If no actions are available at all, returns all-cash
    (assumes index 0 is the canonical cash action per ActionMenu's
    dedup convention).
    """
    masked = policy.copy()
    masked[~mask] = 0.0
    total = masked.sum()
    if total > 0:
        return masked / total
    # No mass on available actions → uniform over available
    n_avail = int(mask.sum())
    if n_avail > 0:
        out = np.zeros_like(policy)
        out[mask] = 1.0 / n_avail
        return out
    # Nothing available → all cash (index 0)
    out = np.zeros_like(policy)
    out[0] = 1.0
    return out


@dataclass
class WindowResult:
    """Per-window record. Mirrors the gate/pairs/vol leaderboard shape."""
    window_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train_rebals: int
    n_val_rebals: int
    # Per-arm val annualized Sharpe.
    cfr_sharpe: float
    passive_ew_sharpe: float
    trailing_best_sharpe: float
    naive_uniform_sharpe: float
    # alpha = arm - passive_ew
    cfr_alpha: float
    trailing_best_alpha: float
    naive_uniform_alpha: float
    # Diagnostics
    avg_policy_entropy_train: float
    cfr_avg_gross: float
    cfr_avg_turnover: float
    visited_infosets: int
    # Per-infoset table fingerprint for debugging (top action by avg policy)
    top_action_per_infoset: list[int] = field(default_factory=list)


@dataclass
class CFRWalkForward:
    """Run the walk-forward and produce per-window + summary stats.

    Configuration mirrors `apps/gate/scripts/run_walkforward.py` so
    apps share a canonical windowing pattern: 5y train / 3y val / 3y
    step is the default that gate / pairs / vol used; CFR follows
    the same so per-window alpha is directly comparable across the
    pivot-arc apps.

    Hyperparameters that affect the CFR specifically:
      `rebal_days`         : block size for regret computation
      `n_training_passes`  : how many times to walk train chronologically
                             (1 is the algorithmic default; more passes
                             help with sparse infoset visits)
      `commission_bps`     : 10 bps round-trip canonical
      `top_k`              : top-K for TopKMode actions
    """
    menu_builder: callable                 # () -> ActionMenu
    infoset_builder_factory: callable      # () -> InfosetBuilder

    train_window_days: int = 1260   # ~5y
    val_window_days:   int = 780    # ~3y
    step_window_days:  int = 780
    rebal_days:        int = 20
    commission_bps:    float = 10.0
    n_training_passes: int = 1
    rng_seed:          int = 0

    def windows(self, n: int) -> list[tuple[int, int, int]]:
        out = []
        start = 0
        while start + self.train_window_days + self.val_window_days <= n:
            out.append((start,
                        start + self.train_window_days,
                        start + self.train_window_days + self.val_window_days))
            start += self.step_window_days
        return out

    def _train_table(
        self,
        train_prices: pd.DataFrame,
        action_weights_train: np.ndarray,
        action_avail_train: np.ndarray,
        infoset_ids_train: np.ndarray,
        rebal_indices_train: np.ndarray,
        menu: ActionMenu,
        builder: InfosetBuilder,
        rng: np.random.Generator,
    ) -> TabularCFR:
        """Run `n_training_passes` chronological passes of CFR updates.

        Per-bar action availability mask `action_avail_train` (`(T,
        n_actions) bool`) gates which actions participate at each
        rebal: unavailable actions are excluded from sampling, do not
        receive regret updates, and are masked out of the played-
        policy used for the cumulative-strategy table. This prevents
        the Phase 2b cash-equivalent contamination bug where a 13F
        mode with no panel data still received policy mass in early
        windows.
        """
        log_p = np.log(train_prices.values,
                       where=(train_prices.values > 0),
                       out=np.full_like(train_prices.values, np.nan, dtype=np.float64))
        table = TabularCFR(
            n_infosets=builder.n_infosets,
            n_actions=menu.n_actions,
        )
        for pass_idx in range(self.n_training_passes):
            for k, t in enumerate(rebal_indices_train):
                t = int(t)
                t_end = t + self.rebal_days
                if t_end >= len(train_prices):
                    break
                infoset = int(infoset_ids_train[t])
                if infoset == builder.warmup_id:
                    continue
                avail_t = action_avail_train[t]            # (A,) bool
                if not avail_t.any():
                    continue
                pi_raw = table.current_policy(infoset)
                pi = _mask_and_renormalize(pi_raw, avail_t)
                block_logret = log_p[t_end] - log_p[t]      # (N,)
                action_w_t = action_weights_train[t]        # (A, N)
                played = int(rng.choice(menu.n_actions, p=pi))
                regrets = compute_block_regrets(
                    block_logret, action_w_t, played)
                # Mask regrets for unavailable actions so they don't
                # accumulate in the table.
                regrets_masked = np.where(avail_t, regrets, 0.0)
                table.update(infoset, regrets_masked, pi)
        return table

    def _eval_cfr(
        self,
        val_prices: pd.DataFrame,
        action_weights_val: np.ndarray,
        action_avail_val: np.ndarray,
        infoset_ids_val: np.ndarray,
        rebal_indices_val: np.ndarray,
        table: TabularCFR,
        builder: InfosetBuilder,
        menu: ActionMenu,
    ) -> tuple[np.ndarray, float, float]:
        """Apply the trained policy to val and return (daily_ret, avg_gross, avg_turnover).

        Honors per-bar action availability: at each val rebal, the
        average policy is masked to the available action subset and
        renormalized before mixing.
        """
        T, A, N = action_weights_val.shape
        target_per_rebal = np.zeros((len(rebal_indices_val), N), dtype=np.float64)
        gross_per_rebal = np.zeros(len(rebal_indices_val), dtype=np.float64)
        for k, t in enumerate(rebal_indices_val):
            t = int(t)
            infoset = int(infoset_ids_val[t])
            avail_t = action_avail_val[t]
            if infoset == builder.warmup_id or not avail_t.any():
                target_per_rebal[k] = 0.0
                gross_per_rebal[k] = 0.0
                continue
            pi_raw = table.average_policy(infoset)
            pi = _mask_and_renormalize(pi_raw, avail_t)
            mixed_w = (pi[:, None] * action_weights_val[t]).sum(axis=0)
            target_per_rebal[k] = mixed_w
            gross_per_rebal[k] = float(mixed_w.sum())
        daily_ret = _portfolio_simulate(
            val_prices, target_per_rebal, rebal_indices_val,
            commission_bps=self.commission_bps,
        )
        # Turnover: mean L1 distance between successive targets.
        if len(target_per_rebal) >= 2:
            turnover = float(np.mean(
                np.abs(target_per_rebal[1:] - target_per_rebal[:-1]).sum(axis=1)))
        else:
            turnover = 0.0
        return daily_ret, float(gross_per_rebal.mean()), turnover

    def run(self, prices: pd.DataFrame) -> tuple[list[WindowResult], dict]:
        """Execute the walk-forward and return per-window + summary."""
        rng = np.random.default_rng(self.rng_seed)
        windows = self.windows(len(prices))
        if not windows:
            raise SystemExit(
                f'no windows fit: have {len(prices)} bars but need '
                f'train+val={self.train_window_days + self.val_window_days}')

        # Precompute menu weights + availability ONCE on the full
        # panel — same modes across windows means same precomputed
        # scores. The slicing below is just into the precomputed
        # tensors.
        menu = self.menu_builder()
        action_weights, action_avail = menu.precompute(prices)  # (T, A, N), (T, A)

        per_window: list[WindowResult] = []
        for w_idx, (lo, mid, hi) in enumerate(windows):
            train_prices = prices.iloc[lo:mid]
            val_prices   = prices.iloc[mid:hi]
            action_weights_train = action_weights[lo:mid]
            action_weights_val   = action_weights[mid:hi]
            action_avail_train   = action_avail[lo:mid]
            action_avail_val     = action_avail[mid:hi]

            builder = self.infoset_builder_factory()
            ids_train = builder.fit_transform(train_prices)
            ids_val   = builder.transform(val_prices)

            min_warmup = 60   # need some history before rebalancing
            rebal_indices_train = np.arange(
                min_warmup, mid - lo - self.rebal_days, self.rebal_days,
                dtype=np.int64,
            )
            rebal_indices_val = np.arange(
                min_warmup, hi - mid, self.rebal_days, dtype=np.int64,
            )

            table = self._train_table(
                train_prices, action_weights_train, action_avail_train,
                ids_train, rebal_indices_train,
                menu, builder, rng,
            )

            cfr_daily, cfr_gross, cfr_turnover = self._eval_cfr(
                val_prices, action_weights_val, action_avail_val,
                ids_val, rebal_indices_val,
                table, builder, menu,
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

            avg_pol_train = table.policy_table_average()
            entropies = []
            for r in avg_pol_train:
                p = r[r > 0]
                entropies.append(float(-(p * np.log(p)).sum()) if len(p) else 0.0)
            avg_entropy = float(np.mean(entropies))

            top_action = [int(np.argmax(avg_pol_train[i]))
                          for i in range(builder.n_infosets)]
            visited = int((table.n_visits > 0).sum())

            wr = WindowResult(
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
                trailing_best_alpha=trailing_metrics['sharpe'] - passive_metrics['sharpe'],
                naive_uniform_alpha=naive_metrics['sharpe'] - passive_metrics['sharpe'],
                avg_policy_entropy_train=avg_entropy,
                cfr_avg_gross=cfr_gross,
                cfr_avg_turnover=cfr_turnover,
                visited_infosets=visited,
                top_action_per_infoset=top_action,
            )
            per_window.append(wr)

        summary = self._summarize(per_window, menu)
        return per_window, summary

    def _summarize(
        self, per_window: list[WindowResult], menu: ActionMenu,
    ) -> dict:
        cfr_alpha = np.array([w.cfr_alpha for w in per_window])
        cfr_sharpe = np.array([w.cfr_sharpe for w in per_window])
        passive_sh = np.array([w.passive_ew_sharpe for w in per_window])
        trailing_sh = np.array([w.trailing_best_sharpe for w in per_window])
        naive_sh = np.array([w.naive_uniform_sharpe for w in per_window])
        mean_alpha = float(cfr_alpha.mean()) if len(cfr_alpha) else 0.0
        pos_frac = float((cfr_alpha > 0).mean()) if len(cfr_alpha) else 0.0
        trailing_lift = float(np.mean(cfr_sharpe - trailing_sh)) if len(cfr_sharpe) else 0.0

        # Pre-registered cuts from apps/cfr TODO Phase 1. Defined for
        # the canonical 6-window walkforward; smaller smoke runs short-
        # circuit to a smoke-only label so a 1-window result isn't
        # mistaken for a verdict.
        n_windows = len(per_window)
        cfr_beats_trailing_frac = float(
            np.mean(cfr_sharpe > trailing_sh)) if n_windows else 0.0
        if n_windows == 0:
            verdict = 'INSUFFICIENT-DATA'
        elif n_windows < 4:
            verdict = (f'SMOKE-ONLY ({n_windows} windows < 4) — '
                       f'numbers are diagnostic, not a verdict. '
                       f'CFR vs trailing-best lift: '
                       f'{trailing_lift:+.3f}, CFR>trailing in '
                       f'{cfr_beats_trailing_frac:.0%} of windows.')
        elif trailing_lift >= 0.10 and cfr_beats_trailing_frac >= 4 / 6:
            verdict = ('PASS — CFR beats trailing-best-greedy by ≥0.10 '
                       'mean and CFR>trailing in ≥4/6 windows; proceed '
                       'to Phase 2 (13F imitation pretrain)')
        elif trailing_lift < -0.10 or cfr_beats_trailing_frac < 3 / 6:
            verdict = ('FAIL (confirmed-null) — CFR underperforms '
                       'trailing-best-greedy by ≥0.10 or beats it in '
                       '<3/6 windows. Park the arc per pre-reg.')
        else:
            verdict = ('MARGINAL — between thresholds; stratify '
                       'per-window before deciding on Phase 2')

        return {
            'n_windows':           n_windows,
            'menu_action_keys':    menu.action_keys,
            'mean_cfr_sharpe':     float(cfr_sharpe.mean()) if n_windows else 0.0,
            'mean_passive_sharpe': float(passive_sh.mean()) if n_windows else 0.0,
            'mean_trailing_sharpe': float(trailing_sh.mean()) if n_windows else 0.0,
            'mean_naive_sharpe':   float(naive_sh.mean()) if n_windows else 0.0,
            'mean_cfr_alpha':      mean_alpha,
            'positive_alpha_fraction': pos_frac,
            'mean_cfr_minus_trailing_best': trailing_lift,
            'cfr_beats_trailing_fraction': cfr_beats_trailing_frac,
            'verdict':             verdict,
        }


__all__ = ['CFRWalkForward', 'WindowResult']
