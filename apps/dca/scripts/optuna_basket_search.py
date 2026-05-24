"""Pre-registered DCA basket Optuna search.

Per `apps/docs/docs/TODO/dca-basket-optuna.md` (committed before this
script ran). The pre-reg locks: bucket-based search space (3,600
combinations), N_TRIALS=200, walk-forward (train 2005-2018 / val
2019-2025), DSR-deflated objective, and the falsification bar
(val_t > canonical_t + 1.0 = confirmed-OOS).

Run from repo root:
    uv run python apps/dca/scripts/optuna_basket_search.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# Quiet Optuna progress logs in this script's stdout.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

from cfr.baselines import PassiveEW
from ss_loaders import load_stooq_matrix
from ss_portfolio import standardize_oos


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_DIR = REPO_ROOT / 'StooqData'

# ---- Locked pre-reg parameters --------------------------------------------

TRAIN_START = pd.Timestamp('2005-02-25')
TRAIN_END   = pd.Timestamp('2018-12-31')
VAL_START   = pd.Timestamp('2019-01-01')
VAL_END     = pd.Timestamp('2025-12-31')

N_TRIALS = 200
SHARPE_STD_ANN = 0.25            # workspace empirical baseline
COMMISSION_BPS = 10.0
PPY = 252.0

# Canonical 13-ETF Phase 4d basket — the reference point.
CANONICAL_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV',
                     'XLY', 'TLT', 'IEF', 'GLD', 'DBC']
CANONICAL_REBAL_DAYS = 80
CANONICAL_DRIFT = 0.05  # documented in `apps/dca` but not used by PassiveEW

# Per the pre-reg's bucket structure
EQUITY_BUCKETS: dict[str, list[str]] = {
    '9-spdr-sectors-EW':  ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
    'SPY-only':           ['SPY'],
    'VTI-only':           ['VTI'],
    'top-3-by-trailing-sharpe': ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
    # ↑ This is a DYNAMIC bucket — at each rebal, pick top-3 of those 9
    #   by trailing 252-day Sharpe. Handled specially in `simulate_basket`.
}
INTL_BUCKETS = {
    'none': [],
    'EFA+EEM': ['EFA', 'EEM'],
    'VEU': ['VEU'],
}
BOND_BUCKETS = {
    'none': [],
    'TLT-only': ['TLT'],
    'TLT+IEF': ['TLT', 'IEF'],
    'AGG': ['AGG'],
    'TLT+IEF+TIP': ['TLT', 'IEF', 'TIP'],
}
COMMOD_BUCKETS = {
    'none': [],
    'GLD-only': ['GLD'],
    'GLD+DBC': ['GLD', 'DBC'],
    'GLD+DBC+USO': ['GLD', 'DBC', 'USO'],
}
REIT_BUCKETS = {
    'none': [],
    'VNQ': ['VNQ'],
}
REBAL_DAYS_GRID = [21, 63, 80, 126, 252]
DRIFT_GRID = [0.03, 0.05, 0.10]    # not used by PassiveEW; recorded only


# ---- Data loading ---------------------------------------------------------

def _all_candidate_tickers() -> list[str]:
    s: set[str] = set()
    for d in (EQUITY_BUCKETS, INTL_BUCKETS, BOND_BUCKETS, COMMOD_BUCKETS, REIT_BUCKETS):
        for vs in d.values():
            s.update(vs)
    s.update(CANONICAL_TICKERS)
    return sorted(s)


def load_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
                ) -> pd.DataFrame:
    print(f'Loading Stooq prices for {len(tickers)} ETFs '
          f'({start.date()} → {end.date()})...', flush=True)
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        str(STOOQ_DIR), min_history=10, include_etfs=True,
        start_date=str(start.date()), end_date=str(end.date()),
        tickers=tickers,
    )
    print(f'  loaded {prices.shape[0]} dates × {prices.shape[1]} symbols '
          f'in {time.perf_counter() - t0:.1f}s', flush=True)
    return prices


# ---- Simulation -----------------------------------------------------------

def _portfolio_returns_from_target(
    prices: pd.DataFrame, target_weights: np.ndarray, rebal_indices: np.ndarray,
    commission_bps: float = COMMISSION_BPS,
) -> np.ndarray:
    """Same accounting as cfr.baselines._portfolio_simulate but
    inlined here for clarity. Daily simple returns; cost on L1 turnover
    at each rebal."""
    closes = prices.values.astype(np.float64)
    T, N = closes.shape
    # Compute daily simple returns
    px_prev = np.maximum(closes[:-1], 1e-12)
    daily_simple = np.zeros_like(closes)
    daily_simple[1:] = closes[1:] / px_prev - 1.0

    ret = np.zeros(T, dtype=np.float64)
    w = np.zeros(N, dtype=np.float64)
    next_rebal = 0
    for t in range(T):
        if next_rebal < len(rebal_indices) and t == rebal_indices[next_rebal]:
            # Rebalance to target
            new_w = target_weights[next_rebal]
            turnover = np.abs(new_w - w).sum()
            cost = commission_bps * 1e-4 * turnover
            ret[t] -= cost
            w = new_w.copy()
            next_rebal += 1
        # Apply daily returns (drift the weights one step forward)
        ret_t = float((w * daily_simple[t]).sum())
        ret[t] += ret_t
        # Drift w by the day's returns (proportionally) for the next bar
        equity_factor = 1.0 + ret_t
        if equity_factor > 1e-12:
            w = w * (1.0 + daily_simple[t]) / equity_factor
    return ret


def simulate_basket(
    prices: pd.DataFrame, equity_bucket: str, intl_bucket: str,
    bond_bucket: str, commod_bucket: str, reit_bucket: str,
    rebal_days: int,
) -> tuple[np.ndarray, list[str]]:
    """Compute daily returns for one basket configuration.

    For non-dynamic equity buckets, the universe is fixed and we
    delegate to PassiveEW. For the dynamic 'top-3-by-trailing-sharpe',
    we pick top-3 of the 9 sectors per rebal date.

    Returns (daily_returns, kept_tickers_actually_present_in_panel).
    """
    intl = INTL_BUCKETS[intl_bucket]
    bonds = BOND_BUCKETS[bond_bucket]
    commod = COMMOD_BUCKETS[commod_bucket]
    reit = REIT_BUCKETS[reit_bucket]

    if equity_bucket == 'top-3-by-trailing-sharpe':
        sector_universe = EQUITY_BUCKETS['9-spdr-sectors-EW']
        fixed_non_equity = intl + bonds + commod + reit
        present_sectors = [t for t in sector_universe if t in prices.columns]
        present_fixed = [t for t in fixed_non_equity if t in prices.columns]
        if not present_sectors:
            return np.zeros(len(prices)), []
        # Compose universe = top-3 sectors (dynamic) + fixed non-equity
        all_tickers = present_sectors + present_fixed
        sub = prices[all_tickers]
        T = len(sub)
        N = sub.shape[1]
        # Rebal indices on the same cadence as PassiveEW
        min_lookback = max(21, 252)  # need 252d trailing for Sharpe
        rebal_indices = np.arange(min_lookback, T, rebal_days, dtype=np.int64)
        if len(rebal_indices) == 0:
            return np.zeros(T), all_tickers
        # Build per-rebal target weights
        sector_cols = [sub.columns.get_loc(t) for t in present_sectors]
        fixed_cols  = [sub.columns.get_loc(t) for t in present_fixed]
        target_per_rebal = np.zeros((len(rebal_indices), N), dtype=np.float64)
        # Pre-compute daily simple returns for trailing-Sharpe calc
        closes = sub.values.astype(np.float64)
        px_prev = np.maximum(closes[:-1], 1e-12)
        daily_simple = np.zeros_like(closes)
        daily_simple[1:] = closes[1:] / px_prev - 1.0

        for k, t in enumerate(rebal_indices):
            lookback = 252
            lo = max(0, int(t) - lookback)
            window = daily_simple[lo:int(t), sector_cols]
            mu = window.mean(axis=0)
            sd = window.std(axis=0, ddof=1)
            sh = np.where(sd > 1e-12, mu / sd, 0.0)
            top3_local = np.argsort(-sh)[:3]
            chosen_sector_cols = [sector_cols[i] for i in top3_local]
            chosen_cols = chosen_sector_cols + fixed_cols
            n_chosen = len(chosen_cols)
            w_each = 1.0 / n_chosen
            row = np.zeros(N, dtype=np.float64)
            for c in chosen_cols:
                row[c] = w_each
            target_per_rebal[k] = row
        ret = _portfolio_returns_from_target(sub, target_per_rebal, rebal_indices)
        return ret, all_tickers

    # Static equity bucket
    equity = EQUITY_BUCKETS[equity_bucket]
    universe = equity + intl + bonds + commod + reit
    present = [t for t in universe if t in prices.columns]
    if not present:
        return np.zeros(len(prices)), []
    sub = prices[present]
    ret = PassiveEW(rebal_days=rebal_days,
                    commission_bps=COMMISSION_BPS).daily_returns(sub)
    return ret, present


# ---- Optuna objective -----------------------------------------------------

@dataclass
class TrialConfig:
    equity_bucket: str
    intl_bucket: str
    bond_bucket: str
    commod_bucket: str
    reit_bucket: str
    rebal_days: int
    drift_threshold: float


def sample_config(trial: optuna.Trial) -> TrialConfig:
    return TrialConfig(
        equity_bucket=trial.suggest_categorical('equity_bucket', list(EQUITY_BUCKETS)),
        intl_bucket=trial.suggest_categorical('intl_bucket', list(INTL_BUCKETS)),
        bond_bucket=trial.suggest_categorical('bond_bucket', list(BOND_BUCKETS)),
        commod_bucket=trial.suggest_categorical('commod_bucket', list(COMMOD_BUCKETS)),
        reit_bucket=trial.suggest_categorical('reit_bucket', list(REIT_BUCKETS)),
        rebal_days=trial.suggest_categorical('rebal_days', REBAL_DAYS_GRID),
        drift_threshold=trial.suggest_categorical('drift_threshold', DRIFT_GRID),
    )


def make_objective(prices_train: pd.DataFrame, sharpe_std_pp: float):
    def objective(trial: optuna.Trial) -> float:
        cfg = sample_config(trial)
        ret, kept = simulate_basket(
            prices_train, cfg.equity_bucket, cfg.intl_bucket, cfg.bond_bucket,
            cfg.commod_bucket, cfg.reit_bucket, cfg.rebal_days)
        ret = ret[np.isfinite(ret)]
        if ret.size < 252:
            return -1e6
        mb = standardize_oos(
            ret, periods_per_year=PPY, n_trials=N_TRIALS,
            sharpe_std=sharpe_std_pp)
        # Store for later inspection
        trial.set_user_attr('kept_tickers', kept)
        trial.set_user_attr('n_kept', len(kept))
        trial.set_user_attr('ann_sharpe', mb.ann_sharpe)
        trial.set_user_attr('max_dd', mb.max_dd)
        return mb.deflated_tstat
    return objective


def evaluate_on_val(
    cfg: TrialConfig, prices_val: pd.DataFrame, sharpe_std_pp: float,
) -> dict:
    ret, kept = simulate_basket(
        prices_val, cfg.equity_bucket, cfg.intl_bucket, cfg.bond_bucket,
        cfg.commod_bucket, cfg.reit_bucket, cfg.rebal_days)
    ret = ret[np.isfinite(ret)]
    mb = standardize_oos(
        ret, periods_per_year=PPY, n_trials=N_TRIALS,
        sharpe_std=sharpe_std_pp)
    return {
        'kept_tickers': kept, 'n_kept': len(kept),
        'ann_sharpe': mb.ann_sharpe, 'deflated_t': mb.deflated_tstat,
        'max_dd': mb.max_dd, 'skew': mb.skew, 'kurtosis': mb.kurtosis,
        'n_obs': mb.n_obs,
    }


def evaluate_canonical(prices: pd.DataFrame, sharpe_std_pp: float) -> dict:
    """Canonical 13-ETF, 80-day rebal, evaluated under identical method."""
    present = [t for t in CANONICAL_TICKERS if t in prices.columns]
    sub = prices[present]
    ret = PassiveEW(rebal_days=CANONICAL_REBAL_DAYS,
                    commission_bps=COMMISSION_BPS).daily_returns(sub)
    ret = ret[np.isfinite(ret)]
    mb = standardize_oos(
        ret, periods_per_year=PPY, n_trials=N_TRIALS,
        sharpe_std=sharpe_std_pp)
    return {
        'kept_tickers': present, 'n_kept': len(present),
        'ann_sharpe': mb.ann_sharpe, 'deflated_t': mb.deflated_tstat,
        'max_dd': mb.max_dd, 'skew': mb.skew, 'kurtosis': mb.kurtosis,
        'n_obs': mb.n_obs,
    }


# ---- Driver --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--train-start', default=str(TRAIN_START.date()))
    p.add_argument('--train-end',   default=str(TRAIN_END.date()))
    p.add_argument('--val-start',   default=str(VAL_START.date()))
    p.add_argument('--val-end',     default=str(VAL_END.date()))
    p.add_argument('--out',
                   default=str(REPO_ROOT / 'Output/dca-basket-optuna.json'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train_start, train_end = pd.Timestamp(args.train_start), pd.Timestamp(args.train_end)
    val_start, val_end = pd.Timestamp(args.val_start), pd.Timestamp(args.val_end)

    tickers = _all_candidate_tickers()
    print(f'\n=== PRE-REGISTERED DCA basket Optuna search ===')
    print(f'  N_TRIALS = {args.n_trials}')
    print(f'  train    = {train_start.date()} → {train_end.date()}')
    print(f'  val      = {val_start.date()} → {val_end.date()}')
    print(f'  tickers  = {len(tickers)} candidates')
    print(f'  pre-reg  = apps/docs/docs/TODO/dca-basket-optuna.md')

    prices_full = load_prices(tickers, train_start, val_end)
    # Verify continuous coverage by training start
    early_coverage = prices_full.loc[:train_start].iloc[-1].notna()
    available = early_coverage[early_coverage].index.tolist()
    dropped = sorted(set(tickers) - set(available))
    print(f'\n  ETFs available at train_start: {len(available)}')
    print(f'  dropped (no 2005-on coverage): {dropped}')

    prices_train = prices_full.loc[train_start:train_end][available]
    prices_val   = prices_full.loc[val_start:val_end][available]
    print(f'  train shape: {prices_train.shape}')
    print(f'  val shape:   {prices_val.shape}')

    sharpe_std_pp = SHARPE_STD_ANN / math.sqrt(PPY)

    # --- Canonical reference under same method ---
    print(f'\n--- Canonical 13-ETF reference (same method) ---')
    canon_train = evaluate_canonical(prices_train, sharpe_std_pp)
    canon_val   = evaluate_canonical(prices_val,   sharpe_std_pp)
    print(f'  canonical train: annSh={canon_train["ann_sharpe"]:+.3f}  '
          f'defl_t={canon_train["deflated_t"]:+.3f}  '
          f'maxDD={canon_train["max_dd"]:+.3f}  '
          f'(n_kept={canon_train["n_kept"]})')
    print(f'  canonical val:   annSh={canon_val["ann_sharpe"]:+.3f}  '
          f'defl_t={canon_val["deflated_t"]:+.3f}  '
          f'maxDD={canon_val["max_dd"]:+.3f}')

    # --- Optuna search ---
    print(f'\n--- Optuna TPE search (n_trials={args.n_trials}) ---')
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = make_objective(prices_train, sharpe_std_pp)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    print(f'  search done in {elapsed:.1f}s')

    best = study.best_trial
    best_cfg = TrialConfig(**best.params)
    print(f'\n  best train deflated_t = {best.value:+.3f}')
    print(f'  best cfg = {best.params}')
    print(f'  kept = {best.user_attrs.get("kept_tickers", [])}')

    # --- Apply winner to val ---
    print(f'\n--- Winner applied to val ---')
    winner_val = evaluate_on_val(best_cfg, prices_val, sharpe_std_pp)
    print(f'  winner val: annSh={winner_val["ann_sharpe"]:+.3f}  '
          f'defl_t={winner_val["deflated_t"]:+.3f}  '
          f'maxDD={winner_val["max_dd"]:+.3f}')

    # --- Verdict per pre-reg ---
    dt = winner_val['deflated_t'] - canon_val['deflated_t']
    dd_diff = winner_val['max_dd'] - canon_val['max_dd']    # signed (more negative = worse)
    if dt > 1.0 and dd_diff > -0.05:
        verdict = 'confirmed-OOS'
    elif dt > 0.0 and dd_diff > -0.05:
        verdict = 'partial-OOS'
    else:
        verdict = 'confirmed-null'
    print(f'\n=== Verdict per pre-reg ===')
    print(f'  Δ deflated_t  (winner_val − canon_val) = {dt:+.3f}')
    print(f'  Δ max_dd      (winner_val − canon_val) = {dd_diff:+.3f}')
    print(f'  -> {verdict}')

    # --- Persist everything ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'pre_reg_page': 'apps/docs/docs/TODO/dca-basket-optuna.md',
        'n_trials': args.n_trials,
        'sharpe_std_ann': SHARPE_STD_ANN,
        'commission_bps': COMMISSION_BPS,
        'train_period': f'{train_start.date()} → {train_end.date()}',
        'val_period':   f'{val_start.date()} → {val_end.date()}',
        'tickers_available': available,
        'tickers_dropped':   dropped,
        'canonical_train':   canon_train,
        'canonical_val':     canon_val,
        'winner_params':     best.params,
        'winner_train':      {
            'deflated_t': best.value,
            'ann_sharpe': best.user_attrs.get('ann_sharpe'),
            'max_dd':     best.user_attrs.get('max_dd'),
            'n_kept':     best.user_attrs.get('n_kept'),
            'kept_tickers': best.user_attrs.get('kept_tickers'),
        },
        'winner_val':        winner_val,
        'delta_deflated_t':  dt,
        'delta_max_dd':      dd_diff,
        'verdict':           verdict,
        'top_10_trials_by_train_t': [
            {
                'params': t.params,
                'train_t': t.value,
                'ann_sharpe': t.user_attrs.get('ann_sharpe'),
                'max_dd': t.user_attrs.get('max_dd'),
            }
            for t in sorted(study.trials, key=lambda x: -(x.value or -1e18))[:10]
        ],
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n→ {out}', flush=True)


if __name__ == '__main__':
    main()
