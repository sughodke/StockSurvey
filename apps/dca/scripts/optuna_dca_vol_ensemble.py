"""Pre-registered DCA × vol overlay joint Optuna search.

Per `apps/docs/docs/TODO/dca-vol-ensemble-optuna.md` (committed before
this script ran). The pre-reg locks: bucket-based DCA basket × vega_scale
joint search space (16,800 combinations), N_TRIALS=200, walk-forward
(train rebal 0-19 / val 20-32 of the 33-obs vol-v3-DoltHub OOS sample),
DSR-deflated objective, capital-free-overlay semantics, and the
falsification bar (val_t > canonical_t + 1.0 AND val maxDD ≤ canon
+5pp = confirmed-OOS).

Capital-free-overlay math:
    r_ens[t] = dca_block[t] + vega_scale × vol_alpha[t]

Run from repo root:
    uv run python apps/dca/scripts/optuna_dca_vol_ensemble.py
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

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

from cfr.baselines import PassiveEW
from ss_loaders import load_stooq_matrix
from ss_portfolio import standardize_oos


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_DIR = REPO_ROOT / 'StooqData'
VOL_NPZ = REPO_ROOT / 'Output/vol-v3-dolthub-oos-returns.npz'

# ---- Locked pre-reg parameters --------------------------------------------

N_TRIALS = 200
SHARPE_STD_ANN = 0.25
COMMISSION_BPS = 10.0
PPY = 12.6                          # ~33 rebals / ~32 calendar months
FORWARD_WINDOW = 20                 # 20-trading-day forward block (matches vol-v3)
TRAIN_LEN = 20                      # train = rebal indices 0..19 (20 obs)
# val = rebal indices 20..32 (13 obs)

CANONICAL_TICKERS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV',
                     'XLY', 'TLT', 'IEF', 'GLD', 'DBC']
CANONICAL_REBAL_DAYS = 80
CANONICAL_VEGA = 3.0

EQUITY_BUCKETS: dict[str, list[str]] = {
    '9-spdr-sectors-EW':  ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
    'SPY-only':           ['SPY'],
    'VTI-only':           ['VTI'],
    'top-3-by-trailing-sharpe': ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY'],
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
VEGA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


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


def load_vol_stream() -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Returns (vol_alpha_per_rebal, rebal_date_list)."""
    d = np.load(VOL_NPZ, allow_pickle=True)
    alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    dates = [pd.Timestamp(str(s)) for s in d['rebal_dates']]
    assert len(alpha) == len(dates) == 33, f'unexpected vol stream length: {len(alpha)}'
    return alpha, dates


# ---- DCA block-return construction ----------------------------------------

def _portfolio_returns_from_target(
    prices: pd.DataFrame, target_weights: np.ndarray, rebal_indices: np.ndarray,
    commission_bps: float = COMMISSION_BPS,
) -> np.ndarray:
    closes = prices.values.astype(np.float64)
    T, N = closes.shape
    px_prev = np.maximum(closes[:-1], 1e-12)
    daily_simple = np.zeros_like(closes)
    daily_simple[1:] = closes[1:] / px_prev - 1.0

    ret = np.zeros(T, dtype=np.float64)
    w = np.zeros(N, dtype=np.float64)
    next_rebal = 0
    for t in range(T):
        if next_rebal < len(rebal_indices) and t == rebal_indices[next_rebal]:
            new_w = target_weights[next_rebal]
            turnover = np.abs(new_w - w).sum()
            ret[t] -= commission_bps * 1e-4 * turnover
            w = new_w.copy()
            next_rebal += 1
        ret_t = float((w * daily_simple[t]).sum())
        ret[t] += ret_t
        eq = 1.0 + ret_t
        if eq > 1e-12:
            w = w * (1.0 + daily_simple[t]) / eq
    return ret


def simulate_basket_daily(
    prices: pd.DataFrame, equity_bucket: str, intl_bucket: str,
    bond_bucket: str, commod_bucket: str, reit_bucket: str,
    rebal_days: int,
) -> tuple[np.ndarray, list[str]]:
    """Daily simple returns for one basket configuration over full panel."""
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
        all_tickers = present_sectors + present_fixed
        sub = prices[all_tickers]
        T = len(sub); N = sub.shape[1]
        min_lookback = max(21, 252)
        rebal_indices = np.arange(min_lookback, T, rebal_days, dtype=np.int64)
        if len(rebal_indices) == 0:
            return np.zeros(T), all_tickers
        sector_cols = [sub.columns.get_loc(t) for t in present_sectors]
        fixed_cols  = [sub.columns.get_loc(t) for t in present_fixed]
        target = np.zeros((len(rebal_indices), N), dtype=np.float64)
        closes = sub.values.astype(np.float64)
        px_prev = np.maximum(closes[:-1], 1e-12)
        daily_simple = np.zeros_like(closes)
        daily_simple[1:] = closes[1:] / px_prev - 1.0
        for k, t in enumerate(rebal_indices):
            lo = max(0, int(t) - 252)
            window = daily_simple[lo:int(t), sector_cols]
            mu = window.mean(axis=0)
            sd = window.std(axis=0, ddof=1)
            sh = np.where(sd > 1e-12, mu / sd, 0.0)
            top3_local = np.argsort(-sh)[:3]
            chosen = [sector_cols[i] for i in top3_local] + fixed_cols
            w_each = 1.0 / len(chosen)
            row = np.zeros(N, dtype=np.float64)
            for c in chosen:
                row[c] = w_each
            target[k] = row
        ret = _portfolio_returns_from_target(sub, target, rebal_indices)
        return ret, all_tickers

    equity = EQUITY_BUCKETS[equity_bucket]
    universe = equity + intl + bonds + commod + reit
    present = [t for t in universe if t in prices.columns]
    if not present:
        return np.zeros(len(prices)), []
    sub = prices[present]
    ret = PassiveEW(rebal_days=rebal_days,
                    commission_bps=COMMISSION_BPS).daily_returns(sub)
    return ret, present


def dca_block_returns(
    daily_ret: np.ndarray, daily_index: pd.DatetimeIndex,
    rebal_dates: list[pd.Timestamp], forward_window: int = FORWARD_WINDOW,
) -> np.ndarray:
    """For each vol rebal_date, compound DCA daily returns over the forward
    `forward_window` trading days *after* that date.

    Returns array of same length as rebal_dates. If a forward window falls
    off the end of the daily panel, the partial window is compounded."""
    blocks = np.zeros(len(rebal_dates), dtype=np.float64)
    for i, d in enumerate(rebal_dates):
        # Find the first daily-index entry >= d
        pos = daily_index.searchsorted(d, side='left')
        # forward block = [pos+1 .. pos+forward_window] (lag 1 to avoid same-day overlap)
        lo = pos + 1
        hi = min(pos + 1 + forward_window, len(daily_ret))
        if lo >= len(daily_ret):
            blocks[i] = 0.0
            continue
        block_slice = daily_ret[lo:hi]
        blocks[i] = float(np.prod(1.0 + block_slice) - 1.0)
    return blocks


# ---- Optuna objective -----------------------------------------------------

@dataclass
class TrialConfig:
    equity_bucket: str
    intl_bucket: str
    bond_bucket: str
    commod_bucket: str
    reit_bucket: str
    rebal_days: int
    vega_scale: float


def sample_config(trial: optuna.Trial) -> TrialConfig:
    return TrialConfig(
        equity_bucket=trial.suggest_categorical('equity_bucket', list(EQUITY_BUCKETS)),
        intl_bucket=trial.suggest_categorical('intl_bucket', list(INTL_BUCKETS)),
        bond_bucket=trial.suggest_categorical('bond_bucket', list(BOND_BUCKETS)),
        commod_bucket=trial.suggest_categorical('commod_bucket', list(COMMOD_BUCKETS)),
        reit_bucket=trial.suggest_categorical('reit_bucket', list(REIT_BUCKETS)),
        rebal_days=trial.suggest_categorical('rebal_days', REBAL_DAYS_GRID),
        vega_scale=trial.suggest_categorical('vega_scale', VEGA_GRID),
    )


def evaluate_ensemble(
    cfg: TrialConfig, prices: pd.DataFrame, vol_alpha: np.ndarray,
    rebal_dates: list[pd.Timestamp], sharpe_std_pp: float,
    train_slice: slice | None = None,
) -> dict:
    """Build the ensemble block-return stream and compute MetricBlock."""
    daily_ret, kept = simulate_basket_daily(
        prices, cfg.equity_bucket, cfg.intl_bucket, cfg.bond_bucket,
        cfg.commod_bucket, cfg.reit_bucket, cfg.rebal_days,
    )
    daily_idx = prices.index
    dca_blocks = dca_block_returns(daily_ret, daily_idx, rebal_dates)
    ens = dca_blocks + cfg.vega_scale * vol_alpha
    if train_slice is not None:
        ens = ens[train_slice]
    ens = ens[np.isfinite(ens)]
    if ens.size < 5:
        return {'deflated_t': -1e6, 'ann_sharpe': 0.0, 'max_dd': 0.0,
                'n_obs': int(ens.size), 'kept': kept}
    mb = standardize_oos(ens, periods_per_year=PPY, n_trials=N_TRIALS,
                        sharpe_std=sharpe_std_pp)
    return {
        'deflated_t': mb.deflated_tstat,
        'ann_sharpe': mb.ann_sharpe,
        'max_dd': mb.max_dd,
        'n_obs': mb.n_obs,
        'skew': mb.skew,
        'kurtosis': mb.kurtosis,
        'kept': kept,
    }


def make_objective(prices, vol_alpha, rebal_dates, sharpe_std_pp):
    train_slc = slice(0, TRAIN_LEN)
    def objective(trial: optuna.Trial) -> float:
        cfg = sample_config(trial)
        r = evaluate_ensemble(cfg, prices, vol_alpha, rebal_dates,
                              sharpe_std_pp, train_slice=train_slc)
        trial.set_user_attr('ann_sharpe', r['ann_sharpe'])
        trial.set_user_attr('max_dd', r['max_dd'])
        trial.set_user_attr('n_kept', len(r['kept']))
        return r['deflated_t']
    return objective


# ---- Driver ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-trials', type=int, default=N_TRIALS)
    p.add_argument('--out',
                   default=str(REPO_ROOT / 'Output/dca-vol-ensemble-optuna.json'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    print(f'\n=== PRE-REGISTERED DCA × vol-overlay joint Optuna search ===')
    print(f'  pre-reg     = apps/docs/docs/TODO/dca-vol-ensemble-optuna.md')
    print(f'  N_TRIALS    = {args.n_trials}')
    print(f'  periods/yr  = {PPY}')
    print(f'  sharpe_std  = {SHARPE_STD_ANN} ann')
    print(f'  commission  = {COMMISSION_BPS} bps')

    # --- Load vol stream ---
    vol_alpha, rebal_dates = load_vol_stream()
    print(f'\n  vol-v3 stream: {len(vol_alpha)} rebals '
          f'({rebal_dates[0].date()} → {rebal_dates[-1].date()})')

    # --- Load DCA prices (cover full forward window past last rebal) ---
    panel_start = pd.Timestamp('2023-01-01')   # buffer for trailing-Sharpe 252d lookback
    # but top-3 needs 252d trailing; widen further
    panel_start = pd.Timestamp('2022-01-01')
    panel_end   = rebal_dates[-1] + pd.Timedelta(days=60)
    tickers = _all_candidate_tickers()
    prices = load_prices(tickers, panel_start, panel_end)
    available = [t for t in tickers if t in prices.columns]
    dropped = sorted(set(tickers) - set(available))
    print(f'  ETFs available: {len(available)}; dropped: {dropped}')

    sharpe_std_pp = SHARPE_STD_ANN / math.sqrt(PPY)
    train_slc = slice(0, TRAIN_LEN)
    val_slc   = slice(TRAIN_LEN, len(rebal_dates))
    print(f'  train rebals: 0..{TRAIN_LEN-1} ({TRAIN_LEN} obs); '
          f'val rebals: {TRAIN_LEN}..{len(rebal_dates)-1} '
          f'({len(rebal_dates)-TRAIN_LEN} obs)')

    # --- Canonical reference under identical method ---
    print(f'\n--- Canonical reference: 13-ETF + vol × {CANONICAL_VEGA} ---')
    canon_cfg = TrialConfig(
        equity_bucket='9-spdr-sectors-EW', intl_bucket='none',
        bond_bucket='TLT+IEF', commod_bucket='GLD+DBC', reit_bucket='none',
        rebal_days=CANONICAL_REBAL_DAYS, vega_scale=CANONICAL_VEGA,
    )
    canon_train = evaluate_ensemble(canon_cfg, prices, vol_alpha, rebal_dates,
                                    sharpe_std_pp, train_slice=train_slc)
    canon_val = evaluate_ensemble(canon_cfg, prices, vol_alpha, rebal_dates,
                                  sharpe_std_pp, train_slice=val_slc)
    print(f'  canonical train: defl_t={canon_train["deflated_t"]:+.3f}  '
          f'annSh={canon_train["ann_sharpe"]:+.3f}  '
          f'maxDD={canon_train["max_dd"]:+.3f}  n={canon_train["n_obs"]}')
    print(f'  canonical val:   defl_t={canon_val["deflated_t"]:+.3f}  '
          f'annSh={canon_val["ann_sharpe"]:+.3f}  '
          f'maxDD={canon_val["max_dd"]:+.3f}  n={canon_val["n_obs"]}')

    # --- Optuna search ---
    print(f'\n--- Optuna TPE search (n_trials={args.n_trials}) ---')
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    obj = make_objective(prices, vol_alpha, rebal_dates, sharpe_std_pp)
    t0 = time.perf_counter()
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)
    print(f'  search done in {time.perf_counter()-t0:.1f}s')

    best = study.best_trial
    best_cfg = TrialConfig(**best.params)
    print(f'\n  best train deflated_t = {best.value:+.3f}')
    print(f'  best cfg = {best.params}')

    # --- Apply winner to val ---
    winner_val = evaluate_ensemble(best_cfg, prices, vol_alpha, rebal_dates,
                                   sharpe_std_pp, train_slice=val_slc)
    print(f'\n--- Winner on val ---')
    print(f'  winner val: defl_t={winner_val["deflated_t"]:+.3f}  '
          f'annSh={winner_val["ann_sharpe"]:+.3f}  '
          f'maxDD={winner_val["max_dd"]:+.3f}  n={winner_val["n_obs"]}')

    # --- Verdict per pre-reg ---
    dt = winner_val['deflated_t'] - canon_val['deflated_t']
    dd_diff = winner_val['max_dd'] - canon_val['max_dd']
    # Pre-reg: vega_scale=0 wins tied to canonical-basket → favor canonical
    if best_cfg.vega_scale == 0.0:
        print(f'\n  NOTE: winner has vega_scale=0.0 (basket-only) — see pre-reg '
              f'"does NOT count as a result"')

    if dt > 1.0 and dd_diff > -0.05:
        verdict = 'confirmed-OOS'
    elif dt >= 0.0 and dd_diff > -0.05:
        verdict = 'partial-OOS'
    else:
        verdict = 'confirmed-null'
    print(f'\n=== Verdict per pre-reg ===')
    print(f'  Δ deflated_t (winner_val − canon_val) = {dt:+.3f}')
    print(f'  Δ max_dd     (winner_val − canon_val) = {dd_diff:+.3f}')
    print(f'  -> {verdict}')

    # --- Persist ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'pre_reg_page': 'apps/docs/docs/TODO/dca-vol-ensemble-optuna.md',
        'n_trials': args.n_trials,
        'sharpe_std_ann': SHARPE_STD_ANN,
        'periods_per_year': PPY,
        'commission_bps': COMMISSION_BPS,
        'train_rebals': [0, TRAIN_LEN - 1],
        'val_rebals': [TRAIN_LEN, len(rebal_dates) - 1],
        'n_train': TRAIN_LEN,
        'n_val': len(rebal_dates) - TRAIN_LEN,
        'tickers_available': available,
        'tickers_dropped': dropped,
        'canonical_cfg': {
            'equity_bucket': canon_cfg.equity_bucket,
            'intl_bucket': canon_cfg.intl_bucket,
            'bond_bucket': canon_cfg.bond_bucket,
            'commod_bucket': canon_cfg.commod_bucket,
            'reit_bucket': canon_cfg.reit_bucket,
            'rebal_days': canon_cfg.rebal_days,
            'vega_scale': canon_cfg.vega_scale,
        },
        'canonical_train': {k: v for k, v in canon_train.items() if k != 'kept'},
        'canonical_val':   {k: v for k, v in canon_val.items() if k != 'kept'},
        'canonical_kept':  canon_train.get('kept'),
        'winner_params': best.params,
        'winner_train': {
            'deflated_t': best.value,
            'ann_sharpe': best.user_attrs.get('ann_sharpe'),
            'max_dd':     best.user_attrs.get('max_dd'),
        },
        'winner_val':  {k: v for k, v in winner_val.items() if k != 'kept'},
        'winner_kept': winner_val.get('kept'),
        'delta_deflated_t': dt,
        'delta_max_dd':     dd_diff,
        'verdict': verdict,
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
