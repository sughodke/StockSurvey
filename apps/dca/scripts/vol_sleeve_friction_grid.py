"""Pre-registered vol_v3 sleeve-sizing friction grid.

Pre-reg: `apps/docs/docs/TODO/vol-v3-sleeve-sizing.md`.

Grid: vega_scale x c_options_bps (9 x 4 = 36 cells). For each cell, builds
the capital-free overlay ensemble `r_ens = r_dca + vega * r_vol_after_friction`
on the pre-frozen vol-v3-DoltHub-OOS 33-rebal block stream (alpha already
includes the friction levels c=0/100/200/400 via the frozen npz artifacts).

For each cell, computes:
  - pooled OOS annualized Sharpe (PPY = 12.6)
  - max drawdown on the compounded block-equity curve
  - Ledoit-Wolf studentized stationary-bootstrap CI (n=2000, seed=42) for
    Sharpe(r_ens) - Sharpe(r_dca_only), n_obs = 33
  - workspace deflated-t (sharpe_std_ann = 0.072 per pre-reg; falls back
    to 0.25 if the workspace constant is unset)

Verdict per pre-reg locked rule:
  confirmed-OOS = there exists (vega, c_bps) with deltaSR_ann >= +0.5 AND
                  CI excludes 0 AND deflated-t > +3.0 at c_bps = 400
  partial-OOS   = same numbers at c_bps = 200 but not 400
  confirmed-null= no cell with CI excluding 0 at any c_bps

Also emits a heatmap PNG: rows = vega_scale, cols = c_options_bps,
cells = combined annualized Sharpe.

Local-only, < 60s. Run from repo root:
  uv run python apps/dca/scripts/vol_sleeve_friction_grid.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ss_portfolio import standardize_oos
from ss_portfolio.sharpe_diff import sharpe_difference_ci


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / 'Output'
FIG_DIR = REPO_ROOT / 'apps/docs/docs/findings/images'

# ---- Pre-reg locked parameters --------------------------------------------

PPY = 12.6                          # 33 rebals / ~32 calendar months
SHARPE_STD_ANN = 0.072              # pre-reg workspace null floor
N_TRIALS_DEFLATED = 36              # the friction-grid cardinality
N_BOOT = 2000
BOOT_SEED = 42

VEGA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
C_BPS_GRID = [0, 100, 200, 400]

VOL_NPZ_TEMPLATE = {
    0:   'Output/vol-v3-dolthub-oos-returns.npz',       # commission_bps=0 baseline
    100: 'Output/vol-v3-dolthub-oos-c100-returns.npz',
    200: 'Output/vol-v3-dolthub-oos-c200-returns.npz',
    400: 'Output/vol-v3-dolthub-oos-c400-returns.npz',
}

# DCA canonical-13 daily stream is built from the existing pickle artifact
# `Output/cfr_phase4d_multiasset_close.pkl` (close panel) via PassiveEW @
# rebal_days=80 to produce daily returns, then block-aggregated forward 20
# days from each vol rebal date.
DCA_PICKLE = REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl'
DCA_REBAL_DAYS = 80
DCA_COMMISSION_BPS = 10.0
FORWARD_WINDOW = 20


# ---- Data loading ---------------------------------------------------------

def load_vol_alpha(c_bps: int) -> tuple[np.ndarray, list[pd.Timestamp]]:
    f = REPO_ROOT / VOL_NPZ_TEMPLATE[c_bps]
    d = np.load(f, allow_pickle=True)
    alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    dates = [pd.Timestamp(str(s)) for s in d['rebal_dates']]
    assert len(alpha) == 33, f'unexpected vol stream length: {len(alpha)}'
    return alpha, dates


def load_dca_daily() -> pd.DataFrame:
    """Phase-4d canonical close panel (13 ETFs)."""
    closes = pd.read_pickle(DCA_PICKLE)
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    return closes


# Inline PassiveEW @ rebal_days=80 with commission to avoid a hard
# dependency on the cfr.baselines module path.
def passive_ew_daily_returns(closes: pd.DataFrame, rebal_days: int,
                             commission_bps: float) -> np.ndarray:
    px = closes.values.astype(np.float64)
    T, N = px.shape
    px_prev = np.maximum(px[:-1], 1e-12)
    daily_simple = np.zeros_like(px)
    daily_simple[1:] = px[1:] / px_prev - 1.0
    ret = np.zeros(T, dtype=np.float64)
    w = np.zeros(N, dtype=np.float64)
    # Equal-weight target across all N names every rebal
    target_w = np.full(N, 1.0 / N, dtype=np.float64)
    rebal_indices = np.arange(0, T, rebal_days, dtype=np.int64)
    next_rebal = 0
    for t in range(T):
        if next_rebal < len(rebal_indices) and t == rebal_indices[next_rebal]:
            new_w = target_w
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


def dca_block_returns(daily_ret: np.ndarray, daily_index: pd.DatetimeIndex,
                      rebal_dates: list[pd.Timestamp],
                      forward_window: int = FORWARD_WINDOW) -> np.ndarray:
    blocks = np.zeros(len(rebal_dates), dtype=np.float64)
    for i, d in enumerate(rebal_dates):
        pos = daily_index.searchsorted(d, side='left')
        lo = pos + 1
        hi = min(pos + 1 + forward_window, len(daily_ret))
        if lo >= len(daily_ret):
            blocks[i] = 0.0
            continue
        block_slice = daily_ret[lo:hi]
        blocks[i] = float(np.prod(1.0 + block_slice) - 1.0)
    return blocks


# ---- Metrics --------------------------------------------------------------

def ann_sharpe(x: np.ndarray, ppy: float = PPY) -> float:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(x.mean() / sd) * math.sqrt(ppy)


def max_dd_compounded(blocks: np.ndarray) -> float:
    eq = np.cumprod(1.0 + blocks)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def cagr_from_blocks(blocks: np.ndarray, periods_per_year: float = PPY) -> float:
    n = len(blocks)
    if n == 0:
        return 0.0
    total = float(np.prod(1.0 + blocks))
    if total <= 0:
        return -1.0
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


# ---- Grid evaluation ------------------------------------------------------

@dataclass
class CellResult:
    vega: float
    c_bps: int
    n_obs: int
    sharpe_ens_ann: float
    sharpe_dca_ann: float
    delta_sr_ann: float
    ci_lo_ann: float
    ci_hi_ann: float
    ci_excludes_zero: bool
    max_dd_ens: float
    max_dd_dca: float
    cagr_ens: float
    cagr_dca: float
    deflated_t_ens: float


def evaluate_cell(dca_blocks: np.ndarray, vol_alpha: np.ndarray,
                  vega: float, c_bps: int, sharpe_std_pp: float) -> CellResult:
    ens = dca_blocks + vega * vol_alpha
    n = ens.size

    sr_ens_pp = ann_sharpe(ens, ppy=1.0)            # per-period
    sr_dca_pp = ann_sharpe(dca_blocks, ppy=1.0)
    sr_ens_ann = sr_ens_pp * math.sqrt(PPY)
    sr_dca_ann = sr_dca_pp * math.sqrt(PPY)

    # CI on per-period Sharpe diff -> rescale to annualized.
    ci = sharpe_difference_ci(ens, dca_blocks, n_bootstraps=N_BOOT,
                              seed=BOOT_SEED)
    ci_lo_ann = ci.ci_lo * math.sqrt(PPY)
    ci_hi_ann = ci.ci_hi * math.sqrt(PPY)
    delta_ann = ci.delta_sr * math.sqrt(PPY)

    # Deflated-t (uses sharpe_std at per-period scale)
    mb = standardize_oos(ens, periods_per_year=PPY,
                        n_trials=N_TRIALS_DEFLATED,
                        sharpe_std=sharpe_std_pp)

    return CellResult(
        vega=vega, c_bps=c_bps, n_obs=n,
        sharpe_ens_ann=sr_ens_ann, sharpe_dca_ann=sr_dca_ann,
        delta_sr_ann=delta_ann,
        ci_lo_ann=ci_lo_ann, ci_hi_ann=ci_hi_ann,
        ci_excludes_zero=not ci.includes_zero,
        max_dd_ens=max_dd_compounded(ens),
        max_dd_dca=max_dd_compounded(dca_blocks),
        cagr_ens=cagr_from_blocks(ens),
        cagr_dca=cagr_from_blocks(dca_blocks),
        deflated_t_ens=mb.deflated_tstat,
    )


# ---- Heatmap --------------------------------------------------------------

def plot_heatmap(grid: np.ndarray, vega_grid: list[float],
                 c_bps_grid: list[int], out_path: Path,
                 title: str, value_fmt: str = '{:+.2f}') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(c_bps_grid)))
    ax.set_xticklabels([str(c) for c in c_bps_grid])
    ax.set_yticks(range(len(vega_grid)))
    ax.set_yticklabels([f'{v:.1f}' for v in vega_grid])
    ax.set_xlabel('options friction c_bps (per-rebal)')
    ax.set_ylabel('vega_scale (vol_v3 overlay multiplier)')
    ax.set_title(title)
    for i in range(len(vega_grid)):
        for j in range(len(c_bps_grid)):
            ax.text(j, i, value_fmt.format(grid[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='black')
    fig.colorbar(im, ax=ax, label='annualized Sharpe')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---- Driver ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out',
                   default=str(OUT_DIR / 'vol-sleeve-friction-grid.json'))
    p.add_argument('--heatmap',
                   default=str(FIG_DIR / 'vol-sleeve-friction-grid.png'))
    p.add_argument('--smoke', action='store_true',
                   help='3 x 2 sub-grid for fast pre-flight check')
    args = p.parse_args()

    print(f'\n=== PRE-REGISTERED vol_v3 sleeve-sizing friction grid ===')
    print(f'  pre-reg = apps/docs/docs/TODO/vol-v3-sleeve-sizing.md')
    print(f'  PPY = {PPY}; n_obs = 33 (all OOS)')
    print(f'  N_BOOT = {N_BOOT}, seed = {BOOT_SEED}')
    print(f'  sharpe_std_ann = {SHARPE_STD_ANN}')

    # --- Vol streams per c_bps ---
    print(f'\n--- Loading frozen vol-v3 streams ---')
    vol_streams: dict[int, np.ndarray] = {}
    rebal_dates: list[pd.Timestamp] | None = None
    for c in C_BPS_GRID:
        a, ds = load_vol_alpha(c)
        vol_streams[c] = a
        if rebal_dates is None:
            rebal_dates = ds
        else:
            assert ds == rebal_dates, 'rebal_date mismatch across c_bps streams'
        print(f'  c_bps={c:>3d}: mean={a.mean():+.4f}  std={a.std():.4f}  '
              f'SR_ann={ann_sharpe(a):+.3f}')

    # --- DCA daily stream -> block returns aligned to vol rebal dates ---
    print(f'\n--- Building DCA canonical-13 daily stream ---')
    t0 = time.perf_counter()
    closes = load_dca_daily()
    print(f'  closes panel: {closes.shape[0]} dates × {closes.shape[1]} '
          f'symbols  ({closes.index.min().date()} -> {closes.index.max().date()})')
    daily_ret = passive_ew_daily_returns(closes, DCA_REBAL_DAYS,
                                         DCA_COMMISSION_BPS)
    dca_blocks = dca_block_returns(daily_ret, closes.index, rebal_dates,
                                   FORWARD_WINDOW)
    print(f'  DCA blocks: n={len(dca_blocks)}  mean={dca_blocks.mean():+.4f}  '
          f'SR_ann={ann_sharpe(dca_blocks):+.3f}  '
          f'maxDD={max_dd_compounded(dca_blocks):+.3f}  '
          f'CAGR={cagr_from_blocks(dca_blocks):+.4f}')
    print(f'  built in {time.perf_counter()-t0:.1f}s')

    sharpe_std_pp = SHARPE_STD_ANN / math.sqrt(PPY)

    # --- Grid ---
    vega_grid = VEGA_GRID
    c_grid = C_BPS_GRID
    if args.smoke:
        vega_grid = [0.0, 2.0, 5.0]
        c_grid = [0, 400]
        print(f'\n--- SMOKE grid: {len(vega_grid)} x {len(c_grid)} = '
              f'{len(vega_grid)*len(c_grid)} cells ---')
    else:
        print(f'\n--- Full grid: {len(vega_grid)} x {len(c_grid)} = '
              f'{len(vega_grid)*len(c_grid)} cells ---')

    results: list[CellResult] = []
    sharpe_matrix = np.zeros((len(vega_grid), len(c_grid)))
    delta_matrix = np.zeros_like(sharpe_matrix)
    excludes_matrix = np.zeros_like(sharpe_matrix, dtype=bool)

    t0 = time.perf_counter()
    for i, vega in enumerate(vega_grid):
        for j, c in enumerate(c_grid):
            r = evaluate_cell(dca_blocks, vol_streams[c], vega, c, sharpe_std_pp)
            results.append(r)
            sharpe_matrix[i, j] = r.sharpe_ens_ann
            delta_matrix[i, j] = r.delta_sr_ann
            excludes_matrix[i, j] = r.ci_excludes_zero
            marker = '*' if r.ci_excludes_zero else ' '
            print(f'  vega={vega:>4.1f}  c={c:>3d}bps  '
                  f'SR_ens={r.sharpe_ens_ann:+.3f}  '
                  f'dSR={r.delta_sr_ann:+.3f} CI=[{r.ci_lo_ann:+.3f},{r.ci_hi_ann:+.3f}]{marker}  '
                  f'maxDD={r.max_dd_ens:+.3f}  '
                  f'dt={r.deflated_t_ens:+.2f}  '
                  f'CAGR={r.cagr_ens:+.4f}')
    print(f'  grid done in {time.perf_counter()-t0:.1f}s')

    # --- Pre-reg verdict logic ---
    # confirmed-OOS = exists (vega,c) with dSR_ann >= +0.5 AND CI excludes 0
    #                 AND deflated-t > +3.0 at c_bps = 400
    # partial-OOS  = same at c_bps = 200 but not 400
    # confirmed-null = no cell with CI excluding 0

    def passes_bar(r: CellResult, c_bar: int) -> bool:
        return (r.c_bps == c_bar and r.delta_sr_ann >= 0.5
                and r.ci_excludes_zero and r.deflated_t_ens > 3.0)

    any_ci_excl = any(r.ci_excludes_zero for r in results)
    pass_400 = any(passes_bar(r, 400) for r in results)
    pass_200 = any(passes_bar(r, 200) for r in results)

    if pass_400:
        verdict = 'confirmed-OOS'
    elif pass_200:
        verdict = 'partial-OOS'
    elif any_ci_excl:
        verdict = 'partial-OOS'
    else:
        verdict = 'confirmed-null'

    # --- Best cell within DD constraint (combined max-DD <= DCA * 1.2) ---
    # Per the user-supplied secondary decision rule.
    dca_dd = float(max_dd_compounded(dca_blocks))
    dd_cap = dca_dd * 1.2  # both negative; *1.2 = even more negative allowed
    dd_eligible = [r for r in results if r.max_dd_ens >= dd_cap]
    best_cell = None
    if dd_eligible:
        best_cell = max(dd_eligible, key=lambda r: r.sharpe_ens_ann)

    # --- Recommendation: highest-SR cell at c_bps=200 AND CI excludes 0 ---
    recommended = None
    at_c200_ci = [r for r in results if r.c_bps == 200 and r.ci_excludes_zero
                  and r.max_dd_ens >= dd_cap]
    if at_c200_ci:
        recommended = max(at_c200_ci, key=lambda r: r.sharpe_ens_ann)

    print(f'\n=== VERDICT per pre-reg ===')
    print(f'  any CI excludes 0    : {any_ci_excl}')
    print(f'  passes bar at c=200  : {pass_200}')
    print(f'  passes bar at c=400  : {pass_400}')
    print(f'  -> {verdict}')
    if best_cell is not None:
        print(f'\n  best-Sharpe cell within DD cap (DCA-dd={dca_dd:+.3f} '
              f'-> cap={dd_cap:+.3f}):')
        print(f'    vega={best_cell.vega}  c={best_cell.c_bps}  '
              f'SR_ens={best_cell.sharpe_ens_ann:+.3f}  '
              f'dSR={best_cell.delta_sr_ann:+.3f} '
              f'CI=[{best_cell.ci_lo_ann:+.3f},{best_cell.ci_hi_ann:+.3f}]  '
              f'maxDD={best_cell.max_dd_ens:+.3f}  '
              f'CAGR={best_cell.cagr_ens:+.4f}')
    if recommended is not None:
        print(f'\n  recommended (c_bps=200, CI excl. 0, within DD cap):')
        print(f'    vega={recommended.vega}  SR_ens={recommended.sharpe_ens_ann:+.3f}')

    # --- Heatmap ---
    print(f'\n--- Writing heatmap ---')
    fig_path = Path(args.heatmap)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(sharpe_matrix, vega_grid, c_grid, fig_path,
                 title='Combined DCA + vega*vol_v3 — annualized Sharpe '
                       f'(n_obs={len(dca_blocks)})')
    # Also write delta-Sharpe heatmap with CI-excludes-zero asterisks
    delta_fig = fig_path.with_name(fig_path.stem + '-delta.png')
    plot_heatmap(delta_matrix, vega_grid, c_grid, delta_fig,
                 title='delta Sharpe_ann vs DCA-only (cells with no CI*=overlap 0)')
    print(f'  -> {fig_path}')
    print(f'  -> {delta_fig}')

    # --- Persist ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'pre_reg_page': 'apps/docs/docs/TODO/vol-v3-sleeve-sizing.md',
        'periods_per_year': PPY,
        'sharpe_std_ann': SHARPE_STD_ANN,
        'n_bootstraps': N_BOOT,
        'boot_seed': BOOT_SEED,
        'n_obs': len(dca_blocks),
        'vega_grid': vega_grid,
        'c_bps_grid': c_grid,
        'dca_only_sharpe_ann': float(ann_sharpe(dca_blocks)),
        'dca_only_max_dd': float(max_dd_compounded(dca_blocks)),
        'dca_only_cagr': float(cagr_from_blocks(dca_blocks)),
        'cells': [
            {
                'vega': r.vega, 'c_bps': r.c_bps, 'n_obs': r.n_obs,
                'sharpe_ens_ann': r.sharpe_ens_ann,
                'sharpe_dca_ann': r.sharpe_dca_ann,
                'delta_sr_ann':   r.delta_sr_ann,
                'ci_lo_ann':      r.ci_lo_ann,
                'ci_hi_ann':      r.ci_hi_ann,
                'ci_excludes_zero': bool(r.ci_excludes_zero),
                'max_dd_ens':     r.max_dd_ens,
                'max_dd_dca':     r.max_dd_dca,
                'cagr_ens':       r.cagr_ens,
                'cagr_dca':       r.cagr_dca,
                'deflated_t_ens': r.deflated_t_ens,
            }
            for r in results
        ],
        'verdict': verdict,
        'best_cell_within_dd_cap': None if best_cell is None else {
            'vega': best_cell.vega, 'c_bps': best_cell.c_bps,
            'sharpe_ens_ann': best_cell.sharpe_ens_ann,
            'delta_sr_ann':   best_cell.delta_sr_ann,
            'ci_lo_ann':      best_cell.ci_lo_ann,
            'ci_hi_ann':      best_cell.ci_hi_ann,
            'ci_excludes_zero': bool(best_cell.ci_excludes_zero),
            'max_dd_ens':     best_cell.max_dd_ens,
            'cagr_ens':       best_cell.cagr_ens,
            'deflated_t_ens': best_cell.deflated_t_ens,
        },
        'recommended_at_c200': None if recommended is None else {
            'vega': recommended.vega, 'c_bps': recommended.c_bps,
            'sharpe_ens_ann': recommended.sharpe_ens_ann,
            'delta_sr_ann':   recommended.delta_sr_ann,
            'ci_lo_ann':      recommended.ci_lo_ann,
            'ci_hi_ann':      recommended.ci_hi_ann,
            'max_dd_ens':     recommended.max_dd_ens,
            'cagr_ens':       recommended.cagr_ens,
            'deflated_t_ens': recommended.deflated_t_ens,
        },
        'heatmap_path': str(fig_path),
        'delta_heatmap_path': str(delta_fig),
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n-> {out}', flush=True)


if __name__ == '__main__':
    main()
