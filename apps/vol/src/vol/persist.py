"""JSON checkpoint persistence for the vol-v3 predictor.

Mirrors `relational/persist.py` and `regime/persist.py`. Stores
exactly what `vol.live.run_live` needs to recompute today's targets
without re-fetching the train substrate:

* the frozen 4-feature OLS predictor (coefs + train z-score stats)
* the universe (list of optionable symbols to consider — derived
  from DoltHub at checkpoint-build time, frozen here so live runs are
  deterministic across days)
* the VIX regime-gate config (rolling-median lookback, FRED series id)
* the top-K + vega budget + strangle-construction knobs
* provenance (train period, val period, OOS Sharpe, source of truth
  for the predictor row)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Order matters — `vol.inference.predict_iv_rv_gap` and the train code
# both depend on this layout. Adding a feature requires re-training.
LIVE_FEATURE_NAMES: list[str] = ['iv_over_hv', 'iv_z', 'iv_change_4w', 'hv_change_4w']


@dataclass(frozen=True)
class StranglesConfig:
    """How a top-K pick translates into an Alpaca multi-leg short strangle.

    Each pick gets a delta-neutral short strangle (sell OTM call + sell
    OTM put) at the target tenor. Vega budget per name caps the size.
    All defaults are conservative paper-trading starting points; tune
    once we observe how the broker fills behave.
    """
    target_tenor_days: int = 30       # ~1-month options, matches v3's 20-trading-day rebal
    tenor_tolerance_days: int = 7     # accept contracts ±N days from target
    target_delta_call: float = 0.20   # OTM call wing |Δ|≈0.20
    target_delta_put:  float = 0.20   # OTM put wing  |Δ|≈0.20
    vega_budget_per_name_usd: float = 100.0  # $ vega risk per strangle
    min_open_interest: int = 100      # skip illiquid contracts
    min_bid_size: int = 10            # skip names with no actual quote depth
    max_bid_ask_spread_pct: float = 0.15  # widest tolerable spread / mid


@dataclass(frozen=True)
class VolCheckpoint:
    """One-row, frozen-after-build state for `ss-vol live`.

    Use `save_checkpoint` to write; `load_checkpoint` to read. The
    predictor is the v2-dolthub-oos 4-feature OLS, retrained on a fixed
    train window. We persist the z-scoring stats alongside the coefs so
    live evaluation does NOT recompute train statistics from a partial
    sample.
    """
    # Predictor (4-feature OLS over `FEATURE_NAMES`)
    feature_names: list[str]
    coefs: list[float]                # length len(features) + 1 (intercept last)
    feat_mean: list[float]            # train z-score mean, len(features)
    feat_std: list[float]             # train z-score std,  len(features)
    # Universe (frozen at checkpoint-build time)
    universe: list[str]               # optionable symbols
    # VIX regime gate
    gate_fred_series: str             # 'VIXCLS'
    gate_lookback_trading_days: int   # 126 (v3 deployment recipe)
    # Sizing / construction
    top_k: int                        # 100 in v3
    strangle: StranglesConfig
    # Provenance
    train_period: str                 # 'YYYY-MM-DD → YYYY-MM-DD'
    val_period: str
    val_pearson_r: float
    n_obs_oos: int
    oos_ann_sharpe: float             # full-panel from non-overlap dump
    oos_deflated_t: float
    saved_at: str = field(default_factory=lambda:
                          datetime.now(timezone.utc).isoformat(timespec='seconds'))
    notes: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        # Nested dataclass -> dict for JSON round-trip
        return d


def save_checkpoint(cp: VolCheckpoint, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cp.to_dict(), indent=2))


def load_checkpoint(path: str | Path) -> VolCheckpoint:
    p = Path(path)
    raw = json.loads(p.read_text())
    # Strict allowlist: drop unknown keys so a forward-compatible file
    # doesn't blow up on `VolCheckpoint(**)`.
    strangle = StranglesConfig(**raw.pop('strangle'))
    known = {f for f in VolCheckpoint.__dataclass_fields__ if f != 'strangle'}
    filtered = {k: v for k, v in raw.items() if k in known}
    return VolCheckpoint(**filtered, strangle=strangle)


def validate(cp: VolCheckpoint) -> None:
    """Raise ValueError if the checkpoint is internally inconsistent."""
    if cp.feature_names != LIVE_FEATURE_NAMES:
        raise ValueError(
            f'feature_names mismatch: got {cp.feature_names}, '
            f'expected {LIVE_FEATURE_NAMES}')
    if len(cp.coefs) != len(cp.feature_names) + 1:
        raise ValueError(
            f'coefs length {len(cp.coefs)} != features+1 '
            f'({len(cp.feature_names) + 1})')
    if len(cp.feat_mean) != len(cp.feature_names):
        raise ValueError(f'feat_mean length mismatch')
    if len(cp.feat_std) != len(cp.feature_names):
        raise ValueError(f'feat_std length mismatch')
    if np.any(np.asarray(cp.feat_std) <= 0):
        raise ValueError(f'feat_std must be positive')
    if not cp.universe:
        raise ValueError(f'universe is empty')
    if not 1 <= cp.top_k <= len(cp.universe):
        raise ValueError(
            f'top_k {cp.top_k} out of bounds for universe size '
            f'{len(cp.universe)}')
    if cp.gate_lookback_trading_days < 1:
        raise ValueError(f'gate_lookback_trading_days must be >= 1')


__all__ = [
    'LIVE_FEATURE_NAMES', 'StranglesConfig', 'VolCheckpoint',
    'save_checkpoint', 'load_checkpoint', 'validate',
]
