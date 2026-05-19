"""Local smoke for the fixed (C,L) encoders + horizon-scaled walk-forward.

Sanctioned local run (small universe, n_steps tiny) per the
compute-placement rule. Proves the load-bearing wiring before any Modal
spend:

  1. build_spectral_features / build_minirocket_features emit (T, C*L)
     with the exact flatten the backbone contract expects (round-trip
     reshape identity).
  2. The (C,L) block flows through align_tickers_at_rebal →
     precompute_inputs → identity_backbone → train_scorer_walkforward
     and yields finite IC/Sharpe.
  3. The year-comparable block scaling produces a sane window count at
     both the 20d anchor and the 5d short horizon.

Run: uv run python apps/factor/scripts/smoke_cl_encoders.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from factor import (
    IndicatorGridConfig, MiniRocketGridConfig, SpectralGridConfig,
    build_minirocket_features, build_spectral_features,
    load_ticker_indicators, load_ticker_minirocket, load_ticker_spectral,
    train_scorer_indicators_walkforward,
    train_scorer_minirocket_walkforward, train_scorer_spectral_walkforward,
)

STOOQ = 'apps/notebook/data/stooq_us_long'
N_TICKERS = 15
N_STEPS = 20

# Year-comparable block scaling: base 63/39/39 at rebal_days=20.
BASE = (63, 39, 39)


def scaled_blocks(rebal_days: int) -> tuple[int, int, int]:
    f = 20.0 / rebal_days
    return tuple(int(round(b * f)) for b in BASE)  # type: ignore[return-value]


def main() -> int:
    manifest = json.loads((Path(STOOQ) / 'manifest.json').read_text())
    names = [t['ticker'] for t in
             sorted(manifest['tickers'], key=lambda t: -t['n_bars'])
             ][:N_TICKERS]
    print(f'smoke universe: {names}')

    spec_cfg = SpectralGridConfig()
    mr_cfg = MiniRocketGridConfig()
    ind_cfg = IndicatorGridConfig()

    # ---- 1. shape + flatten round-trip on one ticker ----
    px = load_ticker_spectral(names[0], stooq_dir=STOOQ).prices
    sf, sv = build_spectral_features(px, spec_cfg)
    mf, mv = build_minirocket_features(px, mr_cfg)
    C_s, L_s = spec_cfg.n_channels(), spec_cfg.compressed_len()
    C_m, L_m = mr_cfg.n_channels(), mr_cfg.compressed_len()
    assert sf.shape == (px.shape[0], C_s * L_s), sf.shape
    assert mf.shape == (px.shape[0], C_m * L_m), mf.shape
    assert sv.sum() > 100, f'spectral valid bars too few: {sv.sum()}'
    assert mv.sum() > 100, f'minirocket valid bars too few: {mv.sum()}'
    # The contract: features.reshape(-1, C, L) must round-trip — this is
    # exactly what align_tickers_at_rebal does before the backbone.
    rt = sf.reshape(-1, C_s, L_s).reshape(sf.shape)
    assert np.array_equal(np.nan_to_num(rt), np.nan_to_num(sf)), 'spectral rt'
    rt = mf.reshape(-1, C_m, L_m).reshape(mf.shape)
    assert np.array_equal(np.nan_to_num(rt), np.nan_to_num(mf)), 'mr rt'
    assert len(spec_cfg.channel_names()) == C_s * L_s
    assert len(mr_cfg.channel_names()) == C_m * L_m
    print(f'  spectral (C,L)=({C_s},{L_s}) width={sf.shape[1]} '
          f'valid={int(sv.sum())}/{sv.size}')
    print(f'  minirocket (C,L)=({C_m},{L_m}) width={mf.shape[1]} '
          f'valid={int(mv.sum())}/{mv.size}')

    # ---- 2 + 3. end-to-end walk-forward at anchor + short horizon ----
    spec_td = [load_ticker_spectral(n, stooq_dir=STOOQ) for n in names]
    mr_td = [load_ticker_minirocket(n, stooq_dir=STOOQ) for n in names]
    ind_td = [load_ticker_indicators(n, stooq_dir=STOOQ) for n in names]

    for rebal in (20, 5):
        tr, va, st = scaled_blocks(rebal)
        print(f'\n=== rebal_days={rebal}  blocks train/val/step={tr}/{va}/{st} ===')
        for tag, fn, td, cfg in (
            ('indicator', train_scorer_indicators_walkforward, ind_td, ind_cfg),
            ('spectral', train_scorer_spectral_walkforward, spec_td, spec_cfg),
            ('minirocket', train_scorer_minirocket_walkforward, mr_td, mr_cfg),
        ):
            wf = fn(td, cfg, rebal_days=rebal,
                    train_window_blocks=tr, val_window_blocks=va,
                    step_window_blocks=st, scorer='linear',
                    n_steps=N_STEPS, learning_rate=1e-2, weight_decay=1e-3,
                    verbose=False)
            mvic, mvsh = wf.mean_val_ic, wf.mean_val_sharpe
            assert wf.n_windows >= 1, f'{tag}: no windows'
            assert np.isfinite(mvic), f'{tag}: non-finite mean val IC'
            assert np.isfinite(mvsh), f'{tag}: non-finite mean val Sharpe'
            print(f'  {tag:<11} {wf.n_windows} win  '
                  f'mean val IC={mvic:+.4f}  mean val Sharpe={mvsh:+.3f}')

    print('\nSMOKE GREEN')
    return 0


if __name__ == '__main__':
    sys.exit(main())
