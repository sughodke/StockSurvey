"""Local smoke for the return-coupled CWT-GRU walk-forward.

Tiny + fast: ~12 tickers, k=4, n_steps=5, short date range. Verifies
wiring + leak-free shapes before any Modal spend. NOT a leaderboard
run. Run: `uv run python apps/factor/scripts/smoke_cwt_gru.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from factor import load_ticker_cwt, train_cwt_gru_walkforward

SUBSET = Path('apps/notebook/data/stooq_us_long')


def main() -> None:
    manifest = json.loads((SUBSET / 'manifest.json').read_text())
    names = [t['ticker'] for t in manifest['tickers']][:12]
    print(f'loading {len(names)} tickers: {names}')
    tds = []
    for n in names:
        try:
            td = load_ticker_cwt(
                n, stooq_dir=str(SUBSET),
                start='2016-01-01', end='2024-01-01', lookback=90)
            if td.valid.any():
                tds.append(td)
        except Exception as e:  # noqa: BLE001
            print(f'  skip {n}: {type(e).__name__}: {e}')
    print(f'usable: {len(tds)} tickers; '
          f'panel shape per ticker e.g. {tds[0].features.shape} '
          f'(expect (*, 13))')
    assert tds[0].features.shape[1] == 13

    res = train_cwt_gru_walkforward(
        tds, k=4, rebal_days=20,
        train_window_blocks=30, val_window_blocks=15, step_window_blocks=15,
        seq_len=16, lookback=90, n_steps=5,
        learning_rate=1e-3, weight_decay=1e-3, seed=0, verbose=True)

    print(f'\nwindows={res.n_windows}  mean_val_ic={res.mean_val_ic:+.4f}  '
          f'pos_frac={res.positive_val_ic_fraction:.2f}  '
          f'mean_val_sharpe={res.mean_val_sharpe:+.3f}')
    assert res.n_windows >= 2, 'expected >=2 walk-forward windows'
    for w in res.windows:
        assert math.isfinite(w.val_ic), f'non-finite val_ic w{w.window_idx}'
        assert math.isfinite(w.train_ic), f'non-finite train_ic w{w.window_idx}'
        assert w.val_block_start >= w.train_block_end, (
            f'LEAK: val starts {w.val_block_start} before train ends '
            f'{w.train_block_end} (w{w.window_idx})')
        print(f'  w{w.window_idx}: train[{w.train_block_start}:'
              f'{w.train_block_end}] val[{w.val_block_start}:'
              f'{w.val_block_end}] '
              f'train_ic={w.train_ic:+.4f} val_ic={w.val_ic:+.4f} '
              f'val_start={w.val_start_date}')
    print('\nSMOKE OK — wiring + leak-free window boundaries verified')


if __name__ == '__main__':
    main()
