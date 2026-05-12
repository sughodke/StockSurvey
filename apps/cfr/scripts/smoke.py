"""Phase 0 smoke. Thin shim that calls `cfr.scripts_smoke.run_smoke`."""
from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--n-tickers', type=int, default=30)
    p.add_argument('--max-bars', type=int, default=2000)
    args = p.parse_args()
    from cfr.scripts_smoke import run_smoke
    return run_smoke(
        data_dir=args.data_dir,
        n_tickers=args.n_tickers,
        max_bars=args.max_bars,
    )


if __name__ == '__main__':
    raise SystemExit(main())
