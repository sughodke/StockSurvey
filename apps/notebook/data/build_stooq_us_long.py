"""Build a curated US-stocks Stooq subset under apps/notebook/data/stooq_us_long/.

Walks the user's local Stooq archive (default `./StooqData/`), keeps
tickers that pass the history + average-price filters, and copies them
to `apps/notebook/data/stooq_us_long/` preserving the
`daily/<country>/<exchange> <type>/<bucket>/<ticker>.<country>.txt`
layout — so anything that takes a `--stooq-dir` argument can point at
the subset and `iter_stooq_ticker_files` walks it the same way.

Why a curated mid-size subset (vs the full 12K-ticker, 1.7GB archive):
git-friendly (~140 MB committed, no LFS) and Modal-image-friendly
(seconds to upload, cached cleanly across rebuilds). Captures the
"wider universe + longer history" the deterministic-indicator path in
apps/factor needs without dragging the entire archive into the repo.

Filter defaults are tuned for the indicator path: `--min-history-years
22` clears the default IndicatorGridConfig's 4978-bar CCI warmup
(~19.7 years) with buffer; `--min-avg-price 5` drops penny stocks where
the indicators behave erratically; `--start-date 2000-01-01` truncates
each ticker to dates >= 2000 because pre-2000 history is gated out by
the warmup anyway, ~halving the on-disk size.

Usage:
    uv run python apps/notebook/data/build_stooq_us_long.py \\
        --src ./StooqData --dst apps/notebook/data/stooq_us_long \\
        --min-history-years 22 --min-avg-price 5

    # Cap to top-N by history-length if the filter still passes too many:
    uv run python apps/notebook/data/build_stooq_us_long.py --max-tickers 500
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ss_loaders import (
    iter_stooq_ticker_files, read_stooq_file, stooq_ticker_from_path,
)


@dataclass
class TickerStats:
    path: Path
    ticker: str
    n_bars: int
    years: float
    avg_close: float
    first_date: str
    last_date: str

    def passes(self, min_years: float, min_avg_price: float) -> bool:
        return (self.years >= min_years and self.avg_close >= min_avg_price
                and self.n_bars > 0)


def scan_archive(
    src: Path, include_etfs: bool, start_date: str | None,
) -> list[TickerStats]:
    """Read every Stooq file once, return TickerStats for downstream filtering.

    `start_date` (YYYY-MM-DD) slices each ticker's history to that date
    forward before computing stats — so the min-history / min-price gates
    are evaluated on the same data we'll actually write. Tickers whose
    truncated history is empty are dropped.

    The bottleneck is the ~12K small-CSV reads — single-threaded does ~30s on
    SSD, ~2-3min on a slow disk. We don't parallelise because pandas's CSV
    reader holds the GIL in the hot path; a thread pool gives no speed-up
    and a process pool's IPC overhead eats it.
    """
    files = iter_stooq_ticker_files(src, include_etfs=include_etfs)
    print(f'scanning {len(files)} ticker files under {src}…',
          file=sys.stderr, flush=True)
    if start_date:
        print(f'  truncating each ticker to dates >= {start_date}',
              file=sys.stderr, flush=True)
    stats: list[TickerStats] = []
    t0 = time.perf_counter()
    for i, path in enumerate(files):
        if i and i % 1000 == 0:
            print(f'  {i:>5d}/{len(files)}  ({time.perf_counter()-t0:.1f}s)',
                  file=sys.stderr, flush=True)
        df = read_stooq_file(path)
        if df is None or df.empty:
            continue
        if start_date:
            df = df.loc[start_date:]
            if df.empty:
                continue
        first = df.index[0]
        last = df.index[-1]
        years = (last - first).days / 365.25
        stats.append(TickerStats(
            path=path,
            ticker=stooq_ticker_from_path(path),
            n_bars=len(df),
            years=years,
            avg_close=float(df['close'].mean()),
            first_date=first.strftime('%Y-%m-%d'),
            last_date=last.strftime('%Y-%m-%d'),
        ))
    print(f'parsed {len(stats)} non-empty files in '
          f'{time.perf_counter()-t0:.1f}s', file=sys.stderr, flush=True)
    return stats


def filter_and_cap(
    stats: list[TickerStats],
    *,
    min_years: float, min_avg_price: float, max_tickers: int | None,
) -> list[TickerStats]:
    """Apply min-history + min-price gates, then optionally cap by
    history length descending so we keep the most data-rich names."""
    passing = [s for s in stats if s.passes(min_years, min_avg_price)]
    print(f'{len(passing)} / {len(stats)} tickers pass '
          f'(min_years={min_years}, min_avg_price={min_avg_price})',
          file=sys.stderr, flush=True)
    if max_tickers is not None and len(passing) > max_tickers:
        passing.sort(key=lambda s: s.years, reverse=True)
        kept = passing[:max_tickers]
        cutoff_years = kept[-1].years
        print(f'capping to top-{max_tickers} by history; '
              f'cutoff history = {cutoff_years:.1f} years',
              file=sys.stderr, flush=True)
        passing = kept
    return sorted(passing, key=lambda s: s.ticker)


def _truncate_stooq_text(path: Path, start_yyyymmdd: str) -> str:
    """Stream `path` line-by-line and keep header + rows whose date field
    is >= `start_yyyymmdd` (lexical compare on YYYYMMDD is monotonic).

    Faster than re-emitting via pandas: avoids the float-formatting
    drift that would otherwise change the on-disk bytes vs the source
    archive (Stooq's float precision varies per ticker; we want to
    preserve it for bit-identical reads downstream).
    """
    out: list[str] = []
    with open(path, 'r') as f:
        header = f.readline()
        out.append(header)
        for line in f:
            # Stooq layout: TICKER,PER,YYYYMMDD,HHMMSS,OPEN,...
            # split with maxsplit=3 — only need fields 0-2 to filter.
            parts = line.split(',', 3)
            if len(parts) >= 3 and parts[2] >= start_yyyymmdd:
                out.append(line)
    return ''.join(out)


def copy_subset(
    stats: list[TickerStats], src: Path, dst: Path,
    *, start_date: str | None,
) -> None:
    """Copy each kept file into the same relative path under `dst`.

    When `start_date` is set, the destination file holds only rows on or
    after that date so the bytes on disk match the manifest's first_date
    / n_bars / years values. Otherwise we copy the source file verbatim
    (preserves mtime + permissions via copy2).
    """
    if dst.exists():
        print(f'WARN: {dst} already exists; new files will be added on top',
              file=sys.stderr, flush=True)
    src = src.resolve()
    start_yyyymmdd = start_date.replace('-', '') if start_date else None
    n_copied = 0
    for s in stats:
        rel = s.path.resolve().relative_to(src)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if start_yyyymmdd:
            dst_path.write_text(_truncate_stooq_text(s.path, start_yyyymmdd))
        else:
            shutil.copy2(s.path, dst_path)
        n_copied += 1
    print(f'copied {n_copied} files to {dst}', file=sys.stderr, flush=True)


def write_manifest(stats: list[TickerStats], dst: Path) -> None:
    """Write a JSON inventory next to the data so the curation gate is
    inspectable after the fact (which tickers, what filter, what cutoff)."""
    manifest = {
        'count': len(stats),
        'tickers': [
            {
                'ticker': s.ticker,
                'n_bars': s.n_bars,
                'years': round(s.years, 2),
                'avg_close': round(s.avg_close, 2),
                'first_date': s.first_date,
                'last_date': s.last_date,
            }
            for s in stats
        ],
    }
    manifest_path = dst / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'wrote manifest with {len(stats)} entries to {manifest_path}',
          file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=Path('./StooqData'),
                   help='Source archive root (default: ./StooqData)')
    p.add_argument('--dst', type=Path,
                   default=Path('./apps/notebook/data/stooq_us_long'),
                   help='Destination subset root '
                        '(default: ./apps/notebook/data/stooq_us_long)')
    p.add_argument('--min-history-years', type=float, default=22.0,
                   help='Minimum history span in years (default: 22 — clears '
                        'the default IndicatorGridConfig 4978-bar warmup '
                        'with buffer)')
    p.add_argument('--min-avg-price', type=float, default=5.0,
                   help='Minimum average close price (default: 5 — drops '
                        'penny stocks)')
    p.add_argument('--max-tickers', type=int, default=None,
                   help='Cap total tickers (default: no cap; recommended ~500 '
                        'to keep the subset under ~50 MB)')
    p.add_argument('--include-etfs', action='store_true',
                   help='Also include ETF subdirectories (default: stocks only)')
    p.add_argument('--start-date', default='2000-01-01',
                   help='Truncate each ticker to dates >= YYYY-MM-DD before '
                        'computing stats and writing (default: 2000-01-01 — '
                        'pre-2000 history is gated out by the indicator-path '
                        'warmup anyway, so dropping it ~halves the on-disk '
                        'size). Pass --start-date \'\' to keep full history.')
    p.add_argument('--dry-run', action='store_true',
                   help='Scan + filter + print stats; do not copy or write')
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f'ERROR: source archive not found at {args.src}',
              file=sys.stderr)
        return 1

    start_date = args.start_date or None
    stats = scan_archive(args.src, include_etfs=args.include_etfs,
                         start_date=start_date)
    kept = filter_and_cap(
        stats,
        min_years=args.min_history_years,
        min_avg_price=args.min_avg_price,
        max_tickers=args.max_tickers,
    )

    if not kept:
        print('ERROR: no tickers passed the filter — relax thresholds',
              file=sys.stderr)
        return 1

    print(f'\nselected {len(kept)} tickers '
          f'(median history {sorted(s.years for s in kept)[len(kept)//2]:.1f} y, '
          f'median avg price ${sorted(s.avg_close for s in kept)[len(kept)//2]:.2f})')

    if args.dry_run:
        print('\n--dry-run: skipping copy. First 10 selected:', file=sys.stderr)
        for s in kept[:10]:
            print(f'  {s.ticker:<8s}  {s.years:5.1f}y  '
                  f'${s.avg_close:>8.2f}  {s.first_date} -> {s.last_date}')
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    copy_subset(kept, args.src, args.dst, start_date=start_date)
    write_manifest(kept, args.dst)
    return 0


if __name__ == '__main__':
    sys.exit(main())
