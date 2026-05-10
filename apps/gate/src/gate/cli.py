"""`ss-gate` CLI shim. Subcommands proxy to scripts/run_*.py for v0."""
from __future__ import annotations

import sys


def main() -> int:
    print(
        'ss-gate v0 — drivers live in apps/gate/scripts:\n'
        '  uv run python apps/gate/scripts/run_baseline.py\n'
        '  uv run python apps/gate/scripts/run_walkforward.py\n',
        file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
