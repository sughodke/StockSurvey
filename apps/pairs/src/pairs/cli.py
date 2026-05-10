"""`ss-pairs` CLI shim. Subcommands proxy to scripts/run_*.py for v0."""
from __future__ import annotations

import sys


def main() -> int:
    print(
        'ss-pairs v0 — drivers live in apps/pairs/scripts:\n'
        '  uv run python apps/pairs/scripts/run_walkforward.py\n',
        file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
