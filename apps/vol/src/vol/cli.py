"""`ss-vol` CLI shim. Subcommands live in apps/vol/scripts/."""
from __future__ import annotations

import sys


def main() -> int:
    print(
        'ss-vol v0 — drivers live in apps/vol/scripts:\n'
        '  uv run python apps/vol/scripts/audit_data.py\n'
        '  uv run python apps/vol/scripts/run_walkforward.py\n',
        file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
