"""Entry point used by the finding — runs all 3 folds + pools."""
from __future__ import annotations

import sys

from e2e_portfolio.cli import main

if __name__ == '__main__':
    sys.argv = ['ss-e2e', 'eval', '--pooled']
    main()
