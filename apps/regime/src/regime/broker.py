"""Back-compat re-export shim. The Alpaca adapter now lives in
`ss_portfolio.broker` so it can be shared with `apps/relational`'s
live-trading path. New code should import from there directly."""

from ss_portfolio.broker import (
    PAPER_BASE_URL,
    Account,
    AlpacaBroker,
    Trade,
)

__all__ = ['Account', 'Trade', 'AlpacaBroker', 'PAPER_BASE_URL']
