"""Counterfactual regret math for trading.

The key insight that makes CFR exceptionally tractable in our setup
is the price-taker assumption: our trade does not move the market,
so future prices are action-independent. Given the realized
per-ticker forward return over a rebal block, we can compute the
*realized* return of every action in the menu in closed form — no
re-simulation required.

```
realized_return_block[a] = log( weights_a · exp(per_ticker_logret) )
                         ≈ weights_a · per_ticker_logret    (1st-order)
instantaneous_regret[a]  = realized_return_block[a]
                         - realized_return_block[σ_played]
```

The 1st-order approximation is exact for small returns and accurate
to ~1bp at typical 20-day-block magnitudes. For larger blocks the
log-sum-exp form is preferred — we expose both.

The `regret_matching` helper turns a cumulative-regret vector into
a probability distribution: positive regrets are normalized;
all-non-positive regrets fall back to uniform.
"""
from __future__ import annotations

import numpy as np


def compute_block_log_returns(
    block_per_ticker_logret: np.ndarray,
    action_weights: np.ndarray,
) -> np.ndarray:
    """Realized portfolio log return for each action over one block.

    Parameters
    ----------
    block_per_ticker_logret : `(N,)`
        Per-ticker log return over the rebal block. NaN positions
        are treated as zero (the ticker delisted mid-block or was
        otherwise missing — the action either weighted it zero
        anyway, or paid the price of weighting a missing ticker as
        zero return, which is the conservative default).
    action_weights : `(n_actions, N)`
        Per-action target weight vector at the start of the block.

    Returns
    -------
    `(n_actions,)` log returns.

    Uses the exact log-of-weighted-sum-of-exponentials form so the
    answer is correct for arbitrary block lengths. For short blocks
    where `per_ticker_logret` is small this is essentially the same
    as `action_weights @ per_ticker_logret`; for long blocks
    (quarterly, annual) the difference can be non-trivial.
    """
    if block_per_ticker_logret.ndim != 1:
        raise ValueError(
            f'block_per_ticker_logret must be 1D, got shape '
            f'{block_per_ticker_logret.shape}')
    if action_weights.shape[1] != block_per_ticker_logret.shape[0]:
        raise ValueError(
            f'action_weights second dim {action_weights.shape[1]} != '
            f'block_per_ticker_logret {block_per_ticker_logret.shape[0]}')
    r = np.where(np.isnan(block_per_ticker_logret), 0.0, block_per_ticker_logret)
    er = np.exp(r)
    portfolio_simple = action_weights @ er  # (n_actions,)
    # Cash / short-circuit: weights sum to ~0 → log of ~0 → -inf, which
    # corrupts regret. The right answer for an all-zero weight (cash)
    # is "no return" = 0. Subtract the gross from the simple return so
    # that `gross=1, return=0` is the do-nothing baseline.
    gross = action_weights.sum(axis=1)
    safe_gross = np.where(gross > 0, gross, 1.0)
    excess = portfolio_simple - gross
    out = np.log1p(excess / safe_gross) * gross
    out = np.where(gross > 0, out, 0.0)
    return out


def compute_block_regrets(
    block_per_ticker_logret: np.ndarray,
    action_weights: np.ndarray,
    played_action: int,
) -> np.ndarray:
    """Instantaneous regret per action = R(a) - R(played).

    `(n_actions,)` regrets. Positive entry = "we should have played
    that action instead." Zero at index `played_action` by
    construction.
    """
    realized = compute_block_log_returns(block_per_ticker_logret, action_weights)
    return realized - realized[played_action]


def regret_matching(
    cumulative_regret: np.ndarray, *,
    uniform_fallback: bool = True,
) -> np.ndarray:
    """Probability distribution from positive cumulative regret.

    Standard regret-matching from Hart & Mas-Colell (2000):
      π(a) = max(R(a), 0) / Σ_a max(R(a), 0)         if Σ > 0
           = 1/N                                       otherwise

    Setting `uniform_fallback=False` returns an all-zero vector
    when no action has positive regret — useful for callers that
    want to distinguish "haven't visited this infoset yet" from
    "actively chose uniform."
    """
    pos = np.maximum(cumulative_regret, 0.0)
    total = pos.sum()
    if total > 0:
        return pos / total
    if uniform_fallback:
        n = len(cumulative_regret)
        return np.full(n, 1.0 / n)
    return np.zeros_like(cumulative_regret)


def sample_action(
    policy: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample an integer action index from a probability vector."""
    if policy.sum() <= 0:
        return int(rng.integers(len(policy)))
    return int(rng.choice(len(policy), p=policy / policy.sum()))


__all__ = [
    'compute_block_log_returns',
    'compute_block_regrets',
    'regret_matching',
    'sample_action',
]
