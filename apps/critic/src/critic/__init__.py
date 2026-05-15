"""apps/critic — Φ(state, action) → predicted-deployment-Sharpe value function.

Trained on (state, action, realized-Sharpe) triples consolidated from
the walk-forward outputs of factor, vol, gate, and pairs. The goal is a
differentiable training signal for downstream policy networks: train π
against -Φ rather than via post-hoc classifier heuristics.

See `apps/docs/docs/TODO/critic-phi-value-function.md` for the
pre-registered Φ-quality experiment.
"""
