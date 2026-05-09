# Review follow-ups — paper-trade can proceed without these

Inline `TODO(review #N)` markers point to the file:line. Grep
`TODO(review` to surface the full backlog.

- **#4** — walk-forward train/val slices double-count the boundary bar
  via end-inclusive pandas `.loc` (`apps/regime/src/regime/trainer.py:522`).
  Research-only, no live impact.
- **#5** — `submit_orders` swallows transport-layer (5xx / connection)
  errors the same way it swallows fractionability rejections. Distinguish
  4xx (skip+log, current) from 5xx (re-raise+abort) so an Alpaca outage
  doesn't silently zero every order. (`ss_portfolio/broker.py:submit_orders`)
- **#6** — `gmm_cluster_pair_weights` produces signed long/short. Not
  exposed via the relational dispatch but importable; document /
  assert long-only invariant at the inference boundary if anyone wires
  it in. (`relational/empirical_sectors_gmm.py:409`)
- **#7** — `rsi` (matrix) and `rsi_strided` use different lag conventions
  (`up[t-1]` vs `up[t]`). Both causal but not interchangeable.
  (`ss_indicators/rsi.py:17`)
- **#10** — `precompute_windows` per-ticker mean is over ALL TIME, not
  causal. Safe under scale-axis-normalized divergences (KL/JS/cosine/L2)
  only — assert that at call sites, or refactor to a causal rolling
  mean before exposing to non-scale-invariant downstream ops.
  (`ss_wavelets/windowing.py:42`)
- **#3 follow-up** — `min_notional` gate in `build_trades` uses full-
  precision notional but ships `round(qty_diff, 6)` qty. Penny-stock
  edge case; surfaced via `rejected_orders` already.
  (`ss_portfolio/broker.py:build_trades`)
