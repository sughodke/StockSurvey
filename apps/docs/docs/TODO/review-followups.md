# Review follow-ups — paper-trade can proceed without these

Inline `TODO(review #N)` markers point to the file:line. Grep
`TODO(review` to surface the full backlog.

- **#5** — `submit_orders` swallows transport-layer (5xx / connection)
  errors the same way it swallows fractionability rejections. Distinguish
  4xx (skip+log, current) from 5xx (re-raise+abort) so an Alpaca outage
  doesn't silently zero every order. (`ss_portfolio/broker.py:submit_orders`)
- **#6** — `gmm_cluster_pair_weights` produces signed long/short. Not
  exposed via the relational dispatch but importable; document /
  assert long-only invariant at the inference boundary if anyone wires
  it in. (`relational/empirical_sectors_gmm.py:409`)
- **#10** — `precompute_windows` per-ticker mean is over ALL TIME, not
  causal. Safe under scale-axis-normalized divergences (KL/JS/cosine/L2)
  only — assert that at call sites, or refactor to a causal rolling
  mean before exposing to non-scale-invariant downstream ops.
  (`ss_wavelets/windowing.py:42`)
- **#3 follow-up** — `min_notional` gate in `build_trades` uses full-
  precision notional but ships `round(qty_diff, 6)` qty. Penny-stock
  edge case; surfaced via `rejected_orders` already.
  (`ss_portfolio/broker.py:build_trades`)

## ss_loaders load_cryptocompare v2 endpoint fix

**Discovered 2026-05-22 via the crypto venue port** (see
[`factor-crypto-venue-test`](factor-crypto-venue-test.md)).
`ss_loaders.load_cryptocompare` hits CryptoCompare's retired v1
response shape — top-level `Data` as a list keyed by `time`. The v2
endpoint nests at `Data.Data` and the per-row timestamp is `time`
inside that nested list. The v1 path raises
`AttributeError: 'DataFrame' object has no attribute 'time'` on any
non-empty response.

Repro:
```python
from ss_loaders import load_cryptocompare
load_cryptocompare('BTC')  # AttributeError
```

Fix (small):
- Switch the request to the v2 endpoint (still
  `min-api.cryptocompare.com/data/v2/histoday` style — already what the
  paginator hits, the bug is in the response unpacking).
- Read `payload['Data']['Data']` (list of dicts) instead of
  `payload['Data']` (assumed-list-of-dicts).
- Paginate via the `toTs` query param using the *last row's* `time`
  field. The crypto venue prep script
  (`apps/factor/scripts/prep_crypto_universe.py`) inlines a working
  v2-compatible fetcher — copy that paging loop into the library
  function and add a small unit test that mocks one page of the v2
  response shape.

Out of scope for the venue test (the prep script's inline fetcher
unblocks the crypto run). One-day pickup before the next consumer of
the library function.
