"""v1: legacy single-ticker workflow + aiohttp web service.

Parked subsystem. New work should not extend this — use the regime app
or build a new app under `apps/`. Subpackages:

  * `v1.models`  — Security/Span/Decider/Evaluator/Plotter (1D, per-ticker)
  * `v1.util`    — original Yahoo/CryptoCompare loaders + 1D indicators
  * `v1.scripts` — runnable entry points (webservice, evaluate_securities,
                   sort_securities, plot_*, etc.)

Imports in `v1.util.indicators` predate the workspace `ss_indicators`
package. New callers should prefer `ss_indicators`; the v1 versions are
preserved here only to keep the legacy scripts running without changes.
"""
