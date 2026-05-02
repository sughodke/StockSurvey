# `ss_indicators`

Pure-numpy technical indicators for the StockSurvey workspace.
Matrix-form (`(T, ...)` → `(T, ...)`) so single-ticker callers pass
`(T,)` and multi-ticker callers pass `(T, N)`. All operations are
causal along axis 0 (the date axis).

Migrated off JAX in the cci+rsi_strided refactor — there's no autograd
path through these anymore. The legacy JAX-Adam regime trainer
(`apps/regime/research/optimize_adam.py`) was the only consumer that
needed differentiability and is now parked. The default Optuna+vectorbt
trainer just consumes scalar scores and is unaffected.

## Surface

| Function | Output | Notes |
| --- | --- | --- |
| `rsi(prices, n=7)` | `(T, ...)` ∈ [0, 100] | Wilder smoothing; first `n` rows = 50 (neutral seed). Vectorized over trailing axes. |
| `rsi_strided(prices, n, w=1)` | `(T,)` ∈ [0, 100] ∪ {NaN} | 1-D only. RSI(n) over stride-w returns; w=1 reduces to canonical daily RSI. Used by FiLM heads for dense (n, w) supervision. |
| `cci(prices, n=20)` | `(T, ...)` (≈ ±300 typical, not strictly bounded) | Lambert (1980) close-only CCI. First `n − 1` rows = NaN. |
| `cci_strided(prices, n, w=1)` | `(T,)` (same range) | 1-D only. CCI(n) over stride-w price history `[t, t-w, ..., t-(n-1)*w]`. Sibling of `rsi_strided` for FiLM grids. |
| `macd(prices, fast=12, slow=26, signal=9)` | `(line, signal, histogram)` | EMA difference + signal smoothing. |
| `bbands(prices, window=21, nsd=2.0)` | `(middle, upper, lower)` | SMA ± nsd × rolling std. |
| `sma(x, window)` | `(T, ...)` | Cumsum trick, O(T) regardless of window. Expanding window during warm-up. |
| `ema(x, span)` | `(T, ...)` | Pandas-style `ewm(span, adjust=False)`. Loop along axis 0; per-step is vectorized over the cross-section. |
| `rolling_std(x, window)` | `(T, ...)` | Population std (ddof=0). Promotes to fp64 internally to avoid `mu² − μ²` cancellation. |
| `corwin_schultz_spread(highs, lows, window=21)` | DataFrame ∈ [0, 0.20] | Pandas-based, since high/low inputs come in as DataFrames anyway. Liquidity proxy when no volume is available. |
| `symmetric_kl_divergence` / `js_divergence` / `cosine_divergence` / `l2_divergence` | scalar or `(...)` | Regime-shift scores between two CWT power distributions. `get_divergence(name)` looks one up by short name. |
| `fibonacci_retracement(prices, n=90)` | `(t1, t2, [levels])` | Legacy v1 plotting helper; numpy. |

## Conventions

- **Time leads.** Axis 0 is dates. Multi-ticker is `(T, N)` so the same
  function compiles down to a single vectorized op across the
  cross-section.
- **Causal everywhere.** Indicators only see data up through index `t`
  when producing the value at `t` — required for backtest honesty and
  for `causal_cwt` interop downstream.
- **Warm-up convention varies by indicator.** RSI fills with the
  neutral value 50; CCI / `*_strided` fill with NaN; SMA / EMA /
  rolling_std use an expanding window during warm-up (matches pandas
  `rolling(window, min_periods=1)`).
- **dtype preservation.** Outputs are returned in the input dtype where
  possible; `rolling_std` is the only one that promotes internally
  (then casts back) for numerical reasons.
- **No autograd.** If you need differentiable indicators, either reach
  for `ss_portfolio` (still JAX) or write a forward-only loss that
  consumes indicator outputs as constants.

## Stride-w convention (FiLM-conditioned heads)

`rsi_strided` and `cci_strided` evaluate their indicator at every bar
using a stride-w sub-sampled history. With `w=1` they reduce exactly
to the canonical daily indicator. With `w=5` you get
weekly-cadence-evaluated-at-every-bar (aligns with discretely-resampled
RSI/CCI on the resampling boundaries; smoothly interpolates
off-boundary). This gives dense per-bar supervision for FiLM heads
that condition over the (n, w) cross-product, without the
discontinuities you'd get from naive resampling.

Both functions are 1-D only (per-ticker call) since the multi-head
trainer pools across tickers as separate rows in its training matrix
rather than as a stacked `(T, N)` axis.

## Tests

```bash
uv run pytest packages/indicators/
```

22 tests, ~0.75s. Covers SMA/EMA/rolling_std vs pandas, RSI matrix vs
hand-coded reference, RSI/CCI strided ⇔ matrix equivalence at w=1,
CCI warmup + range, MACD / Bollinger identities, divergence-zero on
matched distributions, Corwin-Schultz on a hand-checkable input.
