# regime

A pair of CWT-based long-only equity ranking strategies — search
hyperparameters offline, trade live via Alpaca.

Two strategies share the same training infrastructure and live
trading path; the only differences are the score function and the
ranking direction. Pick which one to train via `--strategy`.

  * **regime** (default) — buys stocks whose **wavelet-power
    distribution across timescales** has shifted most over the last
    few weeks. Momentum-of-volatility-shift idea. Picks highest
    divergence (descending rank).
  * **scalogram** — buys stocks whose **direction − momentum ×
    coherence** is most negative. Mean-reversion idea: recent
    weakness on incoherent timescales. Picks lowest score
    (ascending rank).

Operates on the Stooq daily archive (split/dividend-adjusted, has
volume, includes delistings) or the legacy Kaggle dump. Production
training is Optuna walk-forward search over discrete hyperparameters
using a vectorbt backtest engine; an alternative gradient-descent
trainer (JAX-Adam) for regime lives under `research/`.

> **Setup**: this app needs the workspace's nix devShell to provide
> numba/llvmlite for vectorbt. See the top-level `README.md` for
> first-time setup. After that, `uv run regime ...` works from any
> shell.

## Layout

```
apps/regime/src/regime/
  trainer.py           production trainer: Optuna + vectorbt walk-forward
  cli.py               argparse subcommands `train` and `live`
  __main__.py          `python -m regime` entry
  inference.py         pure forward pass: Checkpoint + prices -> target weights
  persist.py           JSON checkpoint round-trip (currently JAX-Adam format only)
  reporting.py         TrainResult -> ss_plotting wrappers
  broker.py            re-export shim — canonical AlpacaBroker now lives
                       at `ss_portfolio.broker` (shared with apps/relational)
  live.py              orchestration with risk rails (kill-switch / freshness
                       / per-name cap / dry-run default). Bar fetch covers
                       KERNEL_HALF_EXTENT*max(scales) + lookback so the
                       latest CWT has full kernel support.
  research/
    optimize_adam.py    JAX-Adam gradient-descent alternative trainer
    optimize_regime.py  legacy reference: Optuna + bt-library walk-forward
    backtest_bt.py      bt-library multi-strategy comparison (rsi/scalogram/regime/equal)
    backtest_ranking.py plain-numpy long/short walk-forward
```

**Productionized from `research/`:** the `weights_regime` /
`weights_scalogram` / `weights_rsi` builders were first written in
`research/backtest_bt.py` and the Optuna walk-forward harness was
prototyped in `research/optimize_regime.py`; both were promoted into
`trainer.py` (Optuna + vectorbt) as the canonical production path.
The original `research/` files are kept as bt-library reference
implementations for cross-checking — `optimize_regime.py` still
imports `make_commission_fn` / `select_top_n_matrix` from
`backtest_bt.py`, so the two are a unit.

Math primitives live in workspace packages (`ss_loaders`,
`ss_indicators`, `ss_wavelets`, `ss_portfolio`, `ss_plotting`) so they
can be reused outside this app.

## Method

For each ticker, the system computes a **scale fingerprint** — what
fraction of recent price activity sits at each timescale — and ranks
tickers by how much that fingerprint has *shifted* relative to the
preceding history. The biggest shifts are bought.

### Pipeline

```
prices                                      # (n_dates, n_tickers)
   ↓ ss_wavelets.causal_cwt
coefficients[scale, t, ticker]              # 13 scales: 3..126 days
   ↓ |·|²
power[scale, t, ticker]
   ↓ ss_wavelets.precompute_windows
recent[scale, t, ticker]      = mean over the last n_tail days
historical[scale, t, ticker]  = mean over the prior (lookback - n_tail) days
   ↓ Σ-normalize over scale axis
rd[scale, ...]                              # discrete distribution over scales
hd[scale, ...]                              # same, but historical
   ↓ ss_indicators.{kl,js,cosine,l2}_divergence
score[t, ticker]                            # bigger = bigger regime shift
   ↓ ss_portfolio.select_top_n_matrix
weights[t, ticker]                          # 1/top_n on the top names, 0 elsewhere
```

### Intuition: scale fingerprints

Most of the time a stock's distribution of variance across timescales
is roughly stable — say, 15% short / 30% mid / 55% long. A "regime
change" is when that distribution shifts (e.g., short doubles to 45%
while long falls to 35%). The chosen divergence quantifies "how
different is this distribution from that one":

| | Stock A (no change) | Stock B (regime shift) |
|---|---|---|
| Historical | 5d=15%, 21d=30%, 90d=55% | 5d=15%, 21d=30%, 90d=55% |
| Recent     | 5d=14%, 21d=32%, 90d=54% | 5d=45%, 21d=20%, 90d=35% |
| KL         | ≈ 0 | ~0.4 |

The strategy holds Stock B. Direction (up vs down) doesn't matter to
the score — it's a momentum-of-volatility-shift idea.

### What the search optimizes

Optuna searches a 7-dimensional discrete space per walk-forward window:

| Hyperparameter | Range / values | Role |
|---|---|---|
| `lookback` | int [40, 252] | Total CWT z-norm + historical-window length, in days |
| `n_tail` | int [3, lookback//2] | Length of the *recent* window |
| `top_n` | int [5, 30] | Number of names held each rebalance |
| `divergence` | `{kl, js, cosine, l2}` | Which distance metric to use |
| `use_short_scales` | bool | Include scales [3, 5, 7]? |
| `use_mid_scales` | bool | Include scales [10, 12, 15, 21, 26]? |
| `use_long_scales` | bool | Include scales [42, 50, 63, 90, 126]? |

Per-scale weights are *equal* within the chosen subset — Optuna picks
which scales to include, not how to weight them. (The JAX-Adam
trainer in `research/optimize_adam.py` does the opposite: fixed
hyperparameters, learned per-scale weights.)

The walk-forward driver rolls a 5y train / 3y val window forward by
2y at a time (configurable), reporting per-window best params and
their out-of-sample Sharpe.

### Why search (Optuna), not optimize (Adam)?

The production trainer is a Bayesian search, not gradient descent.
Both are wired in (Adam lives at `research/optimize_adam.py`); search
won the bake-off because of the structure of *this* problem, not
because gradients are bad in general.

1. **The decisions are discrete.** Choice of divergence (kl/js/cosine/l2),
   choice of scale subset, choice of `top_n` — these are
   non-differentiable. Adam can only optimize continuous knobs, so it
   has to settle for a softmax over scales + a temperature-softened
   top-N, which is a *different and weaker* strategy than the hard
   selections Optuna tries.

2. **Sharpe through real costs is non-differentiable.** Per-side
   commissions, per-name spread costs, equal-weight allocation, and
   integer rebalances all introduce kinks in the objective. Adam needs
   a smooth surrogate (`block_sharpe_with_costs` in `ss_portfolio`)
   which approximates daily-return Sharpe but doesn't equal it.

3. **Returns are noisy; overfitting risk &gt; gradient efficiency.**
   Search with walk-forward windows naturally validates each candidate
   on held-out periods. Adam optimizes a single train/val split — one
   chance to overfit, one chance to validate.

4. **Empirical bake-off (post strict-causality fix)**:

   | Trainer | Best val Sharpe | Notes |
   |---|---|---|
   | Optuna + vectorbt (`regime.trainer`) | **+0.46** | hard top-N, cosine, mid scales |
   | Optuna + bt (`research/optimize_regime`) | +0.46 | reference; same math, slower engine |
   | JAX-Adam (`research/optimize_adam`) | +0.16 | matched window, KL only |
   | JAX-Adam, full data | -0.33 | overfits to single train slice |

   Adam can't reach the search result because soft-top-N over
   ~1000 names spreads weight across dozens of names even at low
   temperature, while Optuna's hard top-N=5 puts 20% on each of 5
   names. That concentration is where the alpha lives in this
   strategy.

When would Adam win? If we ever stack a **learned-feature backbone**
on top of the regime score — e.g., a CNN on the CWT scalogram
(`apps/notebook/notebooks/cwt_vision_multihead.ipynb`) feeding into
the same rank-and-hold pipeline. Search can't enumerate over tens of
thousands of neural-net weights; gradients are mandatory there. Adam
is research scaffolding for *that* future, not an alternative to the
current production path.

### Liquidity handling

There's no upstream liquidity filter at training time. Instead,
Corwin-Schultz spread is folded into the per-side fee in
`ss_portfolio.vbt_backtest`:

```
fee[t, ticker] = commission_bps / 10000 + spread[t, ticker] / 2
```

This is the canonical "cross-half-spread" assumption — each rebalance
trade pays half the relative bid-ask spread of *that* name on *that*
date. The optimizer doesn't have a direct gradient on liquidity (the
optuna search ranks names purely by divergence), but it has a strong
*indirect* signal: any hyperparameter combination that ends up holding
wide-spread names eats the cost on every rebalance and produces a
worse Sharpe, so TPE drifts away from those configs.

Live trading still applies a binary `max_spread` gate at inference
(default 2%). The reason it's asymmetric: the live strategy has no
Sharpe-optimization step that can absorb a bad pick — it just submits
the top-N names ranked by score. The gate exists as a safety floor
against unconscionable trades, not as a strategy filter.

### Why "regime"

There's no Hidden Markov Model or Bayesian state inference here — the
name comes from the *idea* of detecting a state change in the
distribution of price energy across timescales. Mechanically it's
divergence-between-two-distributions, computed every rebalance bar,
ranked, top-N held.

## Usage

### Training (Optuna + vectorbt)

```
# Default: regime strategy on Stooq archive (split/div-adjusted, has volume).
uv run regime train --data-dir ./StooqData --n-trials 50

# Train the scalogram strategy instead:
uv run regime train --data-dir ./StooqData --strategy scalogram --n-trials 50

# All knobs explicit:
uv run regime train --strategy regime --source stooq --data-dir ./StooqData \
    --n-trials 100 --metric sharpe --seed 42 --jobs 4 \
    --train-years 5 --val-years 3 --step-years 3 \
    --start 2010-01-01 --end 2025-12-31

# Legacy Kaggle layout:
uv run regime train --source kaggle --data-dir ./Nasdaq3347 --n-trials 50
```

The CLI prints a per-window summary plus aggregate stats like:

```
Window                     Train       Val     Div   LB   NT  TopN   Scales
2013-2021                +0.8835   +0.5122      js   55    9    23       ML
2015-2023                +1.3250   -0.6641      js  153   63     6      def
2017-2025                +1.1167   -0.0205      kl  156   52     8      SML

Val sharpe stats across 3 windows:
  best   = +0.5122  (window 2013-2021)
  median = -0.0205
  mean   = -0.0575
  worst  = -0.6641
```

The "best" line is max-of-N and is upward-biased; the median is the
more honest single-number summary. `--save-params` serializes the
best window's hyperparameters into a checkpoint for `regime live`.

### Going live (paper trading)

```
export ALPACA_API_KEY=...        # paper-trading keys from app.alpaca.markets
export ALPACA_SECRET_KEY=...
# ALPACA_BASE_URL defaults to paper; set to https://api.alpaca.markets for real money

uv run regime live --params Output/regime-v1.json --dry-run
uv run regime live --params Output/regime-v1.json --live
```

Cron entry that rebalances every weekday at 09:35 ET:

```
35 9 * * 1-5  cd /path/to/StockSurvey && \
    uv run regime live --params Output/regime-v1.json --live \
    >> Output/regime-live.log 2>&1
```

### Alternative trainers (research)

```
# JAX-Adam: gradient descent over per-scale weights and temperature.
uv run python -c "from regime.research.optimize_adam import train; ..."

# Original Optuna + bt-library reference (slower; kept for comparison).
uv run python -m regime.research.optimize_regime --data-dir ./Nasdaq3347
```

## Honest evaluation

The recent search peaks at **val Sharpe +0.46** (cosine divergence,
lookback=116, n_tail=16, top_n=5, mid scales) on window 2015-2023,
with other windows landing +0.21 / +0.34. The signal is real but
weak — and the previous CLAUDE.md headline numbers (+0.80 to +1.78
val Sharpe) were inflated by a CWT slicing bug that let the wavelet
peek ~`4·scale` days into the future. Strict causality cuts the
reported Sharpe by roughly 3–4×.

The search picks **different** divergence + lookback per window —
the underlying signal is non-stationary across regimes, so any single
"best" config is a snapshot of the most-recent fit window.

### Backtest realism

The pipeline has been progressively de-illusioned. What changed and
why, in priority order:

| Tier | Issue | Status | Where |
|---|---|---|---|
| 1 | CWT peeked `4·scale` days into the future via `full[-n:]` slice | **fixed** | `ss_wavelets.causal_cwt` uses `full[:n]` |
| 1 | Splits / dividends not adjusted (Kaggle had no `adj_close`) | **fixed** | switched to Stooq archive (`load_stooq_matrix`); split/div-adjusted |
| 1 | Spread filtered upstream but not charged as a cost | **fixed** | per-(date, ticker) fees in `ss_portfolio.vbt_backtest` |
| 1 | Fill at the same close that produced the signal | **fixed** | `vbt_backtest(fill_lag=1)` shifts orders to the next close |
| 1 | Unlimited `ffill()` papered over multi-year delistings as fake calm | **fixed** | Stooq loader uses `ffill(limit=5)`; longer gaps stay NaN |
| 1 | Best-of-N walk-forward windows is upward-biased | **mitigated** | `print_summary` reports median + mean + worst alongside best |
| 2 | Optuna seed not pinned → ±0.1-0.3 Sharpe noise per re-run | **fixed** | `train(seed=42)`, exposed as `--seed` |
| 2 | Walk-forward windows overlapped (val[N] leaked into train[N+1]) | **fixed** | default `step_years=val_years=3` |
| 2 | CWT operates on raw close prices, not log-returns | **known, unfixed** | `causal_cwt` math change, not yet validated |
| 2 | `commission_bps=10` is an opinion, not a measurement | **known, unfixed** | venue-dependent; document and move on |
| 3 | Equal-weight within top-N basket | **known, by design** | `select_top_n_matrix` puts 1/top_n on each pick |
| 3 | Annualized Sharpe assumes daily-iid returns | **known, unfixed** | Lo (2002) autocorr adjustment not applied |

Survivorship bias is the elephant we *partially* addressed: the Stooq
archive includes delisted tickers, so the universe no longer
mechanically conditions on "you only ever held survivors." It's not
a clean point-in-time membership reconstruction (we don't have S&P
500 / Nasdaq 100 join/leave dates), but it's substantially closer to
honest than the prior Kaggle-survivors-only setup.

## Live-trading risk rails

`regime live` enforces four checks before submitting any orders. Each
aborts the run with a clear reason rather than silently coercing values:

1. **Kill-switch file** — if `~/.regime-killswitch` (or
   `--killswitch PATH`) exists, the run aborts. Lets an operator halt
   trading without touching the cron entry.
2. **Data freshness** — if the latest bar from Alpaca is older than
   `--max-data-age-days` (default 3), abort. Prevents trading on a
   frozen feed.
3. **Per-name cap** — `--max-position` (default 0.25) clips and
   redistributes target weights so no single name exceeds the cap
   (water-fill via `ss_portfolio.apply_position_cap`). Cap distributes
   among nonzero-weight names only — illiquid names already zeroed out
   by an upstream gate cannot be re-introduced.
4. **Dry-run by default** — `--dry-run` is the default; `--live` is
   opt-in. A misconfigured cron entry never accidentally trades.

Two implicit prerequisites:
- **Wavelet support** — bar fetch covers
  `KERNEL_HALF_EXTENT*max(scales) + lookback + bar_buffer_days` trading
  bars so the latest-bar CWT runs on full kernel support, not zero-
  padded history.
- **Order-rejection observability** — `submit_orders` returns
  `(order_ids, rejections)`; per-symbol failures (non-fractionable,
  sub-cent notional, etc.) are logged + captured into
  `LiveRunResult.rejected_orders` rather than silently dropped.

The `AlpacaBroker` itself now lives at `ss_portfolio.broker` (shared
with `apps/relational`); `regime/broker.py` is a re-export shim so
existing imports keep working. `alpaca-py` is gated behind the
`ss-portfolio[alpaca]` optional extra.

Always paper-trade for a full rebalance cycle before pointing
`ALPACA_BASE_URL` at the live endpoint.

## Checkpoint format

Two checkpoint *modes* share the same JSON schema, distinguished by
the `mode` field:

  * **`adam`** — JAX-Adam output. `scale_log_weights` (13 floats,
    softmaxed at inference) and `log_temperature` (1 float, controls
    soft-top-N sharpness) carry the model. Produced by
    `regime.research.optimize_adam.train()` via `save_checkpoint()`.
  * **`optuna`** — Optuna+vectorbt output. `top_n` (int), `divergence`
    (`kl|js|cosine|l2`), and a resolved `scales` subset carry the
    model. Produced by `regime.trainer.train()` via
    `save_checkpoint_from_window()` — `regime train --save-params`
    serializes the highest-val-Sharpe window.

Old checkpoints written before the `mode` field existed default to
`adam` on load — fully backwards-compatible.

`regime.inference.target_weights` dispatches on `cp.mode`:
soft-top-N via temperature softmax for `adam`, hard-top-N
equal-weight basket for `optuna`. `regime live` consumes either
without caring which trainer produced it.

Common fields (both modes):

  * scale grid (`scales`)
  * strategy hyperparameters (`lookback`, `n_tail`, `rebal_days`,
    `max_spread`, `commission_bps`)
  * training-time universe (`universe`)
  * provenance (`trained_at`, `train_start/end`, `val_start/end`,
    `train_sharpe`, `val_sharpe`)

It's plain JSON: human-readable, diffable, and safe to load without
arbitrary-code-execution risk. Forward-compatible — `load_checkpoint`
ignores unknown keys, so future schema additions don't break old
readers.
