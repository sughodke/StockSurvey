# regime

Differentiable CWT-regime equity portfolio strategy — train offline,
trade live via Alpaca.

A standalone implementation of a regime-divergence long-only strategy
trained end-to-end via JAX autograd. Takes per-ticker OHLC CSVs (Kaggle
`svaningelgem/nasdaq-daily-stock-prices` schema), computes a causal
continuous wavelet transform across 13 scales, and learns a
14-parameter soft-top-N strategy that maximizes annualized Sharpe net
of transaction costs. The trained checkpoint is portable JSON and is
consumed by a live runner that fetches recent OHLC from Alpaca,
computes target weights, and submits market orders.

## Layout

    regime/
      data.py        load CSVs; Corwin-Schultz spread (no-volume liquidity proxy)
      cwt.py         causal Ricker CWT + windowed power means
      strategy.py    JAX regime score + portfolio Sharpe with costs
      trainer.py     Adam loop, train/val split, TrainResult dataclass
      reporting.py   scale-weight printout + training plot
      persist.py     JSON checkpoint round-trip
      inference.py   pure forward pass: prices -> target weights
      broker.py      Alpaca adapter (account, positions, bars, orders)
      live.py        orchestration with risk rails
      cli.py         argparse subcommands: `train` and `live`
      __main__.py    `python -m regime` entry

## Usage

### Training

    python -m regime train --data-dir ./Nasdaq3347
    python -m regime train --data-dir ./Nasdaq3347 --lookback 229 --n-tail 106 \
        --n-steps 500 --save-params Output/regime-v1.json --save

### Going live (paper trading)

    export ALPACA_API_KEY=...        # paper-trading keys from app.alpaca.markets
    export ALPACA_SECRET_KEY=...
    # ALPACA_BASE_URL defaults to paper; set to https://api.alpaca.markets for real money

    # Dry-run: print what would be traded.
    python -m regime live --params Output/regime-v1.json --dry-run

    # Submit orders.
    python -m regime live --params Output/regime-v1.json --live

A typical cron entry that rebalances every weekday at 09:35 ET:

    35 9 * * 1-5  cd /path/to/StockSurvey && \
        uv run python -m regime live --params Output/regime-v1.json --live \
        >> Output/regime-live.log 2>&1

## Method

For each rebalance date, build two distributions over CWT scales for
each ticker:

  * **recent** — mean wavelet power over the last `n_tail` days
  * **historical** — mean wavelet power over the prior `lookback - n_tail` days

The per-ticker score is the symmetric KL divergence between these two
distributions, weighted by a learned softmax over scales. Tickers are
ranked via temperature-scaled softmax of (score / liquidity-mask), held
for `rebal_days`, and re-ranked. P&L is reported net of `commission_bps`
of one-sided turnover at each rebalance.

## Learned parameters

  * `scale_log_weights` (13) — pre-softmax weights over CWT scales
  * `log_temperature`   (1)  — softmax concentration; ~0 ≈ argmax (hard top-1)

Empirically the optimizer concentrates weight on the 26-126d band and
drops temperature near zero, i.e. it converges to a near-discrete
top-1-by-regime-shift policy on monthly-to-biannual horizons.

## Honest evaluation

The first `--train-frac` of rebalance blocks (default 70%) is used to
optimize; the remainder is held-out validation. Both Sharpes are
reported at every log step so overfitting is visible during training.

## Live-trading risk rails

`regime live` enforces four checks before submitting any orders. Each
aborts the run with a clear reason rather than silently coercing values:

  1. **Kill-switch file** — if `~/.regime-killswitch` (or `--killswitch PATH`)
     exists, the run aborts. Lets an operator halt trading without
     touching the cron entry.
  2. **Data freshness** — if the latest bar from Alpaca is older than
     `--max-data-age-days` (default 3), abort. Prevents trading on a
     frozen feed.
  3. **Per-name cap** — `--max-position` (default 0.25) clips and
     renormalizes target weights so no single name exceeds the cap.
  4. **Dry-run by default** — `--dry-run` is the default; `--live` is
     opt-in. A misconfigured cron entry never accidentally trades.

Always paper-trade for a full rebalance cycle before pointing
`ALPACA_BASE_URL` at the live endpoint.

## Checkpoint format

The JSON written by `--save-params` captures:

  * learned params (`scale_log_weights`, `log_temperature`)
  * the scale grid (`scales`)
  * strategy hyperparameters (`lookback`, `n_tail`, `rebal_days`,
    `max_spread`, `commission_bps`)
  * the training-time universe (`universe`) — fetched from Alpaca at
    inference time
  * provenance (`trained_at`, `train_start/end`, `val_start/end`,
    `train_sharpe`, `val_sharpe`)

It is plain JSON: human-readable, diffable, and safe to load without
arbitrary-code-execution risk.
