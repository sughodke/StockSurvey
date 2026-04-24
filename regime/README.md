# regime

Differentiable CWT-regime equity portfolio strategy.

A standalone implementation of a regime-divergence long-only strategy
trained end-to-end via JAX autograd. Takes per-ticker OHLC CSVs (Kaggle
`svaningelgem/nasdaq-daily-stock-prices` schema), computes a causal
continuous wavelet transform across 13 scales, and learns a
14-parameter soft-top-N strategy that maximizes annualized Sharpe net
of transaction costs.

## Layout

    regime/
      data.py        load CSVs; Corwin-Schultz spread (no-volume liquidity proxy)
      cwt.py         causal Ricker CWT + windowed power means
      strategy.py    JAX regime score + portfolio Sharpe with costs
      trainer.py     Adam loop, train/val split, TrainResult dataclass
      reporting.py   scale-weight printout + training plot
      cli.py         argparse main
      __main__.py    `python -m regime` entry

## Usage

    python -m regime --data-dir ./Nasdaq3347
    python -m regime --data-dir ./Nasdaq3347 --lookback 229 --n-tail 106 --save
    python -m regime --data-dir ./Nasdaq3347 --rebal-days 5 --n-steps 1000

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
