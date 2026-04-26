"""regime.trainer: Optuna + vectorbt walk-forward search.

Production trainer for the regime strategy. Combines:
  * Optuna TPE search over discrete hyperparameters (lookback, n_tail,
    top_n, divergence, scale subsets), seeded for reproducibility.
  * Hard top-N equal-weight allocation
    (`ss_portfolio.select_top_n_matrix`).
  * vectorbt backtest engine (`ss_portfolio.vbt_backtest`) — much faster
    than bt-library, JIT-compiled by numba.
  * Walk-forward train/val rolling windows with non-overlapping val
    periods (step_years defaults to val_years).
  * Fill-at-next-bar execution (`fill_lag=1` in vbt_backtest) so
    signals computed at close[t] trade at close[t+1].
  * Per-(date, ticker) spread cost via the Corwin-Schultz panel —
    no upstream binary liquidity filter.

This is the spiritual successor to `regime.research.optimize_regime`
(Optuna + bt-library) — same approach, faster engine, plus we
vectorize the per-date score computation through `precompute_windows`
+ JAX divergences instead of looping in Python.

For the gradient-descent alternative on continuous params, see
`regime.research.optimize_adam`. For the bt-library reference
implementation, see `regime.research.optimize_regime`.

Requires the nix devShell (provides numba/llvmlite for vectorbt). See
README.md.

Known limitations not yet fixed
-------------------------------
The pipeline below still embeds a few honest assumptions that inflate
the reported Sharpe vs what real trading would deliver. Documented
here so future readers don't mistake them for invariants:

  * **Equal-weight within the top-N basket** (`select_top_n_matrix`
    puts 1/top_n on each pick). Score-weighted allocation would be a
    different strategy with potentially different Sharpe. Not tested.
  * **Annualized Sharpe assumes daily-iid returns.** Universal stat
    issue. Real returns have autocorrelation and vol clustering, both
    of which inflate the naive Sharpe. The Lo (2002) autocorrelation
    adjustment isn't applied.
  * **`commission_bps=10` (default) is an opinion, not measurement.**
    Retail Alpaca is $0; institutional is 1-3 bps; market impact
    varies. Combined with the spread cost, we may be double-counting
    or undercounting depending on the venue.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd

from ss_indicators import corwin_schultz_spread, get_divergence
from ss_loaders import load_price_matrix
from ss_portfolio import (
    apply_nan_mask,
    select_top_n_matrix,
    vbt_backtest,
)
from ss_wavelets import KERNEL_HALF_EXTENT, causal_cwt, precompute_windows


# Scale-subset groups Optuna picks among via 3 boolean flags.
SHORT_SCALES = [3, 5, 7]
MID_SCALES = [10, 12, 15, 21, 26]
LONG_SCALES = [42, 50, 63, 90, 126]
DEFAULT_SCALES = [5, 21, 90]  # fallback when all three subsets are off

# Optuna search range for the rolling z-norm + recent/historical-split
# window. Single source of truth — also used to derive the per-window
# survivorship floor below.
LOOKBACK_SEARCH_MIN = 40
LOOKBACK_SEARCH_MAX = 252

# Worst-case per-window history needed for ALL scored bars to land on
# fully-supported wavelets:
#
#   * causal_cwt's rolling z-norm needs `lookback` prior bars
#   * the largest Ricker kernel needs `KERNEL_HALF_EXTENT * max_scale`
#     prior bars (=3 * max_scale post the |t|=3 truncation; was 4 *
#     max_scale before)
#
# Combined: KERNEL_HALF_EXTENT * max_scale + lookback. Since both
# lookback and the scale subset are Optuna-searched, we floor on the
# worst case of each: longest possible lookback (LOOKBACK_SEARCH_MAX)
# and largest possible scale (max(LONG_SCALES)). At trainer defaults
# this is 3 * 126 + 252 = 630 bars (~2.5 trading years), down from
# 756 when KERNEL_HALF_EXTENT was 4.
DEFAULT_PER_WINDOW_MIN_HISTORY = (
    KERNEL_HALF_EXTENT * max(LONG_SCALES) + LOOKBACK_SEARCH_MAX
)


@dataclass
class WindowResult:
    """Best params + scores for one walk-forward window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    best_params: dict
    train_score: float
    val_score: float
    strategy: str = 'regime'  # which weight builder produced these params
    use_log_returns: bool = False  # CWT input mode at train time


@dataclass
class TrainResult:
    """Output of `train()` — one entry per walk-forward window."""
    windows: list[WindowResult]
    metric: str
    rebalance_days: int
    commission_bps: float
    strategy: str = 'regime'  # which weight builder was searched
    use_log_returns: bool = False  # CWT input mode used by this run

    @property
    def best_window(self) -> WindowResult:
        """Window with the highest validation score."""
        return max(self.windows, key=lambda w: w.val_score)


def _resolve_scales(params: dict) -> list[int]:
    """Build the scale list from Optuna's three boolean subset flags."""
    scales: list[int] = []
    if params.get('use_short_scales', False):
        scales += SHORT_SCALES
    if params.get('use_mid_scales', False):
        scales += MID_SCALES
    if params.get('use_long_scales', False):
        scales += LONG_SCALES
    return scales or DEFAULT_SCALES


def _log_returns(prices_arr: np.ndarray) -> np.ndarray:
    """`(T, N) -> (T, N)` log-returns; first row padded with 0.

    Available behind the `use_log_returns` flag (default OFF on this
    strategy — see comment block below). Daily log returns are roughly
    stationary (zero-mean, vol-clustered but not trended), so feeding
    them to `causal_cwt` instead of raw close prices means the rolling
    z-norm is doing vol normalization rather than trend removal — the
    long-scale wavelet power is no longer dominated by persistent
    multi-year drift in the price level.

    NaN propagation: a NaN at price index k makes log-returns NaN at
    indices k and k+1 (both `np.log(NaN/x)` and `np.log(x/NaN)` resolve
    to NaN). `apply_nan_mask` downstream still keys off the original
    `prices.values`, so masking still happens at the right cells; the
    transform doesn't introduce NEW NaN regions, only widens existing
    ones by one bar.
    """
    out = np.zeros_like(prices_arr, dtype=np.float64)
    out[1:] = np.log(prices_arr[1:] / prices_arr[:-1])
    return out


# Why `use_log_returns` defaults to False
# ---------------------------------------
# Mathematically, log-returns is the "right" CWT input — it's stationary,
# the rolling z-norm becomes a clean vol normalizer, and the wavelet
# power is a pure regime-change signal. Empirically, for THIS strategy
# (cross-sectional equity ranking via top-N), it hurt out-of-sample
# performance.
#
# Controlled walk-forward eval, Stooq 2010-2024, 20 trials/window,
# kernel half-extent 3 fixed, only the CWT input changed:
#
#                       log-returns      raw close
#   best   val sharpe    +0.16           +0.46
#   median val sharpe    +0.03           +0.15
#   mean   val sharpe    -0.29           +0.07
#   worst  val sharpe    -1.06           -0.41
#
# Per window: log-returns had HIGHER train sharpe but LOWER val sharpe
# in every window — classic overfitting signature. Raw close artifacts
# in `Output/regime-eval-rawclose-kernel3.{log,json}`; log-returns
# artifacts in `Output/regime-eval-logreturns.{log,json}`.
#
# Theory: the raw-close CWT bleeds price-level trend into long-scale
# wavelet power, embedding an implicit cross-sectional momentum factor
# (Jegadeesh-Titman). Log-returns purifies trend out, leaving only
# "vol regime shift" — direction-agnostic and not a known cross-
# sectional return predictor. We were getting momentum for free; log-
# returns took it away.
#
# The flag is preserved (not reverted) so future research can re-enable
# it for non-ranking objectives — vol forecasting, regime-break
# detection, etc. — where stationarity actually matters.


def _filter_window_universe(panel: pd.DataFrame, *, min_bars: int) -> pd.Index:
    """Per-walk-forward survivorship filter: keep tickers that are
    eligible to trade for *this specific* window slice.

    A ticker passes if both:
      1. It has a valid value at the window's first bar — required by
         the causal CWT, whose cumsum-based rolling z-norm propagates
         NaN forward forever once it hits one. A ticker that IPO'd
         after the window started can't be scored in this window.
      2. It has at least `min_bars` valid observations within the
         window — enough history for the wavelet to produce a
         meaningful score. A ticker that delists 30 days into a 5-year
         window doesn't have enough data to participate.

    Returns the surviving column index. The caller slices `panel` and
    any aligned auxiliary frames (spread, etc.) by this index.

    This is the survivorship-bias fix: instead of a panel-wide rule
    that requires every ticker to exist for the entire date range, each
    walk-forward window defines its own eligible universe based on who
    was trading during that window.
    """
    has_valid_start = panel.iloc[0].notna()
    valid_count = panel.notna().sum(axis=0)
    return panel.columns[has_valid_start & (valid_count >= min_bars)]


def weights_regime(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """Hard-top-N basket ranked by CWT-power-distribution divergence.

    The score per (date, ticker) is the chosen divergence between the
    recent vs historical CWT-power distributions across `scales`. We
    compute scores for all valid dates in one vectorized pass via
    `precompute_windows` + JAX divergence (much faster than the Python
    date-loop in `regime.research.optimize_regime`).

    `scale_log_weights = zeros` makes the divergence's internal softmax
    uniform — Optuna chooses *which* scales to include, but each
    included scale contributes equally (matching the legacy behavior).

    Picks **highest-divergence** names (`ascending=False`): biggest
    regime shift wins. Direction (price up vs down) doesn't enter — it's
    a momentum-of-volatility-shift idea.

    Liquidity is not filtered here; it enters the objective via per-
    (date, ticker) fees in `vbt_backtest`. Wide-spread names get
    ranked normally and then naturally tank the realized Sharpe of any
    config that picks them.

    CWT input is **raw close** by default. `use_log_returns=True` runs
    on log returns — preserved as a flag for future research, but
    empirically worse on this objective. See the comment block above
    `_log_returns` for the eval evidence and theory.
    """
    cwt_input = _log_returns(prices.values) if use_log_returns else prices.values
    coeffs = causal_cwt(cwt_input, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)

    div_fn = get_divergence(divergence)
    scale_log_weights = jnp.zeros(len(scales), dtype=jnp.float32)
    # `np.array(jnp_array)` (not `asarray`) forces a writable host copy
    # so `apply_nan_mask` can NaN cells in place.
    scores = np.array(div_fn(
        jnp.asarray(recent), jnp.asarray(historical), scale_log_weights))

    scores = apply_nan_mask(scores, prices.values, lookback)

    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


def weights_scalogram(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """Hard-top-N basket ranked by direction − momentum × coherence.

    A second CWT-based ranking idea, distinct from regime divergence.
    For each (date, ticker), the score is:

        score = direction − momentum × coherence

    where:
      direction = trailing-`n_tail` mean of the **shortest-scale signed**
        wavelet coefficient (sign matters here, unlike regime which
        squares the coefficient). Negative = recent price weakness.
      momentum = trailing-`n_tail` mean of |coeffs|² averaged across
        all scales — total recent volatility magnitude.
      coherence = Pearson correlation between shortest-scale power and
        longest-scale power over the trailing `n_tail` window, clipped
        to [0, 1]. High coherence = timeframes agree (confirmed move).
        Low coherence = timeframes disagree (transition / noise).

    Picks **lowest-score** names (`ascending=True`): negative direction
    on incoherent timescales = potential mean-reversion candidates.
    A counter-trend bet, not a momentum bet.

    Vectorized via cumulative-sum trailing means; same speed
    characteristics as `weights_regime`.

    CWT input is **raw close** by default; `use_log_returns=True`
    is preserved for symmetry with `weights_regime` but has the same
    out-of-sample drawback documented in the comment above
    `_log_returns`.
    """
    cwt_input = _log_returns(prices.values) if use_log_returns else prices.values
    coeffs = causal_cwt(cwt_input, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    n_scales, n_dates, n_tickers = power.shape

    def _trail_mean(arr: np.ndarray) -> np.ndarray:
        """Trailing-`n_tail` mean along axis=1 for valid dates only.
        Input shape (..., n_dates, ...), output (..., n_valid, ...) where
        `n_valid = n_dates - lookback`."""
        cs = np.cumsum(arr.astype(np.float64), axis=1)
        zero = np.zeros_like(cs[:, :1])
        cs = np.concatenate([zero, cs], axis=1)
        end = cs[:, lookback + 1: n_dates + 1]
        start = cs[:, lookback - n_tail + 1: n_dates - n_tail + 1]
        return ((end - start) / n_tail).astype(np.float32)

    momentum = _trail_mean(power).mean(axis=0)
    direction = _trail_mean(coeffs[:1])[0]

    short = power[:1]
    long = power[-1:]
    e_s = _trail_mean(short)[0]
    e_l = _trail_mean(long)[0]
    e_ss = _trail_mean(short ** 2)[0]
    e_ll = _trail_mean(long ** 2)[0]
    e_sl = _trail_mean(short * long)[0]
    cov = e_sl - e_s * e_l
    var_s = np.maximum(e_ss - e_s ** 2, 1e-12)
    var_l = np.maximum(e_ll - e_l ** 2, 1e-12)
    coherence = np.clip(cov / (np.sqrt(var_s * var_l) + 1e-9), 0.0, 1.0)

    scores = (direction - momentum * coherence).astype(np.float32)
    scores = apply_nan_mask(scores, prices.values, lookback)

    weights = select_top_n_matrix(scores, top_n, ascending=True)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


# Strategy dispatch table. Each entry is the weight builder for a
# strategy name; the trainer + inference both look it up here.
STRATEGIES = ('regime', 'scalogram')


def _build_weights(
    strategy: str, prices: pd.DataFrame, params: dict, *,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """Dispatch (strategy, params) → weight matrix. Single source of truth
    for what each strategy's hyperparameter dict means; used by both the
    Optuna objective (train slice) and the val-window evaluation.

    `use_log_returns` is a run-time flag (not an Optuna-searched
    hyperparameter) that toggles the CWT input. See the comment block
    above `_log_returns` for why this defaults to False.
    """
    scales = _resolve_scales(params)
    common = dict(
        lookback=int(params['lookback']), n_tail=int(params['n_tail']),
        top_n=int(params['top_n']), scales=scales,
        use_log_returns=use_log_returns)
    if strategy == 'regime':
        return weights_regime(prices, **common, divergence=str(params['divergence']))
    if strategy == 'scalogram':
        return weights_scalogram(prices, **common)
    raise ValueError(f'unknown strategy {strategy!r}; available: {STRATEGIES}')


def _make_objective(
    prices, *, strategy, rebalance_days, metric, commission_bps, spread_df,
    use_log_returns: bool = False,
):
    """Build an Optuna objective closed over the train slice and strategy.

    Hyperparameters shared by all strategies: `lookback`, `n_tail`,
    `top_n`, and the three scale-subset booleans. Regime adds
    `divergence` (kl/js/cosine/l2) — scalogram has no divergence knob,
    its score is fixed.
    """
    def objective(trial: optuna.Trial) -> float:
        lookback = trial.suggest_int(
            'lookback', LOOKBACK_SEARCH_MIN, LOOKBACK_SEARCH_MAX)
        n_tail = trial.suggest_int('n_tail', 3, lookback // 2)
        top_n = trial.suggest_int('top_n', 5, 30)
        use_short = trial.suggest_categorical('use_short_scales', [True, False])
        use_mid = trial.suggest_categorical('use_mid_scales', [True, False])
        use_long = trial.suggest_categorical('use_long_scales', [True, False])
        params: dict = dict(
            lookback=lookback, n_tail=n_tail, top_n=top_n,
            use_short_scales=use_short, use_mid_scales=use_mid,
            use_long_scales=use_long)
        if strategy == 'regime':
            params['divergence'] = trial.suggest_categorical(
                'divergence', ['kl', 'js', 'cosine', 'l2'])

        try:
            weight_df = _build_weights(
                strategy, prices, params, use_log_returns=use_log_returns)
            if weight_df.values.sum() == 0:
                return float('-inf')
            metrics = vbt_backtest(
                prices, weight_df,
                rebalance_days=rebalance_days,
                commission_bps=commission_bps,
                spread_df=spread_df)
            score = metrics[metric]
            return score if np.isfinite(score) else float('-inf')
        except Exception:
            return float('-inf')

    return objective


def train(
    prices: pd.DataFrame,
    spread_df: pd.DataFrame | None = None,
    *,
    strategy: str = 'regime',
    n_trials: int = 50,
    n_jobs: int = 1,
    rebalance_days: int = 20,
    metric: str = 'sharpe',
    commission_bps: float = 10.0,
    train_years: int = 5,
    val_years: int = 3,
    step_years: int = 3,
    seed: int = 42,
    per_window_min_history: int = DEFAULT_PER_WINDOW_MIN_HISTORY,
    use_log_returns: bool = False,
) -> TrainResult:
    """Walk-forward Optuna+vectorbt search over regime hyperparameters.

    Rolls a `train_years`/`val_years` window forward by `step_years` at
    a time. For each window: TPE-search `n_trials` hyperparameter combos
    on the train slice, then evaluate the best params out-of-sample on
    val. Returns one `WindowResult` per window.

    `metric` is the key from `vbt_backtest`'s output dict to maximize:
    `sharpe`, `cagr`, or `max_drawdown` (note max_drawdown is negative,
    so maximizing it minimizes the worst drawdown).

    `spread_df` is optional but recommended: when provided, per-side
    fees become `commission_bps/10000 + spread/2`, so wide-spread names
    automatically depress the Sharpe of any config that picks them.
    No upstream binary spread filter is applied — the cost matrix
    carries the liquidity signal end-to-end.

    `step_years` defaults to `val_years` (3) so consecutive windows do
    not overlap on the validation axis — window N's val period ends at
    window N+1's train start. With step < val_years, val[N] would leak
    into train[N+1], and any aggregate "average val Sharpe" would
    double-count those bars.

    `seed` pins Optuna's TPE sampler RNG so two runs with the same
    inputs return the same hyperparameters. Re-runs without a pinned
    seed differ by ±0.1-0.3 Sharpe per window from sampler noise alone.

    `n_jobs` runs Optuna trials in parallel via joblib threads. JAX,
    scipy FFT, and numba (vbt's hot path) all release the GIL during
    heavy work, so threads scale close to linearly with core count.
    Note that parallel trials breaks strict reproducibility — TPE sees
    completed trials in non-deterministic order — but the seed still
    keeps the search reasonably stable. Set `n_jobs=1` if you need
    bit-for-bit reproducibility across runs.

    `strategy` selects the weight builder: `'regime'` (CWT-power-
    distribution divergence, momentum-of-volatility-shift idea) or
    `'scalogram'` (direction − momentum × coherence, mean-reversion
    idea). Both share lookback / n_tail / top_n / scale-subset
    hyperparameters; regime additionally searches over divergence
    function (kl/js/cosine/l2).

    `per_window_min_history` enforces the per-walk-forward
    survivorship filter (see `_filter_window_universe`). A ticker
    must have a valid first bar AND >= this many valid bars in each
    window to be eligible for that window. Default
    `DEFAULT_PER_WINDOW_MIN_HISTORY = KERNEL_HALF_EXTENT *
    max(LONG_SCALES) + LOOKBACK_SEARCH_MAX = 630` (~2.5 trading
    years) — the worst-case history needed for the largest possible
    (lookback, scale) Optuna might pick to be fully supported. The
    floor moved from 756 → 630 when the Ricker kernel half-extent
    dropped from 4 to 3 (negligible energy past |t|=3).

    This is also the survivorship-bias fix — the loader returns a
    point-in-time panel (with leading/trailing NaN for IPO/delist
    events) and the trainer constructs a per-window eligible universe
    from it, instead of imposing a panel-wide "must exist for the
    full range" rule.

    Tradeoff to be aware of: raising the floor from 504 → 756
    excludes tickers that IPO'd within the last ~3 years of each
    window. Names like SNOW (2020), CRWD (2019), ZM (2019) post their
    debuts had outsized scalogram-detectable runs that this filter
    discards. The principled fix is per-trial scale-aware filtering
    (Optuna picks scales, then universe is filtered to whoever has
    enough bars for THAT trial's max scale + lookback) rather than
    a global per-window worst-case floor — but that's a bigger
    refactor not done here.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown strategy {strategy!r}; available: {STRATEGIES}')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start, end = prices.index[0], prices.index[-1]
    windows: list[WindowResult] = []
    window_start = start

    while True:
        train_end = window_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        if val_end > end:
            break

        prices_train_full = prices.loc[window_start:train_end]
        prices_val_full = prices.loc[train_end:val_end]

        if len(prices_train_full) < 252 or len(prices_val_full) < 126:
            window_start += pd.DateOffset(years=step_years)
            continue

        # Per-window survivorship filter — each slice picks its own
        # tradeable universe from the point-in-time panel. Train and
        # val are filtered independently so a name that delists between
        # them only affects the slice it stops trading in.
        keep_train = _filter_window_universe(
            prices_train_full, min_bars=per_window_min_history)
        keep_val = _filter_window_universe(
            prices_val_full, min_bars=per_window_min_history)

        if len(keep_train) == 0 or len(keep_val) == 0:
            print(f'\nWindow {window_start.date()}-{val_end.date()}: '
                  f'empty universe (train={len(keep_train)}, '
                  f'val={len(keep_val)}); skipping')
            window_start += pd.DateOffset(years=step_years)
            continue

        prices_train = prices_train_full[keep_train]
        prices_val = prices_val_full[keep_val]
        spread_train = (spread_df.loc[window_start:train_end, keep_train]
                        if spread_df is not None else None)
        spread_val = (spread_df.loc[train_end:val_end, keep_val]
                      if spread_df is not None else None)

        print(f'\nWindow: train {window_start.date()}-{train_end.date()} '
              f'({len(keep_train)} tickers), '
              f'val {train_end.date()}-{val_end.date()} '
              f'({len(keep_val)} tickers)')

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed))
        objective = _make_objective(
            prices_train, strategy=strategy,
            rebalance_days=rebalance_days, metric=metric,
            commission_bps=commission_bps,
            spread_df=spread_train,
            use_log_returns=use_log_returns)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                       show_progress_bar=True)

        best = study.best_params
        train_score = study.best_value
        print(f'  Train {metric}: {train_score:+.4f}, params: {best}')

        try:
            weight_df = _build_weights(
                strategy, prices_val, best, use_log_returns=use_log_returns)
            metrics = vbt_backtest(
                prices_val, weight_df,
                rebalance_days=rebalance_days,
                commission_bps=commission_bps,
                spread_df=spread_val)
            val_score = metrics[metric]
        except Exception:
            val_score = float('nan')
        print(f'  Val   {metric}: {val_score:+.4f}')

        windows.append(WindowResult(
            train_start=window_start, train_end=train_end, val_end=val_end,
            best_params=best, train_score=train_score, val_score=val_score,
            strategy=strategy, use_log_returns=use_log_returns))
        window_start += pd.DateOffset(years=step_years)

    return TrainResult(
        windows=windows, metric=metric,
        rebalance_days=rebalance_days,
        commission_bps=commission_bps,
        strategy=strategy, use_log_returns=use_log_returns)


def print_summary(result: TrainResult) -> None:
    """Pretty-print the per-window summary + best-window highlight."""
    print(f'\n{"=" * 80}')
    print(f'Walk-Forward Summary  (strategy: {result.strategy}, '
          f'metric: {result.metric}, '
          f'commission: {result.commission_bps} bps + half-spread per side)')
    print('=' * 80)
    print(f'{"Window":<22} {"Train":>9} {"Val":>9} {"Div":>7} '
          f'{"LB":>4} {"NT":>4} {"TopN":>5} {"Scales":>8}')
    print('-' * 80)
    for w in result.windows:
        p = w.best_params
        scales_str = ''.join(
            c for c, on in [
                ('S', p.get('use_short_scales')),
                ('M', p.get('use_mid_scales')),
                ('L', p.get('use_long_scales')),
            ] if on) or 'def'
        label = f'{w.train_start.year}-{w.val_end.year}'
        # Scalogram has no divergence knob; print '—' so the column lines up.
        div_str = str(p.get('divergence', '—'))
        print(f'{label:<22} {w.train_score:>+9.4f} {w.val_score:>+9.4f} '
              f'{div_str:>7} {p["lookback"]:>4} {p["n_tail"]:>4} '
              f'{p["top_n"]:>5} {scales_str:>8}')
    if result.windows:
        bw = result.best_window
        val_scores = np.array([w.val_score for w in result.windows
                               if np.isfinite(w.val_score)])
        if len(val_scores):
            print(f'\nVal {result.metric} stats across {len(val_scores)} windows:')
            print(f'  best   = {val_scores.max():+.4f}  '
                  f'(window {bw.train_start.year}-{bw.val_end.year})')
            print(f'  median = {np.median(val_scores):+.4f}')
            print(f'  mean   = {val_scores.mean():+.4f}')
            print(f'  worst  = {val_scores.min():+.4f}')
            print('  (best is max-of-N; expect ~0.2-0.3σ upward bias '
                  'vs the median, which is the more honest single-number summary)')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Optuna + vectorbt walk-forward search for the regime strategy.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory of per-ticker OHLC CSVs (e.g. ./Nasdaq3347).')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna trials per walk-forward window.')
    parser.add_argument('--metric', default='sharpe',
                        choices=['sharpe', 'cagr', 'max_drawdown', 'total_return'])
    parser.add_argument('--rebalance-days', type=int, default=20)
    parser.add_argument('--commission-bps', type=float, default=10.0)
    parser.add_argument('--train-years', type=int, default=5)
    parser.add_argument('--val-years', type=int, default=3)
    parser.add_argument('--step-years', type=int, default=2)
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--min-history', type=int, default=504)
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=args.min_history,
        start_date=args.start, end_date=args.end)
    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)

    result = train(
        prices, spread_df,
        n_trials=args.n_trials,
        rebalance_days=args.rebalance_days,
        metric=args.metric,
        commission_bps=args.commission_bps,
        train_years=args.train_years,
        val_years=args.val_years,
        step_years=args.step_years)

    print_summary(result)


if __name__ == '__main__':
    main()
