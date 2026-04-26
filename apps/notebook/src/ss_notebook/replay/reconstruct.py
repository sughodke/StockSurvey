"""Multi-ticker fit + per-ticker reconstruction orchestrator.

`fit_and_evaluate` is the single entry point: it pools features from
every train ticker into one decoder fit, then applies that fitted
decoder to all train *and* val tickers (each ticker independently). The
return shape is `{ticker_name: {target: {'recon': ndarray, 'stats':
{...}}}}` so callers can plot / report per ticker.

`reconstruct_indicators` is a backward-compat wrapper around
`fit_and_evaluate` that takes a single train price series (and optional
val price series) and unpacks results into the legacy tuple shape.
"""
from __future__ import annotations

import numpy as np

from ss_notebook.replay.decoders import fit_cnn, fit_mlp, fit_ols
from ss_notebook.replay.features import (
    TARGET_NAMES, TickerData, build_features_and_targets,
)
from ss_notebook.replay.metrics import fit_stats


def fit_and_evaluate(
    train_data: list[TickerData],
    val_data: list[TickerData], *,
    decoder: str,
    cnn_channels_per_lag: int,
    mlp_hidden: int = 128,
    mlp_layers: int = 2,
    mlp_steps: int = 2000,
    cnn_hidden: int = 64,
    cnn_kernel: int = 5,
    cnn_layers: int = 2,
    cnn_steps: int = 2000,
) -> dict[str, dict[str, dict]]:
    """Pool train tickers into one decoder fit, predict per-ticker.

    Returns
    -------
    `{ticker_name: {target_name: {'recon': ndarray, 'stats': dict}}}`
    where `recon` is full-length (NaN outside that ticker's valid mask)
    and `stats` holds R² / RMSE / max-|Δ|.
    """
    if not train_data:
        raise ValueError('fit_and_evaluate needs at least one train ticker')
    n_features = train_data[0].features.shape[1]
    for d in train_data + val_data:
        if d.features.shape[1] != n_features:
            raise ValueError(
                f'ticker {d.name!r} has {d.features.shape[1]} features but '
                f'train ticker {train_data[0].name!r} has {n_features}; '
                'feature shapes must match (same scales, window_cols, '
                'include_zscore_stats).')

    # Pool train rows.
    X_pools = [d.features[d.valid] for d in train_data]
    X_train = np.vstack(X_pools)
    y_train = {
        n: np.concatenate([d.targets[n][d.valid] for d in train_data])
        for n in TARGET_NAMES
    }

    # Concatenate every ticker's full feature block for one prediction pass.
    all_data = list(train_data) + list(val_data)
    n_per = [d.features.shape[0] for d in all_data]
    X_predict = np.vstack([d.features for d in all_data])

    results: dict[str, dict[str, dict]] = {d.name: {} for d in all_data}
    for target_name in TARGET_NAMES:
        y = y_train[target_name]
        if decoder == 'linear':
            w, b = fit_ols(X_train, y)
            yhat_all = X_predict @ w + b
        elif decoder == 'mlp':
            yhat_all = fit_mlp(
                X_train, y, X_predict,
                hidden=mlp_hidden, n_layers=mlp_layers, n_steps=mlp_steps)
        elif decoder == 'cnn':
            yhat_all = fit_cnn(
                X_train, y, X_predict,
                n_channels_per_lag=cnn_channels_per_lag,
                hidden=cnn_hidden, kernel=cnn_kernel,
                n_layers=cnn_layers, n_steps=cnn_steps)
        else:
            raise ValueError(f'unknown decoder: {decoder!r}')

        offset = 0
        for d, n in zip(all_data, n_per):
            yhat_t = yhat_all[offset:offset + n]
            recon_full = np.full(n, np.nan, dtype=np.float64)
            recon_full[d.valid] = yhat_t[d.valid]
            stats = fit_stats(yhat_t[d.valid], d.targets[target_name][d.valid])
            results[d.name][target_name] = {
                'recon': recon_full, 'stats': stats}
            offset += n

    return results


def reconstruct_indicators(
    prices: np.ndarray, *,
    val_prices: np.ndarray | None = None,
    scales: list[int],
    lookback: int,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    window_cols: int = 1,
    include_zscore_stats: bool = False,
    decoder: str = 'linear',
    mlp_hidden: int = 128, mlp_layers: int = 2, mlp_steps: int = 2000,
    cnn_hidden: int = 64, cnn_kernel: int = 5,
    cnn_layers: int = 2, cnn_steps: int = 2000,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray],
    dict[str, dict[str, float]], int,
    dict[str, np.ndarray] | None, dict[str, np.ndarray] | None,
    dict[str, dict[str, float]] | None,
]:
    """Single-train-ticker convenience wrapper around `fit_and_evaluate`.

    Returns `(gt, recon, stats, n_features, val_gt, val_recon,
    val_stats)`. The val triple is non-None iff `val_prices` is given.
    """
    feat_kwargs = dict(
        scales=scales, lookback=lookback, window_cols=window_cols,
        include_zscore_stats=include_zscore_stats, decoder=decoder,
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal)
    train_features, train_gt, train_valid = build_features_and_targets(
        prices, **feat_kwargs)
    train_td = TickerData(
        name='train', prices=prices, dates=np.arange(len(prices)),
        features=train_features, targets=train_gt, valid=train_valid)
    train_list = [train_td]

    val_list: list[TickerData] = []
    val_gt: dict[str, np.ndarray] | None = None
    if val_prices is not None:
        val_features, v_gt, v_valid = build_features_and_targets(
            val_prices, **feat_kwargs)
        val_td = TickerData(
            name='val', prices=val_prices,
            dates=np.arange(len(val_prices)),
            features=val_features, targets=v_gt, valid=v_valid)
        val_list = [val_td]
        val_gt = v_gt

    cnn_channels_per_lag = 2 * len(scales)
    results = fit_and_evaluate(
        train_list, val_list,
        decoder=decoder, cnn_channels_per_lag=cnn_channels_per_lag,
        mlp_hidden=mlp_hidden, mlp_layers=mlp_layers, mlp_steps=mlp_steps,
        cnn_hidden=cnn_hidden, cnn_kernel=cnn_kernel,
        cnn_layers=cnn_layers, cnn_steps=cnn_steps)

    recon = {n: results['train'][n]['recon'] for n in train_gt}
    stats = {n: results['train'][n]['stats'] for n in train_gt}

    if val_list:
        val_recon = {n: results['val'][n]['recon'] for n in val_gt}
        val_stats = {n: results['val'][n]['stats'] for n in val_gt}
    else:
        val_recon = None
        val_stats = None

    return (train_gt, recon, stats, train_features.shape[1],
            val_gt, val_recon, val_stats)
