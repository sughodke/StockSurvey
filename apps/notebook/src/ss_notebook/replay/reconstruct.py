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

from ss_notebook.replay.decoders import (
    fit_cnn, fit_cnn_masked_ae, fit_cnn_multihead, fit_mlp, fit_ols,
)
from ss_notebook.replay.features import (
    TARGET_NAMES, TickerData, build_features_and_targets,
)
from ss_notebook.replay.metrics import fit_stats


def fit_and_evaluate(
    train_data: list[TickerData],
    val_data: list[TickerData], *,
    decoder: str,
    cnn_channels_per_lag: int,
    targets: tuple[str, ...] = TARGET_NAMES,
    mlp_hidden: int = 128,
    mlp_layers: int = 2,
    mlp_steps: int = 2000,
    mlp_batch_size: int | None = None,
    cnn_hidden: int = 64,
    cnn_kernel: int = 5,
    cnn_layers: int = 2,
    cnn_steps: int = 2000,
    cnn_batch_size: int | None = None,
    cnn_microbatch_size: int | None = None,
    cnn_film_hidden: int = 32,
    rsi_n_grid: tuple[int, ...] = (),
    rsi_w_grid: tuple[int, ...] = (),
    rsi_anchor_n: int | None = None,
    rsi_anchor_w: int = 1,
    frozen_backbone_path: str | None = None,
    use_bf16: bool = True,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, np.ndarray]]]:
    """Pool train tickers into one decoder fit, predict per-ticker.

    `targets` selects which targets to fit/predict; defaults to all of
    `TARGET_NAMES`. For `linear` and `mlp` each target gets its own
    independent fit. For `cnn` all targets share one conv backbone with
    per-target linear heads (multi-head, joint Adam fit) — see
    `fit_cnn_multihead`. The on-disk `params_per_target` shape is the
    same in both modes; the cnn case duplicates shared backbone weights
    under each target so each per-target dict is self-contained at
    inference time.

    Returns
    -------
    `(per_ticker, params_per_target)` where:
      - `per_ticker` = `{ticker_name: {target_name: {'recon': ndarray,
        'stats': dict}}}`. `recon` is full-length (NaN outside that
        ticker's valid mask); `stats` holds R²/RMSE/max-|Δ|.
      - `params_per_target` = `{target_name: {array_name: ndarray}}` —
        flat numpy params suitable for `np.savez`. One entry per
        fitted target.
    """
    if not train_data:
        raise ValueError('fit_and_evaluate needs at least one train ticker')
    unknown = set(targets) - set(TARGET_NAMES)
    if unknown:
        raise ValueError(
            f'unknown targets {sorted(unknown)!r}; valid: {TARGET_NAMES}')
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
        for n in targets
    }

    # RSI parameter conditioning. When `rsi_n_grid` is set we *logically*
    # replicate every pooled row once per grid value: each logical row
    # maps to its physical pool row via `train_pool_idx`, while the 1-D
    # target / conditioning arrays carry the grid-indexed values. The
    # large `X_train` feature matrix is never materialized in augmented
    # form — `fit_cnn_multihead` gathers `X_train[pool_idx[idx]]` per
    # minibatch from the small unique-pool buffer. Memory cost of the
    # augmentation is O(n_pool * n_replicas) for 1-D arrays, not
    # O(n_pool * n_replicas * K * F).
    #
    # `rsi_w_grid` (resampling stride) extends conditioning to the
    # cross product (w, n) — `n_replicas = n_w * n_n` and the
    # conditioning vector becomes p_dim=2 = `(n_norm, w_norm)`. Row
    # layout of the flattened target_grids matches: outer = w,
    # inner = n, so `rsi_grid_pooled[w_idx * n_n + n_idx]` is the
    # series for that (w, n) pair.
    head_cond_train: dict[str, np.ndarray] = {}
    head_cond_predict: dict[str, np.ndarray] = {}
    train_pool_idx: np.ndarray | None = None
    rsi_grid_active = bool(rsi_n_grid) and 'rsi' in targets and decoder == 'cnn'
    if rsi_w_grid and not rsi_grid_active:
        raise ValueError(
            'rsi_w_grid given but rsi_n_grid is empty — w-conditioning '
            'requires the n-conditioning to be active too.')
    if rsi_grid_active:
        n_n = len(rsi_n_grid)
        n_w = len(rsi_w_grid) if rsi_w_grid else 1
        n_replicas = n_n * n_w
        for d in train_data:
            if 'rsi' not in d.target_grids:
                raise ValueError(
                    f'rsi_n_grid={rsi_n_grid!r} requested but train ticker '
                    f'{d.name!r} has no target_grids["rsi"] — was load_ticker '
                    'called with rsi_n_grid?')
            if d.target_grids['rsi'].shape[0] != n_replicas:
                raise ValueError(
                    f'train ticker {d.name!r} target_grids["rsi"] shape '
                    f'{d.target_grids["rsi"].shape} disagrees with '
                    f'expected n_replicas={n_replicas} '
                    f'(n_w={n_w} * n_n={n_n}).')
        # Stack grid-indexed RSI values matching X_train's pooled row
        # order. Shape (n_replicas, n_pool); reshape(-1) lays out as
        # [replica[0]_pool[0..n_pool-1], replica[1]_pool[0..], ...].
        rsi_grid_pooled = np.concatenate(
            [d.target_grids['rsi'][:, d.valid] for d in train_data], axis=1)
        n_pool = X_train.shape[0]
        train_pool_idx = np.tile(np.arange(n_pool, dtype=np.int64), n_replicas)
        for t in targets:
            if t == 'rsi':
                y_train[t] = rsi_grid_pooled.reshape(-1)
            else:
                y_train[t] = np.tile(y_train[t], n_replicas)
        # Build conditioning vector. Row order matches replica layout
        # (outer w, inner n): for replica = w_idx * n_n + n_idx the
        # conditioning is (n_grid[n_idx]/max_n, w_grid[w_idx]/max_w).
        n_max = float(max(rsi_n_grid))
        n_values = np.array(rsi_n_grid, dtype=np.float32) / n_max
        if rsi_w_grid:
            w_max = float(max(rsi_w_grid))
            w_values = np.array(rsi_w_grid, dtype=np.float32) / w_max
            # Cross product, outer w / inner n. Per replica replicate
            # n_pool times to match the augmented training row count.
            n_col = np.tile(n_values, n_w)             # (n_replicas,)
            w_col = np.repeat(w_values, n_n)           # (n_replicas,)
            cond_per_replica = np.stack([n_col, w_col], axis=1)  # (n_replicas, 2)
        else:
            cond_per_replica = n_values[:, None]       # (n_replicas, 1)
        # Replicate per replica across n_pool rows. `np.repeat` along
        # axis=0 expands (n_replicas, p_dim) -> (n_replicas * n_pool,
        # p_dim) with each replica row repeated n_pool times — matches
        # the train_pool_idx layout (replica[0] occupies the first
        # n_pool augmented rows, etc.).
        head_cond_train['rsi'] = np.repeat(cond_per_replica, n_pool, axis=0)

    # Concatenate every ticker's full feature block for one prediction pass.
    all_data = list(train_data) + list(val_data)
    n_per = [d.features.shape[0] for d in all_data]
    X_predict = np.vstack([d.features for d in all_data])

    if rsi_grid_active:
        # Predict at the anchor (w, n). The 1-D `gt['rsi']` ground-truth
        # the figure compares against was computed at (w=1, n=rsi_n) in
        # features.py, so the default anchor reproduces that.
        anchor_n = (rsi_anchor_n if rsi_anchor_n is not None
                    else int(sorted(rsi_n_grid)[len(rsi_n_grid) // 2]))
        n_norm = anchor_n / float(max(rsi_n_grid))
        if rsi_w_grid:
            w_norm = rsi_anchor_w / float(max(rsi_w_grid))
            head_cond_predict['rsi'] = np.tile(
                np.array([[n_norm, w_norm]], dtype=np.float32),
                (X_predict.shape[0], 1))
        else:
            head_cond_predict['rsi'] = np.full(
                (X_predict.shape[0], 1), n_norm, dtype=np.float32)

    results: dict[str, dict[str, dict]] = {d.name: {} for d in all_data}
    params_per_target: dict[str, dict[str, np.ndarray]] = {}
    yhats_all: dict[str, np.ndarray] = {}

    if decoder == 'cnn':
        # One shared backbone, per-target heads, joint Adam fit.
        # Optional `frozen_backbone_path` flips this into the SSL probe:
        # backbone weights are loaded from a previously-trained npz and
        # held fixed; only heads + FiLM train.
        frozen_backbone = None
        if frozen_backbone_path is not None:
            from ss_notebook.scoring.backbone import load_backbone
            frozen_backbone, _bb_meta = load_backbone(frozen_backbone_path)
        yhats_all, params_per_target = fit_cnn_multihead(
            X_train, {t: y_train[t] for t in targets}, X_predict,
            n_channels_per_lag=cnn_channels_per_lag,
            hidden=cnn_hidden, kernel=cnn_kernel,
            n_layers=cnn_layers, n_steps=cnn_steps,
            batch_size=cnn_batch_size,
            microbatch_size=cnn_microbatch_size,
            film_hidden=cnn_film_hidden,
            head_conditioning_train=head_cond_train,
            head_conditioning_predict=head_cond_predict,
            train_pool_idx=train_pool_idx,
            frozen_backbone=frozen_backbone,
            use_bf16=use_bf16)
    else:
        for target_name in targets:
            y = y_train[target_name]
            if decoder == 'linear':
                w, b = fit_ols(X_train, y)
                yhats_all[target_name] = X_predict @ w + b
                params_per_target[target_name] = {
                    'w': np.asarray(w, dtype=np.float64),
                    'b': np.array([b], dtype=np.float64),
                }
            elif decoder == 'mlp':
                yhats_all[target_name], params_per_target[target_name] = fit_mlp(
                    X_train, y, X_predict,
                    hidden=mlp_hidden, n_layers=mlp_layers, n_steps=mlp_steps,
                    batch_size=mlp_batch_size,
                    use_bf16=use_bf16)
            else:
                raise ValueError(f'unknown decoder: {decoder!r}')

    for target_name in targets:
        yhat_all = yhats_all[target_name]
        offset = 0
        for d, n in zip(all_data, n_per):
            yhat_t = yhat_all[offset:offset + n]
            recon_full = np.full(n, np.nan, dtype=np.float64)
            recon_full[d.valid] = yhat_t[d.valid]
            stats = fit_stats(yhat_t[d.valid], d.targets[target_name][d.valid])
            results[d.name][target_name] = {
                'recon': recon_full, 'stats': stats}
            offset += n

    return results, params_per_target


def fit_and_evaluate_ssl(
    train_data: list[TickerData],
    val_data: list[TickerData],
    *,
    cnn_channels_per_lag: int,
    cnn_hidden: int = 64,
    cnn_kernel: int = 5,
    cnn_layers: int = 2,
    cnn_steps: int = 10_000,
    cnn_batch_size: int | None = None,
    cnn_microbatch_size: int | None = None,
    ssl_decoder_hidden: int = 256,
    ssl_decoder_layers: int = 2,
    mask_ratio: float = 0.4,
    seed: int = 0,
    use_bf16: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Self-supervised pretrain entry point — masked CWT autoencoding.

    Pools `train_data` features (no targets used) and trains the conv
    backbone via masked-cell reconstruction. `val_data` is *not* used
    for training; if non-empty, its full reconstruction MSE is reported
    so the caller can spot generalization gaps.

    Returns `(params_per_target, stats)` where:
      - `params_per_target = {'ssl': params_dict}` so the CLI's existing
        npz writer (which prefixes by target name) drops everything
        under `ssl__` — matching what `load_backbone` expects (≥1
        prefix containing `feat_mu`, `feat_sd`, `conv{i}_W/b`; the
        decoder is stored under `head_dec*` and skipped by the loader).
      - `stats` holds final masked + unmasked train MSE plus an
        optional unmasked val MSE.
    """
    if not train_data:
        raise ValueError('fit_and_evaluate_ssl needs at least one train ticker')
    n_features = train_data[0].features.shape[1]
    for d in train_data + val_data:
        if d.features.shape[1] != n_features:
            raise ValueError(
                f'ticker {d.name!r} has {d.features.shape[1]} features but '
                f'train ticker {train_data[0].name!r} has {n_features}; '
                'feature shapes must match.')

    X_train = np.vstack([d.features[d.valid] for d in train_data])
    X_full = np.vstack(
        [d.features[d.valid] for d in train_data]
        + [d.features[d.valid] for d in val_data])

    params, stats = fit_cnn_masked_ae(
        X_train, X_full,
        n_channels_per_lag=cnn_channels_per_lag,
        hidden=cnn_hidden, kernel=cnn_kernel, n_layers=cnn_layers,
        decoder_hidden=ssl_decoder_hidden,
        decoder_layers=ssl_decoder_layers,
        n_steps=cnn_steps, batch_size=cnn_batch_size,
        microbatch_size=cnn_microbatch_size,
        mask_ratio=mask_ratio, seed=seed,
        use_bf16=use_bf16)
    return {'ssl': params}, stats


def reconstruct_indicators(
    prices: np.ndarray, *,
    val_prices: np.ndarray | None = None,
    scales: list[int],
    lookback: int,
    rsi_n: int, macd_fast: int, macd_slow: int, macd_signal: int,
    window_cols: int = 1,
    include_zscore_stats: bool = False,
    include_returns: bool = False,
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
        include_zscore_stats=include_zscore_stats,
        include_returns=include_returns, decoder=decoder,
        rsi_n=rsi_n, macd_fast=macd_fast, macd_slow=macd_slow,
        macd_signal=macd_signal)
    train_features, train_gt, train_valid, train_grids = (
        build_features_and_targets(prices, **feat_kwargs))
    train_td = TickerData(
        name='train', prices=prices, dates=np.arange(len(prices)),
        features=train_features, targets=train_gt, valid=train_valid,
        target_grids=train_grids)
    train_list = [train_td]

    val_list: list[TickerData] = []
    val_gt: dict[str, np.ndarray] | None = None
    if val_prices is not None:
        val_features, v_gt, v_valid, v_grids = build_features_and_targets(
            val_prices, **feat_kwargs)
        val_td = TickerData(
            name='val', prices=val_prices,
            dates=np.arange(len(val_prices)),
            features=val_features, targets=v_gt, valid=v_valid,
            target_grids=v_grids)
        val_list = [val_td]
        val_gt = v_gt

    cnn_channels_per_lag = train_features.shape[1] // window_cols
    results, _params = fit_and_evaluate(
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
