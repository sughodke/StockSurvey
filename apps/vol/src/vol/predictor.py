"""OLS linear predictor for forward IV/RV gap.

Pooled cross-section + time-series regression: rows are
`(date, symbol)` cells, features are z-scored on the train pool,
target is `iv_rv_gap`. Mirrors the predictor convention from
`apps/gate/src/gate/predictor.py` for cross-app consistency.

Linear is sufficient for v0 because (a) it's the cheapest direct
test of the user's "untested feature class" hypothesis, and (b) if
the linear fit shows positive val R² we know the features carry
signal — *then* MLP / nonlinear is justified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictorResult:
    feature_names: list[str]
    coefficients:  np.ndarray   # (n_features,)
    intercept:     float
    feat_mean:     np.ndarray
    feat_std:      np.ndarray
    train_r2:      float
    train_rmse:    float
    n_train:       int


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def train_predictor(
    X_train: np.ndarray, y_train: np.ndarray,
    feature_names: list[str],
) -> PredictorResult:
    if X_train.ndim != 2:
        raise ValueError(f'X_train must be 2D, got {X_train.shape}')
    if len(X_train) != len(y_train):
        raise ValueError(
            f'X_train rows {len(X_train)} != y_train {len(y_train)}')
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f'feature_names len {len(feature_names)} != n_features '
            f'{X_train.shape[1]}')
    mu, sd = _zscore_fit(X_train)
    Xz = _zscore_apply(X_train, mu, sd)
    Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
    coefs, *_ = np.linalg.lstsq(Xa, y_train, rcond=None)
    coefficients = coefs[:-1]
    intercept = float(coefs[-1])

    pred = Xa @ coefs
    resid = y_train - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    train_rmse = float(np.sqrt(np.mean(resid ** 2)))
    return PredictorResult(
        feature_names=feature_names,
        coefficients=coefficients, intercept=intercept,
        feat_mean=mu, feat_std=sd,
        train_r2=train_r2, train_rmse=train_rmse,
        n_train=int(len(y_train)),
    )


def predict(result: PredictorResult, X: np.ndarray) -> np.ndarray:
    Xz = _zscore_apply(X, result.feat_mean, result.feat_std)
    return Xz @ result.coefficients + result.intercept


def evaluate_r2(predictions: np.ndarray, actuals: np.ndarray) -> float:
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_res = float(np.sum((actuals - predictions) ** 2))
    return 1.0 - ss_res / ss_tot


__all__ = ['PredictorResult', 'evaluate_r2', 'predict', 'train_predictor']
