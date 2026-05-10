"""Linear regression predictor for forward drawdown.

Numpy-only — no tinygrad. Linear predictor is sufficient for v0
because (a) the feature set is small and the data is single-series
(no batch dimension to amortize a deep model's overhead), (b) the
target is a real-valued regression problem where ordinary
least-squares has a closed-form solution, and (c) we don't yet know
if there's *any* signal — bigger models can come if linear shows R²
above zero.

Standardization is fit on train and applied to val. Bias term is
included via an explicit intercept column.

`apply_gate(predicted_dd, threshold)` converts the continuous
prediction into a `gate_t ∈ [0, 1]` exposure scaler. Two modes:
binary (0 or 1) and smooth-sigmoid. Binary is the cleanest test
of "does the predictor have *any* skill"; smooth-sigmoid is closer
to a deployable strategy but adds a hyperparameter (sigmoid slope).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictorResult:
    """Fit summary + parameters."""
    feature_names: list[str]
    coefficients:  np.ndarray   # shape (n_features,)
    intercept:     float
    feat_mean:     np.ndarray   # train-set mean per feature
    feat_std:      np.ndarray   # train-set stdev per feature
    train_r2:      float
    train_rmse:    float


def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def train_predictor(
    X_train: np.ndarray, y_train: np.ndarray,
    feature_names: list[str],
) -> PredictorResult:
    """OLS linear regression with z-scored features + intercept."""
    if X_train.ndim != 2:
        raise ValueError(f'X_train must be 2D, got {X_train.shape}')
    if len(X_train) != len(y_train):
        raise ValueError(
            f'X_train rows {len(X_train)} != y_train {len(y_train)}')
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f'feature_names {len(feature_names)} != n_features '
            f'{X_train.shape[1]}')

    mu, sd = _zscore_fit(X_train)
    Xz = _zscore_apply(X_train, mu, sd)
    # Augment with intercept column for closed-form OLS.
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
        coefficients=coefficients,
        intercept=intercept,
        feat_mean=mu, feat_std=sd,
        train_r2=train_r2, train_rmse=train_rmse,
    )


def predict(result: PredictorResult, X: np.ndarray) -> np.ndarray:
    """Apply a trained predictor to fresh feature rows.

    Standardizes with the train-set stats baked into `result`, then
    applies linear coefficients + intercept. No clipping — caller
    decides what to do with negative or out-of-distribution
    predictions (typically convert via `apply_gate`).
    """
    Xz = _zscore_apply(X, result.feat_mean, result.feat_std)
    return Xz @ result.coefficients + result.intercept


def evaluate_r2(predictions: np.ndarray, actuals: np.ndarray) -> float:
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_res = float(np.sum((actuals - predictions) ** 2))
    return 1.0 - ss_res / ss_tot


def apply_gate(
    predicted_dd: np.ndarray,
    threshold: float,
    mode: str = 'binary',
    sigmoid_slope: float = 50.0,
) -> np.ndarray:
    """Convert predicted drawdown to exposure gate `g ∈ [0, 1]`.

    `mode='binary'` — `g = 1` if predicted_dd <= threshold else 0.
    `mode='sigmoid'` — `g = 1 - sigmoid(slope * (predicted_dd −
    threshold))`. Slope=50 gives a transition over ~5% of drawdown
    units (so threshold=0.04 means `g=0.5` at predicted DD=0.04,
    `g≈0.99` at DD=0.02, `g≈0.01` at DD=0.06). Higher slope is
    closer to binary; lower slope is a softer gate that hedges
    rather than fully exits.
    """
    if mode == 'binary':
        return (predicted_dd <= threshold).astype(np.float64)
    if mode == 'sigmoid':
        z = sigmoid_slope * (predicted_dd - threshold)
        # Numerically-safe sigmoid.
        return 1.0 - 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    raise ValueError(f"mode must be 'binary' or 'sigmoid', got {mode!r}")


__all__ = [
    'PredictorResult', 'apply_gate', 'evaluate_r2', 'predict',
    'train_predictor',
]
