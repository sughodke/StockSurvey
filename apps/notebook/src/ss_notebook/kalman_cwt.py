"""Kalman / linear-dynamical-system compression of the causal CWT panel.

Diagnostic, not a strategy. Question answered: how few recursive
fixed-dim state coordinates can reconstruct the 13-scale causal CWT of a
single ticker with near-lossless fidelity?

Model (linear-Gaussian state space over the CWT observation panel
`y_t in R^p`, `p = len(ALL_SCALES) = 13`):

    s_t = A s_{t-1} + w_t      w_t ~ N(0, Q)      s_t in R^k, k <= p
    y_t = C s_t   + v_t        v_t ~ N(0, R)

Fit is the pragmatic subspace route (numpy-only, no EM iteration):

  1. per-scale z-score the panel (scales differ in power by orders of
     magnitude — without this PCA collapses onto scale=126),
  2. `C` = top-k right singular vectors of the standardized panel,
  3. `A` = least-squares VAR(1) on the latent scores `Z = Y_std @ C`,
  4. `Q`, `R` = residual covariances of the latent step and the
     observation reconstruction.

Then the Kalman filter is run forward to produce the *recursive* state
`s_hat_t = E[s_t | y_{:t}]` — a strictly-causal fixed-dim summary, not
the batch PCA projection. Three reconstructions are scored per `k`:

  * `pca`     — batch projection `Z C^T` (non-causal upper bound).
  * `kalman`  — filtered `C s_hat_t` (causal, recursive — the
                "near-lossless compression" number being asked for).
  * `predict` — one-step-ahead `C A s_hat_{t-1}` (uses `y_{:t-1}`).

The `kalman` vs `predict` gap is the point of contact with the
sufficient-statistic discussion: a recursive state can be near-lossless
for *compression* while carrying almost no *predictive* information —
the same state, scored two ways, in one table.

    uv run python -m ss_notebook.kalman_cwt --stooq-dir ./StooqData AAPL
    uv run ss-kalman-cwt --stooq-dir ./StooqData --save AAPL
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ss_cli import add_save_args, add_single_ticker_loader_args
from ss_features import load_prices
from ss_wavelets import ALL_SCALES, KERNEL_HALF_EXTENT, causal_cwt

DEFAULT_KS = [1, 2, 3, 4, 5, 6, 8, 10, 13]


def cwt_panel(
    prices: np.ndarray,
    *,
    lookback: int = 90,
    use_log_returns: bool = True,
) -> np.ndarray:
    """`(T, p)` causal-CWT observation panel over `ALL_SCALES`.

    Mirrors the scalogram convention: log-returns input + lookback=90,
    so the rolling z-norm is a light vol normalization rather than a
    trend remover. The first `KERNEL_HALF_EXTENT*max(scale) + lookback`
    rows have reduced wavelet support and are dropped by the caller.
    """
    if use_log_returns:
        signal = np.zeros_like(prices, dtype=np.float64)
        signal[1:] = np.log(prices[1:] / prices[:-1])
    else:
        signal = prices.astype(np.float64)
    px = signal.astype(np.float32).reshape(-1, 1)
    coeffs = causal_cwt(px, list(ALL_SCALES), lookback=lookback)[:, :, 0]
    return coeffs.T.astype(np.float64)  # (T, p)


def fit_lds(Y: np.ndarray, k: int) -> dict:
    """Subspace-fit a k-dim linear-Gaussian state space to `Y` (T, p)."""
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0) + 1e-12
    Ys = (Y - mu) / sd

    # C = top-k principal axes (orthonormal columns, p x k).
    _, _, Vt = np.linalg.svd(Ys, full_matrices=False)
    C = Vt[:k].T
    Z = Ys @ C  # latent scores (T, k)

    # A = least-squares VAR(1): Z[1:] ~ Z[:-1] A^T.
    A = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)[0].T
    res_z = Z[1:] - Z[:-1] @ A.T
    Q = np.cov(res_z, rowvar=False).reshape(k, k) + 1e-9 * np.eye(k)

    res_y = Ys - Z @ C.T
    R = np.cov(res_y, rowvar=False) + 1e-9 * np.eye(Y.shape[1])

    return {'mu': mu, 'sd': sd, 'C': C, 'A': A, 'Q': Q, 'R': R}


def kalman_filter(Ys: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Forward filter. Returns filtered and one-step-predicted states."""
    A, C, Q, R = p['A'], p['C'], p['Q'], p['R']
    k = A.shape[0]
    x = np.zeros(k)
    P = np.eye(k)
    filt = np.empty((Ys.shape[0], k))
    pred = np.empty((Ys.shape[0], k))
    I = np.eye(k)
    for t, y in enumerate(Ys):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        pred[t] = x_pred
        S = C @ P_pred @ C.T + R
        K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        x = x_pred + K @ (y - C @ x_pred)
        P = (I - K @ C) @ P_pred
        filt[t] = x
    return filt, pred


def _rel_err(Y: np.ndarray, Yhat: np.ndarray) -> tuple[float, float]:
    """(relative Frobenius error, R^2) on the original CWT scale."""
    sse = float(np.sum((Y - Yhat) ** 2))
    sst = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return (sse / sst) ** 0.5, 1.0 - sse / sst


def evaluate(Y: np.ndarray, ks: list[int]) -> list[dict]:
    rows = []
    for k in ks:
        if k > Y.shape[1]:
            continue
        p = fit_lds(Y, k)
        Ys = (Y - p['mu']) / p['sd']
        Z = Ys @ p['C']
        filt, pred = kalman_filter(Ys, p)

        def back(Zlike: np.ndarray) -> np.ndarray:
            return (Zlike @ p['C'].T) * p['sd'] + p['mu']

        e_pca, r2_pca = _rel_err(Y, back(Z))
        e_kf, r2_kf = _rel_err(Y, back(filt))
        e_pr, r2_pr = _rel_err(Y, back(pred))
        rows.append({
            'k': k,
            'pca_err': e_pca, 'pca_r2': r2_pca,
            'kf_err': e_kf, 'kf_r2': r2_kf,
            'pr_err': e_pr, 'pr_r2': r2_pr,
        })
    return rows


def _print_table(ticker: str, rows: list[dict]) -> None:
    print(f'\n{ticker}: LDS compression of causal CWT '
          f'(p={len(ALL_SCALES)} scales)')
    print('  k | PCA(batch)   | Kalman(causal) | 1-step(predict)')
    print('    | relerr   R2  | relerr    R2   | relerr    R2')
    print('  ' + '-' * 56)
    for r in rows:
        print(f'  {r["k"]:>2d} | {r["pca_err"]:.4f} {r["pca_r2"]:+.3f} '
              f'| {r["kf_err"]:.4f} {r["kf_r2"]:+.3f}  '
              f'| {r["pr_err"]:.4f} {r["pr_r2"]:+.3f}')


def _plot(ticker: str, rows: list[dict], out_dir: str) -> str:
    ks = [r['k'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [r['pca_err'] for r in rows], 'o-', label='PCA (batch upper bound)')
    ax.plot(ks, [r['kf_err'] for r in rows], 's-', label='Kalman filtered (causal compression)')
    ax.plot(ks, [r['pr_err'] for r in rows], '^--', label='Kalman 1-step (predictive)')
    ax.set_xlabel('state dim k')
    ax.set_ylabel('relative Frobenius error')
    ax.set_title(f'{ticker} — CWT reconstruction vs LDS state dim')
    ax.axhline(0.05, color='gray', ls=':', alpha=0.6, label='5% (near-lossless)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ticker}-kalman-cwt.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Kalman/LDS compression diagnostic for the causal CWT panel.')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    add_single_ticker_loader_args(parser)
    parser.add_argument('--lookback', type=int, default=90,
                        help='Causal z-norm window for the CWT (default 90).')
    parser.add_argument('--raw-close', action='store_true',
                        help='Feed raw close instead of log-returns to the CWT.')
    parser.add_argument('--ks', type=int, nargs='+', default=DEFAULT_KS,
                        help='State dims to sweep (default: 1..13).')
    add_save_args(parser)
    args = parser.parse_args()

    warmup = KERNEL_HALF_EXTENT * max(ALL_SCALES) + args.lookback
    for ticker in args.tickers:
        try:
            series = load_prices(
                ticker, stooq_dir=args.stooq_dir, kaggle_dir=args.kaggle_dir,
                start=args.start, end=args.end)
        except (KeyError, RuntimeError) as exc:
            print(f'Skipping {ticker}: {exc}')
            continue
        prices = series.values.astype(np.float64)
        Y = cwt_panel(prices, lookback=args.lookback,
                      use_log_returns=not args.raw_close)
        Y = Y[warmup:]
        if Y.shape[0] < 500:
            print(f'Skipping {ticker}: only {Y.shape[0]} bars after warm-up.')
            continue
        rows = evaluate(Y, args.ks)
        _print_table(ticker, rows)
        if args.save:
            print(f'Saved {_plot(ticker, rows, args.output_dir)}')


if __name__ == '__main__':
    main()
