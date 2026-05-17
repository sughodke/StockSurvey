"""RNN compression of the causal CWT panel — nonlinear counterpart to
`kalman_cwt`.

Diagnostic, not a strategy. It answers the falsifiable question the
Kalman diagnostic gated: *was the no-low-rank result a property of the
CWT panel, or just of the linear-Gaussian model?* Everything is held
identical to `kalman_cwt` — same `cwt_panel`, same warm-up, same
`_rel_err` metric, same `k` sweep — and only the recurrence is swapped
linear `A` → a GRU. The emission stays linear (a single readout matrix,
the analogue of Kalman `C`) so the *only* moving part is recurrence
nonlinearity. That isolates the comparison.

Model (GRU recurrent autoencoder over `y_t in R^p`, `p = 13`):

    h_t = GRU(h_{t-1}, x_t)            h_t in R^k, k <= p   (nonlinear)
    y_hat_t = h_t W_o + b_o                                 (linear emission)

`k < p` is the bottleneck: reconstructing `y_t in R^13` from `h_t in
R^k` forces compression exactly as Kalman reconstructs from an
`R^k` latent.

Two trained objectives mirror the two scored Kalman columns:

  * `recon`   — autoencode `y_t` from `h_t` (causal, `h_t` summarizes
                `y_{:t}`): the near-lossless-compression number.
  * `predict` — one-step-ahead `y_t` from `h_t` built on `y_{:t-1}`:
                the predictive number.

A third, separately-flagged arm (`--seq-bottleneck`) does explicit
*length* compression rather than per-bar width compression: the GRU
encodes an `L`-bar window down to only its final hidden state
`h_L in R^k`, and a linear decoder reconstructs the *entire* `(L, p)`
window from that single vector. Recon error vs `k` (and vs `L`, by
re-running with a different `--seq-len`) answers "how many bars can
one fixed `k`-vector hold". Matched linear baseline: PCA on flattened
windows.

The linear PCA batch baseline (instant SVD) is printed in the same
table so "does nonlinearity buy back the rank the linear model
needed?" is readable in one row. Hand-rolled numpy + Adam + truncated
BPTT — keeps `apps/notebook` numpy-only (tinygrad stays in
factor/replay) and single-ticker keeps it well under the local budget.

    uv run python -m ss_notebook.rnn_cwt --stooq-dir ./StooqData AAPL
    uv run ss-rnn-cwt --stooq-dir ./StooqData --save AAPL
    uv run ss-rnn-cwt --stooq-dir ./StooqData --seq-bottleneck AAPL
    uv run ss-rnn-cwt --stooq-dir ./StooqData --vs GLD --seq-bottleneck AAPL
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ss_cli import add_save_args, add_single_ticker_loader_args
from ss_features import load_prices
from ss_wavelets import ALL_SCALES, KERNEL_HALF_EXTENT
from ss_notebook.kalman_cwt import DEFAULT_KS, _rel_err, cwt_panel


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _init_gru(p: int, k: int, rng: np.random.Generator) -> dict:
    s_in = 1.0 / np.sqrt(p)
    s_h = 1.0 / np.sqrt(k)
    g = lambda a, b, s: rng.uniform(-s, s, (a, b))  # noqa: E731
    return {
        'Wz': g(p, k, s_in), 'Wr': g(p, k, s_in), 'Wn': g(p, k, s_in),
        'Uz': g(k, k, s_h), 'Ur': g(k, k, s_h), 'Un': g(k, k, s_h),
        'bz': np.zeros(k), 'br': np.zeros(k), 'bn': np.zeros(k),
        'Wo': g(k, p, s_h), 'bo': np.zeros(p),
    }


def _gru_scan(P: dict, X: np.ndarray, h0: np.ndarray) -> tuple:
    """Run the GRU recurrence. X: (B, L, p). Returns (H, cache) with
    H of shape (L+1, B, k) — shared by every arm so the recurrence and
    its BPTT are defined once."""
    B, L, _ = X.shape
    k = P['bz'].shape[0]
    H = np.empty((L + 1, B, k))
    H[0] = h0
    cache = []
    for t in range(L):
        x = X[:, t, :]
        hp = H[t]
        z = _sigmoid(x @ P['Wz'] + hp @ P['Uz'] + P['bz'])
        r = _sigmoid(x @ P['Wr'] + hp @ P['Ur'] + P['br'])
        n = np.tanh(x @ P['Wn'] + (r * hp) @ P['Un'] + P['bn'])
        h = (1.0 - z) * n + z * hp
        H[t + 1] = h
        cache.append((x, hp, z, r, n))
    return H, cache


def _forward(P: dict, X: np.ndarray, h0: np.ndarray) -> tuple:
    """X: (B, L, p). Returns Yhat (B,L,p) and a cache for BPTT."""
    H, cache = _gru_scan(P, X, h0)
    Yhat = H[1:].transpose(1, 0, 2) @ P['Wo'] + P['bo']
    return Yhat, (H, cache)


def _backward(P: dict, X: np.ndarray, Y: np.ndarray, Yhat: np.ndarray,
              fc: tuple) -> tuple[dict, float]:
    H, cache = fc
    B, L, p = X.shape
    k = P['bz'].shape[0]
    G = {kk: np.zeros_like(v) for kk, v in P.items()}
    diff = (Yhat - Y) / (B * L)
    loss = float(np.sum((Yhat - Y) ** 2) / (B * L))

    Hs = H[1:].transpose(1, 0, 2)              # (B,L,k)
    G['Wo'] = np.einsum('blk,blp->kp', Hs, diff)
    G['bo'] = diff.sum(axis=(0, 1))
    dH_from_out = diff @ P['Wo'].T             # (B,L,k)

    dh_next = np.zeros((B, k))
    for t in range(L - 1, -1, -1):
        x, hp, z, r, n = cache[t]
        dh = dH_from_out[:, t, :] + dh_next
        dn = dh * (1.0 - z)
        dz = dh * (hp - n)
        dhp = dh * z
        dn_raw = dn * (1.0 - n ** 2)
        G['Wn'] += x.T @ dn_raw
        G['Un'] += (r * hp).T @ dn_raw
        G['bn'] += dn_raw.sum(axis=0)
        d_rh = dn_raw @ P['Un'].T
        dr = d_rh * hp
        dhp += d_rh * r
        dz_raw = dz * z * (1.0 - z)
        G['Wz'] += x.T @ dz_raw
        G['Uz'] += hp.T @ dz_raw
        G['bz'] += dz_raw.sum(axis=0)
        dhp += dz_raw @ P['Uz'].T
        dr_raw = dr * r * (1.0 - r)
        G['Wr'] += x.T @ dr_raw
        G['Ur'] += hp.T @ dr_raw
        G['br'] += dr_raw.sum(axis=0)
        dhp += dr_raw @ P['Ur'].T
        dh_next = dhp
    return G, loss


def _windows(A: np.ndarray, L: int, stride: int) -> np.ndarray:
    """Overlapping (stride < L) length-L windows — more gradient steps
    per epoch than disjoint chunking, which is what lets the k=p model
    actually reach the near-lossless anchor instead of underfitting."""
    starts = range(0, A.shape[0] - L + 1, stride)
    return np.stack([A[s:s + L] for s in starts])


def _train(Y: np.ndarray, k: int, *, predict: bool, seed: int,
           epochs: int, L: int, B: int, lr: float, stride: int) -> dict:
    rng = np.random.default_rng(seed)
    p = Y.shape[1]
    Xs = Y[:-1] if predict else Y
    Ts = Y[1:] if predict else Y
    Xw = _windows(Xs, L, stride)
    Tw = _windows(Ts, L, stride)
    nb = Xw.shape[0]

    P = _init_gru(p, k, rng)
    m = {kk: np.zeros_like(v) for kk, v in P.items()}
    v = {kk: np.zeros_like(v) for kk, v in P.items()}
    step = 0
    best = np.inf
    patience = 0
    for ep in range(epochs):
        order = rng.permutation(nb)
        ep_loss = 0.0
        for i in range(0, nb, B):
            idx = order[i:i + B]
            Xb, Tb = Xw[idx], Tw[idx]
            h0 = np.zeros((Xb.shape[0], k))
            Yhat, fc = _forward(P, Xb, h0)
            G, loss = _backward(P, Xb, Tb, Yhat, fc)
            ep_loss += loss * Xb.shape[0]
            step += 1
            for kk in P:
                m[kk] = 0.9 * m[kk] + 0.1 * G[kk]
                v[kk] = 0.999 * v[kk] + 0.001 * G[kk] ** 2
                mh = m[kk] / (1.0 - 0.9 ** step)
                vh = v[kk] / (1.0 - 0.999 ** step)
                P[kk] -= lr * mh / (np.sqrt(vh) + 1e-8)
        ep_loss /= nb
        if ep_loss < best - 1e-6:
            best, patience = ep_loss, 0
        else:
            patience += 1
            if patience >= 40:
                break
    return P


def _infer(P: dict, Y: np.ndarray, predict: bool) -> np.ndarray:
    """Single full-sequence forward, hidden carried over all history."""
    Xs = (Y[:-1] if predict else Y)[None]      # (1, T, p)
    Yhat, _ = _forward(P, Xs, np.zeros((1, P['bz'].shape[0])))
    return Yhat[0]


def _pca_baseline(Ys: np.ndarray, k: int) -> np.ndarray:
    _, _, Vt = np.linalg.svd(Ys, full_matrices=False)
    C = Vt[:k].T
    return (Ys @ C) @ C.T


# --- Sequence-bottleneck arm -------------------------------------------------
# Explicit length compression: the GRU encodes an L-bar window down to
# ONLY its final hidden state h_L in R^k (the whole L*p window lives in
# k numbers), then a linear decoder reconstructs the entire (L, p)
# window from that single vector. This is the harsh test — recon error
# vs k *and* vs L answers "how many bars can one fixed k-vector hold".
# Linear emission again, so the only nonlinearity is the encoder
# recurrence; the matched linear baseline is PCA on flattened windows.

GRU_KEYS = ['Wz', 'Wr', 'Wn', 'Uz', 'Ur', 'Un', 'bz', 'br', 'bn']


def _backward_enc(P: dict, cache: list, dhL: np.ndarray) -> dict:
    """BPTT through the encoder only — the loss touches the GRU solely
    via the final hidden state, so the output gradient `dhL` enters at
    the last step and propagates back purely through the recurrence."""
    L = len(cache)
    B, k = dhL.shape
    G = {kk: np.zeros_like(P[kk]) for kk in GRU_KEYS}
    dh_next = np.zeros((B, k))
    for t in range(L - 1, -1, -1):
        x, hp, z, r, n = cache[t]
        dh = dh_next + (dhL if t == L - 1 else 0.0)
        dn = dh * (1.0 - z)
        dz = dh * (hp - n)
        dhp = dh * z
        dn_raw = dn * (1.0 - n ** 2)
        G['Wn'] += x.T @ dn_raw
        G['Un'] += (r * hp).T @ dn_raw
        G['bn'] += dn_raw.sum(axis=0)
        d_rh = dn_raw @ P['Un'].T
        dr = d_rh * hp
        dhp += d_rh * r
        dz_raw = dz * z * (1.0 - z)
        G['Wz'] += x.T @ dz_raw
        G['Uz'] += hp.T @ dz_raw
        G['bz'] += dz_raw.sum(axis=0)
        dhp += dz_raw @ P['Uz'].T
        dr_raw = dr * r * (1.0 - r)
        G['Wr'] += x.T @ dr_raw
        G['Ur'] += hp.T @ dr_raw
        G['br'] += dr_raw.sum(axis=0)
        dhp += dr_raw @ P['Ur'].T
        dh_next = dhp
    return G


def _train_seqbottleneck(Y: np.ndarray, k: int, *, L: int, B: int,
                         lr: float, epochs: int, stride: int,
                         seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    p = Y.shape[1]
    W = _windows(Y, L, stride)                       # (nb, L, p)
    nb = W.shape[0]
    P = _init_gru(p, k, rng)
    s = 1.0 / np.sqrt(k)
    D = {'Wd': rng.uniform(-s, s, (k, L * p)), 'bd': np.zeros(L * p)}
    params = GRU_KEYS + ['Wd', 'bd']
    ref = {**{kk: P for kk in GRU_KEYS}, 'Wd': D, 'bd': D}
    m = {kk: np.zeros_like(ref[kk][kk]) for kk in params}
    v = {kk: np.zeros_like(ref[kk][kk]) for kk in params}
    step = 0
    best = np.inf
    patience = 0
    for _ in range(epochs):
        order = rng.permutation(nb)
        ep_loss = 0.0
        for i in range(0, nb, B):
            idx = order[i:i + B]
            Xb = W[idx]
            bsz = Xb.shape[0]
            H, cache = _gru_scan(P, Xb, np.zeros((bsz, k)))
            hL = H[L]                                # (bsz, k)
            flat = hL @ D['Wd'] + D['bd']            # (bsz, L*p)
            tgt = Xb.reshape(bsz, L * p)
            resid = (flat - tgt) / bsz
            ep_loss += float(np.sum((flat - tgt) ** 2) / bsz) * bsz
            Genc = _backward_enc(P, cache, resid @ D['Wd'].T)
            G = {**Genc, 'Wd': hL.T @ resid, 'bd': resid.sum(axis=0)}
            step += 1
            for kk in params:
                m[kk] = 0.9 * m[kk] + 0.1 * G[kk]
                v[kk] = 0.999 * v[kk] + 0.001 * G[kk] ** 2
                mh = m[kk] / (1.0 - 0.9 ** step)
                vh = v[kk] / (1.0 - 0.999 ** step)
                ref[kk][kk] -= lr * mh / (np.sqrt(vh) + 1e-8)
        ep_loss /= nb
        if ep_loss < best - 1e-6:
            best, patience = ep_loss, 0
        else:
            patience += 1
            if patience >= 40:
                break
    return P, D


def _infer_seqbottleneck(P: dict, D: dict, Y: np.ndarray,
                         L: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping L-blocks; reconstruct each from its own h_L."""
    p = Y.shape[1]
    k = P['bz'].shape[0]
    n = (Y.shape[0] // L) * L
    W = Y[:n].reshape(-1, L, p)
    H, _ = _gru_scan(P, W, np.zeros((W.shape[0], k)))
    flat = H[L] @ D['Wd'] + D['bd']
    return flat.reshape(-1, L, p), W


def _pca_flat_baseline(Ys: np.ndarray, L: int, k: int) -> np.ndarray:
    """Optimal linear k-compression of the whole flattened window."""
    p = Ys.shape[1]
    n = (Ys.shape[0] // L) * L
    F = Ys[:n].reshape(-1, L * p)
    mu = F.mean(axis=0)
    Fc = F - mu
    _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
    C = Vt[:k].T
    return ((Fc @ C) @ C.T + mu).reshape(-1, L, p)


def evaluate_seqbottleneck(Y: np.ndarray, ks: list[int], *, L: int,
                           seed: int, epochs: int, B: int, lr: float,
                           stride: int) -> list[dict]:
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0) + 1e-12
    Ys = (Y - mu) / sd
    p = Y.shape[1]
    n = (Y.shape[0] // L) * L
    Yw = Y[:n].reshape(-1, L, p)                     # original scale
    rows = []
    for k in ks:
        pca = _pca_flat_baseline(Ys, L, k) * sd + mu
        e_pca, r2_pca = _rel_err(Yw, pca)
        P, D = _train_seqbottleneck(Ys, k, L=L, B=B, lr=lr,
                                    epochs=epochs, stride=stride,
                                    seed=seed)
        rec, _ = _infer_seqbottleneck(P, D, Ys, L)
        e_gru, r2_gru = _rel_err(Yw, rec * sd + mu)
        rows.append({'k': k, 'in': L * p, 'ratio': (L * p) / k,
                     'pca_err': e_pca, 'pca_r2': r2_pca,
                     'gru_err': e_gru, 'gru_r2': r2_gru})
        print(f'  ...k={k:>2d} done '
              f'(L*p={L * p} → {k}, {(L * p) / k:.1f}× | '
              f'pca-flat {r2_pca:+.3f} | gru-seq {r2_gru:+.3f})')
    return rows


def _print_table_sb(ticker: str, L: int, rows: list[dict]) -> None:
    print(f'\n{ticker}: sequence-bottleneck — one h_L in R^k '
          f'reconstructs a whole {L}-bar × {len(ALL_SCALES)}-scale '
          f'window ({rows[0]["in"]} numbers)')
    print('  k | compress | PCA-flat (linear) | GRU seq-bottleneck')
    print('    |  ratio   | relerr     R2     | relerr     R2')
    print('  ' + '-' * 58)
    for r in rows:
        print(f'  {r["k"]:>2d} | {r["ratio"]:>6.1f}×  '
              f'| {r["pca_err"]:.4f} {r["pca_r2"]:+.3f}   '
              f'| {r["gru_err"]:.4f} {r["gru_r2"]:+.3f}')


def _plot_sb(ticker: str, L: int, rows: list[dict], out_dir: str) -> str:
    ks = [r['k'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [r['pca_err'] for r in rows], 'o--', color='gray',
            label='PCA on flattened windows (linear upper bound)')
    ax.plot(ks, [r['gru_err'] for r in rows], 's-',
            label='GRU seq-bottleneck (one h_L → whole window)')
    ax.set_xlabel('state dim k')
    ax.set_ylabel('relative Frobenius error')
    ax.set_title(f'{ticker} — whole {L}-bar window from one k-vector')
    ax.axhline(0.05, color='gray', ls=':', alpha=0.6,
               label='5% (near-lossless)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ticker}-rnn-cwt-seqbottleneck.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _persistence(Y: np.ndarray) -> tuple[float, float]:
    """Lag-1 baseline for the predict arm: y_hat_{t+1} = y_t. The CWT
    is a slowly-varying wavelet transform, so this is a strong naive
    forecast — the predict latent only "learns" the residual above it."""
    return _rel_err(Y[1:], Y[:-1])


def evaluate(Y: np.ndarray, ks: list[int], *, seed: int, epochs: int,
             L: int, B: int, lr: float, stride: int) -> list[dict]:
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0) + 1e-12
    Ys = (Y - mu) / sd
    pers_err, pers_r2 = _persistence(Y)
    rows = []
    for k in ks:
        if k > Y.shape[1]:
            continue
        e_pca, r2_pca = _rel_err(Y, _pca_baseline(Ys, k) * sd + mu)

        Pr = _train(Ys, k, predict=False, seed=seed,
                    epochs=epochs, L=L, B=B, lr=lr, stride=stride)
        rec = _infer(Pr, Ys, predict=False) * sd + mu
        e_rec, r2_rec = _rel_err(Y, rec)

        Pp = _train(Ys, k, predict=True, seed=seed,
                    epochs=epochs, L=L, B=B, lr=lr, stride=stride)
        prd = _infer(Pp, Ys, predict=True) * sd + mu
        e_prd, r2_prd = _rel_err(Y[1:], prd)

        rows.append({'k': k, 'pca_err': e_pca, 'pca_r2': r2_pca,
                     'rec_err': e_rec, 'rec_r2': r2_rec,
                     'prd_err': e_prd, 'prd_r2': r2_prd,
                     'pers_r2': pers_r2,
                     'prd_minus_pers': r2_prd - pers_r2})
        print(f'  ...k={k:>2d} done '
              f'(pca {r2_pca:+.3f} | rnn {r2_rec:+.3f} | pred {r2_prd:+.3f})')
    return rows


def _print_table(ticker: str, rows: list[dict]) -> None:
    print(f'\n{ticker}: GRU compression of causal CWT '
          f'(p={len(ALL_SCALES)} scales)')
    if rows:
        print(f'  persistence (lag-1, y_hat_t+1 = y_t): '
              f'R2={rows[0]["pers_r2"]:+.3f}  '
              f'— predict latent must beat THIS to have learned anything')
    print('  k | PCA(linear)  | RNN recon(causal) | RNN 1-step(predict) '
          '| pred−persist')
    print('    | relerr   R2  | relerr     R2     | relerr     R2      '
          '|   ΔR2')
    print('  ' + '-' * 74)
    for r in rows:
        print(f'  {r["k"]:>2d} | {r["pca_err"]:.4f} {r["pca_r2"]:+.3f} '
              f'| {r["rec_err"]:.4f} {r["rec_r2"]:+.3f}   '
              f'| {r["prd_err"]:.4f} {r["prd_r2"]:+.3f}     '
              f'| {r["prd_minus_pers"]:+.3f}')


def _plot(ticker: str, rows: list[dict], out_dir: str) -> str:
    ks = [r['k'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [r['pca_err'] for r in rows], 'o--', color='gray',
            label='PCA (linear batch baseline)')
    ax.plot(ks, [r['rec_err'] for r in rows], 's-',
            label='GRU recon (causal, nonlinear compression)')
    ax.plot(ks, [r['prd_err'] for r in rows], '^-',
            label='GRU 1-step (predictive)')
    ax.set_xlabel('state dim k')
    ax.set_ylabel('relative Frobenius error')
    ax.set_title(f'{ticker} — CWT reconstruction vs RNN state dim')
    ax.axhline(0.05, color='gray', ls=':', alpha=0.6,
               label='5% (near-lossless)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{ticker}-rnn-cwt.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='GRU compression diagnostic for the causal CWT panel.')
    parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    add_single_ticker_loader_args(parser)
    parser.add_argument('--lookback', type=int, default=90)
    parser.add_argument('--vs', metavar='TICKER', default=None,
                        help='Denominate price in units of this ticker '
                             "before the CWT (e.g. --vs GLD = Apple "
                             'priced in gold). Inner-aligned on dates.')
    parser.add_argument('--raw-close', action='store_true')
    parser.add_argument('--ks', type=int, nargs='+', default=DEFAULT_KS)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--stride', type=int, default=8,
                        help='TBPTT window stride; < seq-len overlaps for '
                             'more gradient steps (anchor-fit critical).')
    parser.add_argument('--max-bars', type=int, default=4000,
                        help='Cap post-warmup bars to bound local runtime.')
    parser.add_argument('--seq-bottleneck', action='store_true',
                        help='Run ONLY the length-compression arm: encode '
                             'a --seq-len window to one h_L in R^k and '
                             'reconstruct the whole window from it.')
    parser.add_argument('--seed', type=int, default=0)
    add_save_args(parser)
    args = parser.parse_args()

    warmup = KERNEL_HALF_EXTENT * max(ALL_SCALES) + args.lookback
    for ticker in args.tickers:
        try:
            series = load_prices(
                ticker, stooq_dir=args.stooq_dir, kaggle_dir=args.kaggle_dir,
                start=args.start, end=args.end)
            if args.vs:
                denom = load_prices(
                    args.vs, stooq_dir=args.stooq_dir,
                    kaggle_dir=args.kaggle_dir,
                    start=args.start, end=args.end)
                series, denom = series.align(denom, join='inner')
                series = series / denom
        except (KeyError, RuntimeError) as exc:
            print(f'Skipping {ticker}: {exc}')
            continue
        label = ticker if not args.vs else f'{ticker}-over-{args.vs}'
        prices = series.values.astype(np.float64)
        Y = cwt_panel(prices, lookback=args.lookback,
                      use_log_returns=not args.raw_close)[warmup:]
        if Y.shape[0] < 500:
            print(f'Skipping {label}: only {Y.shape[0]} bars after warm-up.')
            continue
        Y = Y[-args.max_bars:]
        if args.seq_bottleneck:
            print(f'{label}: training seq-bottleneck sweep on '
                  f'{Y.shape[0]} bars (L={args.seq_len})...')
            rows = evaluate_seqbottleneck(
                Y, args.ks, L=args.seq_len, seed=args.seed,
                epochs=args.epochs, B=args.batch, lr=args.lr,
                stride=args.stride)
            _print_table_sb(label, args.seq_len, rows)
            if args.save:
                print(f'Saved {_plot_sb(label, args.seq_len, rows, args.output_dir)}')
            continue
        print(f'{label}: training GRU sweep on {Y.shape[0]} bars '
              f'(this is the slow part)...')
        rows = evaluate(Y, args.ks, seed=args.seed, epochs=args.epochs,
                        L=args.seq_len, B=args.batch, lr=args.lr,
                        stride=args.stride)
        _print_table(label, rows)
        if args.save:
            print(f'Saved {_plot(label, rows, args.output_dir)}')


if __name__ == '__main__':
    main()
