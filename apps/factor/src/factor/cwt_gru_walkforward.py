"""Return-coupled recurrent CWT embedding — rank-IC-trained GRU walk-forward.

The one live path out of the
`apps/docs/docs/findings/cwt-recursive-compression.md` arc. That arc
closed *reconstruction* and *CWT-self-prediction* negative; the only
untested target is forward returns. Here the GRU recurrence is **in the
trained graph** (not a frozen reservoir): per walk-forward window a
fresh GRU encoder over the last `L` causal-CWT vectors plus a linear
head are trained jointly against `pearson_rank_ic` at horizon
`rebal_days`, then frozen for that window's val.

Why a new module and not `train_scorer_walkforward`: that path
*precomputes a frozen backbone latent* and trains only the head. A
random frozen GRU there is a reservoir, not a return-coupled embedding —
it cannot test the hypothesis ("does a *trained* recurrent CWT state
extract a compact predictive statistic"). This module keeps the encoder
in the autograd graph and BPTTs through the `L`-step recurrence, per
window, leak-free.

Leak-freedom invariants (asserted):
  * `causal_cwt` is strictly causal — `output[t]` depends on
    `input[:t+1]`. The CWT panel carries no look-ahead.
  * The `L`-bar GRU input window for rebal bar `i` is `panel[i-L+1 :
    i+1]` — entirely at or before `i`.
  * Forward returns are strictly future (`forward_log_returns`) and
    masked out where the window doesn't fit.
  * Per window: fresh seed-derived GRU+head init, trained on the train
    rebal slice only, frozen for val. No cross-window weight carry.
  * The mandatory per-scale standardisation (without it the CWT panel
    is dominated by scale=126's magnitude — see the diagnostic) is fit
    on the **train rebal slice only** and applied to val. A global
    z-score would leak val-period scale magnitudes into train
    normalisation.

See `apps/docs/docs/TODO/factor-cwt-return-coupled.md` for the
pre-registered hypothesis + kill criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW

from ss_features import TickerData, block_windows, load_prices
from ss_wavelets import ALL_SCALES, KERNEL_HALF_EXTENT, causal_cwt
from factor.data import align_tickers, forward_log_returns
from factor.objectives import block_ir_vs_ew, block_sharpe, pearson_rank_ic
from factor.scorers import apply_linear, init_linear


# Same scalogram convention as the cwt-recursive-compression diagnostic
# so `k` vs `p = len(ALL_SCALES)` stays apples-to-apples with its
# recon/predict tables.
DEFAULT_LOOKBACK: int = 90
DEFAULT_SEQ_LEN: int = 32
N_SCALES: int = len(ALL_SCALES)  # p = 13

# Bars of reduced wavelet support to drop before a CWT vector is
# trustworthy: the largest Ricker kernel reaches back
# `KERNEL_HALF_EXTENT * max(scale)` and the rolling z-norm needs
# `lookback`. Matches the diagnostic's `3*126 + 90 = 468`.
def _warmup_floor(lookback: int) -> int:
    return KERNEL_HALF_EXTENT * max(ALL_SCALES) + lookback


def build_cwt_panel(
    prices: np.ndarray, *, lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """`(T, p)` causal-CWT-of-log-returns panel + `(T,)` valid mask.

    Input series to the CWT is the per-bar log return (the diagnostic's
    "causal Ricker CWT of log-returns input"); `causal_cwt` rolling-
    z-norms that series internally with `lookback`, so absolute price
    scale is irrelevant. `valid[t]` is False for the warmup region
    where the kernel/​z-norm support is incomplete.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    if T == 0:
        return (np.empty((0, N_SCALES), dtype=np.float32),
                np.empty((0,), dtype=bool))
    log_p = np.log(np.maximum(prices, 1e-12))
    log_ret = np.zeros(T, dtype=np.float64)
    log_ret[1:] = log_p[1:] - log_p[:-1]
    # causal_cwt wants `(n_dates, n_tickers)`; one ticker here.
    coeffs = causal_cwt(log_ret[:, None], list(ALL_SCALES), lookback)
    # (n_scales, T, 1) -> (T, n_scales)
    panel = coeffs[:, :, 0].T.astype(np.float32)
    valid = np.zeros(T, dtype=bool)
    floor = _warmup_floor(lookback)
    if T > floor:
        valid[floor:] = True
    return panel, valid


def load_ticker_cwt(
    name: str, *,
    stooq_dir: str | None = None,
    kaggle_dir: str | None = None,
    use_yahoo: bool = False,
    start: str | None = None,
    end: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
) -> TickerData:
    """Load one ticker and build its causal-CWT panel as `TickerData`.

    `features` is `(T, p)` (K=1, F=p) so it composes with
    `align_tickers(K=1, F=N_SCALES)`. The GRU input *sequence* is
    reconstructed from the daily panel inside the trainer (the last `L`
    rows ending at each rebal bar) — the panel is stored daily, not
    pre-lagged, to avoid an `(T, L*p)` blow-up at universe scale.
    """
    series = load_prices(
        name, stooq_dir=stooq_dir, kaggle_dir=kaggle_dir,
        use_yahoo=use_yahoo, start=start, end=end)
    prices = series.values.astype(np.float64)
    dates = np.asarray(series.index)
    panel, valid = build_cwt_panel(prices, lookback=lookback)
    return TickerData(
        name=name, prices=prices, dates=dates,
        features=panel, targets={}, valid=valid,
    )


# ---------------------------------------------------------------------------
# GRU encoder (tinygrad). Same gate algebra as the numpy reference in
# `apps/notebook/src/ss_notebook/rnn_cwt.py::_gru_scan` so the
# return-coupled result is comparable to the reconstruction diagnostic.
# ---------------------------------------------------------------------------
_GRU_KEYS = ('Wz', 'Wr', 'Wn', 'Uz', 'Ur', 'Un', 'bz', 'br', 'bn')


def init_cwt_gru(
    rng: np.random.Generator, p: int, k: int,
) -> dict[str, Tensor]:
    """GRU params, matched-uniform init (`±1/sqrt(fan_in)`) per
    `rnn_cwt._init_gru` so the encoder family is identical to the
    diagnostic's, only the training target differs."""
    s_in = 1.0 / np.sqrt(p)
    s_h = 1.0 / np.sqrt(k)

    def u(a: int, b: int, s: float) -> Tensor:
        return Tensor(rng.uniform(-s, s, (a, b)).astype(np.float32),
                      requires_grad=True)

    def z(n: int) -> Tensor:
        return Tensor(np.zeros(n, dtype=np.float32), requires_grad=True)

    return {
        'Wz': u(p, k, s_in), 'Wr': u(p, k, s_in), 'Wn': u(p, k, s_in),
        'Uz': u(k, k, s_h), 'Ur': u(k, k, s_h), 'Un': u(k, k, s_h),
        'bz': z(k), 'br': z(k), 'bn': z(k),
    }


def gru_final_hidden(params: dict[str, Tensor], X: Tensor) -> Tensor:
    """`X` shape `(M, L, p)` → final hidden state `h_L` shape `(M, k)`.

    The recurrence is unrolled in Python over the `L` axis; tinygrad
    builds the static BPTT graph (`L` is fixed across steps so the JIT
    kernel cache warms once). Gate algebra:
        z = σ(x Wz + h Uz + bz)
        r = σ(x Wr + h Ur + br)
        n = tanh(x Wn + (r⊙h) Un + bn)
        h = (1−z)⊙n + z⊙h
    """
    M = X.shape[0]
    L = X.shape[1]
    k = params['bz'].shape[0]
    h = Tensor.zeros(M, k)
    for t in range(L):
        x = X[:, t, :]
        z = (x @ params['Wz'] + h @ params['Uz'] + params['bz']).sigmoid()
        r = (x @ params['Wr'] + h @ params['Ur'] + params['br']).sigmoid()
        n = (x @ params['Wn'] + (r * h) @ params['Un'] + params['bn']).tanh()
        h = (1.0 - z) * n + z * h
    return h


@dataclass(frozen=True)
class CwtGruWindow:
    """One walk-forward window's metrics for a fixed `k`."""
    window_idx: int
    train_block_start: int
    train_block_end: int
    val_block_start: int
    val_block_end: int
    train_ic: float
    val_ic: float
    train_sharpe: float
    val_sharpe: float
    val_ir_vs_ew: float
    n_train_bars: int
    n_val_bars: int
    val_start_date: str = ''


@dataclass
class CwtGruResult:
    """Aggregate of `train_cwt_gru_walkforward` for one `k`."""
    k: int
    seq_len: int
    lookback: int
    rebal_days: int
    n_steps: int
    learning_rate: float
    weight_decay: float
    universe_size: int
    windows: list[CwtGruWindow] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def mean_val_ic(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ic for w in self.windows]))

    @property
    def positive_val_ic_fraction(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ic > 0 for w in self.windows]))

    @property
    def mean_val_sharpe(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_sharpe for w in self.windows]))

    @property
    def mean_val_ir_vs_ew(self) -> float:
        if not self.windows:
            return float('nan')
        return float(np.mean([w.val_ir_vs_ew for w in self.windows]))


def _gather_rebal_windows(
    panel: np.ndarray, rebal_idx: np.ndarray, seq_len: int,
) -> np.ndarray:
    """`(D, N, p)` daily panel → `(Dp, N, L, p)` GRU input sequences.

    For rebal bar `i` the sequence is `panel[i-L+1 : i+1]` (causal —
    every row ≤ `i`). Caller guarantees `rebal_idx` only holds bars with
    `i - L + 1 >= 0`.
    """
    D, N, p = panel.shape
    Dp = len(rebal_idx)
    out = np.empty((Dp, N, seq_len, p), dtype=np.float32)
    for b, i in enumerate(rebal_idx):
        # (L, N, p) -> (N, L, p)
        out[b] = np.transpose(panel[i - seq_len + 1: i + 1], (1, 0, 2))
    return out


def train_cwt_gru_walkforward(
    tickers: list[TickerData], *,
    k: int,
    rebal_days: int = 20,
    train_window_blocks: int = 63,
    val_window_blocks: int = 39,
    step_window_blocks: int = 39,
    seq_len: int = DEFAULT_SEQ_LEN,
    lookback: int = DEFAULT_LOOKBACK,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    commission_bps: float = 10.0,
    seed: int = 0,
    verbose: bool = True,
) -> CwtGruResult:
    """Leak-free walk-forward of a rank-IC-trained GRU-over-CWT embedding.

    `tickers` must carry the daily causal-CWT panel in `.features`
    (`(T, p)`), e.g. from `load_ticker_cwt`. Per window a fresh GRU
    (hidden `k`) + linear head is trained jointly on the train rebal
    slice against `-pearson_rank_ic`, then frozen for val. Hyperparams
    are fixed (no val-based tuning) per the pre-registration.
    """
    if k < 1:
        raise ValueError(f'k={k} must be >= 1')
    aligned = align_tickers(tickers, K=1, F=N_SCALES)
    D = len(aligned.dates)
    N = len(aligned.names)
    panel = aligned.features[:, :, 0, :].astype(np.float32)  # (D, N, p)

    # Rebal grid: every `rebal_days` bars where both the forward-return
    # window AND the trailing L-window fit.
    rebal_idx = np.arange(0, D, rebal_days, dtype=np.int64)
    rebal_idx = rebal_idx[(rebal_idx + rebal_days < D)
                          & (rebal_idx - seq_len + 1 >= 0)]
    if len(rebal_idx) < train_window_blocks + val_window_blocks:
        raise ValueError(
            f'only {len(rebal_idx)} rebal blocks fit '
            f'(D={D}, rebal_days={rebal_days}, seq_len={seq_len}); '
            f'need >= {train_window_blocks + val_window_blocks}')
    Dp = len(rebal_idx)

    # Forward log returns (strictly future, daily grid) → rebal slice.
    fwd_daily = forward_log_returns(aligned.prices, rebal_days=rebal_days)
    fwd_rb = fwd_daily[rebal_idx]  # (Dp, N) f64

    # Block log returns for the eval-only Sharpe / IR columns.
    log_p = np.log(np.maximum(aligned.prices, 1e-12))
    daily_log_ret = np.zeros_like(aligned.prices, dtype=np.float64)
    daily_log_ret[1:] = log_p[1:] - log_p[:-1]
    blr_rb = np.empty((Dp, N), dtype=np.float64)
    for b, i in enumerate(rebal_idx):
        blr_rb[b] = daily_log_ret[i + 1: i + rebal_days + 1].sum(axis=0)

    # GRU input sequences, gathered causally.
    Xrb = _gather_rebal_windows(panel, rebal_idx, seq_len)  # (Dp,N,L,p)

    # Liquidity / validity mask at rebal bars: ticker valid at `i`, the
    # whole L-window finite, and a finite forward return.
    win_finite = np.isfinite(Xrb).all(axis=(2, 3))           # (Dp, N)
    mask_rb = (
        aligned.valid[rebal_idx]
        & np.isfinite(fwd_rb)
        & win_finite
    ).astype(np.float32)

    slices = block_windows(
        Dp, train_window_blocks, val_window_blocks, step_window_blocks)
    if not slices:
        raise ValueError(
            f'no walk-forward windows: {Dp} blocks but each needs '
            f'train+val={train_window_blocks + val_window_blocks}')

    result = CwtGruResult(
        k=k, seq_len=seq_len, lookback=lookback, rebal_days=rebal_days,
        n_steps=n_steps, learning_rate=learning_rate,
        weight_decay=weight_decay, universe_size=N,
    )
    commission_frac = commission_bps / 1e4
    log_temp = Tensor(np.array([0.0], dtype=np.float32), requires_grad=False)

    pbar = tqdm(list(enumerate(slices)), desc=f'cwt-gru k={k}',
                unit='window', disable=not verbose)
    for w_idx, (train_slc, val_slc) in pbar:
        rng = np.random.default_rng(seed + w_idx)

        X_tr = Xrb[train_slc]                # (ntr,N,L,p) f32, may hold NaN
        X_va = Xrb[val_slc]
        # Per-scale standardisation FIT ON TRAIN ONLY (leak-free). The
        # diagnostic flagged this as mandatory — without it scale=126's
        # magnitude dominates the recurrence.
        flat_tr = X_tr.reshape(-1, N_SCALES)
        finite = np.isfinite(flat_tr).all(axis=1)
        ref = flat_tr[finite] if finite.any() else flat_tr
        mu = ref.mean(axis=0).astype(np.float32)
        sd = ref.std(axis=0).astype(np.float32)
        sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

        def _norm(a: np.ndarray) -> np.ndarray:
            return np.nan_to_num((a - mu) / sd, nan=0.0).astype(np.float32)

        Xtr_n = _norm(X_tr)
        Xva_n = _norm(X_va)
        ntr, nva = Xtr_n.shape[0], Xva_n.shape[0]

        gru = init_cwt_gru(rng, N_SCALES, k)
        head = init_linear(rng, k)
        params = {**gru, **head}
        opt = AdamW(list(params.values()), lr=learning_rate,
                    weight_decay=weight_decay)

        fwd_tr_t = Tensor(np.nan_to_num(fwd_rb[train_slc], nan=0.0)
                          .astype(np.float32))
        mask_tr_t = Tensor(mask_rb[train_slc])
        Xtr_t = Tensor(Xtr_n.reshape(ntr * N, seq_len, N_SCALES))

        def _scores(X_t: Tensor, n_bars: int) -> Tensor:
            h = gru_final_hidden(gru, X_t)             # (n_bars*N, k)
            s = apply_linear(head, h)                  # (n_bars*N,)
            return s.reshape(n_bars, N)

        # The 32-step GRU unroll builds a ~300-op autograd graph. Without
        # JIT, tinygrad re-traces + re-schedules it every one of
        # `n_steps` updates × 6 windows × |k| — that is what timed the
        # un-jitted run out at 3h (k=2 ~830s/window, k=4 ~1565s/window).
        # `TinyJit` records the kernel schedule on the 3rd call and
        # replays it (incl. AdamW's in-place param assigns) for every
        # subsequent step. Inputs are full-batch constants; passing the
        # same Tensors each call is the documented "jit a param-updating
        # step" pattern. Fresh TinyJit per window (closure captures this
        # window's params/opt) → ~6 compiles per k, not n_steps×6.
        @TinyJit
        def _train_step(X: Tensor, y: Tensor, m: Tensor) -> Tensor:
            opt.zero_grad()
            loss = -pearson_rank_ic(_scores(X, ntr), y, m)
            loss.backward()
            opt.step()
            return loss.realize()

        Tensor.training = True
        for _ in range(n_steps):
            _train_step(Xtr_t, fwd_tr_t, mask_tr_t)

        Tensor.training = False
        s_tr = _scores(Xtr_t, ntr)
        Xva_t = Tensor(Xva_n.reshape(nva * N, seq_len, N_SCALES))
        s_va = _scores(Xva_t, nva)

        fwd_va_t = Tensor(np.nan_to_num(fwd_rb[val_slc], nan=0.0)
                          .astype(np.float32))
        mask_va_t = Tensor(mask_rb[val_slc])
        blr_tr_t = Tensor(np.nan_to_num(blr_rb[train_slc], nan=0.0)
                          .astype(np.float32))
        blr_va_t = Tensor(np.nan_to_num(blr_rb[val_slc], nan=0.0)
                          .astype(np.float32))

        train_ic = float(pearson_rank_ic(s_tr, fwd_tr_t, mask_tr_t).item())
        val_ic = float(pearson_rank_ic(s_va, fwd_va_t, mask_va_t).item())
        train_sh = float(block_sharpe(
            s_tr, log_temp, blr_tr_t, mask_tr_t,
            rebal_days, commission_frac).item())
        val_sh = float(block_sharpe(
            s_va, log_temp, blr_va_t, mask_va_t,
            rebal_days, commission_frac).item())
        val_ir = float(block_ir_vs_ew(
            s_va, log_temp, blr_va_t, mask_va_t,
            rebal_days, commission_frac).item())

        val_start_idx = int(rebal_idx[val_slc.start])
        val_start_date = str(np.asarray(
            aligned.dates[val_start_idx]).astype('datetime64[D]'))

        if verbose:
            pbar.set_postfix(tr_ic=f'{train_ic:+.4f}',
                             val_ic=f'{val_ic:+.4f}',
                             val_sh=f'{val_sh:+.2f}')

        result.windows.append(CwtGruWindow(
            window_idx=w_idx,
            train_block_start=train_slc.start,
            train_block_end=train_slc.stop,
            val_block_start=val_slc.start,
            val_block_end=val_slc.stop,
            train_ic=train_ic, val_ic=val_ic,
            train_sharpe=train_sh, val_sharpe=val_sh,
            val_ir_vs_ew=val_ir,
            n_train_bars=train_slc.stop - train_slc.start,
            n_val_bars=val_slc.stop - val_slc.start,
            val_start_date=val_start_date,
        ))

    if verbose:
        print(f'cwt-gru k={k}: {result.n_windows} windows, '
              f'mean val IC={result.mean_val_ic:+.4f}, '
              f'pos-frac={result.positive_val_ic_fraction:.2f}, '
              f'mean val Sharpe={result.mean_val_sharpe:+.3f}, '
              f'mean val IR-vs-EW={result.mean_val_ir_vs_ew:+.3f}')
    return result


__all__ = [
    'DEFAULT_LOOKBACK', 'DEFAULT_SEQ_LEN', 'N_SCALES',
    'build_cwt_panel', 'load_ticker_cwt',
    'init_cwt_gru', 'gru_final_hidden',
    'CwtGruWindow', 'CwtGruResult', 'train_cwt_gru_walkforward',
]
