"""End-to-end causality regression for the polar Morlet path.

The packages-level helper has its own causality bound
(`packages/features/tests/test_causal_polar_morlet_matrix.py`); this
test extends the audit through `extract_fingerprints` (uncompressed
and per-channel-block DWT-L1) so a regression in either layer
surfaces here.

Audit recipe: build two synthetic price panels that are bit-identical
on `prices[:T0]` and totally divergent on `prices[T0:]`, run the full
relational pipeline on each, and assert past-bar fingerprints match
modulo fp32 FFT noise.

Noise budget breakdown (with `atol=2e-3` per the matrix-form test):

  - `|c|` block: bit-exact (same as Ricker baseline ~4e-15) —
    `fftconvolve` magnitude is energy-conserving across the
    butterfly chain, so future perturbations don't perturb `|c|` at
    past bars.
  - `cos` / `sin` block: ~1e-3 fp32 FFT phase noise — the complex
    coefficient's phase angle picks up rounding even where its
    magnitude doesn't. This is precision, not information flow:
    the noise floor is independent of what the future perturbation
    actually is, whereas a real leak would scale with the
    perturbation.
  - `g` block: bit-exact — Gaussian on cumulative log-returns has
    the same one-sided convolution structure.
  - L2-normalized fingerprint: ~1e-4 (the per-element averaging
    knocks the unnormalized phase noise down by ~10×).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ss_features import (
    Compression, RELATIONAL_CHANNELS_PER_SCALE,
    causal_polar_morlet_matrix,
)

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


_SCALES = [5, 7, 10, 21, 50, 90]
_LOOKBACK = 120
_FP_WINDOW = 21
_T = 600
_T0 = 400          # split point — bars [0, T0) shared between the two panels
_N = 4
# Generous bound to cover fp32 FFT phase noise at past bars; the
# observed noise floor is ~1e-3 unnormalized, ~1e-4 after L2 znorm.
_ATOL_PANEL = 2e-3
_ATOL_FP = 5e-4


def _two_perturbed_panels() -> tuple[np.ndarray, np.ndarray]:
    """Return `(px_a, px_b)` with `px_a[:T0] == px_b[:T0]` (bit-exact)
    and `px_a[T0:] != px_b[T0:]` (independent random walks).
    """
    rng = np.random.default_rng(0)
    log_steps = rng.normal(0.0005, 0.012, (_T, _N))
    px_a = 100.0 * np.exp(np.cumsum(log_steps, axis=0))
    px_b = px_a.copy()
    rng2 = np.random.default_rng(99)
    px_b[_T0:] = (
        px_b[_T0 - 1:_T0]
        * np.exp(np.cumsum(rng2.normal(0.0, 0.03, (_T - _T0, _N)), axis=0)))
    assert np.array_equal(px_a[:_T0], px_b[:_T0])
    return px_a, px_b


def test_polar_morlet_panel_past_bars_invariant_under_future_perturbation():
    """Matrix-form helper: future-bar perturbation must not move past-bar
    panel values beyond fp32 FFT noise. Per-channel breakdown to make
    failure modes legible — `|c|`/`g` should be bit-exact, only `cos`/
    `sin` carry the phase noise."""
    px_a, px_b = _two_perturbed_panels()
    panel_a = causal_polar_morlet_matrix(px_a, _SCALES, lookback=_LOOKBACK)
    panel_b = causal_polar_morlet_matrix(px_b, _SCALES, lookback=_LOOKBACK)

    diff = np.abs(panel_a[:, :_T0, :] - panel_b[:, :_T0, :])
    assert diff.max() < _ATOL_PANEL, (
        f'polar Morlet panel leaks future data into past bars: '
        f'max diff {diff.max():.2e} (atol={_ATOL_PANEL:.0e})')

    S = len(_SCALES)
    abs_diff_max = diff[0:S].max()
    g_diff_max = diff[3 * S:4 * S].max()
    # `|c|` and `g` come straight out of one-sided causal convolves
    # whose magnitude is energy-conserving — these blocks should
    # match Ricker's bit-exact precision floor (~1e-15 on fp32).
    assert abs_diff_max < 1e-6, (
        f'|c| block past-bar diff {abs_diff_max:.2e} > 1e-6 — '
        f'expected bit-exact; FFT magnitude regression suspected')
    assert g_diff_max < 1e-6, (
        f'g block past-bar diff {g_diff_max:.2e} > 1e-6 — '
        f'expected bit-exact; cumulative-log-returns Gaussian path '
        f'regression suspected')


def test_uncompressed_morlet_fingerprint_past_bars_invariant():
    """`extract_fingerprints(channels_per_scale=4)` on the polar
    Morlet panel: past-bar fingerprints must be invariant beyond the
    L2-normalized fp32 noise floor."""
    px_a, px_b = _two_perturbed_panels()
    panel_a = causal_polar_morlet_matrix(px_a, _SCALES, lookback=_LOOKBACK)
    panel_b = causal_polar_morlet_matrix(px_b, _SCALES, lookback=_LOOKBACK)

    fps_a = extract_fingerprints(
        panel_a, w=_FP_WINDOW, znorm=True,
        channels_per_scale=RELATIONAL_CHANNELS_PER_SCALE)
    fps_b = extract_fingerprints(
        panel_b, w=_FP_WINDOW, znorm=True,
        channels_per_scale=RELATIONAL_CHANNELS_PER_SCALE)

    # Bars `[fp_window-1, T0)` are the meaningful past-bar fingerprints —
    # earlier bars have zero-padded tails that are identical by
    # construction.
    fp_diff = np.abs(fps_a[_FP_WINDOW - 1:_T0] - fps_b[_FP_WINDOW - 1:_T0])
    assert fp_diff.max() < _ATOL_FP, (
        f'uncompressed Morlet fingerprint leaks future data into past '
        f'bars: max diff {fp_diff.max():.2e} (atol={_ATOL_FP:.0e})')


def test_dwt_l1_morlet_fingerprint_past_bars_invariant():
    """Per-channel-block DWT-L1 keep-LL preserves the same past-bar
    invariance — DWT operates per-tile, never crosses the time axis,
    so it can't introduce its own causality leak."""
    px_a, px_b = _two_perturbed_panels()
    panel_a = causal_polar_morlet_matrix(px_a, _SCALES, lookback=_LOOKBACK)
    panel_b = causal_polar_morlet_matrix(px_b, _SCALES, lookback=_LOOKBACK)

    comp = Compression(kind='dwt', levels=1, wavelet='haar',
                       pad_mode='periodization')
    fps_a = extract_fingerprints(
        panel_a, w=_FP_WINDOW, znorm=True, compression=comp,
        channels_per_scale=RELATIONAL_CHANNELS_PER_SCALE)
    fps_b = extract_fingerprints(
        panel_b, w=_FP_WINDOW, znorm=True, compression=comp,
        channels_per_scale=RELATIONAL_CHANNELS_PER_SCALE)

    fp_diff = np.abs(fps_a[_FP_WINDOW - 1:_T0] - fps_b[_FP_WINDOW - 1:_T0])
    assert fp_diff.max() < _ATOL_FP, (
        f'DWT-L1 Morlet fingerprint leaks future data into past bars: '
        f'max diff {fp_diff.max():.2e} (atol={_ATOL_FP:.0e})')


def test_scalogram_cache_morlet_path_is_causal(tmp_path):
    """The `load_or_compute_cwt` wrapper does no math of its own — it
    just routes to `causal_polar_morlet_matrix` on the morlet path —
    so a wrapper-level regression would surface as a past-bar leak
    here. Cache directory is per-test (`tmp_path`) so the two panels
    don't share files."""
    px_a, px_b = _two_perturbed_panels()
    columns = [f'T{i}' for i in range(_N)]
    index = pd.bdate_range('2020-01-01', periods=_T)
    df_a = pd.DataFrame(px_a, columns=columns, index=index)
    df_b = pd.DataFrame(px_b, columns=columns, index=index)

    cache_a = tmp_path / 'a'
    cache_b = tmp_path / 'b'
    panel_a = load_or_compute_cwt(
        df_a, _SCALES, _LOOKBACK, wavelet='morlet',
        cache_dir=cache_a, verbose=False)
    panel_b = load_or_compute_cwt(
        df_b, _SCALES, _LOOKBACK, wavelet='morlet',
        cache_dir=cache_b, verbose=False)

    diff = np.abs(panel_a[:, :_T0, :] - panel_b[:, :_T0, :])
    assert diff.max() < _ATOL_PANEL, (
        f'scalogram_cache morlet path leaks future data into past '
        f'bars: max diff {diff.max():.2e}')
