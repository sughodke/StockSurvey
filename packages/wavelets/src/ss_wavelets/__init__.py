"""ss_wavelets: causal continuous wavelet transform + windowed power means.

Routines:

  * `causal_cwt`          — real Ricker (Mexican-hat) CWT, bandpass.
                            Causal rolling z-norm + FFT convolution.
  * `causal_cwt_morlet`   — complex Morlet CWT, bandpass with phase.
                            Same rolling z-norm path; complex output.
  * `causal_cwt_gaussian` — real Gaussian scaling function, lowpass.
                            No z-norm — caller passes stationary series
                            (e.g. cumulative log-returns).
  * `precompute_windows`  — windowed mean of (CWT power) over a recent
                            / historical split, used by regime
                            divergence scoring downstream.

Implementation runs on numpy + scipy (FFT convolution + cumsum tricks).
The output arrays are plain numpy; CWT itself is a heavy one-shot
precompute that the rest of the monorepo treats as fixed input.

`ALL_SCALES` provides a sensible 13-point logarithmic grid from 3 days
to 126 days for equity strategies.
"""

from ss_wavelets.cwt import (
    ALL_SCALES,
    DEFAULT_MORLET_OMEGA0,
    KERNEL_HALF_EXTENT,
    causal_cwt,
    causal_cwt_gaussian,
    causal_cwt_morlet,
)
from ss_wavelets.windowing import precompute_windows

__all__ = [
    'ALL_SCALES',
    'DEFAULT_MORLET_OMEGA0',
    'KERNEL_HALF_EXTENT',
    'causal_cwt',
    'causal_cwt_gaussian',
    'causal_cwt_morlet',
    'precompute_windows',
]
