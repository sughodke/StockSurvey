"""ss_wavelets: causal continuous wavelet transform + windowed power means.

Two routines:

  * `causal_cwt`         — Ricker (Mexican-hat) CWT where output[t] depends
                           only on input[:t+1]. Causal rolling z-norm + FFT
                           convolution.
  * `precompute_windows` — windowed mean of (CWT power) over a recent /
                           historical split, used by regime divergence
                           scoring downstream.

Implementation runs on numpy + scipy (FFT convolution + cumsum tricks).
The output arrays are plain numpy, not JAX — these are heavy one-shot
precomputes, not part of any autograd path. Cast to `jnp.asarray(...)`
at the JAX boundary in your training loop.

`ALL_SCALES` provides a sensible 13-point logarithmic grid from 3 days
to 126 days for equity strategies.
"""

from ss_wavelets.cwt import ALL_SCALES, KERNEL_HALF_EXTENT, causal_cwt
from ss_wavelets.windowing import precompute_windows

__all__ = [
    'ALL_SCALES',
    'KERNEL_HALF_EXTENT',
    'causal_cwt',
    'precompute_windows',
]
