# ss_wavelets

Causal continuous wavelet transform and the windowed power means
that consume it. The CWT layer for the `regime` strategy.

## What's here

  * `causal_cwt(prices, scales, lookback)` — Ricker (Mexican-hat)
    CWT where `output[t]` depends only on `input[:t+1]`. Each ticker
    is z-normalized over a causal rolling window before convolution.
  * `precompute_windows(power, lookback, n_tail)` — recent vs
    historical means of CWT power, the inputs to `regime`'s
    divergence-based scoring.
  * `ALL_SCALES` — 13-point logarithmic grid from 3 to 126 days.

The two functions form the regime strategy's front half:

```
prices                                # (n_dates, n_tickers)
  ↓ causal_cwt(scales=ALL_SCALES, lookback=L)
coefficients                          # (n_scales, n_dates, n_tickers)
  ↓ |·|²
power
  ↓ precompute_windows(lookback=L, n_tail=N)
recent, historical                    # (n_scales, n_valid, n_tickers)
```

`recent` and `historical` are the two distributions that downstream
divergence (`ss_indicators.symmetric_kl_divergence` etc.) compares.

## Usage

```python
import numpy as np
from ss_wavelets import ALL_SCALES, causal_cwt, precompute_windows

prices = ...                          # (T, N) float
coeffs = causal_cwt(prices, ALL_SCALES, lookback=120)
power = coeffs ** 2
recent, hist = precompute_windows(power, lookback=120, n_tail=20)
```

## Strict causality

This package was originally non-causal. The CWT used
`fftconvolve(..., mode='full')[-n:]`, taking the *right* `n` outputs
of the full convolution. That slice contained the kernel's left
edge centered on the price at time `t + 4·scale`, leaking up to
~504 days of future information at the longest scale.

The fix in `causal_cwt`:

```python
full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
coeffs[si] = full[:n_dates]           # NOT full[-n_dates:]
```

with a one-sided kernel built on `t in [-points, 0]` rather than the
symmetric `[-points, points]`. Each output index `t` aligns with the
kernel's right edge at time `t`, summing `x_norm[max(0, t-points)..t]`
— strictly past samples.

This single change cut reported `regime` validation Sharpe by 3–4×.
Anywhere you see CWT-based numbers in the project, check whether
they came from before or after this fix.

## Per-ticker normalization caveat

`precompute_windows` divides each ticker's power by its mean over
`(scales, time)` before the cumsum. This keeps the float32 cumsum
well-conditioned — raw CWT power on high-priced names exceeds 1e11
and overflows.

The normalizer technically uses full-history information, but the
symmetric-KL divergence downstream is invariant to a per-ticker
uniform rescaling of power, so no future information actually leaks
into training scores. **If the divergence is ever swapped for a
non-scale-invariant one, this normalizer must be made causal.**
This is a deliberate compromise documented in the source — it's
correct for the current consumer, brittle for future ones.

## ALL_SCALES

```python
ALL_SCALES = [3, 5, 7, 10, 12, 15, 21, 26, 42, 50, 63, 90, 126]
```

13 logarithmically spaced lookbacks from 3 days (intra-week noise)
to 126 days (~half year). Trained `regime` models concentrate
weight on the 26–126 day band; the short scales [3, 5, 7] usually
collapse to <1% weight in the JAX-Adam run.

The grid is grouped into three tiers that the Optuna search can
toggle on/off:

  * **short** (S): 3, 5, 7
  * **mid**   (M): 10, 12, 15, 21, 26
  * **long**  (L): 42, 50, 63, 90, 126

`regime`'s `--scales` legend in window summaries (`ML`, `def`, `SML`,
etc.) refers to these subsets.

## Why numpy, not JAX

The two routines are heavy one-shot precomputes called once per
training window, not part of any autograd path. Running them on
numpy + scipy avoids forcing a JAX dependency on consumers that
just want a CWT panel, and the FFT path in scipy (pocketfft) is
faster than `jax.numpy.fft` on CPU for the sizes involved.

Cast to `jnp.asarray(...)` at the JAX boundary in the training loop:

```python
coeffs = causal_cwt(prices, ALL_SCALES, lookback=120)  # numpy
recent, hist = precompute_windows(coeffs ** 2, 120, 20)
recent_j = jnp.asarray(recent)                          # cross to JAX here
```
