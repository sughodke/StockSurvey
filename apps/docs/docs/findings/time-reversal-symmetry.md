---
tags:
  - diagnostic
---

# Time-reversal symmetry — what reversed-price training would tell us about the encoder

A diagnostic about *what kind* of information the
[supervised-`cnn`](replay-decoders.md) encoder + rank-IC factor head
is actually learning. The analysis is conceptual; the falsifiable
experiment that resolves it is in
[`TODO/reversed-price-experiment.md`](../TODO/reversed-price-experiment.md).

## The question

If we reversed every price series in time and re-ran the entire
pipeline (replay pretrain → factor walk-forward), would the result
be a clean inversion of the forward-time strategy — short-where-it-
was-long, equal-magnitude val IC with flipped sign?

If yes, the system is mostly learning *timeless chart shapes* and
the asymmetric statistics of forward-time markets are decorative.
If no, the system is using genuinely asymmetric information, and
the *places* asymmetry leaks in tell us specifically what kind.

## Why the answer is no — three breakages

**1. The CWT is causal.** The Ricker / polar Morlet kernels only
integrate over `t ≤ b` (`KERNEL_HALF_EXTENT * scale + lookback` bars
back). On reversed prices, "the past in the reversed series" is
*the future in the original frame* — so the encoder is now eating
a feature that's a different *kind* of object, not the same shape
with a sign flip. Ricker is even-symmetric in time so its kernel
shape survives reversal, but Morlet carries phase: reversed prices
rotate the complex argument the opposite direction, and the polar
bundle's `cos(arg) / sin(arg)` channels are no longer the same
input the network was trained on. The
[`masked-ae`](replay-decoders.md) pretext would partly survive
this; the supervised-`cnn` reconstruction targets do not.

**2. The indicator targets are themselves causal.** RSI's gain/loss
split is causal; MACD's EMA forgets old bars exponentially (you
cannot run an EMA backward and recover the same shape); rolling vol
trails. Replay's reconstruction objective is *predict
RSI / MACD / vol / CCI from a causal CWT slice* — reverse the prices
and the target series isn't the time-reverse of the original target
series, so the encoder's pretext-trained geometry no longer
corresponds to anything the reversed indicators want.

**3. Markets are not time-symmetric.** Crashes are faster than
rallies; vol clusters on the downside; gap-down dynamics differ
from gap-up. A causal CWT trained on forward prices has implicitly
learned *the shape of how vol arrives*. That pattern is genuinely
different in reverse time, not just sign-flipped. The encoder
isn't learning chart geometry in the abstract; it's learning the
asymmetric statistics of how price moves arrive.

## The piece that *does* invert cleanly

The **target** does invert: `forward_log_returns` on reversed prices
equals `-backward_log_returns` on the original. So if everything
upstream of the loss were time-symmetric, rank-IC would flip sign
and a freshly-trained model on reversed data would recover the same
val IC magnitude with sign inverted. The fact that you cannot
actually run this experiment and recover the inverted strategy is
the diagnostic — it tells you the system is using asymmetric
information (causal-window shapes, indicator-trail dynamics,
crash/rally asymmetry), which is the *correct* thing for a
return-prediction model to be using.

## Why this matters for the broader research

The
[supervised-`cnn` walk-forward result](factor-ssl-walkforward.md)
shows mean val IC of +0.0031 — barely above zero. One reading is
*the encoder isn't extracting return-predictive geometry from the
CWT*. Another is *the encoder is extracting something genuinely
asymmetric, but most of that asymmetry is in features that don't
predict 20-day cross-sectional returns* (e.g. the encoder learned
crash-shape vs rally-shape, but the cross-sectional return spread
isn't dominated by who crashed-vs-rallied recently).

Reversed-price training disambiguates these. If the reversed model
gets val IC ≈ −0.0031 (clean inversion), the encoder was operating
on roughly time-symmetric chart shapes and the asymmetric statistics
are inert noise on top. If the reversed model gets val IC nowhere
near −0.0031 (fails to invert), the encoder *is* using asymmetric
information — and the +0.0031 ceiling is a statement about *which*
asymmetric features survive into 20-day return prediction, not
about the encoder's general capacity.

## Connection to the masked-AE open question

This diagnostic is independent of but related to the
[supervised-`cnn` vs strict-SSL `masked-ae`](factor-ssl-walkforward.md#outstanding-questions)
head-to-head. Masked-AE pretext is more time-symmetric than
indicator-reconstruction (predicting masked CWT bins is a roughly
in-distribution task that doesn't bias toward causal-trail
shapes), so a masked-AE encoder *should* show a cleaner inversion
under reversal than a supervised-`cnn` encoder. Running both halves
of the inversion test on both encoder objectives gives a 2×2 that
locates exactly where the asymmetric information is being injected
(input bundle, pretext objective, factor head, or some combination).

## Master walk-forward log pointer

No leaderboard row yet — this is a `diagnostic`-style hypothesis
without an empirical run. The follow-up experiment that would land
the row is in
[`TODO/reversed-price-experiment.md`](../TODO/reversed-price-experiment.md);
the resulting verdict will be `confirmed-OOS` (clean inversion,
encoder is symmetric-feature-bound) or `reversed-OOS` (no
inversion, encoder is asymmetric-feature-bound) per the
[verdict-label vocabulary](../leaderboard.md#verdict-labels).
