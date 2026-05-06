"""NHC↔NCHW permute wrapper around tinygrad's `Tensor.conv2d`."""
from __future__ import annotations

from tinygrad.tensor import Tensor


def conv1d_nhc(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """1-D conv over an NHC input + HIO weight layout.

    Args:
      x: shape `(B, L, Cin)`           (NHC — batch, length, channels-in).
      W: shape `(kernel, Cin, Cout)`   (HIO — kernel, in, out).
      b: shape `(Cout,)`.

    Returns:
      shape `(B, L_post, Cout)`        (NHC).

    Tinygrad's conv2d is NCHW + OIK — we permute in, run the conv, and
    permute back. Centralized here so callers don't reinvent the
    transpose dance every time they load a JAX-shaped backbone.
    """
    x_bcl = x.permute(0, 2, 1)               # (B, Cin, L)
    W_oik = W.permute(2, 1, 0)               # (Cout, Cin, kernel)
    y_bcl = x_bcl.conv2d(W_oik)              # (B, Cout, L_post)
    return y_bcl.permute(0, 2, 1) + b        # (B, L_post, Cout)
