"""Tinygrad layout-glue helpers used by both `factor` and `replay`.

Both apps store conv weights on disk in JAX-style NHC/HIO layout
(`x: (B, L, Cin)`, `W: (kernel, Cin, Cout)`) — that's the format the
SSL-pretrained backbone npzs hand back. Tinygrad's `Tensor.conv2d`
expects NCHW, so every conv-using forward pass needs the same
permute-in / conv / permute-out wrapper. This module is that wrapper,
factored out so the two implementations don't drift.
"""
from ss_tg_ops.conv import conv1d_nhc

__all__ = ['conv1d_nhc']
