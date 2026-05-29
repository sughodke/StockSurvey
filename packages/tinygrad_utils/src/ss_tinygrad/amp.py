"""Activation mixed-precision helpers for tinygrad.

Pattern: cast activations + matmul inputs to bf16, keep weights + optimizer
state in fp32, cast outputs back to fp32 before the loss.

Backend notes:
- Native bf16: Ampere A100/A40, Ada L4/L40, Hopper H100 — 2-4x tensor-core
  speedup on bf16 matmul vs fp32.
- **Turing T4 / V100: NO bf16 support in tinygrad's CUDA path.** Compile
  throws `Nvrtc Error NVRTC_ERROR_COMPILATION, Unresolved extern function
  '_ZN13__nv_bfloat16C1Ef'`. Confirmed 2026-05-29 on Modal T4 with the
  current tinygrad pin. **Always set `use_bf16=False` on T4** — it does
  not silently fall back. The bf16 flag is opt-in, not opt-out.
- CPU/Metal on Intel Mac: bf16 may not be supported; pass `use_bf16=False`.

Reference: `apps/replay/decoders.py` (`_maybe_bf16` origin pattern),
`apps/factor/cwt_gru_walkforward.py` (TinyJit-with-bf16 pattern).
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes


def maybe_bf16(x: Tensor, use_bf16: bool = True) -> Tensor:
    """Cast tensor activations to bf16 if `use_bf16` is True, else passthrough.

    Typical usage in a forward pass:
        h = maybe_bf16(x, hp.use_bf16)
        h = conv1(h).relu()
        h = conv2(h).relu()
        h = cast_back_fp32(h.mean(axis=-1))
    """
    return x.cast(dtypes.bfloat16) if use_bf16 else x


def cast_back_fp32(x: Tensor) -> Tensor:
    """Always cast back to fp32 at a numerically-sensitive boundary.

    Used between a bf16 forward block and any subsequent stat reduction
    (Sharpe ratio, IC computation) or accumulator op that benefits from
    full mantissa width.
    """
    return x.cast(dtypes.float32)
