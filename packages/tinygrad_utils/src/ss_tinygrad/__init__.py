"""ss_tinygrad — shared tinygrad runtime utilities across apps.

Currently exports mixed-precision helpers. Add more (JIT step factories,
common loss wrappers) here rather than copy-pasting across apps.

Tinygrad-runtime-only — apps consume; numpy packages should not.
"""
from ss_tinygrad.amp import maybe_bf16, cast_back_fp32

__all__ = ['maybe_bf16', 'cast_back_fp32']
