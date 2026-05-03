"""Back-compat shim. `fit_stats` was promoted to `ss_features.fit_stats`.

Existing `from replay.metrics import fit_stats` keeps working through
the re-export below; new code should import directly from `ss_features`.
"""
from ss_features.metrics import fit_stats

__all__ = ['fit_stats']
