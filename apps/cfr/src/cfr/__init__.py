"""Deep CFR meta-allocator across the existing strategy menu.

The architectural premise of `apps/cfr`:

- Predictions with non-zero multivariate signal but regime-conditional
  deployment performance need a regime filter, not a richer
  predictor. The
  [prediction-problem pivot arc](../../docs/findings/prediction-problem-pivot-arc.md)
  established this for `apps/gate`, `apps/pairs`, `apps/vol`.
- Macro features are a real but graduated regime signal — use them as
  a continuous deployment scaler at the meta-level, not as direct
  predictor inputs. The
  [macro regime diagnostic](../../docs/findings/macro-regime-diagnostic.md)
  established this.
- Trading is an imperfect-information game with closed-form
  counterfactual regret (the price-taker can replay every action's
  realized return from history without re-simulating the future).
  That makes Counterfactual Regret Minimization unusually tractable
  here.

Phase 1 (this module) uses **tabular CFR** over discrete (infoset,
action) buckets. The infoset is a coarse market-regime label
(trailing vol × cross-sectional dispersion); the action is a (mode,
gross-bucket) tuple over a handful of universe-agnostic modes (EW,
top-K momentum / reversal / low-vol / high-vol, cash). Deep CFR is
the Phase 2+ extension once 13F imitation pretraining + a learned
encoder land.

Public API:

- `ActionMenu` — enumerates discrete actions and resolves each to a
  weight vector at a given bar.
- `InfosetBuilder` — buckets recent market state into a discrete
  regime label.
- `compute_block_regrets` — closed-form counterfactual regret over a
  forward block.
- `TabularCFR` — regret-matching policy with cumulative regret
  table + time-averaged policy table.
- `CFRWalkForward` — multi-window train/eval driver with passive-EW
  + trailing-best-greedy baselines.

See [`apps/cfr` TODO](../../docs/TODO/apps-cfr.md) for the phased
falsifiable-experiment ladder and pre-registered cuts.
"""

from cfr.menu import (
    ActionMenu, BaseMode, CashMode, EqualWeightMode, TopKMode,
    default_phase1_menu, default_phase2a_menu,
)
from cfr.state import InfosetBuilder, default_infoset_builder
from cfr.regret import compute_block_regrets, regret_matching
from cfr.buffer import ReplayBuffer
from cfr.tabular import TabularCFR
from cfr.baselines import (
    PassiveEW, TrailingBestGreedy, NaiveUniform, evaluate_baseline,
)
from cfr.walkforward import CFRWalkForward, WindowResult

__all__ = [
    'ActionMenu', 'BaseMode', 'CashMode', 'EqualWeightMode', 'TopKMode',
    'default_phase1_menu', 'default_phase2a_menu',
    'InfosetBuilder', 'default_infoset_builder',
    'compute_block_regrets', 'regret_matching',
    'ReplayBuffer',
    'TabularCFR',
    'PassiveEW', 'TrailingBestGreedy', 'NaiveUniform', 'evaluate_baseline',
    'CFRWalkForward', 'WindowResult',
]
