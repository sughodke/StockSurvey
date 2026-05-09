# Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad

Last load-bearing JAX site in the workspace. After the
ss_indicators numpy migration + replay/factor on tinygrad,
`packages/portfolio/src/ss_portfolio/sharpe.py` is the only file
that genuinely needs JAX (for `jax.grad` over a differentiable
Sharpe-with-costs loss). Everything else that imports `jax` /
`jnp` does so as residue or as a parked-research dep.

**Live consumers** (full grep, post-extraction):

- `apps/regime/research/optimize_adam.py` — calls
  `block_sharpe_with_costs` via `jax.value_and_grad`. **Parked**
  per CLAUDE.md (gradient flow is already broken at the
  `get_divergence` boundary since `ss_indicators` went numpy, so
  `jax.grad` produces zero/NaN through the divergence call).
- `apps/factor/src/factor/objectives.py` — docstring reference
  only ("matching the JAX `ss_portfolio.block_sharpe_with_costs`
  definition"); does not actually call the function.
- `packages/portfolio/tests/test_portfolio.py` — exercises with
  `jax.grad` (1 shape test + 1 differentiability test).

So the live consumer is one parked file whose gradient is already
broken. Porting `sharpe.py` alone "kills JAX" in the sense that no
non-parked code imports it; but `optimize_adam.py` will fail at
import (it `import jax` + calls `jax.value_and_grad` on a now-
tinygrad function — pure-functional autograd doesn't compose with
tinygrad's stateful `Tensor.backward()`).

Three honest paths, in increasing scope:

1. **B (recommended): port + delete `optimize_adam.py`** (~1-2 h).
   Parked-and-broken-anyway gets removed; the deletion is the most
   honest acknowledgement of CLAUDE.md's status. Tests in
   `test_portfolio.py` switch to tinygrad's `requires_grad=True`
   + `loss.backward()` pattern.
2. **C: port + skeleton stub `optimize_adam.py`** (~1.5 h). Same
   as B but leaves a 10-line file pointing at the last working JAX
   commit so the historical context survives a `git log` search.
3. **A: port everything including a tinygrad rewrite of the
   JAX-Adam optimization loop** (~3-4 h). Most thorough; preserves
   the differentiable-regime-trainer story end-to-end. Requires
   replacing `jax.value_and_grad` + `optax.adam` with tinygrad's
   `Tensor.backward()` + `tinygrad.nn.optim.Adam`. Worth doing
   only if the differentiable optimizer is actually going to be
   used again — otherwise B is honester.

**Mechanical port of `sharpe.py` itself** (independent of which
optimize_adam path is chosen):

- Replace `jax.Array` annotations with `tinygrad.Tensor`.
- Replace `jnp.{exp, log, abs, concatenate, sqrt}` with the
  tinygrad equivalents (mostly identical names, all on `Tensor`).
- `s - s.max(axis=1, keepdims=True)` already a tensor op in both.
- Soft-top-N math is unchanged; only the framework changes.
- Tests that did `jax.grad(loss)(jnp.log(jnp.asarray(0.5)))`
  become:

  ```python
  log_t = Tensor(np.log([0.5]).astype(np.float32),
                 requires_grad=True)
  loss = -block_sharpe_with_costs(..., log_temperature=log_t, ...)
  loss.backward()
  grad = log_t.grad.numpy()
  assert np.isfinite(grad).all()
  ```

- The `jnp.sqrt(TRADING_DAYS / rebal_days)` constant should be
  pre-computed at module load (it's a Python int → Python float;
  no need to wrap in any tensor type).

**Why this isn't done yet:** the only live consumer is parked +
broken. Doing the port without a path B/C/A choice strands
`optimize_adam.py` in import-error territory (worse than parked).
The port is mechanical; the prerequisite is a decision on what
happens to `optimize_adam.py`.

**Side cleanup that should ride along** (not blocking; pick up
during B): the `regime/trainer.py` + `inference.py` + `persist.py`
+ `reporting.py` files import `jax.numpy as jnp` for handful of
type hints / `jnp.zeros` / `jnp.asarray` calls that became
no-ops once `ss_indicators` went numpy. Each is a 2-line cleanup;
together they remove the residual JAX imports across `apps/regime`
that aren't `optimize_adam.py`.
