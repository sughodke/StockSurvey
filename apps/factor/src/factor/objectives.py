"""Differentiable training + eval objectives for the scoring head.

`pearson_rank_ic` is the training signal — per-rebalance Pearson
correlation of the head's score vector with forward log-returns,
masked to the liquid universe at that bar, then averaged across bars.
Pearson on raw scores is the Grinold "information coefficient": a
*per-decision* signal, dense and well-conditioned, vs Sharpe which is
one number per backtest.

`block_sharpe` is the eval-only annualized Sharpe at rebalance
granularity, with one-sided turnover costs at `commission_frac`.
Block returns are converted to a daily-equivalent annualized Sharpe
via `sqrt(TRADING_DAYS / rebal_days)` (exact under iid block returns,
matching the JAX `ss_portfolio.block_sharpe_with_costs` definition).
"""
from __future__ import annotations

import numpy as np
from tinygrad.tensor import Tensor


TRADING_DAYS: int = 252


def _isfinite(x: Tensor) -> Tensor:
    inf = float('inf')
    return (x == x) & (x < inf) & (x > -inf)


def pearson_rank_ic(
    scores: Tensor, fwd_returns: Tensor, mask: Tensor,
) -> Tensor:
    """Mean over rebalance bars of Pearson(scores[bar], fwd_returns[bar]).

    All inputs shape `(n_bars, n_tickers)`. `mask` is 1.0 for liquid
    tickers at that bar, 0.0 otherwise. Bars with fewer than 2 valid
    tickers contribute 0 to the mean (correlation undefined).

    Inputs are sanitized: NaN/Inf values are replaced with 0 before any
    arithmetic. Callers are still expected to mask them out, but the
    guard avoids gradient contamination if a caller forgets — under
    autograd, `0 * NaN = NaN` and the head's update goes NaN.

    Assumes `fwd_returns` is the cumulative log-return over the same
    rebalance horizon used to subsample `scores` and `mask`; period
    selection happens upstream in `precompute_inputs`.
    """
    scores = _isfinite(scores).where(scores, 0.0)
    fwd_returns = _isfinite(fwd_returns).where(fwd_returns, 0.0)
    counts = mask.sum(axis=1)
    safe_counts = counts.maximum(1.0)
    s_mean = (scores * mask).sum(axis=1) / safe_counts
    r_mean = (fwd_returns * mask).sum(axis=1) / safe_counts
    s_dev = (scores - s_mean.reshape(-1, 1)) * mask
    r_dev = (fwd_returns - r_mean.reshape(-1, 1)) * mask
    cov = (s_dev * r_dev).sum(axis=1)
    s_var = (s_dev * s_dev).sum(axis=1)
    r_var = (r_dev * r_dev).sum(axis=1)
    denom = (s_var * r_var).maximum(1e-18).sqrt()
    per_bar_ic = cov / denom
    bar_valid = (counts >= 2).cast(scores.dtype)
    return (per_bar_ic * bar_valid).sum() / bar_valid.sum().maximum(1.0)


def per_bar_pearson_ic(
    scores: Tensor, fwd_returns: Tensor, mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-bar Pearson IC + validity mask. Inputs `(n_bars, n_tickers)`.

    Returns `(ic_per_bar, bar_valid)` both shape `(n_bars,)`:
      - `ic_per_bar[t]` is Pearson(scores[t], fwd_returns[t]) on the
        masked liquid universe at that bar. Zero when fewer than 2
        valid tickers OR when either var is degenerate.
      - `bar_valid[t]` is 1.0 for bars with >=2 valid tickers and
        non-degenerate score/return variance, else 0.0.

    Same NaN/Inf sanitization as `pearson_rank_ic`. Used by
    `horizon_mixture_loss` — the mixture loss weights per-bar IC by the
    horizon head's per-bar π_t, so the gradient that flows back into π_t
    sees state-conditional IC contributions rather than the aggregate.
    """
    scores = _isfinite(scores).where(scores, 0.0)
    fwd_returns = _isfinite(fwd_returns).where(fwd_returns, 0.0)
    counts = mask.sum(axis=1)
    safe_counts = counts.maximum(1.0)
    s_mean = (scores * mask).sum(axis=1) / safe_counts
    r_mean = (fwd_returns * mask).sum(axis=1) / safe_counts
    s_dev = (scores - s_mean.reshape(-1, 1)) * mask
    r_dev = (fwd_returns - r_mean.reshape(-1, 1)) * mask
    cov = (s_dev * r_dev).sum(axis=1)
    s_var = (s_dev * s_dev).sum(axis=1)
    r_var = (r_dev * r_dev).sum(axis=1)
    denom = (s_var * r_var).maximum(1e-18).sqrt()
    ic = cov / denom
    # Bars with <2 valid tickers OR degenerate variance produce 0 IC.
    # `(counts >= 2)` gates the first; `s_var * r_var > 1e-18` gates the
    # second. Combined as float so the multiply zeroes out invalid bars.
    bar_valid = (counts >= 2).cast(scores.dtype) * (
        (s_var * r_var) > 1e-18).cast(scores.dtype)
    return ic * bar_valid, bar_valid


def horizon_mixture_loss(
    scores: Tensor,
    fwd_multi: Tensor,
    mask: Tensor,
    pi: Tensor,
    *,
    entropy_weight: float = 0.0,
) -> Tensor:
    """State-conditional mixture-of-horizons IC loss (negate for minimization).

    Shapes:
      - `scores`     : `(n_bars, n_tickers)` — score head output at the
                       fine rebal grid.
      - `fwd_multi`  : `(K, n_bars, n_tickers)` — forward log returns at
                       each of K horizons. Edge cells (where the horizon
                       runs past the data) should already be 0-masked by
                       the caller (precompute fills with 0 + tightens
                       `mask`).
      - `mask`       : `(K, n_bars, n_tickers)` — per-horizon liquidity
                       mask. Per-horizon because the trailing edge
                       differs across horizons (h=5 has 5 trailing rows
                       masked; h=60 has 60).
      - `pi`         : `(n_bars, K)` — horizon distribution at each bar
                       (rows sum to 1).
      - `entropy_weight` : optional regularization pushing `pi` away
                       from one-hot. `loss = base_loss - entropy_weight
                       * H(pi)`. Default 0 (no regularization).
                       Negative loss term because higher entropy is
                       preferred, but the *returned* loss is what we
                       minimize, so we subtract entropy.
    The returned scalar is `-mean_t Σ_k π_t[k] · IC_k_t` (plus optional
    entropy reg). Negative to make minimization correspond to maximizing
    state-weighted IC. Mean-over-bars uses only bars with at least one
    horizon contributing a valid per-bar IC.
    """
    K = int(fwd_multi.shape[0])
    # Per-horizon per-bar IC: stack into `(K, n_bars)`.
    ic_per_horizon = []
    valid_per_horizon = []
    for k in range(K):
        ic_k, v_k = per_bar_pearson_ic(scores, fwd_multi[k], mask[k])
        ic_per_horizon.append(ic_k.reshape(1, -1))
        valid_per_horizon.append(v_k.reshape(1, -1))
    ic_kt = ic_per_horizon[0].cat(*ic_per_horizon[1:], dim=0)
    valid_kt = valid_per_horizon[0].cat(*valid_per_horizon[1:], dim=0)

    # `(K, n_bars)` × `(K, n_bars)` (transpose of pi) → mixture-weighted
    # per-bar IC: `mix_t = Σ_k π_t[k] · IC_k_t · valid_k_t`. The valid
    # mask absorbs per-horizon edge bars (e.g. last 60 bars are invalid
    # for h=60 but fine for h=5). pi rows still sum to 1 even when some
    # horizons are masked — that's the correct economic semantics ("if
    # I'd picked horizon h here, the IC would have been undefined") but
    # the *loss* contribution from those (k, t) cells is correctly zero.
    pi_kt = pi.transpose(0, 1)            # `(K, n_bars)`
    mix_per_bar = (pi_kt * ic_kt * valid_kt).sum(axis=0)   # `(n_bars,)`

    # Bar is "loss-contributing" if any horizon is valid at that bar.
    any_valid = (valid_kt.sum(axis=0) > 0).cast(scores.dtype)
    n_valid_bars = any_valid.sum().maximum(1.0)
    mean_mix_ic = (mix_per_bar * any_valid).sum() / n_valid_bars

    loss = -mean_mix_ic
    if entropy_weight > 0.0:
        # H(pi_t) = -Σ_k π_t[k] log π_t[k]. Reduce over bars by mean.
        # 1e-12 floor for numerical stability (matches softmax-eps used
        # elsewhere in this file).
        log_pi = (pi + 1e-12).log()
        H = -(pi * log_pi).sum(axis=1).mean()
        loss = loss - Tensor(entropy_weight) * H
    return loss


def horizon_mixture_loss_bilevel(
    scores: Tensor,
    fwd_multi: Tensor,
    mask: Tensor,
    pi: Tensor,
    *,
    horizons: tuple[int, ...],
    commission_frac: float = 0.001,
    entropy_weight: float = 0.0,
    deployment_reward_weight: float = 0.0,
) -> Tensor:
    """Bilevel mixture loss: rank-IC for score head, IC + λ·deployment-return for π.

    Same `(scores, fwd_multi, mask, pi)` shape contract as
    `horizon_mixture_loss`. Additional inputs:

      - `horizons`: tuple of horizon lengths matching `fwd_multi`'s K
        axis. Used to convert cumulative log returns to per-day rates
        so the cost term and the return signal are scale-comparable
        across horizons.
      - `commission_frac`: per-rebalance commission (e.g. 0.001 = 10 bps),
        amortized over each horizon's holding period in the deployment-
        reward term.
      - `deployment_reward_weight` (λ): scalar weight on the
        deployment-return term. λ=0 reduces to `horizon_mixture_loss`
        exactly. Both terms are per-batch std-normalized (with the
        std *detached* so it's a constant in the gradient graph) so
        λ is dimensionless.

    Loss structure:

        L = -mean_t Σ_k π_t[k] · IC_k_t / std_detached(IC)         ← scores + pi see this
            -λ · mean_t Σ_k π_t[k] · ret_k_t / std_detached(ret)    ← only pi sees this
            -entropy_weight · H(pi)

    where `ret_k_t = (centered_score_detached · fwd_k_t · mask_k_t).mean_over_tickers / horizon_k - commission_frac/horizon_k`.

    The score-head detach inside the return term is the bilevel split:
    the score head is trained on rank-IC's stability signal only, while
    the horizon head sees deployment-return supervision in addition.
    """
    K = int(fwd_multi.shape[0])

    # --- IC term (gradient flows into scores + pi) ---
    ic_per_horizon = []
    valid_per_horizon = []
    for k in range(K):
        ic_k, v_k = per_bar_pearson_ic(scores, fwd_multi[k], mask[k])
        ic_per_horizon.append(ic_k.reshape(1, -1))
        valid_per_horizon.append(v_k.reshape(1, -1))
    ic_kt = ic_per_horizon[0].cat(*ic_per_horizon[1:], dim=0)
    valid_kt = valid_per_horizon[0].cat(*valid_per_horizon[1:], dim=0)

    pi_kt = pi.transpose(0, 1)              # (K, n_bars)
    mix_ic_per_bar = (pi_kt * ic_kt * valid_kt).sum(axis=0)
    any_valid = (valid_kt.sum(axis=0) > 0).cast(scores.dtype)
    n_valid_bars = any_valid.sum().maximum(1.0)
    mean_mix_ic = (mix_ic_per_bar * any_valid).sum() / n_valid_bars

    # Per-batch std for scale normalization (detach: pure scale factor,
    # no gradient through it).
    ic_for_std = (ic_kt * valid_kt).detach()
    ic_std = (ic_for_std.std() + 1e-8)
    L_IC = -mean_mix_ic / ic_std

    if deployment_reward_weight > 0.0:
        # --- Deployment-return term (gradient flows ONLY into pi) ---
        scores_d = scores.detach()
        ret_per_horizon = []
        for k in range(K):
            h = horizons[k]
            fwd_k = fwd_multi[k]
            mask_k = mask[k]
            counts = mask_k.sum(axis=1)
            safe_counts = counts.maximum(1.0)
            # Centered score per-bar (matches IC centering convention)
            s_mean = (scores_d * mask_k).sum(axis=1) / safe_counts
            s_dev = (scores_d - s_mean.reshape(-1, 1)) * mask_k
            # Per-bar score-weighted forward log return, masked + per-day rate
            cov_k = (s_dev * fwd_k * mask_k).sum(axis=1) / safe_counts
            per_day_ret_k = cov_k / float(h) - commission_frac / float(h)
            ret_per_horizon.append(per_day_ret_k.reshape(1, -1))
        ret_kt = ret_per_horizon[0].cat(*ret_per_horizon[1:], dim=0)

        mix_ret_per_bar = (pi_kt * ret_kt * valid_kt).sum(axis=0)
        mean_mix_ret = (mix_ret_per_bar * any_valid).sum() / n_valid_bars

        ret_for_std = (ret_kt * valid_kt).detach()
        ret_std = (ret_for_std.std() + 1e-8)
        L_RET = -mean_mix_ret / ret_std

        loss = L_IC + L_RET * deployment_reward_weight
    else:
        loss = L_IC

    if entropy_weight > 0.0:
        log_pi = (pi + 1e-12).log()
        H = -(pi * log_pi).sum(axis=1).mean()
        loss = loss - Tensor(entropy_weight) * H
    return loss


def horizon_mixture_loss_target(
    scores: Tensor,
    fwd_multi: Tensor,
    mask: Tensor,
    pi: Tensor,
    *,
    horizons: tuple[int, ...],
    logits: Tensor | None = None,
    rng: np.random.Generator | None = None,
    commission_frac: float = 0.001,
    entropy_weight: float = 0.0,
    reinforce_weight: float = 0.0,
) -> Tensor:
    """Target-side REINFORCE loss: rank-IC for score head + per-bar
    Sharpe-residual REINFORCE on π's training signal.

    Loss structure:

        L = -mean_t Σ_k π_t[k] · IC_k_t / std_detached(IC)            ← score head + π
            +β · -mean_t [log π_t[a_t] · advantage_t]                   ← π only (scores detached)

    where `advantage_t = (ret_at_sampled_t.detach() − m) / s`, with
    `ret_at_sampled_t = (centered_scores_detached · fwd_log_return_{a_t})_t / count_t / horizon_{a_t} − commission_frac/horizon_{a_t}`
    and `(m, s)` are the trajectory's mean and std of `ret_at_sampled`.

    The sampling step `a_t ~ Categorical(π_t)` is performed INSIDE this
    function via `pi.numpy()` — bringing pi into numpy land for the
    categorical draw. Doing this outside (in the trainer) breaks the
    upstream autograd graph in tinygrad: calling `pi.detach().numpy()`
    on the same lazy buffer that feeds the loss appears to prune the
    gradient path back to the horizon head's Wh/bh parameters even
    when the loss is computed from the original (non-detached) `pi`
    tensor. Keeping the materialization local to this function (and
    creating a fresh sampling rng per call) avoids the issue.

    Score-head gradient comes ONLY from the IC term (scores.detach()
    inside the REINFORCE term). π head sees both signals.
    """
    if rng is None:
        rng = np.random.default_rng()
    K = int(fwd_multi.shape[0])

    # --- IC term (same as horizon_mixture_loss_bilevel) ---
    ic_per_horizon = []
    valid_per_horizon = []
    for k in range(K):
        ic_k, v_k = per_bar_pearson_ic(scores, fwd_multi[k], mask[k])
        ic_per_horizon.append(ic_k.reshape(1, -1))
        valid_per_horizon.append(v_k.reshape(1, -1))
    ic_kt = ic_per_horizon[0].cat(*ic_per_horizon[1:], dim=0)
    valid_kt = valid_per_horizon[0].cat(*valid_per_horizon[1:], dim=0)

    pi_kt = pi.transpose(0, 1)
    mix_ic_per_bar = (pi_kt * ic_kt * valid_kt).sum(axis=0)
    any_valid = (valid_kt.sum(axis=0) > 0).cast(scores.dtype)
    n_valid_bars = any_valid.sum().maximum(1.0)
    mean_mix_ic = (mix_ic_per_bar * any_valid).sum() / n_valid_bars

    ic_for_std = (ic_kt * valid_kt).detach()
    ic_std = (ic_for_std.std() + 1e-8)
    L_IC = -mean_mix_ic / ic_std

    if reinforce_weight > 0.0:
        # --- Per-bar reward at sampled action (using detached scores) ---
        scores_d = scores.detach()
        n_bars = int(pi.shape[0])

        # Sample one horizon per bar INSIDE the loss using the
        # caller-provided logits (NOT pi). Materializing pi truncates
        # autograd back to Wh/bh in tinygrad; materializing logits is
        # safe (linear-only chain to Wh/bh).
        if logits is None:
            raise ValueError(
                "horizon_mixture_loss_target requires `logits` when "
                "reinforce_weight > 0 — call apply_mlp_horizon_full to "
                "get them.")
        logits_np = logits.detach().numpy().astype(np.float64)
        # Numerically stable softmax in numpy for sampling
        logits_np = logits_np - logits_np.max(axis=1, keepdims=True)
        pi_np_for_sampling = np.exp(logits_np)
        pi_np_for_sampling = pi_np_for_sampling / pi_np_for_sampling.sum(axis=1, keepdims=True)
        u = rng.random(n_bars)
        cum = np.cumsum(pi_np_for_sampling, axis=1)
        sampled_actions = (cum > u[:, None]).argmax(axis=1).astype(np.int32)

        # Build one-hot for sampled actions: shape (K, n_bars)
        one_hot = np.zeros((K, n_bars), dtype=np.float32)
        for t in range(n_bars):
            one_hot[int(sampled_actions[t]), t] = 1.0
        one_hot_t = Tensor(one_hot)  # (K, n_bars)

        # Per-bar per-horizon return signal (same as bilevel ret_kt)
        ret_per_horizon = []
        for k in range(K):
            h = horizons[k]
            fwd_k = fwd_multi[k]
            mask_k = mask[k]
            counts = mask_k.sum(axis=1)
            safe_counts = counts.maximum(1.0)
            s_mean = (scores_d * mask_k).sum(axis=1) / safe_counts
            s_dev = (scores_d - s_mean.reshape(-1, 1)) * mask_k
            cov_k = (s_dev * fwd_k * mask_k).sum(axis=1) / safe_counts
            per_day_ret_k = cov_k / float(h) - commission_frac / float(h)
            ret_per_horizon.append(per_day_ret_k.reshape(1, -1))
        ret_kt = ret_per_horizon[0].cat(*ret_per_horizon[1:], dim=0)

        # Reward at sampled action per bar — (one_hot * ret_kt).sum(axis=0) gives (n_bars,)
        ret_at_sampled = (one_hot_t * ret_kt).sum(axis=0)  # (n_bars,)

        # Per-bar Sharpe-residual advantage: (ret_t - mean) / std, all detached
        ret_d = ret_at_sampled.detach()
        m_d = ret_d.mean()
        s_d = ret_d.std() + 1e-8
        advantage = (ret_d - m_d) / s_d  # (n_bars,) zero-mean unit-std

        # log π at sampled action — gradient flows through pi
        log_pi = (pi + 1e-12).log()                          # (n_bars, K)
        log_pi_at_a = (one_hot_t.transpose(0, 1) * log_pi).sum(axis=1)  # (n_bars,)

        # Score-function REINFORCE loss; mask invalid bars (where neither
        # horizon contributed any IC, the reward signal is also degenerate).
        L_REINFORCE = -((log_pi_at_a * advantage) * any_valid).sum() / n_valid_bars

        loss = L_IC + L_REINFORCE * reinforce_weight
    else:
        loss = L_IC

    if entropy_weight > 0.0:
        log_pi = (pi + 1e-12).log()
        H = -(pi * log_pi).sum(axis=1).mean()
        loss = loss - Tensor(entropy_weight) * H
    return loss


def masked_mse(
    scores: Tensor, targets: Tensor, mask: Tensor,
) -> Tensor:
    """Mean squared error averaged over `(bar, ticker)` cells with `mask=1`.

    All inputs shape `(n_bars, n_tickers)`. `mask` is 1.0 for cells the
    auxiliary loss should see, 0.0 otherwise. NaN/Inf values are
    sanitized to 0 before the diff so a missed mask upstream cannot
    NaN-poison the gradient — same defensive pattern as
    `pearson_rank_ic`.

    Used as the auxiliary loss in the multi-task path: the aux head
    predicts cross-sectionally winsorized + z-scored forward returns
    (see `factor.data.forward_robust_z`). Mean-over-cells rather than
    mean-over-bars-then-mean keeps gradient magnitude proportional to
    actual sample count, which matters for early bars with fewer valid
    tickers.
    """
    scores = _isfinite(scores).where(scores, 0.0)
    targets = _isfinite(targets).where(targets, 0.0)
    sq = (scores - targets) ** 2 * mask
    n_valid = mask.sum().maximum(1.0)
    return sq.sum() / n_valid


def block_sharpe(
    rebal_scores: Tensor,
    log_temperature: Tensor,
    block_log_ret: Tensor,
    rebal_mask: Tensor,
    rebal_days: int,
    commission_frac: float,
) -> Tensor:
    """Annualized portfolio Sharpe at block (rebalance) granularity.

    Mirrors the contract of `ss_portfolio.block_sharpe_with_costs`
    (which still runs JAX in the regime app). Soft top-N is implemented
    as a temperature-scaled softmax of the regime score; small
    temperature approaches a hard argmax over the liquid universe.

    Costs: initial entry from cash incurs full one-sided turnover
    (sum of weights = 1). Subsequent rebalances pay `0.5 * L1(delta_w)`
    per period — the factor of 0.5 converts bidirectional L1 to a
    one-sided cost.

    Block returns are converted to daily-equivalent annualized Sharpe
    via `sqrt(TRADING_DAYS / rebal_days)`.
    """
    temp = log_temperature.exp()
    # `+ log(mask + eps)` drives masked entries to -inf in the softmax,
    # so they get zero weight regardless of score. Subtract row-max for
    # numerical stability before exp.
    s = rebal_scores / temp + (rebal_mask + 1e-12).log()
    s = s - s.max(axis=1, keepdim=True)
    exp_s = s.exp() * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdim=True) + 1e-12)

    port_block_ret = (w * block_log_ret).sum(axis=1)

    init_cost = w[0].abs().sum()
    diff_cost = 0.5 * (w[1:] - w[:-1]).abs().sum(axis=1)
    costs = commission_frac * init_cost.reshape(1).cat(diff_cost, dim=0)
    port_block_ret = port_block_ret - costs

    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    return mean / std * Tensor((TRADING_DAYS / rebal_days) ** 0.5)


def long_short_weights(
    rebal_scores: Tensor, rebal_mask: Tensor,
    leverage: float = 1.0, clip_sigma: float = 3.0,
) -> Tensor:
    """Per-bar market-neutral long-short weights via z-score → clip → L1-normalize.

    Closes the rank-IC / long-only-top-N mismatch documented in
    `apps/docs/docs/findings/factor-rankic-long-only-mismatch.md`:
    rank-IC trains a sign-symmetric signal that softmax-top-N can only
    half-execute. This constructor uses both tails.

    Returns weights with `sum(w_t) ≈ 0` and `sum(|w_t|) ≈ leverage` per
    bar, where equality is exact for bars with ≥2 valid tickers and a
    non-degenerate score distribution. Degenerate bars (constant scores
    or <2 valid tickers) return all-zero weights — no position taken.

    Inputs sanitized to 0 outside `mask=1` cells, matching the defensive
    pattern in `pearson_rank_ic` / `masked_mse`.
    """
    counts = rebal_mask.sum(axis=1, keepdim=True)
    safe_counts = counts.maximum(1.0)
    s_mean = (rebal_scores * rebal_mask).sum(axis=1, keepdim=True) / safe_counts
    s_dev = (rebal_scores - s_mean) * rebal_mask
    s_var = (s_dev * s_dev).sum(axis=1, keepdim=True) / safe_counts
    s_std = (s_var + 1e-12).sqrt()
    z = s_dev / s_std
    z = z.maximum(-clip_sigma).minimum(clip_sigma) * rebal_mask
    z_mean = z.sum(axis=1, keepdim=True) / safe_counts
    z = (z - z_mean) * rebal_mask
    l1 = z.abs().sum(axis=1, keepdim=True)
    safe_l1 = l1.maximum(1e-12)
    w = leverage * z / safe_l1
    valid_row = (l1 > 1e-9).cast(w.dtype)
    return w * valid_row


def block_sharpe_long_short(
    rebal_scores: Tensor,
    block_log_ret: Tensor,
    rebal_mask: Tensor,
    rebal_days: int,
    commission_frac: float,
    leverage: float = 1.0,
    clip_sigma: float = 3.0,
) -> Tensor:
    """Annualized Sharpe for the market-neutral long-short constructor.

    Mirrors `block_sharpe`'s reduction (mean / std over per-bar block
    returns, then `sqrt(TRADING_DAYS / rebal_days)` annualization), but
    swaps the constructor. Costs use `commission_frac * L1(delta_w)`
    *without* the 0.5 factor: for a market-neutral book the L1 of the
    delta is already the one-sided turnover (the 0.5 factor in
    `block_sharpe` exists because long-only L1(delta) double-counts
    given `sum(w) = 1`). Initial entry from cash pays full leverage.

    `leverage = 1.0` is the comparison gate — Sharpe scales roughly
    linearly with leverage before costs, so larger values are
    informative as a sensitivity check, not a shippable result.
    """
    w = long_short_weights(rebal_scores, rebal_mask, leverage, clip_sigma)
    port_block_ret = (w * block_log_ret).sum(axis=1)
    init_cost = w[0].abs().sum()
    diff_cost = (w[1:] - w[:-1]).abs().sum(axis=1)
    costs = commission_frac * init_cost.reshape(1).cat(diff_cost, dim=0)
    port_block_ret = port_block_ret - costs
    mean = port_block_ret.mean()
    std = port_block_ret.std() + 1e-9
    return mean / std * Tensor((TRADING_DAYS / rebal_days) ** 0.5)


def block_ir_vs_ew(
    rebal_scores: Tensor,
    log_temperature: Tensor,
    block_log_ret: Tensor,
    rebal_mask: Tensor,
    rebal_days: int,
    commission_frac: float,
) -> Tensor:
    """Annualized Information Ratio of the softmax-LO portfolio vs the
    universe's EW benchmark.

    `IR_t = (port_after_cost_t − ew_pre_cost_t) / TE`. EW benchmark is
    the per-bar mean of `block_log_ret` over `rebal_mask=1` cells —
    treated as frictionless (the standard convention; matched-friction
    EW would require modeling EW's drift-rebalance turnover, which
    only adds bias in the same direction for both arms).

    Mirrors `block_sharpe`'s constructor and cost treatment exactly,
    so this is a drop-in replacement loss for that path. The optimum
    of this loss differs from the optimum of `block_sharpe`: this one
    rewards alpha per unit *tracking error*, not per unit total
    volatility — directly aligned with the EW gate operational rule
    in `CLAUDE.md` that requires `alpha = model_val_sharpe −
    passive_val_sharpe ≥ 0` for shippability.

    Why it matters: rank-IC training is scale-invariant and rewards
    spreading thin information across the cross-section. Sharpe-as-loss
    rewards risk-adjusted absolute return. IR-as-loss rewards
    risk-adjusted *active* return, which is the metric the EW gate
    measures. See `findings/factor-rankic-long-only-mismatch.md` for
    the diagnostic that motivated this loss.
    """
    temp = log_temperature.exp()
    s = rebal_scores / temp + (rebal_mask + 1e-12).log()
    s = s - s.max(axis=1, keepdim=True)
    exp_s = s.exp() * rebal_mask
    w = exp_s / (exp_s.sum(axis=1, keepdim=True) + 1e-12)

    port_block_ret = (w * block_log_ret).sum(axis=1)
    counts = rebal_mask.sum(axis=1).maximum(1.0)
    ew_block_ret = (block_log_ret * rebal_mask).sum(axis=1) / counts

    init_cost = w[0].abs().sum()
    diff_cost = 0.5 * (w[1:] - w[:-1]).abs().sum(axis=1)
    costs = commission_frac * init_cost.reshape(1).cat(diff_cost, dim=0)
    port_block_ret = port_block_ret - costs

    active = port_block_ret - ew_block_ret
    mean = active.mean()
    std = active.std() + 1e-9
    return mean / std * Tensor((TRADING_DAYS / rebal_days) ** 0.5)


__all__ = [
    'TRADING_DAYS', 'block_sharpe', 'block_sharpe_long_short',
    'block_ir_vs_ew', 'horizon_mixture_loss', 'horizon_mixture_loss_bilevel',
    'horizon_mixture_loss_target', 'long_short_weights',
    'masked_mse', 'pearson_rank_ic', 'per_bar_pearson_ic',
]
