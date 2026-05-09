# Modal-cron live deployment for ss-relational

Goal: replace the laptop-or-VPS execution model with a Modal cron
that fires daily, with monthly trade actions (subject to the
rebal-days sweep above). Deploy plan, in order:

1. **Wait for walk-forward eval to confirm OOS edge.** If the
   2021-2025 val-period Sharpe collapses, none of the rest matters.
2. **Cloud-native kill-switch.** Replace the `~/.relational-killswitch`
   file rail with a `modal.Dict["killswitch_active"]` boolean. Add a
   tiny side-CLI (`ss-relational-killswitch {on,off,status}`) that
   reads/writes the Dict. Without this rail you've removed the
   operator override.
3. **Secrets via `modal.Secret`.** Create `modal.Secret.from_name(
   "alpaca-keys")` containing `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL`. The cron function reads
   them at runtime. Default `BASE_URL` to paper until the pilot
   completes.
4. **Idempotency guard.** `modal.Dict["last_run_date"]` — abort at
   function start if today already ran successfully. Protects against
   Modal retry storms.
5. **Failure webhook.** Slack / Discord / email URL in another
   `modal.Secret`. Fire on: any uncaught exception, kill-switch hit,
   non-empty `rejected_orders`, or weight-diff above some sanity
   threshold (e.g. all-new top-N versus yesterday's top-N — could
   indicate a feed glitch).
6. **Run record persistence.** Each cron invocation appends
   `(date, target_weights, executed_orders, rejections)` to a
   `modal.Volume`-backed parquet so you can reproduce decisions for
   compliance/audit. Modal's stdout logs are useful but ephemeral.
7. **Schedule.** `Cron("30 21 * * 1-5")` — 21:30 UTC fires after
   NYSE close in both winter (21:00) and summer (20:00) DST regimes.
   Don't use a wall-clock-naive cron unless you want to trade an hour
   before close half the year.
8. **Phased rollout.**
    - Week 1-2: cron in `--dry-run`, log decisions, no orders.
      Compare logged decisions against backtest expected positions.
    - Week 3-6: `--live` with `--max-position 0.05` (5% per name)
      pilot. Limit damage from latent bugs.
    - Week 7+: full size (`--max-position 0.25`).

The non-trade Modal cron (daily checks: kill-switch, data freshness,
position drift vs target) is a separable, lower-risk first deploy —
ship it before the trade-submitting cron lands.

Cost: ~1 day of code + ~1 day of testing. The minimum-viable scaffold
above is the deliverable.
