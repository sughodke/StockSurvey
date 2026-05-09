# StockSurvey

![AAPL price seen through a continuous wavelet transform — multi-scale power, 2000-2026](apps/images/notebook-aapl-scalogram-morlet.png)

What does AAPL look like at every timescale at once? Above is one
ticker's price decomposed into a continuous wavelet transform — the
daily price on the top strip, multi-scale power as a heatmap below.
The horizontal bands are price-trend regimes living at different
timescales; the vertical streaks at known volatility events (2018 Q4,
2020 March, 2022 Q3) reach across the scale axis because volatility
touches every horizon at once.

This site is the source of truth for **StockSurvey** — a research
monorepo that asks what cross-sectional alpha lives in those bands,
and what doesn't.

## What we've found

- A **+0.012 cross-sectional return-IC ceiling** on the canonical
  297-ticker universe at 20-day horizon — *data-side, not
  architecture-side*. Holds across deterministic indicator stacks, raw
  CWT, SSL-pretrained CNN encoders, 7× wider universes, and quarterly
  horizons. Eight orthogonal arms have all hit the same number. See
  the [factor indicator-IC baseline](findings/factor-indicator-baseline.md).
- A relational-CWT analog-kNN strategy posting **val Sharpe 1.146** on
  a 21-name mega-cap pool — the only walk-forward arm in the codebase
  whose val *exceeds* train. Off mega-caps it collapses to 0.48,
  [confirming the macro-tailwind hypothesis](findings/relational-universe-shift.md).
  The [Leaderboard](leaderboard.md) holds every other arm we've tried.
- A multi-head CNN that learns the indicator family from masked CWT
  reconstruction and generalises zero-shot from one training ticker
  to the rest of the universe. SSL reconstruction R² is high, IC
  isn't — the [Notes](notes.md#self-supervised-pretrain-why-and-how)
  unpack why that's a finding, not a failure.

## Where to go next

- [Apps](apps/index.md) — five active research apps (regime,
  relational, factor, replay, notebook) plus this docs site.
- [Leaderboard](leaderboard.md) — append-only master table of every
  walk-forward / OOS eval, one row per arm, verdict per row.
- [Notes](notes.md) — durable framings: strategy as a dot product,
  SSL pretrain thesis, search vs optimize, multi-stock CWT.
- [Findings](findings/index.md) — historical eval results and the
  decision rationale they produced.
- [TODO](TODO/index.md) — the active backlog, one page per workstream.

## Conventions

- Distribution names are hyphenated (`ss-indicators`); import names
  are underscored (`ss_indicators`).
- Each app under `apps/` is a uv workspace member with its own
  `pyproject.toml`. Source layout is `src/<importname>/`.
- This site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
  Run locally with `uv run ss-docs-serve`.
