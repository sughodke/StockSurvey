# StockSurvey

A `uv`-workspace monorepo for trading-strategy research and live execution.

This site is the **source of truth** for the project. Architecture sits
in `CLAUDE.md` (the LLM-facing operational reference); everything else
— findings, workflows, the active backlog — lives here.

## Sections

- [Apps](apps/index.md) — runnable strategies, trainers, CLI tools.
- [Packages](packages/index.md) — shared numpy primitives.
- [Leaderboard](leaderboard.md) — append-only master table of every
  walk-forward / OOS eval, with verdict per row.
- [Findings](findings/index.md) — historical eval results and the
  decision rationale they produced.
- [Workflows](workflows.md) — how to add an indicator, swap brokers,
  author a relational checkpoint, etc.
- [TODO](TODO/index.md) — active backlog, one page per workstream.

## Conventions

- Distribution names are hyphenated (`ss-indicators`); import names
  are underscored (`ss_indicators`).
- Each app under `apps/` is a uv workspace member with its own
  `pyproject.toml`. Source layout is `src/<importname>/`.
- This site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
  Run locally with `uv run ss-docs-serve`.
