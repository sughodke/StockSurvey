# apps/docs

Material for MkDocs site for the StockSurvey monorepo.

## Local development

```bash
uv sync --all-packages --inexact   # picks up ss-docs as a workspace member
uv run ss-docs-serve               # http://127.0.0.1:8000 with live reload
uv run ss-docs-build               # static site → apps/docs/site/
```

The `ss-docs-serve` / `ss-docs-build` scripts are thin wrappers that `chdir`
into `apps/docs/` and invoke `mkdocs serve` / `mkdocs build`. You can also
run `mkdocs` directly from this directory.

## Layout

- `mkdocs.yml` — site config (Material theme + extensions).
- `docs/` — markdown content. `index.md` is the landing page.
- `src/ss_docs/` — tiny CLI shim so the workspace member is buildable.
- `site/` — build output (gitignored).
