"""Thin wrappers so `uv run ss-docs-serve` / `ss-docs-build` work from the repo root.

`mkdocs` reads `mkdocs.yml` from the working directory; we chdir into `apps/docs`
before delegating to the standard `mkdocs` CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mkdocs.__main__ import cli as mkdocs_cli


def _chdir_to_app() -> None:
    os.chdir(Path(__file__).resolve().parents[2])


def serve() -> None:
    _chdir_to_app()
    sys.argv = ["mkdocs", "serve", *sys.argv[1:]]
    mkdocs_cli()


def build() -> None:
    _chdir_to_app()
    sys.argv = ["mkdocs", "build", *sys.argv[1:]]
    mkdocs_cli()
