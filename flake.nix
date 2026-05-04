{
  description = "StockSurvey dev shell: Python 3.13 + LLVM 19 (for numba/llvmlite/vectorbt) + uv.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # Python with the numba/llvmlite stack pre-built by nix. PyPI has no
        # llvmlite wheel for Python 3.13 / Intel-macOS and the source build
        # fails (LLVM 19 vs llvmlite-0.47 CMake mismatch). nixpkgs' versions
        # are pinned to compatible LLVM and Just Work on this platform.
        pythonWithNumba = pkgs.python313.withPackages (ps: [
          ps.numba
          ps.llvmlite
          ps.numpy
          ps.scipy
          ps.pandas
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonWithNumba
            pkgs.uv
            pkgs.dolt
          ];

          # uv should use the nix Python (which already has numba) instead
          # of downloading its own. Together with `uv venv --system-site-
          # packages`, this lets pure-Python deps (vectorbt itself) install
          # via uv while the C-heavy numba stack stays in /nix/store.
          UV_PYTHON_PREFERENCE = "only-system";

          shellHook = ''
            echo "StockSurvey dev shell"
            echo "  python : $(python3.13 --version)"
            echo "  uv     : $(uv --version)"
            echo "  dolt   : $(dolt version 2>/dev/null | head -1 || echo not yet importable)"
            echo "  numba  : $(python3.13 -c 'import numba; print(numba.__version__)' 2>/dev/null || echo not yet importable)"
            echo ""
            echo "First-time setup:"
            echo "  rm -rf .venv && uv venv --system-site-packages && uv sync --all-packages --inexact"
          '';
        };
      });
}
