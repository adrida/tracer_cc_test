#!/usr/bin/env bash
# tracerCC bootstrap — one command, opens your AI-coding Wrapped.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/adrida/tracer_cc_test/main/install.sh | bash
#
# What it does:
#   1. Installs `uv` (~10 MB, isolated Python toolchain) if missing.
#   2. Runs `tracercc` via `uvx` — no global install, no venv, no creds.
#
# Override target source for local dev:
#   TRACERCC_SOURCE=/path/to/local/tracerCC bash install.sh
#
# Override the analysis backend (rare; default is hosted):
#   TRACERCC_BACKEND_URL=https://my.backend ./install.sh

set -euo pipefail

C_BOLD="$(tput bold 2>/dev/null || true)"
C_DIM="$(tput dim 2>/dev/null || true)"
C_GRN="$(tput setaf 2 2>/dev/null || true)"
C_RED="$(tput setaf 1 2>/dev/null || true)"
C_RST="$(tput sgr0 2>/dev/null || true)"

say()  { printf '%s\n' "${C_DIM}→${C_RST} $*"; }
ok()   { printf '%s\n' "${C_GRN}✓${C_RST} $*"; }
err()  { printf '%s\n' "${C_RED}✗${C_RST} $*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (Python toolchain, ~10 MB)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  err "uv install failed — please install manually: https://docs.astral.sh/uv/"
  exit 1
fi
ok "uv ready ($(uv --version 2>&1 | head -1))"

SOURCE="${TRACERCC_SOURCE:-git+https://github.com/adrida/tracer_cc_test.git}"
say "running tracerCC from ${SOURCE} ..."
echo
exec uvx --python 3.11 --from "$SOURCE" tracercc "$@"
