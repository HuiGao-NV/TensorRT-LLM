#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NVInfo CLI health check — diagnostic only, makes NO network calls.
#
# Checks:
#   1. Is `nvinfo` on PATH?
#   2. Is Node.js >= 18 installed?
#   3. Is the user authenticated?
#
# Exit codes:
#   0 — CLI found and ready (or found + auth check completed)
#   1 — CLI missing or Node.js prerequisite not met
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MIN_NODE_MAJOR=18
NVINFO_SERVER="https://nvinfo.nvidia.com"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[  ok]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Check: CLI present ─────────────────────────────────────────────────────
if command -v nvinfo &>/dev/null; then
  CLI_VERSION="$(nvinfo --version 2>/dev/null || echo 'ok')"
  info "NVInfo CLI found (version: $CLI_VERSION)"

  # Non-fatal auth check
  if nvinfo auth status 2>/dev/null | grep -qi "signed in\|active\|authenticated"; then
    info "Authentication: active"
  else
    warn "Not signed in. Run: nvinfo login ${NVINFO_SERVER}"
  fi

  exit 0
fi

# ── CLI not found — provide diagnostics ────────────────────────────────────
fail "NVInfo CLI is not installed."
echo ""

if ! command -v node &>/dev/null; then
  fail "Node.js is not installed (required for CLI)."
  echo ""
  echo "  To install NVInfo CLI:"
  echo "    1. Install Node.js 18+:"
  echo "       macOS:   brew install node@20"
  echo "       Linux:   sudo apt install nodejs npm"
  echo "       Windows: winget install OpenJS.NodeJS.LTS"
  echo "       All:     https://nodejs.org"
  echo ""
  echo "    2. Re-run the NVInfo deployment policy (contact IT)"
  echo "       or run: bash install.sh from the distribution package"
  exit 1
fi

NODE_VERSION="$(node --version | sed 's/^v//')"
NODE_MAJOR="${NODE_VERSION%%.*}"

if [[ "$NODE_MAJOR" -lt "$MIN_NODE_MAJOR" ]]; then
  fail "Node.js $NODE_VERSION found, but ${MIN_NODE_MAJOR}+ is required."
  echo ""
  echo "  Upgrade: nvm install 20 && nvm use 20"
  echo "  Or: brew upgrade node"
  exit 1
fi

info "Node.js $NODE_VERSION found (meets minimum)"
echo ""
echo "  Node.js is ready but the NVInfo CLI is not installed."
echo "  Ask your IT admin to re-run the NVInfo deployment policy,"
echo "  or install manually:"
echo ""
echo "    curl -fsSL ${NVINFO_SERVER}/api/cli/install | sh"
echo ""
exit 1
