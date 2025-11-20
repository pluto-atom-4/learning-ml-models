#!/usr/bin/env bash
# scripts/setup.sh
# Git Bash friendly helper to create a venv, install dependencies from pyproject.toml,
# and (optionally) start Jupyter Lab using the venv's Python.
#
# Usage:
#   bash scripts/setup.sh [--dry-run] [--no-jupyter] [--extras "dev,notebook"]
# Examples:
#   bash scripts/setup.sh                # create venv, install, start jupyter
#   bash scripts/setup.sh --dry-run      # print actions but don't run them
#   bash scripts/setup.sh --no-jupyter   # do setup but don't launch jupyter
#   bash scripts/setup.sh --extras "dev"  # install extras defined in pyproject

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VENV_DIR="${REPO_ROOT}/.venv"
PYPROJECT_FILE="${REPO_ROOT}/pyproject.toml"
DRY_RUN=false
START_JUPYTER=true
EXTRAS=""

print_usage() {
  cat <<'USAGE'
Usage: scripts/setup.sh [--dry-run] [--no-jupyter] [--extras "dev,notebook"]

Options:
  --dry-run        Print actions that would be taken and exit (no changes).
  --no-jupyter     Do not start Jupyter Lab at the end.
  --extras "x,y"     Comma-separated extras to install (e.g. "dev,notebook").
  --help           Show this help and exit.

Examples:
  bash scripts/setup.sh
  bash scripts/setup.sh --dry-run
  bash scripts/setup.sh --extras "dev"
USAGE
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --no-jupyter) START_JUPYTER=false; shift ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown argument: $1"; print_usage; exit 1 ;;
  esac
done

echod() { if [[ "$DRY_RUN" == "true" ]]; then echo "[DRY] $*"; else echo "$@"; fi }
run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY] $*"
  else
    eval "$*"
  fi
}

# Find a python executable
PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else
  echo "No python executable found on PATH. Install Python 3 and retry."; exit 1;
fi

echod "Using Python: ${PY}"

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  echod "Creating virtual environment at $VENV_DIR"
  run_cmd "${PY} -m venv \"${VENV_DIR}\""
else
  echod "Virtual environment already exists: $VENV_DIR"
fi

# Determine venv pip/python paths (works for both Unix-style and Windows Git Bash venv layouts)
VENV_PIP="${VENV_DIR}/bin/pip"
VENV_PY="${VENV_DIR}/bin/python"
if [[ "$DRY_RUN" == "true" ]]; then
  # In dry-run mode we do not touch the filesystem further; just echo what would be checked.
  echod "(dry-run) would check for pip at: ${VENV_PIP} or ${VENV_DIR}/Scripts/pip"
  echod "(dry-run) would check for python at: ${VENV_PY} or ${VENV_DIR}/Scripts/python"
else
  if [[ ! -x "$VENV_PIP" ]]; then
    VENV_PIP="${VENV_DIR}/Scripts/pip"
  fi
  if [[ ! -x "$VENV_PY" ]]; then
    VENV_PY="${VENV_DIR}/Scripts/python"
  fi

  if [[ ! -x "$VENV_PIP" ]]; then
    echo "Cannot find pip inside venv ($VENV_PIP). The venv may not have been created correctly."; exit 1;
  fi
fi

# Upgrade pip/setuptools/wheel inside the venv
echod "Upgrading pip, setuptools, and wheel inside the venv"
run_cmd "\"${VENV_PIP}\" install --upgrade pip setuptools wheel"

# Install project dependencies from pyproject.toml
if [[ -f "$PYPROJECT_FILE" ]]; then
  echod "Installing project from pyproject.toml"
  # Prefer editable install if supported
  if [[ -n "$EXTRAS" ]]; then
    # Convert comma list to bracket form: "dev,notebook" -> "[dev,notebook]"
    extras_bracket="[${EXTRAS}]"
  else
    extras_bracket=""
  fi

  # Try editable install, fall back to normal install
  install_cmd="\"${VENV_PIP}\" install -e .${extras_bracket}"
  if [[ "$DRY_RUN" == "true" ]]; then
    echod "Would run: ${install_cmd}"
  else
    set +e
    eval ${install_cmd}
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echod "Editable install failed, trying a standard PEP517 install"
      run_cmd "\"${VENV_PIP}\" install ."
    fi
  fi
else
  echod "No pyproject.toml found at $PYPROJECT_FILE — skipping project install"
fi

# Optional: start Jupyter Lab
if [[ "$START_JUPYTER" == "true" ]]; then
  echod "Starting Jupyter Lab using venv's Python"
  # Run jupyter via venv python to ensure correct environment
  JUPYTER_CMD="\"${VENV_PY}\" -m jupyter lab --port 8888 --no-browser"
  echod "Jupyter command: ${JUPYTER_CMD}"
  if [[ "$DRY_RUN" == "true" ]]; then
    echod "Dry run: not launching Jupyter"
    exit 0
  else
    # Exec replaces this shell with the jupyter process (useful when calling script directly)
    exec ${VENV_PY} -m jupyter lab --port 8888 --no-browser
  fi
else
  echod "--no-jupyter specified; setup complete. Activate the venv with: source .venv/Scripts/activate"
fi
