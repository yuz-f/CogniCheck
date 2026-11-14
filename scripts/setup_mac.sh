#!/bin/sh
# One-time setup on a Mac: create the project environment (.venv) and install everything.
# Run from the repository folder:  sh scripts/setup_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python 3.10 or newer found. Install one with:  brew install python@3.12"; exit 1
fi
echo "Using $PY ($("$PY" --version))"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt torch zstandard pytest
.venv/bin/python -m pytest -q tests/test_chessnet.py
echo "Done. In VS Code: Cmd+Shift+P, 'Developer: Reload Window'. The Run button then uses .venv."
