#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHERUSAGESTATS=false

if command -v uv >/dev/null 2>&1; then
  uv run streamlit run app/main.py --server.port="${PORT:-8501}" --server.address="${HOST:-127.0.0.1}"
else
  python -m streamlit run app/main.py --server.port="${PORT:-8501}" --server.address="${HOST:-127.0.0.1}"
fi
