#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-$(date +%Y.%m.%d)}"
OUT_DIR="$ROOT/release/CrudeWatch-$VERSION"
ZIP_PATH="$ROOT/release/CrudeWatch-$VERSION.zip"

rm -rf "$OUT_DIR" "$ZIP_PATH"
mkdir -p "$OUT_DIR"

echo "Prebuilding parquet cache..."
if command -v uv >/dev/null 2>&1; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}" uv run python scripts/prebuild_cache.py
else
  python scripts/prebuild_cache.py
fi

echo "Collecting release files..."
rsync -a \
  --exclude '.git' \
  --exclude '.github' \
  --exclude '.agents' \
  --exclude '.codex' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '.mypy_cache' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude 'release' \
  --exclude '.streamlit/secrets.toml' \
  app src data deploy scripts \
  pyproject.toml requirements.txt uv.lock README.md Dockerfile docker-compose.yml run_app.py run.sh CrudeWatch.spec \
  "$OUT_DIR/"

cat > "$OUT_DIR/START_HERE.txt" <<'EOF'
CrudeWatch

Fast start:
  Linux/macOS:
    ./run.sh

  Windows with Python:
    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/main.py

  Docker:
    docker compose up -d --build
    open http://127.0.0.1:8501

The release includes data/processed parquet caches, so the first launch should not rebuild the raw workbook.
EOF

chmod +x "$OUT_DIR/run.sh"

echo "Creating zip..."
(cd "$ROOT/release" && zip -qr "CrudeWatch-$VERSION.zip" "CrudeWatch-$VERSION")

echo "Release ready:"
echo "  $OUT_DIR"
echo "  $ZIP_PATH"
