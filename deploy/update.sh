#!/usr/bin/env bash
#
# Redeploy CrudeWatch after pulling new code/data on the VM.
#   cd crude_watch && bash deploy/update.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

git pull --ff-only
sudo docker build -t crudewatch "$REPO_DIR"
sudo docker rm -f crudewatch >/dev/null 2>&1 || true
sudo docker run -d --restart unless-stopped \
  -p 127.0.0.1:8501:8501 --name crudewatch crudewatch
echo "==> Redeployed. Tailscale serve keeps running; the URL is unchanged."
