#!/usr/bin/env bash
#
# One-shot setup for CrudeWatch on a fresh Ubuntu VM (Oracle Cloud Always Free).
# It installs Docker + Tailscale, builds & runs the app, and serves it to your
# private tailnet only (no public URL, no public ports opened).
#
# Usage (run from inside the cloned repo on the VM):
#   cd crude_watch
#   bash deploy/setup_oracle.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> [1/6] Checking the data workbook is present"
if [ ! -f data/raw_files.xlsx ]; then
  echo "ERROR: data/raw_files.xlsx not found in $REPO_DIR" >&2
  echo "       Commit it to the repo (it is baked into the app), then re-run." >&2
  exit 1
fi

echo "==> [2/6] Installing Docker (if missing)"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "==> [3/6] Building the app image (first build takes a few minutes)"
sudo docker build -t crudewatch "$REPO_DIR"

echo "==> [4/6] (Re)starting the container on 127.0.0.1:8501"
sudo docker rm -f crudewatch >/dev/null 2>&1 || true
sudo docker run -d --restart unless-stopped \
  -p 127.0.0.1:8501:8501 --name crudewatch crudewatch

echo "==> [5/6] Installing Tailscale (if missing)"
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "==> [6/6] Connecting to your tailnet and serving privately"
echo "    If prompted, open the printed URL in a browser to authenticate."
sudo tailscale up
sudo tailscale serve --bg 8501

echo
echo "======================================================================"
echo " Done. Your PRIVATE app URL (reachable only inside your tailnet):"
sudo tailscale serve status || true
echo "======================================================================"
echo "Invite people in the Tailscale admin console; they install Tailscale,"
echo "sign in, and open the URL above. Nobody outside your tailnet can reach it."
