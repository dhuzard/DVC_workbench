#!/usr/bin/env bash
# One-click launcher for non-developer beta testers (macOS / Linux).
# Tries the pre-built image from GHCR first (seconds). Falls back to a
# local build (minutes) if the pull fails — e.g. you are offline, on a
# branch with no published image, or behind a registry block.
#
# Requires: Docker Desktop installed and running.
set -e

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed."
  echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and run this script again."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is installed but not running. Please start it and re-run this script."
  exit 1
fi

echo "Trying the pre-built image from GitHub Container Registry..."
if docker compose -f docker-compose.prebuilt.yml pull; then
  echo "Pre-built image ready. Starting DVC Workbench at http://localhost:8501"
  echo "Press Ctrl+C to stop."
  exec docker compose -f docker-compose.prebuilt.yml up
fi

echo
echo "Pre-built image not available — falling back to a local build."
echo "(This is normal on a private branch or if you are offline.)"
echo "First build can take several minutes."
docker compose build
echo "Starting DVC Workbench at http://localhost:8501"
echo "Press Ctrl+C to stop."
exec docker compose up
