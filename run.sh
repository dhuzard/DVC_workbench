#!/usr/bin/env bash
# One-click launcher for non-developer beta testers (macOS / Linux).
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

echo "Building DVC Workbench (first run can take several minutes)..."
docker compose build

echo "Starting DVC Workbench at http://localhost:8501"
echo "Press Ctrl+C to stop."
docker compose up
