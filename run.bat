@echo off
REM One-click launcher for non-developer beta testers (Windows).
REM Tries the pre-built image from GHCR first (seconds). Falls back to a
REM local build (minutes) if the pull fails.
REM
REM Requires: Docker Desktop installed and running.

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
  echo Docker is not installed.
  echo Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and re-run this script.
  pause
  exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
  echo Docker Desktop is installed but not running. Please start it and re-run this script.
  pause
  exit /b 1
)

echo Trying the pre-built image from GitHub Container Registry...
docker compose -f docker-compose.prebuilt.yml pull
if not errorlevel 1 (
  echo Pre-built image ready. Starting DVC Workbench at http://localhost:8501
  echo Close this window to stop.
  docker compose -f docker-compose.prebuilt.yml up
  exit /b 0
)

echo.
echo Pre-built image not available - falling back to a local build.
echo First build can take several minutes.
docker compose build
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo Starting DVC Workbench at http://localhost:8501
echo Close this window to stop.
docker compose up
