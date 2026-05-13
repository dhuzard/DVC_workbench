@echo off
REM One-click launcher for non-developer beta testers (Windows).
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

echo Building DVC Workbench (first run can take several minutes)...
docker compose build
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo Starting DVC Workbench at http://localhost:8501
echo Close this window to stop.
docker compose up
