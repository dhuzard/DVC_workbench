@echo off
setlocal EnableExtensions
REM One-click launcher for non-developer beta testers (Windows).
REM Tries the pre-built image from GHCR first (seconds). Falls back to a
REM local build (minutes) if the pull fails -- e.g. you are offline, on a
REM branch with no published image, or behind a registry block.
REM
REM Requires: Docker Desktop installed and running.

cd /d "%~dp0"
set "PULL_LOG=%TEMP%\dvc-workbench-pull.log"

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

docker compose version >nul 2>nul
if errorlevel 1 (
  echo Docker is installed, but Docker Compose v2 is not available.
  echo Update Docker Desktop, then re-run this script.
  pause
  exit /b 1
)

echo Trying the pre-built image from GitHub Container Registry...
docker compose -f docker-compose.prebuilt.yml pull >"%PULL_LOG%" 2>&1
if errorlevel 1 goto prebuilt_failed

del "%PULL_LOG%" >nul 2>nul
echo Pre-built image ready. Starting DVC Workbench at http://localhost:8501
echo Press Ctrl+C to stop.
docker compose -f docker-compose.prebuilt.yml up
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0
goto startup_failed

:prebuilt_failed
echo.
echo Pre-built image not available - falling back to a local build.
echo This is normal on a private branch, when offline, or behind a registry block.
echo Pull details were saved to "%PULL_LOG%".
echo First build can take several minutes.
docker compose build
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)
echo Starting DVC Workbench at http://localhost:8501
echo Press Ctrl+C to stop.
docker compose up
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0

:startup_failed
echo.
echo DVC Workbench stopped with an error. Exit code: %EXIT_CODE%
echo If you report this, include the last 20 lines printed above.
pause
exit /b %EXIT_CODE%
