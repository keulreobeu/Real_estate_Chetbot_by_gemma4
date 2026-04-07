@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "PS_SCRIPT=%REPO_ROOT%run_web_demo_public.ps1"

if not exist "%PS_SCRIPT%" (
  echo [ERROR] Script not found: %PS_SCRIPT%
  pause
  exit /b 1
)

powershell -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Backend transformers -Model gemma4_2b -OpenBrowser

echo.
pause
