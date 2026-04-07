@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "PYTHON_EXE=C:\Users\lwwde\miniconda3\envs\py312\python.exe"
set "SCRIPT_PATH=%REPO_ROOT%02_gemma4_generation\demo_chatbot_web_mvp.py"
set "HOST=0.0.0.0"
set "PORT=8787"
set "URL=http://127.0.0.1:%PORT%"
set "BACKEND=transformers"
set "MODEL=gemma4_2b"

if not exist "%SCRIPT_PATH%" (
  echo [ERROR] demo_chatbot_web_mvp.py not found: %SCRIPT_PATH%
  pause
  exit /b 1
)

echo Starting Web Demo MVP...
echo backend=%BACKEND% model=%MODEL% local_url=%URL%
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue ^| ? { $_.IPAddress -ne '127.0.0.1' -and $_.PrefixOrigin -ne 'WellKnown' } ^| select -ExpandProperty IPAddress -Unique; $ips -join ', '"`) do set "LAN_IPS=%%I"
for /f "tokens=1 delims=, " %%A in ("%LAN_IPS%") do set "LAN_IP_FIRST=%%A"
if defined LAN_IPS (
  echo LAN access IPs: %LAN_IPS%
  echo Example URL: http://%LAN_IP_FIRST%:%PORT%
)
echo.
echo Browser will open in 2 seconds.
echo Stop server with Ctrl + C in this window.
echo.

start "" "%URL%"
timeout /t 2 /nobreak >nul

"%PYTHON_EXE%" "%SCRIPT_PATH%" --backend %BACKEND% --model %MODEL% --host %HOST% --port %PORT%

echo.
echo Server stopped.
pause
