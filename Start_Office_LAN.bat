@echo off
setlocal
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] Missing virtual environment Python at %PROJECT_DIR%\.venv\Scripts\python.exe
    pause
    exit /b 1
)

if "%OFFICE_TOKEN%"=="" (
    for /f "usebackq delims=" %%i in (`"%PROJECT_DIR%\.venv\Scripts\python.exe" -c "import secrets; print(secrets.token_hex(32))"`) do set "OFFICE_TOKEN=%%i"
)

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1 -ExpandProperty IPAddress)"`) do set "LAN_IP=%%i"

echo ==================================================
echo  Nepal News Bot HQ - 2D Office (LAN Mode)
echo  Security Token: %OFFICE_TOKEN%
echo  PC URL:    http://127.0.0.1:8765/?token=%OFFICE_TOKEN%
if defined LAN_IP (
    echo  Phone URL: http://%LAN_IP%:8765/?token=%OFFICE_TOKEN%
) else (
    echo  Phone URL: Could not detect LAN IP automatically
)
echo  Note: LAN requests require token authentication.
echo  Press Ctrl+C in this window to stop the server.
echo ==================================================

start "" "http://127.0.0.1:8765/?token=%OFFICE_TOKEN%"
"%PROJECT_DIR%\.venv\Scripts\python.exe" -m uvicorn web_office.server:app --host 0.0.0.0 --port 8765
pause
