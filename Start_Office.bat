@echo off
setlocal
set "PROJECT_DIR=C:\news-auto"
cd /d "%PROJECT_DIR%"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] Missing virtual environment Python at %PROJECT_DIR%\.venv\Scripts\python.exe
    pause
    exit /b 1
)

echo ==================================================
echo  Nepal News Bot HQ - 2D Office Web Interface
echo  URL: http://127.0.0.1:8765
echo  Press Ctrl+C in this window to stop the server.
echo ==================================================

start "" "http://127.0.0.1:8765"
"%PROJECT_DIR%\.venv\Scripts\python.exe" -m uvicorn web_office.server:app --host 127.0.0.1 --port 8765
pause
