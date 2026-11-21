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

echo ==================================================
echo  Nepal News Bot - Finish Vertical (TikTok/Reels 9:16)
echo  Folder: %PROJECT_DIR%
echo ==================================================
"%PROJECT_DIR%\.venv\Scripts\python.exe" "%PROJECT_DIR%\run_daily.py" --finish --orientation vertical %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [FINISHED] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
