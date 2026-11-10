@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Missing virtual environment Python at .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo ==================================================
echo  Nepal News Bot - Morning Prep (--morning)
echo ==================================================
".venv\Scripts\python.exe" run_daily.py --morning %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [FINISHED] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
