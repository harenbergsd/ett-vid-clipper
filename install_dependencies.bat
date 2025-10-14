@echo off
cd /d "%~dp0"

winget install uv

uv venv --clear

call .venv\Scripts\activate.bat

uv pip install -r requirements.txt

echo.
echo Done. Press any key to exit.
pause >nul