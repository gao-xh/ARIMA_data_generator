@echo off
setlocal

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found. Please run run_app.bat first.
    pause
    exit /b 1
)

echo [INFO] Uninstalling current PySide6...
"%PYTHON_EXE%" -m pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6

echo [INFO] Installing stable PySide6 (6.5.3)...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo [INFO] Done. Trying to launch app...
call run_app.bat
