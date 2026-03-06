@echo off
setlocal

:: Configuration
set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS=requirements.txt"
set "MAIN_SCRIPT=run_ui.py"

:: Check if Python VENV exists
if not exist "%PYTHON_EXE%" (
    echo [INFO] Virtual environment not found at %VENV_DIR%
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Please check python installation.
        pause
        exit /b 1
    )
    
    echo [INFO] Upgrade pip...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    
    echo [INFO] Installing dependencies...
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Run the application
echo [INFO] Starting Application...
"%PYTHON_EXE%" "%MAIN_SCRIPT%"
if errorlevel 1 (
    echo [ERROR] Application crashed.
    pause
)

endlocal
