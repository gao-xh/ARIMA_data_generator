@echo off
echo Cleaning up environment...
if exist ".venv" (
    rmdir /s /q .venv
    echo Removed .venv
)
if exist "__pycache__" (
    rmdir /s /q __pycache__
    echo Removed __pycache__
)
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo Re-running setup...
call run_app.bat
