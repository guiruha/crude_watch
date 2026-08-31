@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Launching CrudeWatch with .venv...
    ".venv\Scripts\python.exe" run_app.py
    goto :done
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Launching CrudeWatch with uv...
    uv run python run_app.py
    goto :done
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    echo Launching CrudeWatch with Python launcher...
    py -3 run_app.py
    goto :done
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    echo Launching CrudeWatch with python...
    python run_app.py
    goto :done
)

echo Python was not found.
echo Install Python 3.10 or newer, or install uv, then run this file again.
pause
exit /b 1

:done
if errorlevel 1 (
    echo.
    echo CrudeWatch exited with an error.
    pause
    exit /b %errorlevel%
)

endlocal
