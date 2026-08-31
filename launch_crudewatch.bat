@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="--after-python-install" goto :launch

:launch
if exist ".venv\Scripts\python.exe" (
    echo Launching CrudeWatch with .venv...
    ".venv\Scripts\python.exe" run_app.py
    goto :done
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    echo Launching CrudeWatch with Python 3.11...
    "%LocalAppData%\Programs\Python\Python311\python.exe" run_app.py
    goto :done
)

if exist "%ProgramFiles%\Python311\python.exe" (
    echo Launching CrudeWatch with Python 3.11...
    "%ProgramFiles%\Python311\python.exe" run_app.py
    goto :done
)

if exist "%ProgramFiles(x86)%\Python311\python.exe" (
    echo Launching CrudeWatch with Python 3.11...
    "%ProgramFiles(x86)%\Python311\python.exe" run_app.py
    goto :done
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 --version >nul 2>nul
    if %errorlevel% equ 0 (
        echo Launching CrudeWatch with Python launcher...
        py -3 run_app.py
        goto :done
    )
)

where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo Launching CrudeWatch with uv...
    uv run python run_app.py
    goto :done
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%P in ('where python 2^>nul ^| findstr /i /v "\\WindowsApps\\"') do (
        echo Launching CrudeWatch with python...
        "%%P" run_app.py
        goto :done
    )
)

echo Python was not found.
if "%~1"=="--after-python-install" goto :manual_python_install

where winget >nul 2>nul
if %errorlevel% neq 0 goto :manual_python_install

echo Installing Python 3.11 with winget...
winget install --id Python.Python.3.11 --exact --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo Python installation failed.
    goto :manual_python_install
)

echo.
echo Python installed. Launching CrudeWatch...
call "%~f0" --after-python-install
exit /b %errorlevel%

:done
if errorlevel 1 (
    echo.
    echo CrudeWatch exited with an error.
    pause
    exit /b %errorlevel%
)

endlocal
exit /b 0

:manual_python_install
echo.
echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
echo During install, enable "Add python.exe to PATH".
echo If Windows opens the Microsoft Store, disable the python.exe App execution alias.
pause
exit /b 1
