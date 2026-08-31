@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="--after-python-install" goto :launch

:launch
if exist ".venv\Scripts\python.exe" (
    goto :run_with_venv
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python311\python.exe""
    goto :ensure_venv
)

if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_CMD="%ProgramFiles%\Python311\python.exe""
    goto :ensure_venv
)

if exist "%ProgramFiles(x86)%\Python311\python.exe" (
    set "PYTHON_CMD="%ProgramFiles(x86)%\Python311\python.exe""
    goto :ensure_venv
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 --version >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py -3"
        goto :ensure_venv
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
        set "PYTHON_CMD="%%P""
        goto :ensure_venv
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

:ensure_venv
echo Creating CrudeWatch virtual environment...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo.
    echo Could not create the virtual environment.
    pause
    exit /b %errorlevel%
)

:run_with_venv
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if %errorlevel% equ 0 goto :start_app

echo Installing CrudeWatch dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :pip_error

if exist "requirements.txt" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -e ".[app]"
)
if errorlevel 1 goto :pip_error

:start_app
echo Launching CrudeWatch with .venv...
".venv\Scripts\python.exe" run_app.py
goto :done

:pip_error
echo.
echo Could not install CrudeWatch dependencies.
echo Check your internet connection and try again.
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
exit /b 0

:manual_python_install
echo.
echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
echo During install, enable "Add python.exe to PATH".
echo If Windows opens the Microsoft Store, disable the python.exe App execution alias.
pause
exit /b 1
