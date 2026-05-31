@echo off
cd /d "%~dp0"

set "USERPROFILE=%CD%"
set "HOME=%CD%"
set "PYTHON_EXE="

if exist "C:\Users\UNLOST\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=C:\Users\UNLOST\AppData\Local\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_EXE (
    py -3.12 --version >nul 2>nul
    if %ERRORLEVEL% EQU 0 set "PYTHON_EXE=py -3.12"
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    if exist "C:\Program Files\Python312\python.exe" (
        set "PYTHON_EXE=C:\Program Files\Python312\python.exe"
    )
)

if not defined PYTHON_EXE (
    if exist "C:\Users\UNLOST\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
        set "PYTHON_EXE=C:\Users\UNLOST\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo Python 3.12 not found.
    echo.
    echo Install Python 3.12 and enable "Add python.exe to PATH".
    echo Then run:
    echo python -m pip install -r requirements.txt
    echo python -m streamlit run app.py
    pause
    exit /b 1
)

echo Using Python:
%PYTHON_EXE% --version
echo.
echo Starting Streamlit...
echo.
echo. | %PYTHON_EXE% -m streamlit run app.py --browser.gatherUsageStats false
