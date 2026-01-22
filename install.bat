@echo off
:: Bulletproof Installer for Flux.2 Klein
title Install Flux.2 Klein
cls

echo ---------------------------------------------------------------------
echo   FLUX.2 KLEIN - INSTALLATION
echo ---------------------------------------------------------------------
echo.

:: --- CHECK 1: Python ---
python --version >nul 2>&1
if %errorlevel% EQU 0 goto :FOUND_PYTHON

:: Fallback: Try 'py' launcher
py --version >nul 2>&1
if %errorlevel% EQU 0 goto :FOUND_PY

:: If neither found
color 0c
echo [ERROR] Python not found!
echo Please install Python 3.10+ and add it to PATH.
pause
exit /b

:FOUND_PY
set PYTHON_CMD=py
goto :SETUP_VENV

:FOUND_PYTHON
set PYTHON_CMD=python
goto :SETUP_VENV

:SETUP_VENV
echo Using Python command: %PYTHON_CMD%
echo.

:: --- CHECK 2: Venv ---
if exist "venv" goto :INSTALL_DEPS
echo [1/3] Creating virtual environment...
%PYTHON_CMD% -m venv venv
if %errorlevel% NEQ 0 goto :ERROR_VENV

:INSTALL_DEPS
echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat

%PYTHON_CMD% -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% NEQ 0 goto :ERROR_INSTALL

:: --- CHECK 3: Login ---
echo.
echo [3/3] HUGGING FACE LOGIN
echo.
echo Do you want to log in now? (Required for Gated Models)
set /p login_choice="Type Y for Yes, N for No: "
if /i "%login_choice%"=="Y" goto :DO_LOGIN
goto :FINISH

:DO_LOGIN
huggingface-cli login
goto :FINISH

:ERROR_VENV
color 0c
echo [ERROR] Could not create venv.
pause
exit /b

:ERROR_INSTALL
color 0c
echo [ERROR] Installation failed. Check internet connection.
pause
exit /b

:FINISH
echo.
echo =========================================================
echo   INSTALLATION COMPLETE!
echo   You can now start the app.
echo =========================================================
pause