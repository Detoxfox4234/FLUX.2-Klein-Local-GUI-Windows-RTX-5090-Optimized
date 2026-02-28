@echo off
:: Bulletproof Installer for Flux.2 Klein 9B
title Install Flux.2 Klein 9B
cls

echo ---------------------------------------------------------------------
echo   FLUX.2 KLEIN 9B - INSTALLATION
echo ---------------------------------------------------------------------
echo.

:: 1. Check if Python exists
if exist "python_env\python.exe" (
    echo Python environment found. Skipping download.
    goto :INSTALL_PACKAGES
)

:: 2. Download Python Portable (3.11.9)
echo [1/4] Downloading Python 3.11 Portable...
if not exist "python_env" mkdir "python_env"
curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -o "python_env\python.zip"

:: 3. Unzip
echo [2/4] Extracting Python...
tar -xf "python_env\python.zip" -C "python_env"
del "python_env\python.zip"

:: 4. Patch ._pth file (enable import site)
echo [3/4] Configuring Python...
(
echo python311.zip
echo .
echo import site
) > "python_env\python311._pth"

:: 5. Install Pip
echo [4/4] Installing Pip Package Manager...
curl -L "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
.\python_env\python.exe get-pip.py --no-warn-script-location
del "get-pip.py"

:INSTALL_PACKAGES
echo.
echo ===================================================
echo   Installing Libraries...
echo ===================================================

:: A. PyTorch Nightly (Must be first for RTX 5090)
echo Installing PyTorch Nightly (Blackwell Support)...
.\python_env\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --no-warn-script-location

:: B. Other Requirements
echo Installing Flux.2 Klein and Tools...
.\python_env\python.exe -m pip install -r requirements.txt --no-warn-script-location

:: C. Cleanup
echo Cleaning up download cache...
.\python_env\python.exe -m pip cache purge

echo.
echo ===================================================
echo   INSTALLATION COMPLETE!
echo   You can now run 'start_FLUX2-KLEIN-9B_gui.bat'.
echo ===================================================
pause
