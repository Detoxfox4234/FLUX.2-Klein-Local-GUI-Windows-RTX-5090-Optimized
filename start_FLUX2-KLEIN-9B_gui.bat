@echo off
title FLUX2 KLEIN 9B
echo Starting App...

if not exist "python_env\python.exe" (
    echo ERROR: Python environment not found!
    echo Please run 'install.bat' first.
    pause
    exit
)

.\python_env\python.exe app.py
pause
