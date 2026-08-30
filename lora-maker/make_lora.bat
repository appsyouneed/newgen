@echo off
title LoRA Maker — Wan 2.2 I2V
color 0B

echo.
echo  LoRA Maker — Wan 2.2 I2V Subject Trainer
echo  ==========================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Check Git is installed (needed to clone musubi-tuner)
git --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Git not found.
    echo.
    echo  Please install Git from:
    echo    https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

:: Run the app
echo  Launching GUI...
echo.
python "%~dp0make_lora.py"

if errorlevel 1 (
    echo.
    echo  [ERROR] App exited with an error.
    echo  Check the output above for details.
    pause
)
