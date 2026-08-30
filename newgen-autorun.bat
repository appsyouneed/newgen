@echo off
setlocal EnableDelayedExpansion
title NewGen Autorun

set DOWNLOAD_DIR=D:\Apps\newgen\downloads
set AUTORUN_DIR=D:\Apps\newgen\newgen-autorun
set PYTHON=python
set FEEDER=%~dp0autorun_feeder.py

echo.
echo  ================================================
echo   NewGen Autorun Launcher
echo  ================================================
echo.

:: ---- Prompt for SSH command ----
set /p SSH_CMD=Enter SSH command (e.g. ssh root@159.48.242.33 -p20202): 

:: Strip leading "ssh " to get the rest
set SSH_REST=!SSH_CMD:ssh =!

:: Default values
set SSH_USERHOST=
set SSH_PORT=22

:: Walk every token in the command
set PREV=
for %%T in (!SSH_REST!) do (
    if "!PREV!"=="-p" (
        set SSH_PORT=%%T
    ) else (
        if not "%%T"=="-p" (
            if not "%%T"=="-N" (
                if not "%%T"=="-L" (
                    if "!SSH_USERHOST!"=="" (
                        set SSH_USERHOST=%%T
                    )
                )
            )
        )
    )
    set PREV=%%T
)

echo.
echo  SSH target : !SSH_USERHOST!
echo  SSH port   : !SSH_PORT!
echo  Tunnel     : localhost:7861 ^<-^> VPS 127.0.0.1:7861
echo  Images     : !AUTORUN_DIR!
echo  Videos     : !DOWNLOAD_DIR!
echo.

:: ---- Checks ----
where ssh >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ssh.exe not found. Enable OpenSSH in Windows Settings ^(Apps ^> Optional Features^).
    pause & exit /b 1
)
where !PYTHON! >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause & exit /b 1
)
if not exist "!FEEDER!" (
    echo [ERROR] autorun_feeder.py not found at: !FEEDER!
    pause & exit /b 1
)
if not exist "!AUTORUN_DIR!" (
    echo [ERROR] Autorun image folder not found: !AUTORUN_DIR!
    pause & exit /b 1
)
if not exist "!DOWNLOAD_DIR!" mkdir "!DOWNLOAD_DIR!"

:: ---- Open SSH tunnel in its own visible window ----
:: /k keeps the window open after the command exits so you can see errors.
:: You type your password in that window when it prompts you.
echo  Opening SSH tunnel window...
echo  ^> Type your SSH password in the new window that opens, then come back here.
echo.

start "NewGen SSH Tunnel -- type password here" cmd /k "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=9999 -L 7861:127.0.0.1:7861 -N -p !SSH_PORT! !SSH_USERHOST!"

:: Give the user a moment to see the password prompt
timeout /t 3 /nobreak >nul

:: ---- Wait for tunnel to come up ----
echo  Waiting for tunnel on port 7861...
echo  (Enter your password in the other window now if you have not yet.)
echo.
:wait_tunnel
!PYTHON! -c "import socket,sys;s=socket.socket();s.settimeout(2);sys.exit(0 if s.connect_ex(('127.0.0.1',7861))==0 else 1)" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_tunnel
)
echo  Tunnel is up^^!
echo.

:: ---- Run feeder ----
echo  Starting feeder...
echo  Make sure you clicked [Start Push Autorun] in the Gradio browser tab first.
echo.
!PYTHON! "!FEEDER!" --folder "!AUTORUN_DIR!" --port 7861 --download-dir "!DOWNLOAD_DIR!"

echo.
if errorlevel 1 (
    echo  Feeder finished with errors -- check output above.
) else (
    echo  All done^^!  Videos saved to: !DOWNLOAD_DIR!
)
echo.
echo  The SSH tunnel window is still open. Close it when you are finished.
pause
