@echo off
setlocal EnableDelayedExpansion
title NewGen Autorun

set DOWNLOAD_DIR=D:\Apps\newgen\downloads
set AUTORUN_DIR=D:\Apps\newgen\newgen-autorun
set PYTHON=python
set FEEDER=%~dp0autorun_feeder.py
set API_PORT=7861

echo.
echo  ================================================
echo   NewGen Autorun Launcher
echo  ================================================
echo.

:: ---- Prompt for SSH command ----
echo  Paste your SSH command exactly as you would type it in a terminal.
echo  Examples:
echo    ssh root@1.2.3.4
echo    ssh root@1.2.3.4 -p 20202
echo    ssh -p 20202 root@1.2.3.4
echo    ssh -i C:\keys\mykey.pem root@1.2.3.4 -p 20202
echo.
set /p SSH_CMD=SSH command: 

:: ---- Parse user@host and port from the SSH command ----------------------
:: Strategy: scan every token; the first token that contains '@' is the
:: user@host; the token that follows '-p' is the port.  Identity files
:: (-i path) and other flags are preserved verbatim via SSH_EXTRA_ARGS so
:: the tunnel window can use them.
set SSH_USERHOST=
set SSH_PORT=22
set SSH_EXTRA_ARGS=
set _PREV_TOKEN=

:: Strip the leading "ssh" command word before looping
set _REST=!SSH_CMD!
for /f "tokens=1,* delims= " %%A in ("!_REST!") do (
    if /i "%%A"=="ssh" (set _REST=%%B) else (set _REST=!SSH_CMD!)
)

for %%T in (!_REST!) do (
    :: Token after -p is the port number
    if "!_PREV_TOKEN!"=="-p" (
        set SSH_PORT=%%T
        set _PREV_TOKEN=
        goto :next_token_%%T
    )
    :: Token after -i is an identity file path — preserve it
    if "!_PREV_TOKEN!"=="-i" (
        set SSH_EXTRA_ARGS=!SSH_EXTRA_ARGS! -i %%T
        set _PREV_TOKEN=
        goto :next_token_%%T
    )
    :: Skip flag tokens but remember them for the next iteration
    if "%%T"=="-p" (set _PREV_TOKEN=-p& goto :next_token_%%T)
    if "%%T"=="-i" (set _PREV_TOKEN=-i& goto :next_token_%%T)
    if "%%T"=="-N" (set _PREV_TOKEN=& goto :next_token_%%T)
    if "%%T"=="-L" (set _PREV_TOKEN=-L& goto :next_token_%%T)
    :: First token with '@' is user@host
    echo %%T | findstr "@" >nul 2>&1
    if not errorlevel 1 (
        if "!SSH_USERHOST!"=="" (
            set SSH_USERHOST=%%T
            set _PREV_TOKEN=
            goto :next_token_%%T
        )
    )
    set _PREV_TOKEN=%%T
    :next_token_%%T
)

if "!SSH_USERHOST!"=="" (
    echo [ERROR] Could not find user@host in the SSH command.
    echo         Please include a token like root@1.2.3.4
    pause & exit /b 1
)

echo.
echo  SSH target   : !SSH_USERHOST!
echo  SSH port     : !SSH_PORT!
echo  Extra args   : !SSH_EXTRA_ARGS!
echo  Tunnel       : localhost:!API_PORT! ^<-^> VPS 127.0.0.1:!API_PORT!
echo  Input images : !AUTORUN_DIR!
echo  Videos saved : !DOWNLOAD_DIR!
echo.

:: ---- Prerequisite checks -----------------------------------------------
where ssh >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ssh.exe not found.
    echo         Enable OpenSSH in Windows Settings ^> Apps ^> Optional Features.
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
    echo [ERROR] Input image folder not found: !AUTORUN_DIR!
    pause & exit /b 1
)
if not exist "!DOWNLOAD_DIR!" mkdir "!DOWNLOAD_DIR!"

:: ---- Kill any stale tunnel already using the local port ----------------
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr "127.0.0.1:!API_PORT! "') do (
    echo  Releasing port !API_PORT! (PID %%P)...
    taskkill /PID %%P /F >nul 2>&1
)

:: ---- Open SSH tunnel in its own window ---------------------------------
echo  Opening SSH tunnel window...
echo  ^> If your key requires a passphrase, type it in the new window that opens.
echo.

start "NewGen SSH Tunnel [port !SSH_PORT!] -- enter passphrase here if prompted" ^
    cmd /k "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=9999 !SSH_EXTRA_ARGS! -L !API_PORT!:127.0.0.1:!API_PORT! -N -p !SSH_PORT! !SSH_USERHOST!"

:: ---- Wait for the tunnel to actually forward the port ------------------
echo  Waiting for SSH tunnel on 127.0.0.1:!API_PORT!...
echo  ^(Enter passphrase in the other window now if prompted.^)
echo.
set /a _TUNNEL_WAIT=0
:wait_tunnel
!PYTHON! -c "import socket,sys;s=socket.socket();s.settimeout(2);sys.exit(0 if s.connect_ex(('127.0.0.1',!API_PORT!))==0 else 1)" >nul 2>&1
if errorlevel 1 (
    set /a _TUNNEL_WAIT+=2
    if !_TUNNEL_WAIT! GEQ 120 (
        echo [ERROR] SSH tunnel did not come up after 120 s.
        echo         Check the tunnel window for errors ^(wrong password, host unreachable, etc.^).
        pause & exit /b 1
    )
    timeout /t 2 /nobreak >nul
    goto wait_tunnel
)
echo  Tunnel is up^^!
echo.

:: ---- Run feeder ---------------------------------------------------------
echo  Starting feeder...
echo  Make sure you have clicked [Start Push Autorun] in the Gradio browser tab.
echo.
!PYTHON! "!FEEDER!" --folder "!AUTORUN_DIR!" --port !API_PORT! --download-dir "!DOWNLOAD_DIR!"

echo.
if errorlevel 1 (
    echo  Feeder finished with errors -- check output above.
) else (
    echo  All done^^!  Videos saved to: !DOWNLOAD_DIR!
)
echo.
echo  The SSH tunnel window is still open. Close it when you are finished.
pause
