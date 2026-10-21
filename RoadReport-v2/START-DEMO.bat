@echo off
setlocal
cd /d "%~dp0roadreport\frontend"
where node >nul 2>nul
if errorlevel 1 (
  echo Install Node.js 24, then run this file again.
  pause
  exit /b 1
)
if not exist node_modules (
  call npm ci
  if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
  )
)
echo Opening the synthetic demo at http://127.0.0.1:5173/?demo=1
echo Keep this window open while using the demo. Press Ctrl+C to stop.
call npm run demo
pause
