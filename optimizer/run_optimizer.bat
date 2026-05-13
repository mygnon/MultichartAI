@echo off
REM Self-elevating launcher for the MultiCharts optimizer.
REM If not running as Administrator, re-launch with UAC elevation.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [run_optimizer] Not Administrator -- requesting elevation via UAC...
    powershell -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/K cd /d \"%~dp0\" && py main.py --strategies all' -Verb RunAs"
    exit /b
)

REM Already elevated -- run the optimizer
cd /d "%~dp0"
echo [run_optimizer] Running as Administrator.
py main.py --strategies all
pause
