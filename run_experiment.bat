@echo off
echo ========================================================
echo   Adaptive FL Framework - Federated Experiment Runner
echo ========================================================
cd /d "%~dp0\adaptive-fl-framework"
set PYTHON="..\.venv\Scripts\python.exe"
if not exist %PYTHON% (
    set PYTHON=python
)

%PYTHON% run.py %*
pause
