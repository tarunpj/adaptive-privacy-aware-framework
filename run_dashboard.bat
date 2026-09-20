@echo off
echo ========================================================
echo   Launching Adaptive FL Framework Streamlit Dashboard
echo ========================================================
cd /d "%~dp0\adaptive-fl-framework"
if exist "..\.venv\Scripts\streamlit.exe" (
    "..\.venv\Scripts\streamlit.exe" run dashboard\app.py
) else (
    streamlit run dashboard\app.py
)
pause
