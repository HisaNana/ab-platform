@echo off
setlocal

cd /d "%~dp0"
python -m streamlit run dashboard\experiment_dashboard.py

pause
