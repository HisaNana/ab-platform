#!/bin/zsh

cd "$(dirname "$0")"
python3 -m streamlit run dashboard/experiment_dashboard.py
