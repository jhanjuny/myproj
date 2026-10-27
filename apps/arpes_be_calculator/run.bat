@echo off
chcp 65001 > nul
set "PY=C:\Users\hanjunpy\anaconda3\python.exe"
if not exist "%PY%" set "PY=D:\conda_envs\torch\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0arpes_be_calc.py"
pause
