@echo off
cd /d "%~dp0"
if not exist ".airenv\Scripts\python.exe" (
  echo The AirLLM environment is missing.
  echo Run: C:\Users\PrattayATSarker\AppData\Local\Programs\Python\Python314\python.exe -m venv .airenv
  pause
  exit /b 1
)
start "air local AI" /b ".airenv\Scripts\python.exe" app.py
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:7860/"
