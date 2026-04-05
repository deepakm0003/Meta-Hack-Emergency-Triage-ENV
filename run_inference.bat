@echo off
echo ============================================
echo  Emergency Triage - Starting Server...
echo ============================================

cd /d "%~dp0"

REM ── Set your API credentials here ──────────────────────────────────────────
REM  Get a free Groq key at https://console.groq.com
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=YOUR_GROQ_API_KEY_HERE
set ENV_BASE_URL=http://localhost:7860
REM ───────────────────────────────────────────────────────────────────────────

REM Start uvicorn server in background
start "Triage Server" cmd /k "cd /d %~dp0 && uvicorn main:app --host 0.0.0.0 --port 7860"

REM Wait for server to boot
echo Waiting for server to start...
timeout /t 5 /nobreak > nul

echo ============================================
echo  Running Inference with Groq LLaMA...
echo ============================================

python inference.py

echo.
echo Press any key to close...
pause > nul
