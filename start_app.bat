@echo off
REM start_app.bat -- Windows startup script for Proactive Talent Alignment
REM Starts FastAPI backend (port 8000) and Streamlit frontend (port 8501)

echo ============================================================
echo  Proactive Talent Alignment for Sustainable Employment
echo ============================================================
echo.
echo Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI Backend" cmd /k "uvicorn backend.main:app --reload --port 8000"

timeout /t 3 /nobreak > nul

echo Starting Streamlit frontend on http://localhost:8501 ...
start "Streamlit Frontend" cmd /k "streamlit run frontend/app.py --server.port 8501"

echo.
echo Both services started.
echo Backend API docs: http://localhost:8000/docs
echo Frontend UI:      http://localhost:8501
echo.
pause
