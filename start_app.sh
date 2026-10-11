#!/bin/bash
# start_app.sh -- Unix/Mac startup script for Proactive Talent Alignment
# Starts FastAPI backend (port 8000) and Streamlit frontend (port 8501)

echo "============================================================"
echo " Proactive Talent Alignment for Sustainable Employment"
echo "============================================================"
echo ""
echo "Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 2

echo "Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "Both services started."
echo "Backend API docs: http://localhost:8000/docs"
echo "Frontend UI:      http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait $BACKEND_PID $FRONTEND_PID
