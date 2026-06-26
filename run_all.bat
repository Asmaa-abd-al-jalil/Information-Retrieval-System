@echo off
title Running IR System Microservices

echo ====================================================
echo Starting all IR System services in parallel...
echo ====================================================

echo Starting Preprocessing Service (Port 8001)...
start "Preprocessing Service - 8001" cmd /k "python -m uvicorn apis.api_preprocessing:app --port 8001 --reload"

echo Starting Retrieval Service (Port 8002)...
start "Retrieval Service - 8002" cmd /k "python -m uvicorn apis.api_retrieval:app --port 8002 --reload"

echo Starting Ranking Service (Port 8003)...
start "Ranking Service - 8003" cmd /k "python -m uvicorn apis.api_ranking:app --port 8003 --reload"

echo Starting Refinement Service (Port 8004)...
start "Refinement Service - 8004" cmd /k "python -m uvicorn apis.api_refinement:app --port 8004 --reload"

echo Starting Evaluation Service (Port 8005)...
start "Evaluation Service - 8005" cmd /k "python -m uvicorn apis.api_evaluation:app --port 8005"

echo Starting API Gateway (Port 8006)...
start "API Gateway - 8006" cmd /k "python -m uvicorn apis.gateway:app --port 8006 --reload"

echo Starting Flask Frontend UI (Port 5000)...
start "Flask Frontend UI - 5000" cmd /k "python ui/app_ui.py"

echo ====================================================
echo All services are spawning now! Keep this window open.
echo ====================================================
pause