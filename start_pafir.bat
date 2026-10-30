@echo off
TITLE PAFIR Patent AI Framework Launcher
COLOR 0A

echo ===================================================
echo     PAFIR Patent AI Framework Startup Launcher
echo ===================================================
echo.

echo [1/4] Starting FastAPI backend on http://localhost:8000 ...
start "PAFIR FastAPI Backend" cmd /k "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/4] Waiting for FastAPI backend health check...
powershell -Command "for ($i=0; $i -lt 30; $i++) { try { $res = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2; if ($res.status -eq 'ready' -or $res.backend -eq 'ready') { exit 0 } } catch {} Start-Sleep -Seconds 1 }; exit 1"

echo [3/4] Starting React frontend dev server on http://localhost:3000 ...
start "PAFIR Vite Frontend" cmd /k "cd frontend && npm run dev"

echo [4/4] Launching PAFIR Application in Browser...
powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:3000'"

echo.
echo ===================================================
echo   PAFIR Framework is now active!
echo   Frontend: http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo ===================================================
