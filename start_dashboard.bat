@echo off
echo ==================================================
echo   Starting Heart Disease Prediction Dashboard
echo ==================================================
echo.
echo 1. Launching default browser to: http://localhost:8000/dashboard.html
start "" "http://localhost:8000/dashboard.html"
echo.
echo 2. Starting local HTTP Server on port 8000...
echo [Press Ctrl+C in this terminal window to stop the server]
echo.
.\venv\Scripts\python -m http.server 8000
