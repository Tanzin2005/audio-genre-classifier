@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
if not exist "models\cnn.pt" goto model
echo Open http://127.0.0.1:7860 after startup. Press Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 7860
pause
exit /b
:setup
echo Run setup_windows.bat first.
pause
exit /b 1
:model
echo Train the model first to create models\cnn.pt.
pause
exit /b 1
