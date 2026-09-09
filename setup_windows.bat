@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt -r requirements-dev.txt
if errorlevel 1 goto fail
echo Setup finished. Run run_windows.bat to open the trained classifier.
pause
exit /b 0
:fail
echo Setup failed. Copy the error above for troubleshooting.
pause
exit /b 1
