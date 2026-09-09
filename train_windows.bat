@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
if "%~1"=="" goto usage
if exist "data\cnn_cache\manifest.json" goto train
.venv\Scripts\python.exe -m genre_cnn.prepare --data "%~1"
if errorlevel 1 goto fail
:train
.venv\Scripts\python.exe -m genre_cnn.train --device cpu --threads 2 --output runs/cnn-reproduction
if errorlevel 1 goto fail
echo Training finished. Results are in runs\cnn-reproduction.
pause
exit /b 0
:setup
echo Run setup_windows.bat first.
exit /b 1
:usage
echo Usage: train_windows.bat "C:\path\to\genres_original"
exit /b 1
:fail
echo Training stopped. Copy the error above for troubleshooting.
pause
exit /b 1
