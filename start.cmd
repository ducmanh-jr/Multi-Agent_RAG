@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

title Multi-Agent Research RAG

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHONUTF8=1"
set "PYTHONPATH=%ROOT_DIR%\main"

REM Chon Python tu virtualenv neu co, nguoc lai dung Python he thong
set "PYTHON_EXE=python"
if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\.venv\Scripts\python.exe"
) else if exist "%ROOT_DIR%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\venv\Scripts\python.exe"
)

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Vui long cai dat Python 3.10+ hoac kiem tra PATH.
    pause
    exit /b 1
)

cd /d "%ROOT_DIR%\main"

echo ============================================================
echo   MULTI-AGENT RESEARCH RAG - KHOI DONG HE THONG
echo ============================================================
echo   Python   : %PYTHON_EXE%
echo   Web Demo 2 : http://localhost:8000 (tu dong mo /demo2/)
echo ============================================================
echo.

if "%~1"=="" (
    "%PYTHON_EXE%" main.py --serve --port 8000
) else (
    "%PYTHON_EXE%" main.py %*
)

if errorlevel 1 (
    echo.
    echo [LOI] Ung dung dung voi ma loi %errorlevel%.
    pause
    exit /b %errorlevel%
)

endlocal
