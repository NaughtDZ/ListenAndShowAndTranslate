@echo off
REM ============================================================
REM  ListenAndShowAndTranslate 一键启动
REM  依赖必须装在 .venv 内（红线：禁止全局 pip install）
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv
    echo 请先执行:  uv venv --python 3.12.12 .venv
    echo            uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    pause
    exit /b 1
)

set PYTHONUTF8=1
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
endlocal
