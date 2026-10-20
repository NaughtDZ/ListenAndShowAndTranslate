@echo off
REM ============================================================
REM  听·显·译 · 字幕工坊（离线转写）
REM  · 双击即开：把音频/视频转成 srt / vtt / lrc（可选 LLM 翻译）
REM  · 用 pythonw 启动，窗口开起来后这个黑框立刻关掉
REM  · 只用本目录下的 .venv，绝不碰系统 Python（禁止全局 pip）
REM  · 要盯着报错看，就改用：启动.bat --studio
REM ============================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title 听·显·译 · 字幕工坊

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_PW=%~dp0.venv\Scripts\pythonw.exe"

REM ---- 1. 虚拟环境 ----
if not exist "%VENV_PY%" (
    echo.
    echo 还没安装运行环境（找不到 .venv）。
    echo.
    choice /c YN /m "现在运行首次安装吗"
    if errorlevel 2 (
        echo 已取消。需要时请双击「首次安装.bat」。
        timeout /t 3 >nul
        exit /b 1
    )
    call "%~dp0首次安装.bat"
    exit /b %errorlevel%
)

REM ---- 2. 依赖自检（缺了就直接引导补装）----
"%VENV_PY%" -c "import PySide6, sherpa_onnx" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [提示] 依赖不完整（PySide6 / sherpa-onnx）。
    choice /c YN /m "现在补装依赖吗（会读 requirements.txt）"
    if errorlevel 2 goto :launch
    call "%~dp0首次安装.bat"
    exit /b %errorlevel%
)

:launch
REM  PYTHONUTF8=1 保证中文/日文输出不乱码
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ---- 3. 启动（优先 pythonw：不留常驻黑窗）----
if exist "%VENV_PW%" (
    start "" "%VENV_PW%" "%~dp0main.py" --studio
    endlocal & exit /b 0
)

REM 没有 pythonw 才退回控制台版（正常安装不会走到这里）
"%VENV_PY%" "%~dp0main.py" --studio
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo 程序退出，代码 %RC%。日志见 data\logs\lst.log
    pause
)
endlocal & exit /b %RC%
