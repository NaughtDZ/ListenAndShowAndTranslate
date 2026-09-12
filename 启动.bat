@echo off
REM ============================================================
REM  听·显·译 (ListenAndShowAndTranslate) 启动
REM  · 全部路径基于 %~dp0（脚本所在目录），可整个文件夹搬走
REM  · 只用本目录下的 .venv，绝不碰系统 Python（禁止全局 pip）
REM  · 环境或模型没准备好时会自动引导到首次安装 / 向导
REM ============================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title 听·显·译

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

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

REM ---- 2. 解释器版本自检（3.12 是硬要求）----
"%VENV_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [警告] .venv 里的 Python 不是 3.12。
    echo        proc-tap 与 sherpa-onnx 在 3.13+ 上没有可用轮子。
    echo        建议删除 .venv 后重新运行「首次安装.bat」。
    echo.
    pause
)

REM ---- 3. 依赖自检（缺了就直接引导）----
"%VENV_PY%" -c "import PySide6, proctap, sherpa_onnx, pycaw" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [提示] 依赖不完整。
    choice /c YN /m "现在补装依赖吗（会读 requirements.txt）"
    if errorlevel 2 goto :launch
    call "%~dp0首次安装.bat"
    exit /b %errorlevel%
)

:launch
REM ---- 4. 启动 ----
REM  PYTHONUTF8=1 保证中文/日文输出不乱码
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM  可传参，例如：启动.bat --list-audio / --settings / --run 1234
"%VENV_PY%" "%~dp0main.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo 程序退出，代码 %RC%。日志见 data\logs\lst.log
    pause
)
endlocal & exit /b %RC%
