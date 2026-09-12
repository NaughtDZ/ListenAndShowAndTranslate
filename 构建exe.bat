@echo off
REM ============================================================
REM  构建前端 exe（听显译.exe）—— 只在需要重新打包时跑
REM  · 打包后双击 听显译.exe 启动，不会有控制台黑窗口
REM  · 必须在项目 .venv 里跑（禁止全局 pip）
REM ============================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo 找不到 .venv，请先运行「首次安装.bat」。
    pause
    exit /b 1
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%VENV_PY%" "%~dp0scripts\build_exe.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo 打包失败，代码 %RC%。
)
pause
endlocal & exit /b %RC%
