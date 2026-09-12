@echo off
REM ============================================================
REM  听·显·译 (ListenAndShowAndTranslate) 首次安装
REM  作用：用 uv 在本目录下建 Python 3.12 虚拟环境并装依赖，
REM        然后启动首次运行向导。
REM  说明：全部路径基于 %~dp0（本脚本所在目录），可整个文件夹搬走。
REM ============================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title 听·显·译 - 首次安装

echo ============================================================
echo   听·显·译  首次安装
echo   安装目录：%~dp0
echo ============================================================
echo.

REM ---- 1. 检查 uv ----
where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 没找到 uv。
    echo.
    echo   本项目必须用 uv 建 Python 3.12 环境：
    echo   系统自带的 Python 3.14 装不上 proc-tap / sherpa-onnx（没有对应轮子）。
    echo.
    echo   装 uv 的方法（任选其一）：
    echo     powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo     或到 https://github.com/astral-sh/uv/releases 下载 uv.exe 放到 PATH 里
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('uv --version 2^>nul') do set "UVVER=%%v"
echo [1/4] uv 已就绪：%UVVER%

REM ---- 2. 代理（可选）----
set "PROXY="
echo.
echo [2/4] 下载依赖可能需要代理（例如 127.0.0.1:2333）。
echo       直接回车 = 不用代理。
set /p PROXY=      代理地址: 
if defined PROXY (
    set "HTTP_PROXY=%PROXY%"
    set "HTTPS_PROXY=%PROXY%"
    echo       已设置 HTTP_PROXY=%PROXY%
) else (
    echo       不使用代理
)

REM ---- 3. 建虚拟环境 ----
echo.
echo [3/4] 创建 Python 3.12 虚拟环境 .venv ...
if exist "%~dp0.venv\Scripts\python.exe" (
    echo       已存在，跳过创建（如需重建请先删除 .venv 目录）
) else (
    uv venv --python 3.12.12 .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

REM ---- 4. 装依赖 ----
echo.
echo [4/4] 安装依赖（几十 MB，请稍候）...
uv pip install --python "%~dp0.venv\Scripts\python.exe" -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [错误] 依赖安装失败。若因网络问题，请重跑本脚本并在第 2 步填代理。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成，启动首次运行向导
echo ============================================================
echo.
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" --wizard
if errorlevel 1 pause
endlocal
