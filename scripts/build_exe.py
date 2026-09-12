"""把 ``frontend.py`` 打包成 ``听显译.exe``（PyInstaller，无控制台单文件）。

用法（**必须在项目 venv 里跑**，禁止全局安装）::

    .venv\\Scripts\\python.exe scripts\\build_exe.py
    # 或者直接双击 构建exe.bat

为什么是 onefile + noconsole：

* **noconsole（``--windowed``）**：双击不弹控制台黑窗——做这个 exe 的全部理由。
  启动器内部会用 ``pythonw.exe`` 拉起主程序，所以主程序同样没有黑窗。
* **onefile**：只生成一个 exe，用户不用管旁边那一堆 ``_internal`` 目录。
  代价是首次启动要多花约 0.5~1 秒解包到自己临时目录（启动器本身随即退出）。
* 大小只有几 MB：启动器只用标准库，不把 PySide6 / sherpa-onnx 打进去
  （那些仍然在 ``.venv`` 里跑）。

产物：项目根目录下的 ``听显译.exe``（``data/build/`` 里的是中间产物，已在 .gitignore）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE_NAME = "听显译"
ENTRY = ROOT / "frontend.py"
BUILD_DIR = ROOT / "data" / "build" / "frontend"
PROXY = "http://127.0.0.1:2333"

# 启动器只用标准库；显式排除这些大件，免得被依赖图顺手拖进 exe
EXCLUDES = (
    "PySide6", "shiboken6", "numpy", "sherpa_onnx", "soxr", "soundfile",
    "comtypes", "pycaw", "psutil", "pydantic", "httpx", "tenacity",
    "proctap", "tkinter", "matplotlib", "PIL",
)


def pyinstaller_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("PyInstaller") is not None


def main() -> int:
    if not pyinstaller_available():
        venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
        print("❌ 没装 PyInstaller。请在项目 venv 里装（uv 走环境变量代理）：\n")
        print(f'   $env:HTTP_PROXY="{PROXY}"; $env:HTTPS_PROXY="{PROXY}"')
        print(f'   uv pip install --python "{venv_py}" pyinstaller\n')
        print("或者（同样要装在 .venv 里，别全局装）：")
        print(f'   "{venv_py}" -m pip install --proxy {PROXY} pyinstaller')
        return 2

    if not ENTRY.is_file():
        print(f"❌ 找不到入口脚本：{ENTRY}")
        return 2

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile", "--noconsole",
        "--name", EXE_NAME,
        "--distpath", str(ROOT),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(BUILD_DIR),
        *[arg for mod in EXCLUDES for arg in ("--exclude-module", mod)],
        str(ENTRY),
    ]
    print("运行：", " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    result = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode != 0:
        print(f"❌ 打包失败（exit {result.returncode}）")
        return result.returncode

    exe = ROOT / f"{EXE_NAME}.exe"
    if not exe.is_file():
        print(f"❌ 打包结束但没看到 {exe}")
        return 1

    size_mb = exe.stat().st_size / 1024 / 1024
    print(f"\n✅ 打包完成：{exe}（{size_mb:.1f} MB）")
    print("   双击它即可启动，不会有控制台窗口。")
    tmp = BUILD_DIR
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)  # 中间产物没必要留着
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
