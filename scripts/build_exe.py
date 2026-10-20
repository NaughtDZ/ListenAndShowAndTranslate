"""把启动器打包成**两个**双击即用的 exe（PyInstaller，无控制台单文件）。

用法（**必须在项目 venv 里跑**，禁止全局安装）::

    .venv\\Scripts\\python.exe scripts\\build_exe.py
    # 或者直接双击 构建exe.bat

产物（项目根目录）：

=========================  ===========================  ==================
exe                        入口                          打开什么
=========================  ===========================  ==================
``听显译.exe``             ``frontend.py``              主程序（实时字幕）
``听显译-字幕工坊.exe``     ``frontend_studio.py``       「字幕工坊」（--studio）
=========================  ===========================  ==================

两个 exe 都只是**启动器**：自己不含业务代码，也不把 PySide6 / sherpa-onnx 打进去
（那些仍然在 ``.venv`` 里跑）。启动器只用标准库，所以体积只有几 MB。

为什么一个 exe 不够：PyInstaller 的 ``--onefile`` 不能把"默认参数"塞进 exe，
所以"双击即开字幕工坊"只能再打一个入口脚本（``frontend_studio.py``，逻辑仍复用
``frontend.py``）。参数会原样透传，``听显译.exe --settings`` 照样能用。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 作为脚本直接运行时 sys.path[0] 是 scripts/，import app.* 会失败
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROXY = "http://127.0.0.1:2333"

TARGETS: tuple[tuple[str, Path], ...] = (
    ("听显译", ROOT / "frontend.py"),
    ("听显译-字幕工坊", ROOT / "frontend_studio.py"),
)
"""``(exe 名, 入口脚本)``。加第三个双击入口就往这里加一行。"""

# 启动器只用标准库；显式排除这些大件，免得被依赖图顺手拖进 exe
EXCLUDES = (
    "PySide6", "shiboken6", "numpy", "sherpa_onnx", "soxr", "soundfile",
    "comtypes", "pycaw", "psutil", "pydantic", "httpx", "tenacity",
    "proctap", "tkinter", "matplotlib", "PIL",
)


def pyinstaller_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("PyInstaller") is not None


def build_one(name: str, entry: Path) -> Path:
    """打包一个入口，返回生成的 exe 路径。失败抛 ``RuntimeError``。"""
    build_dir = ROOT / "data" / "build" / name
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile", "--noconsole",
        "--name", name,
        "--distpath", str(ROOT),
        "--workpath", str(build_dir),
        "--specpath", str(build_dir),
        *[arg for mod in EXCLUDES for arg in ("--exclude-module", mod)],
        str(entry),
    ]
    print(f"\n=== 打包 {name} ← {entry.name} ===")
    print("运行：", " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    result = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode != 0:
        raise RuntimeError(f"打包 {name} 失败（exit {result.returncode}）")

    exe = ROOT / f"{name}.exe"
    if not exe.is_file():
        raise RuntimeError(f"打包结束但没看到 {exe}")

    # 中间产物没必要留着（.spec / build 目录）
    shutil.rmtree(build_dir, ignore_errors=True)
    return exe


def main() -> int:
    if not pyinstaller_available():
        venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
        print("❌ 没装 PyInstaller。请在项目 venv 里装（uv 走环境变量代理）：\n")
        print(f'   $env:HTTP_PROXY="{PROXY}"; $env:HTTPS_PROXY="{PROXY}"')
        print(f'   uv pip install --python "{venv_py}" pyinstaller\n')
        print("或者（同样要装在 .venv 里，别全局装）：")
        print(f'   "{venv_py}" -m pip install --proxy {PROXY} pyinstaller')
        return 2

    missing = [str(entry) for _, entry in TARGETS if not entry.is_file()]
    if missing:
        print("❌ 找不到入口脚本：\n  " + "\n  ".join(missing))
        return 2

    built: list[Path] = []
    for name, entry in TARGETS:
        try:
            built.append(build_one(name, entry))
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1

    print("\n✅ 打包完成：")
    for exe in built:
        size_mb = exe.stat().st_size / 1024 / 1024
        print(f"   {exe.name}（{size_mb:.1f} MB）")
    print("\n   双击即可启动，不会有控制台窗口。")
    return 0


if __name__ == "__main__":
    from app.utils.console import enable_utf8_console

    enable_utf8_console()
    raise SystemExit(main())
