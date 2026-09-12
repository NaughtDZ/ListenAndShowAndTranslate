"""前端启动器：双击 exe 就能开程序，**不留任何控制台黑窗口**。

为什么要有它：`启动.bat` 用的是 ``python.exe``（控制台版解释器），
只要程序还在跑，那个黑框就一直戳在任务栏里——把控制窗收进托盘了它还在，
很难看（用户反馈）。这个脚本被打包成 ``听显译.exe``（PyInstaller，无控制台），
它自己**不跑业务**，只做三件事：

1. 找到程序目录（exe 所在目录；也兼容放在 ``dist/`` 或别处，会往上找 ``main.py``）
2. 检查 ``.venv\\Scripts\\pythonw.exe`` 在不在（不在就弹一个系统对话框，问要不要装）
3. 用 ``DETACHED_PROCESS`` 拉起 ``pythonw.exe main.py``，然后**自己立刻退出**

于是：没有控制台、没有常驻的批处理窗口；主程序该有的托盘/悬浮窗行为完全不变。
参数会原样透传，所以 ``听显译.exe --settings``、``听显译.exe --list-audio`` 也能用。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "听·显·译"
INSTALL_BAT = "首次安装.bat"
ENTRY = "main.py"
MAX_UP = 3  # 往上找 main.py 的层数（兼容 dist/ 之类）


def app_root() -> Path:
    """程序根目录：以 exe（或本脚本）所在目录为起点往上找 ``main.py``。"""
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents[:MAX_UP]):
        if (candidate / ENTRY).is_file():
            return candidate
    return start


def _message(text: str, *, title: str = APP_NAME, flags: int = 0x40) -> int:
    """系统级消息框（不依赖 tkinter，打包体积小）。返回用户点了哪个按钮。"""
    try:
        import ctypes

        # MB_ICONINFORMATION / MB_ICONERROR 由 flags 决定；0x40000 = 置顶
        return int(ctypes.windll.user32.MessageBoxW(None, text, title, flags | 0x40000))
    except Exception:  # noqa: BLE001 - 极端情况下连对话框都弹不出来
        print(text, file=sys.stderr)
        return 0


def find_pythonw(root: Path) -> Path | None:
    """找到 venv 里的 pythonw.exe（无控制台解释器）。"""
    for name in ("pythonw.exe", "python.exe"):
        exe = root / ".venv" / "Scripts" / name
        if exe.is_file():
            return exe
    return None


def launch(root: Path, pythonw: Path, argv: list[str]) -> int:
    """用独立进程拉起主程序，然后退出（父进程退出不会带走它）。"""
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    cmd = [str(pythonw), str(root / ENTRY), *argv]
    try:
        subprocess.Popen(
            cmd, cwd=str(root), creationflags=flags, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        _message(
            f"启动失败：{exc}\n\n可以试试双击「{INSTALL_BAT}」重装环境，"
            "或者用命令行运行 启动.bat 看具体报错。",
            flags=0x10,  # MB_ICONERROR
        )
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = app_root()
    pythonw = find_pythonw(root)
    if pythonw is None:
        # 没装环境：问一句要不要现在装（用户不用记着去双击哪个 bat）
        answer = _message(
            f"还没安装运行环境（找不到 {root}\\.venv）。\n\n"
            f"要现在运行「{INSTALL_BAT}」安装吗？\n"
            "（需要联网，首次会下载依赖与模型）",
            flags=0x04 | 0x30,  # MB_YESNO | MB_ICONQUESTION：6=是 7=否
        )
        install = root / INSTALL_BAT
        if answer == 6 and install.is_file():
            subprocess.Popen(
                ["cmd.exe", "/c", str(install)],
                cwd=str(root),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            return 0
        return 2

    code = launch(root, pythonw, args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
