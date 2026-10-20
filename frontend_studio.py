"""``听显译-字幕工坊.exe`` 的入口：与 ``frontend.py`` 同一套启动逻辑，只是固定带 ``--studio``。

为什么需要单独一个文件：PyInstaller 的 ``--onefile`` **不能把"默认参数"塞进 exe**，
所以想打两个双击即用的 exe，就必须有两个入口脚本。真正的逻辑仍然只在
``frontend.py`` 里（找程序目录、检查 .venv、用 pythonw 拉起来、自己退出），
这里只是转发并**固定补上 ``--studio``**。

命令行上额外传的参数会被保留（万一以后要 ``--config`` 之类）。
"""

from __future__ import annotations

import sys

from frontend import main

if __name__ == "__main__":
    extra = [arg for arg in sys.argv[1:] if arg != "--studio"]
    raise SystemExit(main(["--studio", *extra]))
