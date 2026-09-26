"""控制台输出编码：把 ``stdout`` / ``stderr`` 切到 UTF-8。

为什么单独一个模块：不只是 CLI（``main.py``），``scripts/`` 下的验证脚本也会打印
``✓`` / ``✗`` / ``★`` / ``✅`` 这类字符，而 Windows 控制台默认代码页是 GBK（936）
—— 这些字符 GBK 根本编码不了，于是脚本**直接抛 UnicodeEncodeError 崩掉**，
用户看到的是"一敲命令就报错"，而不是正常输出。

入口处调用一次即可（``if __name__ == "__main__":`` 块内），不要在 import 时执行。
"""

from __future__ import annotations

import sys

UTF8_CODE_PAGE = 65001


def enable_utf8_console() -> None:
    """把当前进程的控制台输出切成 UTF-8。

    两件事缺一不可：

    1. **切控制台代码页**（Windows）：否则 Python 输出 UTF-8 字节、控制台按 GBK 解释，
       中文全变乱码；
    2. **重配 ``sys.stdout`` / ``sys.stderr``**：否则 Python 仍按 GBK 编码，
       写 ``✓`` 的瞬间就抛 ``UnicodeEncodeError``。

    ``errors="replace"`` 是最后的保险：万一还有编码不了的东西，也只会显示成 ``?``，不会崩。

    本函数**永不抛异常**——没有控制台（``pythonw``）或被重定向/替换过的流都不该
    影响程序启动。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(UTF8_CODE_PAGE)
        except Exception:  # noqa: BLE001 - 没有真实控制台时这一步必然失败，忽略
            pass

    for stream in (sys.stdout, sys.stderr):
        # pythonw 启动时这两个可能是 None；也可能被第三方包装成没有 reconfigure 的对象
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 只读流 / 已关闭流都不能挡住启动
            pass
