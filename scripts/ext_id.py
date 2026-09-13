"""离线算出**解压缩扩展**的扩展 ID（Chromium 的算法）。

为什么需要它：自动化测试里没法点扩展图标（浏览器要求用户亲手触发），
只能用官方的旁路开关 ``--allowlisted-extension-id=<扩展ID>``；
而那个 ID 以前只能"先启动一遍浏览器、从 CDP 里读出来"。

算法（Chromium `extensions/common/extension_id.cc` 的做法）：
    对扩展目录的**绝对路径**按 **UTF-16LE** 编码取 SHA-256，
    取**前 16 字节**（32 个十六进制字符），每个半字节 0x0–0xf 映射到 'a'–'p'。

实测校验：

    L:\\AI_AudioAndChat\\ListenAndShowAndTranslate\\data\\tmp_tab_capture\\ext
      → dmknbccimoofgfeocikiojbcjpdligic   （与 CDP 里看到的完全一致）

注意：同一个目录路径 → 同一个 ID；换个目录（换个文件夹名）ID 就变了。
所以**不能**在程序里写死一个"我们扩展的 ID"——它取决于用户把扩展放在哪里。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def unpacked_extension_id(path: str | Path) -> str:
    """返回该目录作为"已加载的解压缩扩展"时的扩展 ID。"""
    raw = str(Path(path).resolve()).encode("utf-16-le")
    hexed = hashlib.sha256(raw).hexdigest()[:32]
    return "".join(chr(ord("a") + int(c, 16)) for c in hexed)
