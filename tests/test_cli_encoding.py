"""CLI 的 UTF-8 输出：Windows GBK 控制台不能把命令弄崩。

起因（真实 bug）：``python main.py --models status`` 在 GBK 控制台下直接
``UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'`` ——
状态列表要打印 ``✓`` / ``✗``，而 GBK 没有这两个字；``--models list`` 的 ``★`` 同理。
用户看到的是"命令一敲就报错"，而不是模型列表。

修法见 ``app/utils/console.enable_utf8_console``（``main.py`` 与 ``scripts/*.py``
的入口都会调它）：切控制台代码页 + 重配 stdout/stderr。
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.utils.console import enable_utf8_console

ROOT = Path(__file__).resolve().parent.parent

PASS_MARK = "\u2713"   # ✓
CROSS_MARK = "\u2717"  # ✗


# --------------------------------------------------------------------------- #
# 单元：重配本身
# --------------------------------------------------------------------------- #
def test_reconfigures_gbk_stream_to_utf8(monkeypatch):
    buf = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", buf)

    enable_utf8_console()

    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    print(f"{PASS_MARK} {CROSS_MARK} \u2605")   # 曾经这三种字符都会抛异常
    sys.stdout.flush()
    assert PASS_MARK in buf.buffer.getvalue().decode("utf-8")


def test_missing_streams_do_not_break_startup(monkeypatch):
    """pythonw 起的进程没有控制台，sys.stdout/stderr 可能是 None。"""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    enable_utf8_console()   # 不该抛任何异常


def test_object_without_reconfigure_is_ignored(monkeypatch):
    """被替换成普通对象的流（第三方包装/测试替身）也不能挡启动。"""
    class _Plain:
        def write(self, text: str) -> int:  # pragma: no cover - 只要求存在
            return len(text)

    monkeypatch.setattr(sys, "stdout", _Plain())
    enable_utf8_console()   # 不该抛任何异常


def test_main_entry_calls_it_before_argparse(monkeypatch):
    """CLI 入口必须真的调用它——否则 `--models status` 又会崩回 GBK。"""
    import main as main_mod

    called: list[bool] = []
    monkeypatch.setattr(main_mod, "enable_utf8_console", lambda: called.append(True))

    with pytest.raises(SystemExit):      # --help 由 argparse 抛 SystemExit
        main_mod.main(["--help"])

    assert called, "main() 没有在解析参数前切 UTF-8"


# --------------------------------------------------------------------------- #
# 端到端：真在 GBK stdio 下跑一次 CLI
# --------------------------------------------------------------------------- #
def test_models_status_survives_gbk_stdio():
    """回归守卫：强制 GBK 编码的 stdio 下，--models status 必须正常退出。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"   # 强制 stdout=gbk，复现用户环境
    env["PYTHONUTF8"] = "0"           # 同时关掉 UTF-8 模式

    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--models", "status"],
        cwd=str(ROOT), capture_output=True, env=env, timeout=300,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")

    assert proc.returncode == 0, f"CLI 仍然崩了：\n{err}"
    assert "whisper-vad-asmr" in out, "模型状态列表没打出来"
    assert PASS_MARK in out, "✓ 没能输出（编码被降级成了别的字符）"
