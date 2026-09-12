"""日志脱敏测试 —— 保证密钥不会写进日志文件（工程红线）。

注意：本文件**故意不写任何形似真实凭据的字面量**，
假 token 一律在运行时拼接生成，这样仓库里永远不存在可被扫描器命中的密钥串。
"""

from __future__ import annotations

import logging
import sys

from app.utils.log import install_excepthook, scrub

# 运行时拼接的假凭据（源码里不存在完整 token 形态）
FAKE_OPENAI_KEY = "sk-" + "A1b2C3d4E5f6G7h8I9j0"
FAKE_GITHUB_TOKEN = "ghp_" + "0123456789abcdefghijklmnopqrstuvwxyz"
FAKE_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"


def test_scrub_openai_key():
    out = scrub(f"using api key {FAKE_OPENAI_KEY} for request")
    assert FAKE_OPENAI_KEY not in out
    assert "<redacted>" in out


def test_scrub_github_token():
    out = scrub(f"token={FAKE_GITHUB_TOKEN}")
    assert FAKE_GITHUB_TOKEN not in out
    assert "<redacted>" in out


def test_scrub_bearer_header():
    out = scrub(f"Authorization: Bearer {FAKE_JWT}")
    assert FAKE_JWT not in out
    assert "<redacted>" in out


def test_scrub_key_value_forms():
    cases = [
        ('api_key="abcdef123456"', "abcdef123456"),
        ("app_secret: youdao-secret-9999", "youdao-secret-9999"),
        ("APPKEY=1234567890abcdef", "1234567890abcdef"),
        ("password=hunter2xyz", "hunter2xyz"),
    ]
    for line, secret in cases:
        out = scrub(line)
        assert "<redacted>" in out, line
        assert secret not in out, line


def test_scrub_leaves_normal_text_alone():
    text = "启动 v0.1.0 | Python 3.12.12 | 目标进程 喜马拉雅.exe"
    assert scrub(text) == text


def test_scrub_is_idempotent():
    once = scrub(f"api_key={FAKE_OPENAI_KEY}")
    assert scrub(once) == once


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_hook(fn) -> list[logging.LogRecord]:
    """把日志抓下来看（lst logger 的 propagate=False，caplog 抓不到）。"""
    logger = logging.getLogger("lst")
    handler = _Collect()
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
    return handler.records


def test_excepthook_writes_uncaught_exception_to_log():
    """pythonw 起的子进程没有控制台：未捕获异常必须落到日志，否则凭空消失。"""
    original = sys.excepthook
    try:
        install_excepthook()
        assert sys.excepthook is not original

        def boom() -> None:
            raise ValueError("字幕子进程崩了")

        def run() -> None:
            try:
                boom()
            except ValueError:
                sys.excepthook(*sys.exc_info())

        records = _capture_hook(run)
        assert records, "excepthook 什么都没写"
        assert "未捕获异常" in records[0].getMessage()
        assert records[0].exc_info is not None
    finally:
        sys.excepthook = original


def test_excepthook_keeps_keyboard_interrupt_quiet():
    """Ctrl+C 不该被记成 CRITICAL 异常。"""
    original = sys.excepthook
    try:
        install_excepthook()

        def run() -> None:
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                sys.excepthook(*sys.exc_info())

        records = _capture_hook(run)
    finally:
        sys.excepthook = original
    assert not [r for r in records if r.levelno >= logging.CRITICAL]
