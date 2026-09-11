"""日志脱敏测试 —— 保证密钥不会写进日志文件（工程红线）。

注意：本文件**故意不写任何形似真实凭据的字面量**，
假 token 一律在运行时拼接生成，这样仓库里永远不存在可被扫描器命中的密钥串。
"""

from __future__ import annotations

from app.utils.log import scrub

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
