"""配置模块测试：默认值、往返保存、损坏回退、导出脱敏。"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import AppConfig


def test_defaults_are_sane():
    cfg = AppConfig()
    assert cfg.asr.engine == "sherpa_stream"
    assert cfg.audio.source_mode == "process"
    assert cfg.translate.display_mode == "bilingual"
    assert cfg.overlay.click_through is True
    assert cfg.overlay.always_on_top is True


def test_roundtrip(tmp_path: Path):
    path = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.audio.target_process_name = "喜马拉雅.exe"
    cfg.asr.preset = "mid"
    cfg.overlay.font_size = 48
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.audio.target_process_name == "喜马拉雅.exe"
    assert loaded.asr.preset == "mid"
    assert loaded.overlay.font_size == 48


def test_missing_file_falls_back_to_defaults(tmp_path: Path):
    cfg = AppConfig.load(tmp_path / "nope.json")
    assert cfg.version == 1
    assert cfg.asr.engine == "sherpa_stream"


def test_corrupt_file_falls_back_and_backs_up(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{ this is not json", encoding="utf-8")

    cfg = AppConfig.load(path)

    assert cfg.asr.engine == "sherpa_stream"  # 未抛异常
    backups = list(tmp_path.glob("config.broken-*.json"))
    assert len(backups) == 1, "损坏的配置应被备份而不是被静默丢弃"


def test_invalid_values_fall_back(tmp_path: Path):
    path = tmp_path / "config.json"
    # font_size 超出 [8,200] 上界
    path.write_text(json.dumps({"overlay": {"font_size": 99999}}), encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.overlay.font_size == 34  # 回退默认值


def test_redacted_removes_credentials():
    cfg = AppConfig()
    cfg.translate.providers = {
        "baidu": {"app_id": "12345", "secret_key": "super-secret-value"},
        "youdao": {"app_key": "yd-key-123456", "app_secret": "yd-secret"},
    }
    cfg.translate.llm.api_key = "sk-abcdef1234567890"

    red = cfg.redacted()

    assert red["translate"]["llm"]["api_key"] == "<redacted>"
    assert red["translate"]["providers"]["baidu"]["secret_key"] == "<redacted>"
    assert red["translate"]["providers"]["youdao"]["app_key"] == "<redacted>"
    # 非敏感字段必须保留
    assert red["translate"]["providers"]["baidu"]["app_id"] == "12345"

    # 序列化后的全文里不能出现原始密钥
    blob = json.dumps(red, ensure_ascii=False)
    assert "super-secret-value" not in blob
    assert "yd-secret" not in blob
    assert "sk-abcdef1234567890" not in blob
