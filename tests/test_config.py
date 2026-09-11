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


# --------------------------------------------------------------------------- #
# 多语言路由（计划书第 12 节）
# --------------------------------------------------------------------------- #
def test_language_routing_defaults_cover_common_languages():
    cfg = AppConfig()
    routing = cfg.asr.routing
    for lang in ("zh", "en", "ja", "ko", "yue", "*"):
        assert lang in routing, f"路由表缺少 {lang}"
    assert routing["*"].model, "必须有兜底路由，否则小语种会无处可去"


def test_japanese_routes_to_offline_engine():
    """实测：sherpa-onnx 官方没有日语流式模型，ja 必须走分块引擎。

    这条测试锁住该事实——若哪天官方出了日语流式模型，
    应该是有意识地改这里，而不是悄悄变成流式。
    """
    cfg = AppConfig()
    ja = cfg.asr.routing["ja"]
    assert ja.engine == "sherpa_offline"
    assert ja.streaming is False
    assert "sense-voice" in ja.model


def test_japanese_korean_cantonese_share_one_model():
    """SenseVoice 单模型覆盖 5 语言，不应让用户重复下载三份。"""
    cfg = AppConfig()
    models = {cfg.asr.routing[k].model for k in ("ja", "ko", "yue")}
    assert len(models) == 1, models


def test_chinese_and_english_are_streaming():
    cfg = AppConfig()
    for lang in ("zh", "zh-en", "en"):
        assert cfg.asr.routing[lang].streaming is True, lang


def test_routing_is_user_overridable():
    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "faster_whisper", "model": "large-v3-turbo", "streaming": False})
    cfg.asr.language = "ja"
    dumped = cfg.model_dump(mode="json")
    assert dumped["asr"]["routing"]["ja"]["engine"] == "faster_whisper"
    assert dumped["asr"]["language"] == "ja"


def test_dict_item_assignment_would_bypass_validation_so_we_guard_it():
    """pydantic 不校验字典项赋值——这正是 set_route 存在的原因。"""
    from app.config import LanguageRoute

    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "whispercpp", "model": "whisper-tiny", "streaming": False})
    assert isinstance(cfg.asr.routing["ja"], LanguageRoute)
    assert cfg.asr.route_for("ja").engine == "whispercpp"


def test_routing_survives_json_roundtrip(tmp_path):
    from app.config import LanguageRoute

    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "faster_whisper", "model": "large-v3-turbo", "streaming": False})
    path = tmp_path / "config.json"
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert isinstance(loaded.asr.routing["ja"], LanguageRoute), "载入后必须是对象而非裸 dict"
    assert loaded.asr.routing["ja"].engine == "faster_whisper"


def test_route_for_falls_back_to_wildcard():
    cfg = AppConfig()
    r = cfg.asr.route_for("sw")  # 斯瓦希里语：表里没有，应走 *
    assert r.engine == "whispercpp"


def test_translate_has_source_language_for_multilingual():
    cfg = AppConfig()
    assert cfg.translate.source_language == "auto"
    assert cfg.translate.target_language == "zh"


def test_supported_languages_include_requested_ones():
    from app.config import SUPPORTED_LANGUAGES

    for lang in ("zh", "en", "ja"):
        assert lang in SUPPORTED_LANGUAGES
    assert "auto" in SUPPORTED_LANGUAGES
