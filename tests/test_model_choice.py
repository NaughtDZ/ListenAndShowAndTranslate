"""「识别模型手动指定」的测试（offscreen 平台）。

用户的原话：「既然模型是程序硬编码好的会下载哪些模型，那么哪些模型支持哪些语言，
程序应该是知道的，这样不是不会产生用户乱下载模型然后乱用的情况吗？」

所以这里守两件事：

1. **界面只让选能用的**：每个语言的下拉框里只有"真的能识别这门语言"的识别模型
   （语种识别模型 / VAD / 不支持该语言的模型都不许出现）；
2. **手改配置也乱不了**：运行时再兜一道，模型不能用就退回默认并说明原因。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.models.registry import (  # noqa: E402
    MODELS,
    models_for_language,
    pack_for_model,
    route_kwargs_for,
    supports_language,
)
from app.ui.settings import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def no_save(monkeypatch):
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)


# --------------------------------------------------------------------------- #
# "谁支持谁" 这张表本身
# --------------------------------------------------------------------------- #
def test_supports_language_matches_registry():
    assert supports_language("zipformer-zh-int8", "zh") is True
    assert supports_language("zipformer-zh-int8", "ja") is False
    assert supports_language("parakeet-ja-int8", "ja") is True
    assert supports_language("parakeet-ja-int8", "ko") is False   # 日语专用模型
    assert supports_language("dolphin-base-ctc-int8", "ko") is True
    assert supports_language("dolphin-base-ctc-int8", "yue") is True
    # Whisper turbo 声明了 "*"：什么语言都能塞
    assert supports_language("whisper-turbo-int8", "fr") is True
    assert supports_language("whisper-turbo-int8", "zh") is True
    # 中英混说只能由双语模型（或 Whisper）承担
    assert supports_language("zipformer-zh-en-int8", "zh-en") is True
    assert supports_language("zipformer-zh-int8", "zh-en") is False


def test_non_asr_models_are_never_offered():
    """**核心反例**：语种识别模型和 VAD 不是识别器，喂给它们不会有字幕。"""
    assert supports_language("whisper-tiny-lid", "ja") is False
    assert supports_language("silero-vad", "zh") is False
    for lang in ("zh", "ja", "*"):
        ids = {m.id for m in models_for_language(lang)}
        assert "whisper-tiny-lid" not in ids
        assert "silero-vad" not in ids
    # 不存在的模型也不能"支持"任何语言
    assert supports_language("chatgpt-asr", "zh") is False


def test_models_for_language_contents():
    ja = {m.id for m in models_for_language("ja")}
    assert ja == {"sensevoice-int8", "parakeet-ja-int8", "dolphin-base-ctc-int8",
                  "omnilingual-300m-ctc-int8", "whisper-turbo-int8"}
    zh = {m.id for m in models_for_language("zh")}
    assert {"zipformer-zh-int8", "zipformer-zh-en-int8", "fire-red-asr2-ctc-zh_en-int8",
            "dolphin-base-ctc-int8", "omnilingual-300m-ctc-int8",
            "whisper-turbo-int8"} <= {m.id for m in models_for_language("zh")}
    assert "zipformer-en-int8" not in zh


def test_route_kwargs_follow_registry_engine():
    """引擎类型必须以注册表为准，否则流式模型会被塞进分块引擎。"""
    stream = route_kwargs_for("zipformer-zh-int8")
    assert stream["engine"] == "sherpa_stream"
    assert stream["streaming"] is True
    offline = route_kwargs_for("dolphin-base-ctc-int8")
    assert offline["engine"] == "sherpa_offline"
    assert offline["streaming"] is False
    whisper = route_kwargs_for("whisper-turbo-int8")
    assert whisper["engine"] == "whispercpp"
    assert whisper["streaming"] is False


def test_pack_for_model_gives_download_hint():
    assert pack_for_model("parakeet-ja-int8") == "ja-parakeet"
    assert pack_for_model("dolphin-base-ctc-int8") == "dolphin"
    assert pack_for_model("zipformer-zh-int8") == "zh"
    assert pack_for_model("silero-vad") == "core"
    assert pack_for_model("chatgpt-asr") == ""


# --------------------------------------------------------------------------- #
# 配置：默认路由 & 写入
# --------------------------------------------------------------------------- #
def test_default_route_is_the_builtin_table():
    asr = AppConfig().asr
    assert asr.default_route_for("ja").model == "sensevoice-int8"
    assert asr.default_route_for("zh").model == "zipformer-zh-int8"
    # 没见过的语言落到 "*" 兜底（Whisper turbo，99 语言）
    assert asr.default_route_for("fr").model == "whisper-turbo-int8"


def test_set_route_round_trip():
    asr = AppConfig().asr
    asr.set_route("ja", route_kwargs_for("whisper-turbo-int8"))
    route = asr.route_for("ja")
    assert route.model == "whisper-turbo-int8"
    assert route.engine == "whispercpp" and route.streaming is False


# --------------------------------------------------------------------------- #
# 运行时兜底：手改配置也乱不了
# --------------------------------------------------------------------------- #
def _router(config: AppConfig):
    from app.asr.router import LanguageRouter

    logs: list[str] = []
    router = LanguageRouter(config.asr, on_log=logs.append)
    return router, logs


def test_router_honours_valid_override():
    cfg = AppConfig()
    cfg.asr.set_route("zh", route_kwargs_for("zipformer-zh-en-int8"))
    router, logs = _router(cfg)
    assert router._active_route("zh").model == "zipformer-zh-en-int8"
    assert logs == []


def test_router_ignores_model_that_cannot_do_that_language():
    cfg = AppConfig()
    cfg.asr.set_route("zh", route_kwargs_for("zipformer-en-int8"))  # 英文专用模型
    router, logs = _router(cfg)
    route = router._active_route("zh")
    assert route.model == "zipformer-zh-int8", "应退回默认中文模型"
    assert router._fallback_chain("zh")[0] == "zipformer-zh-int8"
    assert any("不能用" in line for line in logs)


def test_router_ignores_lid_model():
    """有人把语种识别模型填成识别模型：必须拦住，而不是加载失败。"""
    cfg = AppConfig()
    # engine 字段本身受 Literal 校验（下面另有用例），这里用合法引擎配错模型
    cfg.asr.set_route("ja", {"engine": "sherpa_offline", "model": "whisper-tiny-lid",
                             "streaming": False})
    router, logs = _router(cfg)
    assert router._active_route("ja").model == "sensevoice-int8"
    assert any("不能用" in line for line in logs)


def test_config_rejects_bogus_engine():
    """engine 也有一层校验：手写一个不存在的引擎，存都存不进去。"""
    cfg = AppConfig()
    with pytest.raises(Exception):
        cfg.asr.set_route("ja", {"engine": "lid", "model": "whisper-tiny-lid"})


def test_router_ignores_unknown_model():
    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "sherpa_offline", "model": "某不存在的模型",
                             "streaming": False})
    router, logs = _router(cfg)
    assert router._active_route("ja").model == "sensevoice-int8"
    assert any("不能用" in line for line in logs)


def test_router_warns_only_once_per_language():
    """同一门语言只提示一次，别每帧刷屏。"""
    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "sherpa_offline", "model": "silero-vad",
                             "streaming": False})
    router, logs = _router(cfg)
    for _ in range(5):
        router._active_route("ja")
    assert sum("不能用" in line for line in logs) == 1


# --------------------------------------------------------------------------- #
# 设置界面
# --------------------------------------------------------------------------- #
def _settings(qapp, no_save, monkeypatch, *, installed: bool = True) -> SettingsWindow:
    from app.models.downloader import ModelDownloader

    status = "installed" if installed else "missing"
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: status, raising=True)
    win = SettingsWindow(AppConfig())
    return win


def test_ui_only_offers_models_that_support_the_language(qapp, no_save, monkeypatch):
    win = _settings(qapp, no_save, monkeypatch)
    try:
        assert "*" in win.model_combos
        assert list(win.model_combos)[-1] == "*", "兜底那一行要排最后"
        for lang, combo in win.model_combos.items():
            offered = [combo.itemData(i) for i in range(combo.count())]
            assert offered[0] == "", "第一项必须是「自动」"
            for model_id in offered[1:]:
                assert model_id in MODELS
                assert supports_language(model_id, lang), (lang, model_id)
        ja = [win.model_combos["ja"].itemData(i) for i in range(win.model_combos["ja"].count())]
        assert ja == [
            "", "sensevoice-int8", "parakeet-ja-int8", "dolphin-base-ctc-int8",
            "omnilingual-300m-ctc-int8", "whisper-turbo-int8",
        ]
    finally:
        win.hide()
        win.deleteLater()


def test_ui_marks_not_downloaded_models(qapp, no_save, monkeypatch):
    win = _settings(qapp, no_save, monkeypatch, installed=False)
    try:
        combo = win.model_combos["ja"]
        whisper = combo.findData("whisper-turbo-int8")
        assert "未下载" in combo.itemText(whisper)
        combo.setCurrentIndex(whisper)
        assert "还没下载" in win.model_hint.text()
        assert "multilingual" in win.model_hint.text()  # 指出该下哪个包
    finally:
        win.hide()
        win.deleteLater()


def test_ui_no_warning_when_everything_installed(qapp, no_save, monkeypatch):
    win = _settings(qapp, no_save, monkeypatch, installed=True)
    try:
        combo = win.model_combos["ja"]
        combo.setCurrentIndex(combo.findData("omnilingual-300m-ctc-int8"))
        assert win.model_hint.text() == ""
    finally:
        win.hide()
        win.deleteLater()


def test_ui_save_writes_route_with_registry_engine(qapp, no_save, monkeypatch):
    win = _settings(qapp, no_save, monkeypatch)
    try:
        combo = win.model_combos["ja"]
        combo.setCurrentIndex(combo.findData("omnilingual-300m-ctc-int8"))
        win._save()
        route = win.config.asr.route_for("ja")
        assert route.model == "omnilingual-300m-ctc-int8"
        assert route.engine == "sherpa_offline"  # 引擎大类由注册表决定
        assert route.streaming is False
    finally:
        win.hide()
        win.deleteLater()


def test_ui_auto_restores_default_route(qapp, no_save, monkeypatch):
    win = _settings(qapp, no_save, monkeypatch)
    try:
        combo = win.model_combos["ja"]
        combo.setCurrentIndex(combo.findData("omnilingual-300m-ctc-int8"))
        win._save()
        win.reset_model_choices()
        win._save()
        assert win.config.asr.route_for("ja").model == "sensevoice-int8"
    finally:
        win.hide()
        win.deleteLater()


def test_ui_flags_invalid_route_from_hand_edited_config(qapp, no_save, monkeypatch):
    """手改配置写了不能用的模型：界面上要说清楚，并显示成「自动」。"""
    from app.models.downloader import ModelDownloader

    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)
    cfg = AppConfig()
    cfg.asr.set_route("ja", {"engine": "sherpa_offline", "model": "whisper-tiny-lid",
                             "streaming": False})
    win = SettingsWindow(cfg)
    try:
        assert win.model_combos["ja"].currentData() == ""
        assert "whisper-tiny-lid" in win.model_hint.text()
        assert "改回默认" in win.model_hint.text()
        win._save()
        assert win.config.asr.route_for("ja").model == "sensevoice-int8"
    finally:
        win.hide()
        win.deleteLater()
