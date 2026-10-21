"""DeepSeek 官方接口接入。

依据官方文档（2026）：

- ``base_url`` 就是 ``https://api.deepseek.com``（**不加** /v1，SDK 自己拼 /chat/completions）
- 思考开关是官方的 ``{"thinking": {"type": "enabled/disabled"}}``；
  默认**开着**，而翻译短句不需要思考，还会把预算烧在 ``reasoning_content`` 上
  （content 直接空串 → 字幕空白），所以要显式关掉
- ``reasoning_content`` 是多轮对话时要注意的字段；本项目翻译是单轮，不用回传

⚠️ 这里的 UI 测试**必须** monkeypatch 掉 ``AppConfig.save``：
   之前有一次验证脚本直接调了 ``_save()``，把真实 ``data/config.json`` 覆盖了。
"""

from __future__ import annotations

import os

import pytest

from app.translate.openai_compat import thinking_extras


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# 请求参数
# --------------------------------------------------------------------------- #
def test_deepseek_uses_official_thinking_field():
    """DeepSeek 认官方 thinking 字段，不是 LM Studio 那套 chat_template_kwargs。"""
    extras = thinking_extras("https://api.deepseek.com", "deepseek-flash")
    assert extras == {"thinking": {"type": "disabled"}}


def test_deepseek_thinking_survives_trailing_path():
    assert thinking_extras("https://api.deepseek.com/v1", "") == {"thinking": {"type": "disabled"}}
    assert thinking_extras("https://api.deepseek.com/", "") == {"thinking": {"type": "disabled"}}


# --------------------------------------------------------------------------- #
# 快捷预设
# --------------------------------------------------------------------------- #
def test_guess_preset_from_url():
    from app.ui.settings import guess_llm_preset

    assert guess_llm_preset("https://api.deepseek.com") == "deepseek"
    assert guess_llm_preset("http://127.0.0.1:1234/v1") == "local"
    assert guess_llm_preset("https://api.openai.com/v1") == "custom"
    assert guess_llm_preset("") == "custom"


def test_deepseek_models_are_documented_ones():
    from app.ui.settings import DEEPSEEK_BASE_URL, DEEPSEEK_MODELS

    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert "deepseek-flash" in DEEPSEEK_MODELS
    # 视觉型号不该出现在翻译候选里（用不上还更贵）
    assert all("vision" not in m for m in DEEPSEEK_MODELS)


@pytest.fixture()
def settings_win(qapp, monkeypatch):
    """打开设置窗，但**掐掉 save**（不允许碰真实配置文件）。"""
    from app.config import AppConfig
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    from app.models.downloader import ModelDownloader

    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    win = SettingsWindow(cfg)
    yield win
    win.hide()
    win.deleteLater()


def test_choosing_deepseek_preset_fills_url_and_models(settings_win):
    win = settings_win
    win.llm_preset.setCurrentIndex(win.llm_preset.findData("deepseek"))

    assert win.llm_base.text() == "https://api.deepseek.com"
    models = [win.llm_model.itemText(i) for i in range(win.llm_model.count())]
    assert "deepseek-flash" in models
    assert "必填" in win.llm_key.placeholderText()
    assert win.llm_style.currentData() == "chat"


def test_switching_to_local_preset_restores_local_url(settings_win):
    win = settings_win
    win.llm_preset.setCurrentIndex(win.llm_preset.findData("deepseek"))
    win.llm_preset.setCurrentIndex(win.llm_preset.findData("local"))
    assert win.llm_base.text().startswith("http://127.0.0.1")


def test_stale_local_model_is_dropped_when_switching_to_deepseek(settings_win):
    win = settings_win
    win.llm_model.setCurrentText("gemma4-26b-local")
    win.llm_preset.setCurrentIndex(win.llm_preset.findData("deepseek"))

    models = [win.llm_model.itemText(i) for i in range(win.llm_model.count())]
    assert "gemma4-26b-local" not in models, "本地模型名留在 DeepSeek 下拉里会误导"
