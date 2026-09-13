"""向导「翻译通道」页的功能回归：点「列出模型」「测试通道」不能报 NameError。

这两个按钮是**真的去连** LM Studio / Ollama 的（本机 127.0.0.1），
所以测试里把 ``probe_endpoint`` 换掉——验的是"按钮背后的代码路径通不通"，
不是"LM Studio 在不在跑"。

背景（用户实测报的 bug，2026-09-13）：
    ``_list_models()`` 里 ``from … import probe_endpoint``，
    真正干活的却是 ``_do_list()`` —— 局部导入只在那一个函数作用域可见，
    于是点按钮直接 ``NameError: name 'probe_endpoint' is not defined``。
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from app.translate import openai_compat
from app.ui.wizard import TranslatePage


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def page(qapp):
    p = TranslatePage()
    yield p
    p.deleteLater()


def _fake_probe(ok: bool = True, models: list[str] | None = None, msg: str = "连上了"):
    calls: list[tuple] = []

    def probe(base_url: str, api_key: str = "", proxy: str = ""):
        calls.append((base_url, api_key, proxy))
        return ok, msg, list(models if models is not None else ["fake-model-a", "fake-model-b"])

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


# --------------------------------------------------------------------------- #
# 列出模型
# --------------------------------------------------------------------------- #
def test_list_models_does_not_raise_nameerror(page, monkeypatch):
    """核心回归：以前这里必然 NameError（跨方法用了局部导入）。"""
    probe = _fake_probe()
    monkeypatch.setattr(openai_compat, "probe_endpoint", probe)

    page.base_url.setText("http://127.0.0.1:1234/v1")
    text = page._do_list("http://127.0.0.1:1234/v1")

    assert "✅" in text
    assert "fake-model-a" in text and "fake-model-b" in text
    assert probe.calls and probe.calls[0][0] == "http://127.0.0.1:1234/v1"
    assert page._models == ["fake-model-a", "fake-model-b"]


def test_list_models_reports_failure_politely(page, monkeypatch):
    probe = _fake_probe(ok=False, msg="连不上 127.0.0.1:1234")
    monkeypatch.setattr(openai_compat, "probe_endpoint", probe)

    text = page._do_list("http://127.0.0.1:1234/v1")
    assert "❌" in text
    assert "LM Studio" in text  # 失败时给出可操作提示


def test_list_models_truncates_long_lists(page, monkeypatch):
    many = [f"model-{i}" for i in range(25)]
    monkeypatch.setattr(openai_compat, "probe_endpoint", _fake_probe(models=many))

    text = page._do_list("http://127.0.0.1:1234/v1")
    assert "model-19" in text
    assert "另有 5 个" in text


# --------------------------------------------------------------------------- #
# 模型名单过滤：嵌入/重排模型不能当翻译模型
# --------------------------------------------------------------------------- #
def test_is_chat_model_filters_non_chat_models():
    """本机 LM Studio 实测的 16 个模型里有嵌入与重排模型，必须滤掉。"""
    from app.translate.openai_compat import is_chat_model, split_model_list

    assert is_chat_model("qwen3.8-27b-uncensored") is True
    assert is_chat_model("hy-mt2-7b") is True  # 名字里没有关键词，一律算可用
    assert is_chat_model("text-embedding-qwen3-embedding-0.6b") is False
    assert is_chat_model("bge-m3") is False
    assert is_chat_model("qwen3-reranker-0.6b") is False
    assert is_chat_model("whisper-large-v3") is False
    assert is_chat_model("") is False

    chat, others = split_model_list(
        ["text-embedding-x", "qwen3.8-2b", "bge-m3", "hy-mt2-7b"]
    )
    assert chat == ["qwen3.8-2b", "hy-mt2-7b"]
    assert others == ["text-embedding-x", "bge-m3"]


def test_do_list_hides_embedding_models(page, monkeypatch):
    probe = _fake_probe(
        models=["text-embedding-qwen3-embedding-0.6b", "qwen3.8-2b", "bge-m3"]
    )
    monkeypatch.setattr(openai_compat, "probe_endpoint", probe)

    text = page._do_list("http://127.0.0.1:1234/v1")
    assert "  · qwen3.8-2b" in text
    # 不能出现在"可翻译"的列表里（只允许出现在"已滤掉…"那行说明里）
    assert "  · text-embedding" not in text
    assert "  · bge-m3" not in text
    # 但要如实告诉用户滤掉了什么
    assert "已滤掉 2 个" in text
    assert page._models == ["qwen3.8-2b"]


def test_do_list_warns_when_no_chat_model(page, monkeypatch):
    monkeypatch.setattr(
        openai_compat, "probe_endpoint", _fake_probe(models=["bge-m3", "whisper-large-v3"])
    )
    text = page._do_list("http://127.0.0.1:1234/v1")
    assert "没有可翻译的对话模型" in text


def test_do_test_does_not_auto_pick_a_model(page, monkeypatch):
    """不能自动把列表第一个填进去：实测第一个常常是嵌入模型，
    而且在设置窗那边"自动选"还可能是 20GB+ 的大模型（吃满显存）。"""
    monkeypatch.setattr(
        openai_compat,
        "probe_endpoint",
        _fake_probe(models=["text-embedding-x", "qwen3.8-2b", "hy-mt2-7b"]),
    )
    page.provider.setCurrentIndex(0)
    page.model.setText("")

    out = page._do_test()

    assert page.model.text() == "", "不该替用户自动选模型"
    assert "qwen3.8-2b" in out and "hy-mt2-7b" in out, "要把可选名字摆出来"
    assert "text-embedding" not in out, "嵌入模型不该出现在可选项里"


def test_list_models_button_is_wired(page):
    """按钮存在且能触发（不真的连网：只验接线，避免依赖本机服务）。"""
    assert page._list_models.__self__ is page
    assert callable(page._do_list)


# --------------------------------------------------------------------------- #
# 测试通道
# --------------------------------------------------------------------------- #
def test_test_channel_llm_path(page, monkeypatch):
    probe = _fake_probe()
    monkeypatch.setattr(openai_compat, "probe_endpoint", probe)

    # 第 0 项就是「本地大模型（llm）」
    page.provider.setCurrentIndex(0)
    page.base_url.setText("http://127.0.0.1:1234/v1")
    page.model.setText("")

    out = page._do_test()
    assert out.startswith("✅")
    # 不自动替用户选模型，但要把可选名字摆出来（见 test_do_test_does_not_auto_pick_a_model）
    assert "fake-model-a" in out
    assert page.model.text() == ""


def test_test_channel_none_path(page):
    idx = page.provider.findData("none")
    page.provider.setCurrentIndex(idx)
    assert "跳过" in page._do_test()


def test_test_channel_never_raises(page, monkeypatch):
    """任何异常都要变成一行 ❌ 文本，不能把向导带崩。"""

    def boom(*_a, **_k):
        raise RuntimeError("网络炸了")

    monkeypatch.setattr(openai_compat, "probe_endpoint", boom)
    page.provider.setCurrentIndex(0)
    out = page._do_test()
    assert out.startswith("❌") and "网络炸了" in out


def test_proxy_falls_back_to_empty_outside_wizard(page):
    """向导外的独立构造（测试/复用）不能因为 wizard() 为 None 而炸。"""
    assert page._proxy() == ""
