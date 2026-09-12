"""「默认关闭思考」的测试（OpenAI 兼容通道）。

用户问：访问 OpenAI 兼容接口时，是不是应该默认把思考关掉？
答案：**应该**，而且光发一个参数不够——各家开关不一样，还有服务端会 400。
这个文件守住四件事：

1. 默认就带关思考参数，并且**按服务端类型**选对的字段；
2. 严格网关返回 400 时，去掉这些参数重试一次（宁可关不掉思考，也不能翻不出来）；
3. 服务端把思考**内联**在 content 里（``<think>…</think>``）时要剥掉，不能上屏；
4. 空译文 + 有思考时，用 ``/no_think`` 软开关兜底重试。
"""

from __future__ import annotations

from app.config import AppConfig
from app.translate.base import Segment, TranslateRequest
from app.translate.openai_compat import (
    OpenAICompatTranslator,
    strip_thinking,
    thinking_extras,
)


class _Resp:
    def __init__(self, status_code: int, data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = text or ""

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _reply(content: str, *, reasoning: str = "", truncated: bool = False) -> _Resp:
    return _Resp(200, {
        "choices": [{
            "message": {"content": content, "reasoning_content": reasoning},
            "finish_reason": "length" if truncated else "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


class _FakeClient:
    """记录每次请求的 payload，按顺序返回预设响应。"""

    def __init__(self, replies: list[_Resp]) -> None:
        self._replies = list(replies)
        self.payloads: list[dict] = []

    def post(self, _url: str, json: dict):  # noqa: A002 - 模仿 httpx 签名
        self.payloads.append(json)
        return self._replies.pop(0) if self._replies else _reply("兜底译文")

    def get(self, *_a, **_k):  # pragma: no cover
        raise AssertionError("不该调 GET")

    def close(self) -> None:
        ...


def _translator(**kw) -> OpenAICompatTranslator:
    t = OpenAICompatTranslator(base_url="http://127.0.0.1:1234/v1", model="m", **kw)
    t._client.close()
    return t


def _request(text: str = "夜の列車。") -> TranslateRequest:
    return TranslateRequest(
        segments=[Segment(id=1, text=text, language="ja")],
        source_language="ja", target_language="zh",
        context=[], glossary={}, template="subtitle_direct", custom_prompt="",
    )


# --------------------------------------------------------------------------- #
# 参数选择
# --------------------------------------------------------------------------- #
def test_thinking_extras_picks_the_right_knob_per_provider():
    """各家开关不一样，这是 2026 年的现实（写下来免得以后忘了）。"""
    local = thinking_extras("http://127.0.0.1:1234/v1")
    assert local == {"chat_template_kwargs": {"enable_thinking": False}}

    vllm = thinking_extras("http://192.168.1.9:8000/v1")
    assert "chat_template_kwargs" in vllm

    assert thinking_extras("https://api.openai.com/v1") == {"reasoning_effort": "minimal"}
    assert thinking_extras("https://openrouter.ai/api/v1") == {"reasoning": {"enabled": False}}
    assert thinking_extras("https://dashscope.aliyuncs.com/compatible-mode/v1") == {
        "enable_thinking": False
    }


def test_default_sends_disable_thinking():
    t = _translator()
    t._client = _FakeClient([_reply("1. 译文")])
    t.translate(_request())
    assert t._client.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_can_turn_the_knob_off():
    t = _translator(disable_thinking=False)
    t._client = _FakeClient([_reply("1. 译文")])
    t.translate(_request())
    payload = t._client.payloads[0]
    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_plain_style_never_sends_thinking_params():
    """纯翻译模型（sakura 那类）不吃这套参数，别给它塞。"""
    t = _translator(prompt_style="plain")
    t._client = _FakeClient([_reply("译文")])
    t.translate(_request())
    assert "chat_template_kwargs" not in t._client.payloads[0]


def test_strict_gateway_400_falls_back_without_the_params():
    """OpenAI / DeepSeek 官方对未知字段直接 400：去掉参数重试，别一句都翻不出来。"""
    t = _translator()
    t._client = _FakeClient([_Resp(400, text="Unrecognized request argument"), _reply("1. 译文")])
    result = t.translate(_request())
    assert result.translations[1] == "译文"
    assert len(t._client.payloads) == 2
    assert "chat_template_kwargs" not in t._client.payloads[1]


# --------------------------------------------------------------------------- #
# 内联思考
# --------------------------------------------------------------------------- #
def test_strip_inline_thinking():
    assert strip_thinking("<think>先判断语气…</think>译文在这") == "译文在这"
    assert strip_thinking("译文<THINKING>…</THINKING>") == "译文"
    # 被 max_tokens 截断的未闭合思考块：整段丢掉，不能把思考当译文
    assert strip_thinking("<think>嗯，这句话的意思是") == ""
    assert strip_thinking("正常译文") == "正常译文"


def test_inline_thinking_does_not_reach_the_user():
    t = _translator()
    t._client = _FakeClient([_reply("<think>思考中…</think>1. 译文")])
    result = t.translate(_request())
    assert result.translations[1] == "译文"


# --------------------------------------------------------------------------- #
# 空译文 + 思考 → /no_think 兜底
# --------------------------------------------------------------------------- #
def test_reasoning_only_reply_triggers_no_think_retry():
    t = _translator()
    t._client = _FakeClient([
        _reply("", reasoning="我需要先理解这句话的语气……", truncated=True),
        _reply("1. 他乘着夜车，静静地读着书。"),
    ])
    result = t.translate(_request())
    assert result.translations[1].startswith("他乘着夜车")
    assert len(t._client.payloads) == 2
    # 第二次请求把 /no_think 追加进了提示词
    assert "/no_think" in str(t._client.payloads[1]["messages"])
    # 而且统计到了思考字符（设置里的"测试通道"会展示这个数字）
    assert t.reasoning_chars_seen > 0
    assert "/no_think" in result.note


def test_reasoning_seen_is_counted_when_content_is_fine():
    """哪怕有译文，也把思考字符数记下来，方便设置里报告。"""
    t = _translator()
    t._client = _FakeClient([_reply("1. 译文", reasoning="（服务端还是思考了 20 字）")])
    t.translate(_request())
    assert t.reasoning_chars_seen > 0


# --------------------------------------------------------------------------- #
# 配置与界面接线
# --------------------------------------------------------------------------- #
def test_config_defaults_to_no_thinking():
    assert AppConfig().translate.llm.disable_thinking is True


def test_settings_checkbox_reflects_and_saves(qapp=None):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.ui.settings import SettingsWindow

    app = QApplication.instance() or QApplication([])
    cfg = AppConfig()
    cfg.translate.llm.disable_thinking = True
    win = SettingsWindow(cfg)
    try:
        assert win.llm_no_think.isChecked() is True
        win.llm_no_think.setChecked(False)
        win._save()
        assert cfg.translate.llm.disable_thinking is False
    finally:
        win.hide()
        win.deleteLater()
    assert app is not None
