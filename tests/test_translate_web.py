"""网页版（免 key）翻译通道 + 回显检测 + prompt_style 的测试。不联网。"""

from __future__ import annotations

import httpx
import pytest

from app.translate.base import Segment, TranslateRequest
from app.translate.openai_compat import (
    guess_prompt_style,
    looks_like_echo,
    parse_lines,
)
from app.translate.traditional.providers import ALL_PROVIDERS, PROVIDERS, WEB_PROVIDERS
from app.translate.traditional.web import GoogleWebTranslator

# --------------------------------------------------------------------------- #
# Google 网页版端点
# --------------------------------------------------------------------------- #
def test_web_google_registered_separately():
    """网页版通道要单独一张表：它们不是官方 API，设置界面得标注'可能失效'。"""
    assert "web_google" in WEB_PROVIDERS
    assert "web_google" in ALL_PROVIDERS
    assert "web_google" not in PROVIDERS


def test_web_google_parses_nested_response():
    t = GoogleWebTranslator()
    payload = [[["列车？", "列車?", None, None, 10], ["林凡握紧了钥匙。", "リンファン", None, None, 10]]]

    def handler(r: httpx.Request) -> httpx.Response:
        # 参数要对：client=gtx 且带上待翻译文本
        assert "client=gtx" in str(r.url)
        assert "q=" in str(r.url)
        return httpx.Response(200, json=payload)

    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(handler))
    out = t._translate_one("リンファン", "ja", "zh-CN")
    assert out == "列车？林凡握紧了钥匙。"


def test_web_google_raises_on_empty():
    t = GoogleWebTranslator()
    t._client.close()
    t._client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[[]]))
    )
    with pytest.raises(RuntimeError, match="空译文"):
        t._translate_one("x", "ja", "zh-CN")


def test_web_google_raises_on_bad_structure():
    t = GoogleWebTranslator()
    t._client.close()
    t._client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"unexpected": 1}))
    )
    with pytest.raises(RuntimeError):
        t._translate_one("x", "ja", "zh-CN")


def test_web_google_language_mapping():
    t = GoogleWebTranslator()
    assert t.map_language("zh") == "zh-CN"
    assert t.map_language("ja") == "ja"
    assert t.map_language("auto") == "auto"


def test_web_google_low_default_qps():
    """免费端点限流严，默认 QPS 要压低，别把 IP 打黑。"""
    t = GoogleWebTranslator()
    assert t.qps_limit <= 3.0


def test_web_google_translate_result_and_failure_recording():
    t = GoogleWebTranslator(retry_times=0)
    state = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(200, json=[[["ok", "x", None, None, 1]]])

    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(handler))
    r = t.translate(TranslateRequest(
        segments=[Segment(id=1, text="a"), Segment(id=2, text="b")],
        source_language="ja", target_language="zh",
    ))
    assert r.translations == {1: "ok", 2: "ok"}
    assert state["n"] == 2, "网页版接口单条调用，基类应逐条发"


# --------------------------------------------------------------------------- #
# 回显检测（sakura 事故的防线）
# --------------------------------------------------------------------------- #
def test_looks_like_echo_detects_our_prompt_markers():
    """sakura-galtransl-7b 实测会把指令式 prompt 原样吐回来。
    这类输出必须被拦下：上屏是垃圾，进缓存/上下文会污染后续所有请求。"""
    assert looks_like_echo("列车？\n【待翻译（共 1 条）】\n请按相同编号逐条输出译文")
    assert looks_like_echo("硬性要求：忠实原文")
    assert looks_like_echo("【术语表（必须严格遵守）】")
    assert looks_like_echo("【前文（仅供理解上下文，不要翻译它）】")


def test_looks_like_echo_passes_normal_translation():
    assert not looks_like_echo("列车？林凡紧紧握住手中的青铜钥匙。")
    assert not looks_like_echo("")
    assert not looks_like_echo("Chapter one. The night train.")


# --------------------------------------------------------------------------- #
# prompt_style：微调翻译模型不能喂指令式 prompt
# --------------------------------------------------------------------------- #
def test_guess_prompt_style_for_finetuned_translators():
    assert guess_prompt_style("sakura-galtransl-7b-v3.7") == "plain"
    assert guess_prompt_style("Helsinki-NLP/opus-mt-ja-zh") == "plain"
    assert guess_prompt_style("nllb-200-distilled") == "plain"


def test_guess_prompt_style_for_instruct_models():
    assert guess_prompt_style("qwen3.8-2b-uncensored") == "chat"
    assert guess_prompt_style("gpt-4o-mini") == "chat"
    assert guess_prompt_style("") == "chat"


def test_parse_lines_for_plain_batch():
    parsed = parse_lines("第一句\n第二句\n", 2)
    assert parsed == {1: "第一句", 2: "第二句"}
    # 条数不符必须返回 None，让调用方退回逐条
    assert parse_lines("只有一行", 2) is None
    assert parse_lines("", 1) is None


def test_plain_style_sends_only_source_text():
    """plain 模式下请求体里不能出现我们的指令标记，否则微调模型会照抄回来。"""
    from app.translate.openai_compat import OpenAICompatTranslator

    t = OpenAICompatTranslator(model="sakura-galtransl-7b-v3.7", prompt_style="plain")
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "列车？林凡握紧了钥匙。"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        })

    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(handler))

    r = t.translate(TranslateRequest(
        segments=[Segment(id=1, text="リンファンは鍵を握った")],
        source_language="ja", target_language="zh",
        context=[("前文", "译文")],
        glossary={"リンファン": "林凡"},
    ))
    assert r.translations[1] == "列车？林凡握紧了钥匙。"

    body = captured[0].content.decode()
    assert "【术语表" not in body, "plain 模式不该注入指令式段落"
    assert "【前文" not in body
    assert "リンファンは鍵を握った" in body


def test_chat_style_still_injects_glossary_and_context():
    from app.translate.openai_compat import OpenAICompatTranslator

    t = OpenAICompatTranslator(model="qwen3.8-2b-uncensored", prompt_style="chat")
    captured: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        captured.append(r)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(handler))
    t.translate(TranslateRequest(
        segments=[Segment(id=1, text="リンファン")],
        context=[("前文原文", "前文译文")],
        glossary={"リンファン": "林凡"},
    ))
    body = captured[0].content.decode()
    assert "【术语表" in body and "【前文" in body


def test_echo_response_is_rejected_not_returned():
    """即使模型回显，也不能把提示词当译文返回给上层。"""
    from app.translate.openai_compat import OpenAICompatTranslator

    t = OpenAICompatTranslator(model="some-model", prompt_style="chat")
    echoed = "列车？\n【待翻译（共 1 条）】\n1. リンファン"
    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={
            "choices": [{"message": {"content": echoed}, "finish_reason": "stop"}],
            "usage": {},
        })
    ))
    r = t.translate(TranslateRequest(segments=[Segment(id=1, text="リンファン")]))
    assert r.ok_count == 0, "回显内容绝不能当作译文"
    assert r.fail_count == 1
