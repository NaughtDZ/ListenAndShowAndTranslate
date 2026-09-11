"""传统翻译 API 适配器测试（不联网，用 MockTransport 抓请求验签名）。

签名算法写错的典型表现不是崩溃，而是接口回一个 "签名错误" 的 error_code，
或者更糟——静默返回错误结果。所以这里把签名规则逐条钉住。
"""

from __future__ import annotations

import hashlib
import time

import httpx
import pytest

from app.translate.base import Segment, TranslateRequest
from app.translate.traditional.base import TraditionalTranslatorBase, UnsupportedLanguage
from app.translate.traditional.providers import (
    PROVIDERS,
    AzureTranslator,
    BaiduTranslator,
    DeepLTranslator,
    GoogleTranslator,
    YoudaoTranslator,
    build_provider,
)

FIXED_SALT = "abcd1234abcd1234"
FIXED_NOW = 1700000000.0


def install(translator: TraditionalTranslatorBase, handler) -> list[httpx.Request]:
    """把通道的 httpx 客户端换成 MockTransport，返回捕获到的请求列表。"""
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    translator._client.close()
    translator._client = httpx.Client(transport=httpx.MockTransport(_handler))
    return captured


@pytest.fixture(autouse=True)
def fixed_env(monkeypatch):
    """把 salt 与时间固定下来，让签名可复现。"""
    monkeypatch.setattr(TraditionalTranslatorBase, "random_salt", staticmethod(lambda length=16: FIXED_SALT))
    monkeypatch.setattr(time, "time", lambda: FIXED_NOW)


# --------------------------------------------------------------------------- #
# 百度
# --------------------------------------------------------------------------- #
def test_baidu_signature_and_fields():
    t = BaiduTranslator(app_id="myid", secret_key="mysecret")
    captured = install(t, lambda r: httpx.Response(200, json={
        "from": "en", "to": "zh", "trans_result": [{"src": "hello", "dst": "你好"}],
    }))

    out = t._translate_one("hello", "en", "zh")
    assert out == "你好"

    body = captured[0].content.decode()
    expected_sign = hashlib.md5(
        f"myid{'hello'}{FIXED_SALT}mysecret".encode("utf-8")
    ).hexdigest()
    assert f"sign={expected_sign}" in body
    assert "appid=myid" in body
    assert f"salt={FIXED_SALT}" in body
    assert "q=hello" in body


def test_baidu_reports_error_code():
    t = BaiduTranslator(app_id="myid", secret_key="bad")
    install(t, lambda r: httpx.Response(200, json={
        "error_code": "54001", "error_msg": "Invalid Sign",
    }))
    with pytest.raises(RuntimeError, match="54001"):
        t._translate_one("hello", "en", "zh")


def test_baidu_missing_credentials():
    t = BaiduTranslator()
    with pytest.raises(UnsupportedLanguage, match="缺少"):
        t._translate_one("hello", "en", "zh")


def test_baidu_language_mapping():
    t = BaiduTranslator(app_id="a", secret_key="b")
    assert t.map_language("ja") == "jp"     # 百度日语是 jp，不是 ja
    assert t.map_language("ko") == "kor"
    assert t.map_language("fr") == "fra"


# --------------------------------------------------------------------------- #
# 有道（签名最复杂，truncate 规则必须精确）
# --------------------------------------------------------------------------- #
def test_youdao_truncate_rules():
    assert YoudaoTranslator.truncate("短文本") == "短文本"
    assert YoudaoTranslator.truncate("a" * 20) == "a" * 20      # 边界：20 不截断
    long_text = "a" * 25
    got = YoudaoTranslator.truncate(long_text)
    assert got == "a" * 10 + "25" + "a" * 10, got


def test_youdao_signature_and_fields():
    t = YoudaoTranslator(app_key="key1", app_secret="sec1")
    captured = install(t, lambda r: httpx.Response(200, json={
        "errorCode": "0", "translation": ["你好"],
    }))

    out = t._translate_one("hello", "en", "zh-CHS")
    assert out == "你好"

    body = captured[0].content.decode()
    curtime = str(int(FIXED_NOW))
    expected = hashlib.sha256(
        f"key1{'hello'}{FIXED_SALT}{curtime}sec1".encode("utf-8")
    ).hexdigest()
    assert f"sign={expected}" in body
    assert "signType=v3" in body
    assert f"curtime={curtime}" in body
    assert "appKey=key1" in body


def test_youdao_error_code():
    t = YoudaoTranslator(app_key="k", app_secret="s")
    install(t, lambda r: httpx.Response(200, json={"errorCode": "108", "translation": []}))
    with pytest.raises(RuntimeError, match="108"):
        t._translate_one("x", "en", "zh-CHS")


# --------------------------------------------------------------------------- #
# Azure
# --------------------------------------------------------------------------- #
def test_azure_headers_and_body():
    t = AzureTranslator(api_key="key-azure", region="eastasia")
    captured = install(t, lambda r: httpx.Response(200, json=[
        {"translations": [{"text": "你好", "to": "zh-Hans"}]},
    ]))

    out = t._translate_one("hello", "en", "zh-Hans")
    assert out == "你好"

    req = captured[0]
    assert req.headers["Ocp-Apim-Subscription-Key"] == "key-azure"
    assert req.headers["Ocp-Apim-Subscription-Region"] == "eastasia"
    assert "api-version=3.0" in str(req.url)
    assert "to=zh-Hans" in str(req.url)
    assert b'"Text"' in req.content


def test_azure_omits_from_when_auto():
    t = AzureTranslator(api_key="k")
    captured = install(t, lambda r: httpx.Response(200, json=[
        {"translations": [{"text": "x"}]},
    ]))
    t._translate_one("hello", "auto", "zh-Hans")
    assert "from=" not in str(captured[0].url)


# --------------------------------------------------------------------------- #
# Google
# --------------------------------------------------------------------------- #
def test_google_params_and_parsing():
    t = GoogleTranslator(api_key="gkey")
    captured = install(t, lambda r: httpx.Response(200, json={
        "data": {"translations": [{"translatedText": "你好"}]},
    }))
    out = t._translate_one("hello", "en", "zh-CN")
    assert out == "你好"
    body = captured[0].content.decode()
    assert "key=gkey" in body
    assert "target=zh-CN" in body
    assert "source=en" in body


def test_google_omits_source_when_auto():
    t = GoogleTranslator(api_key="gkey")
    captured = install(t, lambda r: httpx.Response(200, json={
        "data": {"translations": [{"translatedText": "x"}]},
    }))
    t._translate_one("hello", "auto", "zh-CN")
    assert "source=" not in captured[0].content.decode()


# --------------------------------------------------------------------------- #
# DeepL
# --------------------------------------------------------------------------- #
def test_deepl_auth_header_and_free_endpoint():
    t = DeepLTranslator(api_key="dk", free=True)
    captured = install(t, lambda r: httpx.Response(200, json={
        "translations": [{"text": "你好"}],
    }))
    out = t._translate_one("hello", "en", "ZH")
    assert out == "你好"
    assert captured[0].headers["Authorization"] == "DeepL-Auth-Key dk"
    assert "api-free.deepl.com" in str(captured[0].url)


def test_deepl_pro_endpoint():
    t = DeepLTranslator(api_key="dk", free=False)
    assert "api.deepl.com" in t.endpoint
    assert "free" not in t.endpoint


def test_deepl_does_not_send_chinese_as_source():
    """DeepL 不支持把中文当源语言，传了会直接报错。"""
    t = DeepLTranslator(api_key="dk")
    captured = install(t, lambda r: httpx.Response(200, json={
        "translations": [{"text": "x"}],
    }))
    t._translate_one("你好", "zh", "EN")
    body = captured[0].content.decode()
    assert "source_lang" not in body
    assert "target_lang=EN" in body


# --------------------------------------------------------------------------- #
# 基类：批量循环、重试、失败记录
# --------------------------------------------------------------------------- #
def test_base_loops_over_segments_and_merges():
    t = BaiduTranslator(app_id="a", secret_key="b")
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = r.content.decode()
        q = [p.split("=", 1)[1] for p in body.split("&") if p.startswith("q=")][0]
        return httpx.Response(200, json={"trans_result": [{"dst": f"译:{q}"}]})

    install(t, handler)
    req = TranslateRequest(
        segments=[Segment(id=1, text="one"), Segment(id=2, text="two")],
        source_language="en", target_language="zh",
    )
    r = t.translate(req)
    assert r.translations == {1: "译:one", 2: "译:two"}
    assert calls["n"] == 2, "传统 API 单条调用，基类应逐条发"


def test_base_records_failure_without_raising():
    t = BaiduTranslator(app_id="a", secret_key="b", retry_times=0)
    install(t, lambda r: httpx.Response(200, json={
        "error_code": "54003", "error_msg": "quota",
    }))
    req = TranslateRequest(segments=[Segment(id=1, text="x")])
    r = t.translate(req)
    assert r.fail_count == 1
    assert "54003" in r.failures[1]


def test_base_retries_then_succeeds():
    t = BaiduTranslator(app_id="a", secret_key="b", retry_times=2)
    state = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 2:
            return httpx.Response(500, json={"error": "server busy"})
        return httpx.Response(200, json={"trans_result": [{"dst": "成功"}]})

    install(t, handler)
    r = t.translate(TranslateRequest(segments=[Segment(id=1, text="x")]))
    assert r.translations[1] == "成功"
    assert state["n"] == 2


def test_base_throttle_disabled_when_qps_zero():
    t = BaiduTranslator(app_id="a", secret_key="b", qps_limit=0)
    start = time.monotonic()
    for _ in range(3):
        t._throttle()
    assert time.monotonic() - start < 0.05


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def test_providers_registry():
    assert set(PROVIDERS) == {"baidu", "youdao", "azure", "google", "deepl"}


def test_build_provider_from_credentials():
    t = build_provider("baidu", {"app_id": "x", "secret_key": "y"})
    assert isinstance(t, BaiduTranslator)
    assert t.app_id == "x"


def test_build_provider_unknown_returns_none():
    assert build_provider("nonexistent", {}) is None


def test_build_provider_ignores_empty_credentials():
    t = build_provider("deepl", {"api_key": "", "free": True})
    assert isinstance(t, DeepLTranslator)
    assert t.api_key == ""
