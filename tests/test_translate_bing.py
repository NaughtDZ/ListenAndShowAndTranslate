"""Bing 网页版翻译通道的测试（不联网）。

重点验证踩坑点：
1. IG / IID / key / token **四个都要**（只给 token 会 205）
2. 401/ShowCaptcha 时要刷新参数重试一次
3. 参数有有效期，过期要自动刷新
"""

from __future__ import annotations

import httpx
import pytest

from app.translate.base import Segment, TranslateRequest
from app.translate.traditional.providers import ALL_PROVIDERS, WEB_PROVIDERS
from app.translate.traditional.web import BingWebTranslator

FAKE_PAGE = """
<html><head><script>
  var IG:"AABBCCDDEEFF00112233445566778899";
  params_AbusePreventionHelper = [1789173064447,"TOKEN-abc_123",3600000];
</script></head>
<body data-iid="translator.5023"></body></html>
"""


def make_translator(handler) -> tuple[BingWebTranslator, list[httpx.Request]]:
    t = BingWebTranslator(retry_times=0)
    captured: list[httpx.Request] = []

    def _wrap(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    t._client.close()
    t._client = httpx.Client(transport=httpx.MockTransport(_wrap))
    return t, captured


def test_bing_registered():
    assert "web_bing" in WEB_PROVIDERS
    assert "web_bing" in ALL_PROVIDERS


# 注意：不能用"URL 里有没有 translator"来区分取页面和翻译请求——
# 翻译请求的 query 里带着 IID=translator.5023，一样含这个词（这个坑把 6 个测试一起干掉了）。
# 用 HTTP 方法区分才可靠：取页面是 GET，翻译是 POST。
def _is_page(r: httpx.Request) -> bool:
    return r.method == "GET"


def test_bing_extracts_all_four_params_from_page():
    t, captured = make_translator(
        lambda r: httpx.Response(200, text=FAKE_PAGE) if _is_page(r)
        else httpx.Response(200, text='[{"translations":[{"text":"你好"}]}]')
    )
    out = t._translate_one("hello", "en", "zh-Hans")
    assert out == "你好"

    # 第一次是取页面，第二次是 POST
    assert len(captured) == 2
    post = captured[1]
    body = post.content.decode()
    # 四个参数都要在
    assert "IG=AABBCCDDEEFF00112233445566778899" in str(post.url)
    assert "IID=translator.5023" in str(post.url)
    assert "key=1789173064447" in body
    assert "token=TOKEN-abc_123" in body
    assert "fromLang" in body and "text=hello" in body


def test_bing_requires_both_key_and_token():
    """只给 token 不给 key 会返回 statusCode 205 —— 最初就卡在这里。"""
    seen: list[str] = []

    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            return httpx.Response(200, text=FAKE_PAGE)
        body = r.content.decode()
        seen.append(body)
        if "key=" not in body:
            return httpx.Response(200, text='{"statusCode":205,"errorMessage":""}')
        return httpx.Response(200, text='[{"translations":[{"text":"你好"}]}]')

    t, _ = make_translator(handler)
    assert t._translate_one("hello", "en", "zh-Hans") == "你好"
    assert all("key=" in b for b in seen), "每次 POST 都必须带 key"


def test_bing_205_raises_clear_error():
    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            return httpx.Response(200, text=FAKE_PAGE)
        return httpx.Response(200, text='{"statusCode":205,"errorMessage":""}')

    t, _ = make_translator(handler)
    with pytest.raises(RuntimeError, match="Bing"):
        t._translate_one("hello", "en", "zh-Hans")


def test_bing_401_refreshes_params_and_retries_once():
    """401 + ShowCaptcha 是'参数被拒/被限流'的典型响应：应刷新参数再试一次。"""
    state = {"page_fetches": 0, "posts": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            state["page_fetches"] += 1
            return httpx.Response(200, text=FAKE_PAGE)
        state["posts"] += 1
        if state["posts"] == 1:
            return httpx.Response(401, text='{"ShowCaptcha":false}')
        return httpx.Response(200, text='[{"translations":[{"text":"你好"}]}]')

    t, _ = make_translator(handler)
    assert t._translate_one("hello", "en", "zh-Hans") == "你好"
    assert state["page_fetches"] == 2, "401 后应重新取一次参数"
    assert state["posts"] == 2


def test_bing_persistent_401_raises():
    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            return httpx.Response(200, text=FAKE_PAGE)
        return httpx.Response(401, text='{"ShowCaptcha":false}')

    t, _ = make_translator(handler)
    with pytest.raises(RuntimeError, match="401"):
        t._translate_one("hello", "en", "zh-Hans")


def test_bing_caches_params_between_calls():
    """参数有有效期（防滥用 interval），不该每次都重取页面。"""
    state = {"pages": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            state["pages"] += 1
            return httpx.Response(200, text=FAKE_PAGE)
        return httpx.Response(200, text='[{"translations":[{"text":"你好"}]}]')

    t, _ = make_translator(handler)
    t._translate_one("a", "en", "zh-Hans")
    t._translate_one("b", "en", "zh-Hans")
    t._translate_one("c", "en", "zh-Hans")
    assert state["pages"] == 1, "参数在有效期内应复用"


def test_bing_refreshes_expired_params():
    t, _ = make_translator(
        lambda r: httpx.Response(200, text=FAKE_PAGE) if _is_page(r)
        else httpx.Response(200, text='[{"translations":[{"text":"x"}]}]')
    )
    t._params_expire = 0  # 人为过期
    t._ensure_params()
    assert t._params_expire > 0


def test_bing_page_missing_params_raises():
    t, _ = make_translator(lambda r: httpx.Response(200, text="<html>nothing here</html>"))
    with pytest.raises(RuntimeError, match="没找到"):
        t._translate_one("hello", "en", "zh-Hans")


def test_bing_language_mapping():
    t = BingWebTranslator()
    assert t.map_language("zh") == "zh-Hans"
    assert t.map_language("ja") == "ja"


def test_bing_low_default_qps():
    """限流很紧，默认 QPS 必须压得很低。"""
    t = BingWebTranslator()
    assert t.qps_limit <= 1.0


def test_bing_translate_records_failure_without_raising():
    def handler(r: httpx.Request) -> httpx.Response:
        if _is_page(r):
            return httpx.Response(200, text=FAKE_PAGE)
        return httpx.Response(401, text='{"ShowCaptcha":false}')

    t, _ = make_translator(handler)
    r = t.translate(TranslateRequest(segments=[Segment(id=1, text="x")],
                                     source_language="ja", target_language="zh"))
    assert r.fail_count == 1
    assert "401" in r.failures[1]
