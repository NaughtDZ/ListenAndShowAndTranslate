"""网页版翻译的"内部接口"通道（不需要 API key）。

实测结论（2026-02-21，本机 + 代理 127.0.0.1:2333）：

====================  ==============  ==================================================
通道                   实测结果        说明
====================  ==============  ==================================================
Google gtx 端点        ✅ **可用**      无 key、无浏览器；热态 248~634ms；130+ 语言
Bing ttranslatev3      ✅ **可用**      参数见下；0.5 QPS 下连打 8 条**零失败**，
                                     中位延迟 1235ms；日译中质量好过本地 2B 模型
百度 v2transapi        ❌ 不可用        需 JS 算出的 sign；首页 token/gtk 由 JS 注入
有道老接口             ❌ 不可用        已改版，返回非 JSON，现在同样需要签名
====================  ==============  ==================================================

**Bing 的参数要求（踩坑记录，省得以后再摸一遍）**

``POST https://www.bing.com/ttranslatev3?isVertical=1&IG=<IG>&IID=<IID>``

必须先 ``GET https://www.bing.com/translator``，从页面里抠出 **四个**值：

============================  ==========================================
值                            来源（正则）
============================  ==========================================
``IG``                        ``IG:"([0-9A-F]+)"``
``IID``                       ``data-iid="([^"]+)"``
``key`` + ``token``           ``params_AbusePreventionHelper = [key, token, interval]``
============================  ==========================================

body 里要同时带 ``fromLang`` / ``to`` / ``text`` / ``key`` / ``token``。

**只用 token 不用 key 会返回 ``statusCode:205``**——这就是最初卡住的地方。

**⚠️ 限流是它唯一的软肋，但比想象中宽松**：
密集探测（几秒内打 5 次以上）会触发 ``HTTP 401 {"ShowCaptcha":false}``，
但**隔一段时间就自动恢复**（并不是封禁）。
按实时字幕的真实节奏（默认 ``qps_limit=0.5``，即每 2 秒一次）连打 8 条**零失败**。
所以它**可以作为实时字幕的主力通道之一**，只要别把 QPS 调高。
连续 401 时 TranslatorHub 的熔断会自动切走，不会卡住字幕。

参数会在有效期内缓存（防滥用 interval 通常 1 小时，代码按 90% 保守取值），
只有 401 时才强制刷新——否则每条字幕都要多花 0.5~1 秒抓页面。

**本机实测：Bing 的 POST 直连（不挂代理）会返回空响应**，
挂上代理才正常，所以国内使用仍需代理。


**关于"要不要用无头浏览器"**（用户问过，这里留个结论，免得以后再走一遍）：

不需要。百度/有道要的是"能跑 JS"，不是"整个浏览器"：
  - 浏览器方案：Playwright 要装 ~150-300MB 浏览器，每次翻译 1~3 秒，
    而且**并不能降低被封风险**（一样是爬），反爬对 headless 反而更敏感
  - JS 引擎方案：``py_mini_racer`` / ``quickjs`` 只有几 MB，直接跑签名函数

但两条路的**真正成本都不是技术，而是维护**：各家会改 JS/端点，
今天能用的明天可能就 997。所以本模块的定位是 **兜底通道**，
主通道仍应是本地 LLM（免费、无限、离线、质量更好），
由 TranslatorHub 的熔断机制在免费通道失效时自动切换。
"""

from __future__ import annotations

import re
import threading
import time

from app.translate.traditional.base import TraditionalTranslatorBase
from app.utils.log import get_logger

log = get_logger(__name__)

# 浏览器特征头：Bing 对缺这些头的请求更敏感（实测带上仍会被限流，但更接近真实浏览器）
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

_XHR_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.bing.com",
    "Referer": "https://www.bing.com/translator",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

_IG_RE = re.compile(r'IG:"([0-9A-F]+)"')
_IID_RE = re.compile(r'data-iid="([^"]+)"')
_ABUSE_RE = re.compile(
    r'params_AbusePreventionHelper\s*=\s*\[(\d+),\s*"([^"]+)",\s*(\d+)\]'
)


class GoogleWebTranslator(TraditionalTranslatorBase):
    """谷歌翻译的公开 gtx 端点（``translate_a/single``）。

    这是网页版翻译器自己用的接口，**无需 API key**。
    非官方、无 SLA，随时可能变；只建议作为兜底。
    国内需要代理（本机 127.0.0.1:2333）。
    """

    name = "web_google"
    ENDPOINT = "https://translate.googleapis.com/translate_a/single"

    LANG_MAP = {
        "zh": "zh-CN", "zh-en": "zh-CN", "en": "en", "ja": "ja", "ko": "ko",
        "yue": "yue", "fr": "fr", "de": "de", "es": "es", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ar", "th": "th", "vi": "vi",
        "id": "id", "tr": "tr", "nl": "nl", "pl": "pl", "hi": "hi",
    }

    def __init__(self, proxy: str = "", qps_limit: float = 3.0, **kw) -> None:
        # 免费端点限流严，默认压低 QPS，别把 IP 打黑
        super().__init__(proxy=proxy, qps_limit=qps_limit, **kw)

    def model_name(self) -> str:
        return "google-web-gtx"

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        params = {
            "client": "gtx",
            "sl": src or "auto",
            "tl": dst,
            "dt": "t",
            "q": text,
        }
        r = self._client.get(self.ENDPOINT, params=params)
        r.raise_for_status()
        data = r.json()
        # 返回结构：[[[译文片段, 原文片段, ...], ...], ...]
        # 注意 dict 上取 [0] 抛的是 KeyError（不是 IndexError），三种都要接住
        try:
            segments = data[0] or []
        except (IndexError, TypeError, KeyError) as exc:
            raise RuntimeError(f"谷歌返回结构异常: {str(data)[:120]}") from exc
        out = "".join(seg[0] for seg in segments if seg and seg[0])
        if not out:
            raise RuntimeError("谷歌返回空译文")
        return out.strip()

    def ping(self) -> tuple[bool, str]:
        try:
            t0 = time.time()
            out = self._translate_one("hello", "en", "zh-CN")
            dt = (time.time() - t0) * 1000
            return (True, f"可用（{dt:.0f}ms），测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:150]
            if "proxy" in msg.lower() or "connect" in msg.lower() or "timeout" in msg.lower():
                msg += "（国内访问谷歌需要代理，请检查设置里的代理地址）"
            return False, msg


class BingWebTranslator(TraditionalTranslatorBase):
    """Bing 网页版翻译（``ttranslatev3``）。

    实测**可用**：0.5 QPS 下连打 8 条零失败，中位延迟 1235ms，
    日译中质量好过本地 qwen3.8-2b。

    ⚠️ 唯一的软肋是限流：密集请求（几秒内 5 次以上）会触发
    ``HTTP 401 {"ShowCaptcha":false}``，但过一会儿会自动恢复（不是封禁）。
    所以默认 ``qps_limit=0.5``（每 2 秒一次），不要调高。
    连续 401 时 TranslatorHub 的熔断会自动切走，字幕不会卡住。

    参数（IG/IID/key/token）会缓存复用，只在 401 时强制刷新——
    否则每条字幕都要多花 0.5~1 秒去抓翻译页。
    """

    name = "web_bing"

    LANG_MAP = {
        "zh": "zh-Hans", "zh-en": "zh-Hans", "en": "en", "ja": "ja", "ko": "ko",
        "yue": "yue", "fr": "fr", "de": "de", "es": "es", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ar", "th": "th", "vi": "vi",
    }

    def __init__(
        self,
        proxy: str = "",
        qps_limit: float = 0.5,
        host: str = "https://www.bing.com",
        **kw,
    ) -> None:
        super().__init__(proxy=proxy, qps_limit=qps_limit, **kw)
        self.host = host.rstrip("/")
        self._params: dict | None = None
        self._params_expire = 0.0
        self._params_lock = threading.Lock()

    def model_name(self) -> str:
        return "bing-web-ttranslatev3"

    # ------------------------------------------------------------------ #
    def _fetch_params(self) -> dict:
        """从翻译页抠出 IG / IID / key / token **四个**值。

        **key 与 token 缺一不可**：只给 token 会返回 ``statusCode:205``，
        这正是当初卡住的地方。
        """
        r = self._client.get(f"{self.host}/translator", headers=_BROWSER_HEADERS)
        r.raise_for_status()
        html = r.text
        ig = _IG_RE.search(html)
        iid = _IID_RE.search(html)
        abuse = _ABUSE_RE.search(html)
        if not (ig and iid and abuse):
            raise RuntimeError(
                "Bing 翻译页里没找到 IG/IID/防滥用参数（页面结构可能已变，或触发了验证）"
            )
        # 防滥用参数自带有效期（秒），提前 10% 刷新
        interval_s = max(60, int(abuse.group(3)) // 1000)
        return {
            "IG": ig.group(1),
            "IID": iid.group(1),
            "key": abuse.group(1),
            "token": abuse.group(2),
            "expire": time.time() + interval_s * 0.9,
        }

    def _ensure_params(self, force: bool = False) -> dict:
        with self._params_lock:
            if force or self._params is None or time.time() >= self._params_expire:
                self._params = self._fetch_params()
                self._params_expire = self._params["expire"]
                log.debug("已刷新 Bing 防滥用参数")
            return self._params

    def _post(self, text: str, src: str, dst: str, params: dict) -> tuple[int, str]:
        r = self._client.post(
            f"{self.host}/ttranslatev3",
            params={"isVertical": "1", "IG": params["IG"], "IID": params["IID"]},
            data={
                "fromLang": src or "auto",
                "to": dst,
                "text": text,
                "key": params["key"],
                "token": params["token"],
            },
            headers=_XHR_HEADERS,
        )
        return r.status_code, r.text

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        import json

        params = self._ensure_params()
        status, body = self._post(text, src, dst, params)

        # 401 + ShowCaptcha 是"参数被拒/被限流"的典型响应：
        # 换一套新参数再试一次；还不行就交给上层熔断
        if status == 401 or "ShowCaptcha" in body:
            log.debug("Bing 拒绝了当前参数（HTTP %s），刷新后重试一次", status)
            params = self._ensure_params(force=True)
            status, body = self._post(text, src, dst, params)

        if status != 200:
            raise RuntimeError(f"Bing HTTP {status}: {body[:120]}")
        if '"translations"' not in body:
            raise RuntimeError(f"Bing 返回错误: {body[:120]}")
        try:
            data = json.loads(body)
            out = data[0]["translations"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Bing 返回结构异常: {body[:120]}") from exc
        if not out:
            raise RuntimeError("Bing 返回空译文")
        return out.strip()

    def ping(self) -> tuple[bool, str]:
        try:
            t0 = time.time()
            out = self._translate_one("hello", "en", "zh-Hans")
            dt = (time.time() - t0) * 1000
            return (True, f"可用（{dt:.0f}ms），测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:160]
            if "401" in msg or "205" in msg or "ShowCaptcha" in msg:
                msg += "。Bing 网页接口限流很紧（实测连续几次后即被拒），建议用 web_google 或本地模型"
            return False, msg
