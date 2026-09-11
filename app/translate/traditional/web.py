"""网页版翻译的"内部接口"通道（不需要 API key）。

实测结论（2026-02-21，走本机代理 127.0.0.1:2333）：

===================  ==========  ================================================
通道                  实测结果    说明
===================  ==========  ================================================
Google gtx 端点       ✅ **可用**  无 key、无浏览器；热态 248~634ms；覆盖 130+ 语言
Bing ttranslatev3     ⚠️ 部分可用  能从翻译页抓到 IG/IID，但 POST 返回 statusCode 205
                                  （参数还需调整）；且每次要抓页面（+763ms）除非缓存
百度 v2transapi       ❌ 不可用    返回 errno 997，**需要 JS 算出的 sign**；
                                  首页里也没有 token/gtk（由 JS 注入）
有道老接口            ❌ 不可用    已改版，返回非 JSON，现在同样需要签名
===================  ==========  ================================================

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

import time

from app.translate.traditional.base import TraditionalTranslatorBase
from app.utils.log import get_logger

log = get_logger(__name__)


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
            if "proxy" in msg.lower() or "connect" in msg.lower():
                msg += "（国内访问谷歌需要代理，请检查设置里的代理地址）"
            return False, msg
