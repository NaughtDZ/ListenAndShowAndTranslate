"""各传统翻译 API 的适配器实现。

签名算法严格按各家文档实现，salt 与时间戳可注入（便于单元测试验签）。
**没有 key 也能导入和实例化**，只有真正调用时才会报鉴权错误——
这样设置界面可以先列出来让用户填 key。
"""

from __future__ import annotations

import hashlib

from app.translate.traditional.base import TraditionalTranslatorBase, UnsupportedLanguage
from app.utils.log import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 百度翻译
# --------------------------------------------------------------------------- #
class BaiduTranslator(TraditionalTranslatorBase):
    """百度翻译（通用文本翻译）。

    签名：``md5(appid + q + salt + secret_key)``
    """

    name = "baidu"
    ENDPOINT = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    LANG_MAP = {
        "zh": "zh", "zh-en": "zh", "en": "en", "ja": "jp", "ko": "kor",
        "yue": "yue", "fr": "fra", "de": "de", "es": "spa", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ara", "th": "th", "vi": "vie",
    }

    def __init__(self, app_id: str = "", secret_key: str = "", **kw) -> None:
        super().__init__(**kw)
        self.app_id = app_id
        self.secret_key = secret_key

    def model_name(self) -> str:
        return "baidu-general"

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if not self.app_id or not self.secret_key:
            raise UnsupportedLanguage("百度翻译缺少 app_id / secret_key")
        salt = self.random_salt()
        sign = hashlib.md5(
            f"{self.app_id}{text}{salt}{self.secret_key}".encode("utf-8")
        ).hexdigest()
        data = {
            "q": text, "from": src or "auto", "to": dst,
            "appid": self.app_id, "salt": salt, "sign": sign,
        }
        r = self._client.post(self.ENDPOINT, data=data)
        r.raise_for_status()
        body = r.json()
        if body.get("error_code"):
            raise RuntimeError(f"百度错误 {body.get('error_code')}: {body.get('error_msg')}")
        items = body.get("trans_result") or []
        if not items:
            raise RuntimeError("百度返回空结果")
        return "\n".join(i.get("dst", "") for i in items).strip()

    def ping(self) -> tuple[bool, str]:
        if not self.app_id or not self.secret_key:
            return False, "未填写 app_id / secret_key"
        try:
            out = self._translate_one("hello", "en", "zh")
            return (True, f"可用，测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]


# --------------------------------------------------------------------------- #
# 有道智云
# --------------------------------------------------------------------------- #
class YoudaoTranslator(TraditionalTranslatorBase):
    """有道智云文本翻译 v3。

    签名：``sha256(appKey + truncate(q) + salt + curtime + appSecret)``
    其中 ``truncate(q)``：长度 ≤20 用原文；否则取「前10 + 长度 + 后10」。
    """

    name = "youdao"
    ENDPOINT = "https://openapi.youdao.com/api"

    LANG_MAP = {
        "zh": "zh-CHS", "zh-en": "zh-CHS", "en": "en", "ja": "ja", "ko": "ko",
        "yue": "yue", "fr": "fr", "de": "de", "es": "es", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ar", "th": "th", "vi": "vi",
    }

    def __init__(self, app_key: str = "", app_secret: str = "", **kw) -> None:
        super().__init__(**kw)
        self.app_key = app_key
        self.app_secret = app_secret

    def model_name(self) -> str:
        return "youdao-v3"

    @staticmethod
    def truncate(text: str) -> str:
        """有道的 q 截断规则（签名的关键，写错就 108 错误）。"""
        if len(text) <= 20:
            return text
        return text[:10] + str(len(text)) + text[-10:]

    def _sign(self, text: str, salt: str, curtime: str) -> str:
        raw = f"{self.app_key}{self.truncate(text)}{salt}{curtime}{self.app_secret}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if not self.app_key or not self.app_secret:
            raise UnsupportedLanguage("有道翻译缺少 app_key / app_secret")
        salt = self.random_salt()
        curtime = str(int(__import__("time").time()))
        data = {
            "q": text, "from": src or "auto", "to": dst,
            "appKey": self.app_key, "salt": salt, "sign": self._sign(text, salt, curtime),
            "signType": "v3", "curtime": curtime,
        }
        r = self._client.post(self.ENDPOINT, data=data)
        r.raise_for_status()
        body = r.json()
        code = str(body.get("errorCode", "0"))
        if code != "0":
            raise RuntimeError(f"有道错误码 {code}")
        items = body.get("translation") or []
        if not items:
            raise RuntimeError("有道返回空结果")
        return "\n".join(items).strip()

    def ping(self) -> tuple[bool, str]:
        if not self.app_key or not self.app_secret:
            return False, "未填写 app_key / app_secret"
        try:
            out = self._translate_one("hello", "en", "zh-CHS")
            return (True, f"可用，测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]


# --------------------------------------------------------------------------- #
# 微软 Azure
# --------------------------------------------------------------------------- #
class AzureTranslator(TraditionalTranslatorBase):
    """Azure AI Translator（免费额度较大，国内可直连）。"""

    name = "azure"
    ENDPOINT = "https://api.cognitive.microsofttranslator.com/translate"

    LANG_MAP = {
        "zh": "zh-Hans", "zh-en": "zh-Hans", "en": "en", "ja": "ja", "ko": "ko",
        "yue": "yue", "fr": "fr", "de": "de", "es": "es", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ar", "th": "th", "vi": "vi",
    }

    def __init__(self, api_key: str = "", region: str = "", **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key
        self.region = region

    def model_name(self) -> str:
        return "azure-v3"

    def _headers_call(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Ocp-Apim-Subscription-Key"] = self.api_key
        if self.region:
            h["Ocp-Apim-Subscription-Region"] = self.region
        return h

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if not self.api_key:
            raise UnsupportedLanguage("Azure 翻译缺少 api_key")
        params = {"api-version": "3.0", "to": dst}
        if src and src != "auto":
            params["from"] = src
        r = self._client.post(
            self.ENDPOINT, params=params,
            headers=self._headers_call(),
            json=[{"Text": text}],
        )
        r.raise_for_status()
        body = r.json()
        if not body or not body[0].get("translations"):
            raise RuntimeError("Azure 返回空结果")
        return body[0]["translations"][0].get("text", "").strip()

    def ping(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "未填写 api_key"
        try:
            out = self._translate_one("hello", "en", "zh-Hans")
            return (True, f"可用，测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]


# --------------------------------------------------------------------------- #
# 谷歌翻译
# --------------------------------------------------------------------------- #
class GoogleTranslator(TraditionalTranslatorBase):
    """谷歌云翻译 v2（需要 API Key）。

    也可以填 ``https://translate.googleapis.com`` 走非官方免费端点，
    但那条路**随时可能失效**，所以默认走官方。
    """

    name = "google"
    ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

    LANG_MAP = {
        "zh": "zh-CN", "zh-en": "zh-CN", "en": "en", "ja": "ja", "ko": "ko",
        "yue": "yue", "fr": "fr", "de": "de", "es": "es", "ru": "ru",
        "pt": "pt", "it": "it", "ar": "ar", "th": "th", "vi": "vi",
    }

    def __init__(self, api_key: str = "", **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key

    def model_name(self) -> str:
        return "google-v2"

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if not self.api_key:
            raise UnsupportedLanguage("谷歌翻译缺少 api_key")
        data = {"q": text, "target": dst, "format": "text", "key": self.api_key}
        if src and src != "auto":
            data["source"] = src
        r = self._client.post(self.ENDPOINT, data=data)
        r.raise_for_status()
        body = r.json()
        items = (body.get("data") or {}).get("translations") or []
        if not items:
            raise RuntimeError("谷歌返回空结果")
        return items[0].get("translatedText", "").strip()

    def ping(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "未填写 api_key"
        try:
            out = self._translate_one("hello", "en", "zh-CN")
            return (True, f"可用，测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]


# --------------------------------------------------------------------------- #
# DeepL
# --------------------------------------------------------------------------- #
class DeepLTranslator(TraditionalTranslatorBase):
    """DeepL。免费版与付费版端点不同，用 ``free`` 开关切换。"""

    name = "deepl"

    LANG_MAP = {
        "zh": "ZH", "zh-en": "ZH", "en": "EN", "ja": "JA", "ko": "KO",
        "fr": "FR", "de": "DE", "es": "ES", "ru": "RU", "pt": "PT",
        "it": "IT", "nl": "NL", "pl": "PL",
    }

    def __init__(self, api_key: str = "", free: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key
        self.free = free
        self.endpoint = (
            "https://api-free.deepl.com/v2/translate" if free
            else "https://api.deepl.com/v2/translate"
        )

    def model_name(self) -> str:
        return "deepl-free" if self.free else "deepl-pro"

    def _translate_one(self, text: str, src: str, dst: str) -> str:
        if not self.api_key:
            raise UnsupportedLanguage("DeepL 缺少 api_key")
        data = {"text": text, "target_lang": dst}
        # DeepL 不支持把中文当源语言（会直接报错），所以 auto 时不传 source
        if src and src != "auto" and src.upper() != "ZH":
            data["source_lang"] = src.upper()
        r = self._client.post(
            self.endpoint, data=data,
            headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"},
        )
        r.raise_for_status()
        body = r.json()
        items = body.get("translations") or []
        if not items:
            raise RuntimeError("DeepL 返回空结果")
        return items[0].get("text", "").strip()

    def ping(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "未填写 api_key"
        try:
            out = self._translate_one("hello", "en", "ZH")
            return (True, f"可用，测试译文：{out}") if out else (False, "返回空译文")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]


#: 通道 id → 类。设置界面按这张表渲染。
PROVIDERS: dict[str, type[TraditionalTranslatorBase]] = {
    "baidu": BaiduTranslator,
    "youdao": YoudaoTranslator,
    "azure": AzureTranslator,
    "google": GoogleTranslator,
    "deepl": DeepLTranslator,
}


def build_provider(provider_id: str, credentials: dict, **kw) -> TraditionalTranslatorBase | None:
    """按配置里的凭据字典构造通道。

    凭据键名故意和各家文档一致（``app_id`` / ``secret_key`` / ``app_key`` ...），
    这样用户对着文档填就行，不用记我们自创的名字。
    """
    cls = PROVIDERS.get(provider_id)
    if cls is None:
        return None
    creds = {k: v for k, v in (credentials or {}).items() if v}
    return cls(**creds, **kw)
