"""传统翻译 API 的公共基类。

这些 API 的特点（与 LLM 通道的区别）：
- 大多**一次只翻一条**文本 → 基类负责循环 + 合并，对上层仍是统一的批量接口
- 语言代码**各家不一样**（日语在百度是 ``jp``、有道是 ``ja``、微软是 ``ja``）→ 统一做映射表
- 按**字符计费** → 基类统计字符数
- 会限流/报错 → 统一的退避重试

签名相关的 salt / 时间戳做成可注入参数，这样单元测试能验签而不依赖当前时间。
"""

from __future__ import annotations

import random
import string
import threading
import time
from abc import ABC, abstractmethod

import httpx

from app.translate.base import Segment, TranslateRequest, TranslateResult, TranslatorStats
from app.utils.log import get_logger

log = get_logger(__name__)


class UnsupportedLanguage(ValueError):
    """该通道不支持这对语言。"""


class TraditionalTranslatorBase(ABC):
    """传统 API 通道基类：子类只需实现 :meth:`_translate_one`。"""

    name = "traditional"
    supports_batch = False
    """传统 API 基本都是单条，由基类循环调用，所以对外仍是批量接口。"""

    supports_streaming = False

    #: 每家的语言代码表：内部代码 → 该家代码
    LANG_MAP: dict[str, str] = {}

    def __init__(
        self,
        proxy: str = "",
        timeout_s: float = 15.0,
        retry_times: int = 2,
        qps_limit: float = 5.0,
        app_name: str = "ListenAndShowAndTranslate",
    ) -> None:
        self.proxy = proxy or ""
        self.timeout_s = timeout_s
        self.retry_times = max(0, retry_times)
        self.qps_limit = qps_limit
        self.app_name = app_name
        self.stats = TranslatorStats()
        self._lock = threading.Lock()
        self._last_call_at = 0.0
        self._client = httpx.Client(
            timeout=timeout_s,
            proxy=self.proxy or None,
            trust_env=not self.proxy,
            headers={"User-Agent": app_name},
        )

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("关闭 %s 客户端失败: %s", self.name, exc)

    # ------------------------------------------------------------------ #
    def map_language(self, code: str, is_source: bool = False) -> str:
        """内部语言代码 → 该家代码。映射表里没有就原样返回（多数家接受通用代码）。"""
        if not code or code == "auto":
            return "auto" if not is_source else "auto"
        return self.LANG_MAP.get(code, code)

    def _code_or_raise(self, code: str, is_source: bool) -> str:
        mapped = self.map_language(code, is_source)
        if mapped is None:
            raise UnsupportedLanguage(
                f"{self.name} 不支持{'源' if is_source else '目标'}语言 {code}"
            )
        return mapped

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        if self.qps_limit <= 0:
            return
        with self._lock:
            interval = 1.0 / self.qps_limit
            wait = interval - (time.time() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.time()

    def translate(self, request: TranslateRequest) -> TranslateResult:
        """逐条调用并合并。**不抛异常**，失败写进 failures。"""
        result = TranslateResult(provider=self.name, model=self.model_name())
        src = self._code_or_raise(request.source_language or "auto", True)
        dst = self._code_or_raise(request.target_language or "zh", False)
        if dst == "auto":
            dst = self.map_language(request.target_language or "zh", False)

        for seg in request.segments:
            if not seg.text:
                continue
            text = ""
            last_err = ""
            for attempt in range(self.retry_times + 1):
                try:
                    self._throttle()
                    text = self._translate_one(seg.text, src, dst)
                    if text:
                        break
                    last_err = "通道返回空译文"
                except UnsupportedLanguage:
                    raise
                except Exception as exc:  # noqa: BLE001 - 网络/配额问题都要重试
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt < self.retry_times:
                        # 指数退避 + 抖动，避免多实例同时重试打爆配额
                        time.sleep(min(8.0, 0.5 * (2 ** attempt)) * (0.5 + random.random()))
            if text:
                result.translations[seg.id] = text
                self.stats.characters += len(seg.text)
            else:
                result.failures[seg.id] = last_err or "翻译失败"
                log.warning("[%s] 字幕 %s 翻译失败: %s", self.name, seg.id, last_err)

        self.stats.requests += 1
        self.stats.segments += result.ok_count
        self.stats.failures += result.fail_count
        return result

    # ------------------------------------------------------------------ #
    @abstractmethod
    def _translate_one(self, text: str, src: str, dst: str) -> str:
        """翻译单条文本。失败应抛异常（基类负责重试与记录）。"""

    def model_name(self) -> str:
        return self.name

    @abstractmethod
    def ping(self) -> tuple[bool, str]:
        """连通性/凭据检查。"""

    # ------------------------------------------------------------------ #
    @staticmethod
    def random_salt(length: int = 16) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))
