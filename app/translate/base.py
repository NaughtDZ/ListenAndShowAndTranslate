"""翻译接口的数据模型。

设计围绕"字幕"这个场景，而不是通用文本翻译：

- **一条字幕一个 segment**，带 id，因为译文可能乱序回来（并发/批量），必须能归位
- **批量翻译**是一等公民：实测把 5 条合成一次请求能省 71% token（见 docs/P4-翻译实测.md），
  所以请求里装的是 **list**，不是单条
- **上下文**用 ``context``（前文原文+译文对）表达，而不是把整篇文章塞进去
- **术语表**是强制映射，与"提示词里提一句"不同，它可被程序校验
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Segment:
    """待翻译的一条字幕。"""

    id: int
    text: str
    language: str = ""

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()


@dataclass
class TranslateRequest:
    """一次翻译请求（可含多条字幕）。"""

    segments: list[Segment]
    source_language: str = "auto"
    target_language: str = "zh"
    context: list[tuple[str, str]] = field(default_factory=list)
    """前文 (原文, 译文) 对，按时间顺序。用于保持人称/称谓/语气一致。"""

    glossary: dict[str, str] = field(default_factory=dict)
    """强制术语映射：原文词 → 指定译法。小说的人名/功法/地名靠它。"""

    template: str = "subtitle_direct"
    custom_prompt: str = ""
    """非空时覆盖模板的 system 部分。"""

    @property
    def is_batch(self) -> bool:
        return len(self.segments) > 1


@dataclass
class TranslateResult:
    """翻译结果。**永远逐条返回**，哪怕底层是批量调的。"""

    translations: dict[int, str] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)

    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    from_cache: int = 0
    """命中缓存的条数。"""

    retries: int = 0
    note: str = ""

    def text_for(self, segment_id: int) -> str:
        return self.translations.get(segment_id, "")

    @property
    def ok_count(self) -> int:
        return len(self.translations)

    @property
    def fail_count(self) -> int:
        return len(self.failures)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: "TranslateResult") -> "TranslateResult":
        """合并另一次请求的结果（批量拆成多次时用）。"""
        self.translations.update(other.translations)
        self.failures.update(other.failures)
        self.latency_ms += other.latency_ms
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.from_cache += other.from_cache
        self.retries += other.retries
        if other.note:
            self.note = (self.note + "；" + other.note) if self.note else other.note
        return self


@dataclass
class TranslatorStats:
    """通道的运行统计，供 UI 与成本护栏用。"""

    requests: int = 0
    segments: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    characters: int = 0
    """累计源文字符数——传统 API 按字符计费，用它做成本统计。"""

    def add(self, result: TranslateResult, source_chars: int = 0) -> None:
        self.requests += 1
        self.segments += result.ok_count
        self.failures += result.fail_count
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.characters += source_chars


@runtime_checkable
class Translator(Protocol):
    """所有翻译通道实现的接口。"""

    name: str
    supports_batch: bool
    supports_streaming: bool

    def translate(self, request: TranslateRequest) -> TranslateResult:
        """翻译一批字幕。实现方**必须**保证不会抛异常打断字幕流水线，
        出错时把失败原因写进 ``TranslateResult.failures``。"""

    def close(self) -> None:
        """释放资源。"""


class TranslateError(RuntimeError):
    """通道级错误（如鉴权失败）。单条失败不该用它，应写进 failures。"""
