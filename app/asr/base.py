"""语音识别引擎抽象层。

设计要点（多语言需求带来的一条硬约束）：
**接口必须同时容纳"流式"与"分块"两种引擎。**

- 流式（中/英 的 zipformer）：边听边出中间结果，能立刻显示没定稿的字幕
- 分块（日/韩/粤 的 SenseVoice，小语种的 Whisper）：必须等一整句说完才出结果

如果只按流式设计接口，分块引擎就只能假装流式（憋着不出），
上层的延迟统计与 UI 状态都会失真。所以事件里明确区分 partial / final，
并提供 ``is_streaming`` 让上层知道"这个引擎在句子说完之前不会有任何输出"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class ASREventType(str, Enum):
    PARTIAL = "partial"
    """未定稿的中间结果，可能被后续 partial 覆盖。"""

    FINAL = "final"
    """已定稿，可以送去翻译。"""

    ERROR = "error"


@dataclass
class ASREvent:
    """识别结果事件。"""

    type: ASREventType
    text: str
    segment_id: int
    language: str = ""
    is_final: bool = False

    # 延迟统计（毫秒）：从"这段音频结束"到"出结果"
    latency_ms: float = 0.0
    audio_start_s: float = 0.0
    audio_end_s: float = 0.0

    model_id: str = ""
    error: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.audio_end_s - self.audio_start_s)


@dataclass
class EngineStats:
    """引擎运行统计，供基准与 UI 显示。"""

    segments: int = 0
    partials: int = 0
    audio_seconds: float = 0.0
    process_seconds: float = 0.0
    last_latency_ms: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """实时率：处理耗时 / 音频时长。< 1 表示比实时快。"""
        if self.audio_seconds <= 0:
            return 0.0
        return self.process_seconds / self.audio_seconds

    def add_latency(self, ms: float) -> None:
        self.last_latency_ms = ms
        self.latencies_ms.append(ms)

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
        return s[idx]


@runtime_checkable
class ASREngine(Protocol):
    """所有识别引擎必须实现的接口。"""

    name: str
    model_id: str
    language: str
    is_streaming: bool
    stats: EngineStats

    def start(self) -> None:
        """加载模型、准备推理。可能耗时数百毫秒。"""

    def feed(self, samples: np.ndarray) -> list[ASREvent]:
        """喂入 16 kHz 单声道 float32 音频，返回本次产生的事件（可能为空）。"""

    def finish(self) -> list[ASREvent]:
        """音频结束，冲出残留结果。"""

    def reset(self) -> None:
        """清空状态，准备下一段音频（不清模型）。"""

    def close(self) -> None:
        """释放资源。"""


class BaseEngine:
    """引擎基类：统一统计与 segment_id 管理，子类只实现真正的识别。"""

    name = "base"
    model_id = ""
    language = "auto"
    is_streaming = False

    def __init__(self, language: str = "auto") -> None:
        self.language = language
        self.stats = EngineStats()
        self._segment_id = 0
        self._audio_pos = 0.0  # 已喂入的音频总秒数
        self._segment_start = 0.0
        self._t0 = 0.0

    # ---- 子类需要实现 ---- #
    def _load(self) -> None:
        raise NotImplementedError

    def _feed(self, samples: np.ndarray) -> list[ASREvent]:
        raise NotImplementedError

    def _flush(self) -> list[ASREvent]:
        return []

    def _reset_state(self) -> None:
        pass

    def _close(self) -> None:
        pass

    # ---- 通用流程 ---- #
    def start(self) -> None:
        self._load()
        self._t0 = time.monotonic()

    def feed(self, samples: np.ndarray) -> list[ASREvent]:
        if samples.size == 0:
            return []
        t0 = time.monotonic()
        events = self._feed(samples)
        self.stats.process_seconds += time.monotonic() - t0
        self.stats.audio_seconds += samples.size / 16000.0
        self._audio_pos += samples.size / 16000.0
        for ev in events:
            ev.model_id = self.model_id
            if not ev.language:
                ev.language = self.language
            if ev.type is ASREventType.PARTIAL:
                self.stats.partials += 1
            elif ev.type is ASREventType.FINAL:
                self.stats.segments += 1
        return events

    def finish(self) -> list[ASREvent]:
        events = self._flush()
        for ev in events:
            ev.model_id = self.model_id
            if not ev.language:
                ev.language = self.language
            if ev.type is ASREventType.FINAL:
                self.stats.segments += 1
        return events

    def reset(self) -> None:
        self._reset_state()
        self._segment_start = self._audio_pos

    def close(self) -> None:
        self._close()

    # ---- 辅助 ---- #
    def next_segment_id(self) -> int:
        self._segment_id += 1
        return self._segment_id

    def mark_segment_start(self) -> None:
        self._segment_start = self._audio_pos

    def final_event(self, text: str, latency_ms: float = 0.0) -> ASREvent:
        ev = ASREvent(
            type=ASREventType.FINAL,
            text=text,
            segment_id=self.next_segment_id(),
            is_final=True,
            latency_ms=latency_ms,
            audio_start_s=self._segment_start,
            audio_end_s=self._audio_pos,
        )
        self.stats.add_latency(latency_ms)
        self.mark_segment_start()
        return ev

    def partial_event(self, text: str) -> ASREvent:
        return ASREvent(
            type=ASREventType.PARTIAL,
            text=text,
            segment_id=self._segment_id + 1,
            audio_start_s=self._segment_start,
            audio_end_s=self._audio_pos,
        )

    def error_event(self, message: str) -> ASREvent:
        return ASREvent(
            type=ASREventType.ERROR,
            text="",
            segment_id=self._segment_id,
            error=message,
        )
