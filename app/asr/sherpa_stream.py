"""流式引擎：sherpa-onnx 在线 zipformer（中文 / 中英 / 英文）。

为什么用它：真流式，边听边出中间结果，延迟 0.3~0.7s，
是目前能拿到的最低延迟路线（官方流式模型覆盖 zh / zh-en / en / ko / fr）。

⚠️ 关于断句：流式引擎不用外挂 VAD，而是用 sherpa-onnx 的
**端点检测（endpoint detection）**，由三条规则触发：
  - rule1: 一直没解出任何内容时的尾部静音上限
  - rule2: 已经解出内容后，尾部静音超过它就算一句结束 ← **对应我们的"断句静音时长"**
  - rule3: 单句最长时长（对应"最长憋句时间"）
这正好和 ``app/config.py`` 的 LATENCY_PRESETS 一一对应。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from app.asr.base import ASREvent, ASREventType, BaseEngine
from app.asr.locate import require
from app.utils.log import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
FEATURE_DIM = 80


class SherpaStreamingEngine(BaseEngine):
    """sherpa-onnx OnlineRecognizer（zipformer transducer）。"""

    name = "sherpa_stream"
    is_streaming = True

    def __init__(
        self,
        model_id: str = "zipformer-zh-int8",
        language: str = "zh",
        num_threads: int = 2,
        provider: str = "cpu",
        models_dir: Path | None = None,
        # 端点检测（= 断句规则）
        min_silence_ms: int = 350,
        max_segment_ms: int = 8000,
        decoding_method: str = "greedy_search",
    ) -> None:
        super().__init__(language=language)
        self.model_id = model_id
        self.num_threads = num_threads
        self.provider = provider
        self.models_dir = models_dir
        self.min_silence_ms = min_silence_ms
        self.max_segment_ms = max_segment_ms
        self.decoding_method = decoding_method

        self._recognizer: sherpa_onnx.OnlineRecognizer | None = None
        self._stream: sherpa_onnx.OnlineStream | None = None
        self._last_text = ""
        self._pending_silence_s = 0.0

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        paths = require(
            self.model_id,
            self.models_dir,
            roles=("encoder", "decoder", "joiner", "tokens"),
        )
        log.info(
            "加载流式模型 %s（threads=%d provider=%s 断句=%dms）",
            self.model_id, self.num_threads, self.provider, self.min_silence_ms,
        )

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=self.decoding_method,
            provider=self.provider,
            enable_endpoint_detection=True,
            # rule1：还没解出任何内容时，静音超过它就断（给较宽松的值，避免噪声切碎）
            rule1_min_trailing_silence=2.4,
            # rule2：已解出内容后的尾部静音上限 ← 用户可调的那颗"断句静音时长"
            rule2_min_trailing_silence=max(0.1, self.min_silence_ms / 1000.0),
            # rule3：单句最长时长 ← 用户可调的"最长憋句时间"
            rule3_min_utterance_length=max(1.0, self.max_segment_ms / 1000.0),
        )
        self._stream = self._recognizer.create_stream()
        self._last_text = ""
        self._pending_silence_s = 0.0

    # ------------------------------------------------------------------ #
    def _feed(self, samples: np.ndarray) -> list[ASREvent]:
        if self._recognizer is None or self._stream is None:
            return [self.error_event("引擎未启动，请先调用 start()")]

        events: list[ASREvent] = []
        self._stream.accept_waveform(SAMPLE_RATE, samples)

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._recognizer.get_result(self._stream).strip()

        # 中间结果：只有变化时才发，避免刷屏
        if text and text != self._last_text and not self._recognizer.is_endpoint(self._stream):
            events.append(self.partial_event(text))
            self._last_text = text

        # 端点检测 → 定稿
        if self._recognizer.is_endpoint(self._stream):
            if text:
                events.append(self.final_event(text))
            self._recognizer.reset(self._stream)
            self._last_text = ""

        return events

    # ------------------------------------------------------------------ #
    def _flush(self) -> list[ASREvent]:
        if self._recognizer is None or self._stream is None:
            return []
        events: list[ASREvent] = []
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream).strip()
        if text:
            events.append(self.final_event(text))
        self._last_text = ""
        return events

    def _reset_state(self) -> None:
        if self._recognizer is not None and self._stream is not None:
            self._recognizer.reset(self._stream)
        self._last_text = ""

    def _close(self) -> None:
        self._stream = None
        self._recognizer = None


def available_streaming_models() -> list[str]:
    """当前已注册的流式模型 id。"""
    from app.models.registry import MODELS

    return [m.id for m in MODELS.values() if m.engine == "sherpa_stream"]
