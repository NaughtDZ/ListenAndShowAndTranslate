"""VAD 语音活动检测：把连续音频切成一句一句。

为什么必须有它：
- 分块引擎（SenseVoice 日/韩/粤、Whisper 小语种）**只能吃整段语音**，
  必须有东西告诉它"这一句从哪到哪"
- 静音段完全不送去识别，低端机 CPU 占用能降一个量级

用 sherpa-onnx 的 ``VoiceActivityDetector``（内部是 Silero VAD）。
延迟相关的三个参数与 ``app/config.py`` 的 LATENCY_PRESETS 对应。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from app.asr.locate import require, ModelNotFound
from app.utils.log import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
VAD_MODEL_ID = "silero-vad"


@dataclass
class SpeechSegment:
    """一段检测到的语音。"""

    samples: np.ndarray
    start_sample: int

    @property
    def duration_s(self) -> float:
        return self.samples.size / SAMPLE_RATE


class VadSegmenter:
    """包一层 VoiceActivityDetector，提供好用的"喂音频 → 取语音段"接口。"""

    def __init__(
        self,
        min_silence_ms: int = 350,
        min_speech_ms: int = 250,
        max_speech_ms: int = 8000,
        threshold: float = 0.5,
        num_threads: int = 1,
        models_dir: Path | None = None,
        buffer_seconds: float = 60.0,
    ) -> None:
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.threshold = threshold
        self.num_threads = num_threads
        self.models_dir = models_dir
        self.buffer_seconds = buffer_seconds
        self._vad: sherpa_onnx.VoiceActivityDetector | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        paths = require(VAD_MODEL_ID, self.models_dir, roles=("model",))
        model_path = paths["model"]

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model_path)
        cfg.silero_vad.threshold = float(self.threshold)
        cfg.silero_vad.min_silence_duration = max(0.05, self.min_silence_ms / 1000.0)
        cfg.silero_vad.min_speech_duration = max(0.05, self.min_speech_ms / 1000.0)
        cfg.silero_vad.max_speech_duration = max(1.0, self.max_speech_ms / 1000.0)
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = self.num_threads

        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, self.buffer_seconds)
        log.info(
            "VAD 就绪（静音 %.0fms / 最短语音 %.0fms / 最长 %.0fs）",
            self.min_silence_ms, self.min_speech_ms, self.max_speech_ms / 1000,
        )

    # ------------------------------------------------------------------ #
    def accept(self, samples: np.ndarray) -> None:
        """喂入 16k 单声道 float32 音频。"""
        if self._vad is None:
            return
        # pybind11 签名是 Sequence[float]；实测 numpy 一维数组也能被接受，
        # 但为了跨版本稳妥，这里显式转成 list（分块很小，开销可忽略）
        self._vad.accept_waveform(samples.tolist() if isinstance(samples, np.ndarray) else samples)

    def pop_ready(self) -> list[SpeechSegment]:
        """取出当前所有已完成的语音段。"""
        out: list[SpeechSegment] = []
        if self._vad is None:
            return out
        while not self._vad.empty():
            seg = self._vad.front
            try:
                raw = getattr(seg, "samples", None)
                start = int(getattr(seg, "start", 0))
                if raw is None:
                    self._vad.pop()
                    continue
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
                if arr.size:
                    out.append(SpeechSegment(samples=arr, start_sample=start))
            finally:
                self._vad.pop()
        return out

    def flush(self) -> list[SpeechSegment]:
        """音频结束：把尾巴上的语音也定稿。"""
        if self._vad is None:
            return []
        self._vad.flush()
        return self.pop_ready()

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset()

    @property
    def ready(self) -> bool:
        return self._vad is not None
