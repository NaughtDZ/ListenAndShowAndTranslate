"""分块引擎：sherpa-onnx 离线识别（SenseVoice / Whisper / Paraformer）。

用途：**日语、韩语、粤语、小语种**——这些语言要么没有流式模型（日语），
要么只有 Whisper 可用。用 VAD 把音频切成句子，再整句送识别。

⚠️ 关于延迟（必须让用户知道）：
分块引擎**在整句说完之前不会有任何输出**，所以：
- 它没有"中间结果"（partial），字幕是整句一次性出现的
- 延迟下限 = VAD 断句静音时长 + 整句推理耗时
- 因此 ``partial_interval_ms`` 对它无效（UI 上应置灰）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from app.asr.base import ASREvent, BaseEngine
from app.asr.locate import require
from app.asr.vad import SAMPLE_RATE, VadSegmenter
from app.utils.log import get_logger

log = get_logger(__name__)

FEATURE_DIM = 80


class SherpaOfflineEngine(BaseEngine):
    """VAD 分句 + 离线识别器（SenseVoice / Whisper）。"""

    name = "sherpa_offline"
    is_streaming = False

    def __init__(
        self,
        model_id: str = "sensevoice-int8",
        language: str = "ja",
        num_threads: int = 2,
        provider: str = "cpu",
        models_dir: Path | None = None,
        use_itn: bool = True,
        min_silence_ms: int = 350,
        min_speech_ms: int = 250,
        max_segment_ms: int = 8000,
    ) -> None:
        super().__init__(language=language)
        self.model_id = model_id
        self.num_threads = num_threads
        self.provider = provider
        self.models_dir = models_dir
        self.use_itn = use_itn
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_segment_ms = max_segment_ms

        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._vad: VadSegmenter | None = None

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        self._vad = VadSegmenter(
            min_silence_ms=self.min_silence_ms,
            min_speech_ms=self.min_speech_ms,
            max_speech_ms=self.max_segment_ms,
            num_threads=1,
            models_dir=self.models_dir,
        )
        self._vad.load()

        is_sense = "sensevoice" in self.model_id or "sense-voice" in self.model_id
        is_whisper = "whisper" in self.model_id
        # 不同模型族的文件结构不同：SenseVoice 单模型，Whisper 是 encoder+decoder
        roles = ("model", "tokens") if is_sense else ("encoder", "decoder", "tokens")
        paths = require(self.model_id, self.models_dir, roles=roles)

        # SenseVoice：language 支持 auto/zh/en/ja/ko/yue，且自带语种识别
        if is_sense:
            lang = self.language if self.language != "zh-en" else "auto"
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(paths["model"]),
                tokens=str(paths["tokens"]),
                num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=FEATURE_DIM,
                decoding_method="greedy_search",
                language=lang,
                use_itn=self.use_itn,
                provider=self.provider,
            )
            log.info("加载 SenseVoice（model=%s language=%s）", self.model_id, lang)
        elif is_whisper:
            # 注意：from_whisper **不接受** sample_rate / feature_dim（与 SenseVoice 不同）
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(paths["encoder"]),
                decoder=str(paths["decoder"]),
                tokens=str(paths["tokens"]),
                num_threads=self.num_threads,
                language=self.language if self.language not in ("zh-en", "") else "auto",
                task="transcribe",
                decoding_method="greedy_search",
                provider=self.provider,
            )
            log.info("加载 Whisper（model=%s language=%s）", self.model_id, self.language)
        else:
            raise ValueError(f"分块引擎暂不支持该模型: {self.model_id}")

    # ------------------------------------------------------------------ #
    def _feed(self, samples: np.ndarray) -> list[ASREvent]:
        if self._vad is None or self._recognizer is None:
            return [self.error_event("引擎未启动，请先调用 start()")]

        self._vad.accept(samples)
        events: list[ASREvent] = []
        for seg in self._vad.pop_ready():
            text = self._recognize(seg.samples)
            if text:
                events.append(self.final_event(text))
        return events

    def _flush(self) -> list[ASREvent]:
        if self._vad is None or self._recognizer is None:
            return []
        events: list[ASREvent] = []
        for seg in self._vad.flush():
            text = self._recognize(seg.samples)
            if text:
                events.append(self.final_event(text))
        return events

    def _reset_state(self) -> None:
        if self._vad is not None:
            self._vad.reset()

    def _close(self) -> None:
        self._recognizer = None
        self._vad = None

    # ------------------------------------------------------------------ #
    def _recognize(self, samples: np.ndarray) -> str:
        assert self._recognizer is not None
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()
