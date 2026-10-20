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
from app.models.registry import MODELS
from app.utils.log import get_logger

log = get_logger(__name__)

FEATURE_DIM = 80

_CTC_FACTORIES: tuple[str, ...] = (
    "nemo_ctc", "dolphin_ctc", "omnilingual_asr_ctc", "fire_red_asr_ctc",
    "paraformer", "zipformer_ctc", "telespeech_ctc", "wenet_ctc",
)
"""单文件（model.onnx + tokens.txt）+ ``from_<factory>`` 的离线识别器家族。"""


def guess_factory(model_id: str) -> str:
    """注册表没写 factory 时的兜底推断（兼容老配置/手写模型 id）。"""
    name = model_id.lower()
    if "whisper" in name:
        return "whisper"
    if "parakeet" in name:
        return "nemo_ctc"
    if "dolphin" in name:
        return "dolphin_ctc"
    if "omnilingual" in name:
        return "omnilingual_asr_ctc"
    if "fire" in name:
        return "fire_red_asr_ctc"
    return ""


def build_offline_recognizer(
    model_id: str,
    language: str = "auto",
    num_threads: int = 2,
    provider: str = "cpu",
    models_dir: Path | None = None,
    use_itn: bool = True,
) -> sherpa_onnx.OfflineRecognizer:
    """按注册表构造一个 sherpa-onnx 离线识别器。

    **实时引擎与离线字幕工具共用这一份**：模型族的文件结构差别很大
    （Whisper 是 encoder+decoder，CTC 家族是单个 model.onnx + tokens.txt），
    但"用哪个工厂"由注册表的 ``ModelSpec.factory`` 说了算。
    两边各写一套的话，加新模型时必漏一处。
    """
    spec = MODELS.get(model_id)
    factory = (spec.factory if spec else "") or guess_factory(model_id)

    if factory == "whisper":
        roles = ("encoder", "decoder", "tokens")
    elif factory in ("sense_voice", *_CTC_FACTORIES):
        roles = ("model", "tokens")
    else:
        raise ValueError(f"不认识的模型：{model_id}（factory={factory or '未声明'}）")

    paths = require(model_id, models_dir, roles=roles)

    if factory == "whisper":
        # 注意：from_whisper **不接受** sample_rate / feature_dim
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            tokens=str(paths["tokens"]),
            num_threads=num_threads,
            language=language if language not in ("zh-en", "") else "auto",
            task="transcribe",
            decoding_method="greedy_search",
            provider=provider,
        )
    elif factory == "sense_voice":
        # SenseVoice：language 支持 auto/zh/en/ja/ko/yue，自带语种识别与标点
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(paths["model"]),
            tokens=str(paths["tokens"]),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method="greedy_search",
            language=language if language != "zh-en" else "auto",
            use_itn=use_itn,
            provider=provider,
        )
    else:
        # NeMo CTC / Dolphin / Omnilingual / FireRedASR CTC 这一族：
        # 单个 model.onnx + tokens.txt，参数完全一致（语言信息在模型内部）。
        build = getattr(sherpa_onnx.OfflineRecognizer, f"from_{factory}", None)
        if build is None:
            raise ValueError(
                f"当前 sherpa-onnx 不支持 factory={factory}（模型 {model_id}）"
            )
        recognizer = build(
            model=str(paths["model"]),
            tokens=str(paths["tokens"]),
            num_threads=num_threads,
            provider=provider,
        )

    log.info(
        "加载分块识别器：model=%s factory=%s language=%s provider=%s",
        model_id, factory, language, provider,
    )
    return recognizer


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
        speech_pad_ms: int = 120,
        vad_model: str = "silero-vad",
        vad_hop_ms: int = 1000,
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
        self.speech_pad_ms = speech_pad_ms
        self.vad_model = vad_model
        self.vad_hop_ms = vad_hop_ms

        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._vad: VadSegmenter | None = None

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        self._vad = VadSegmenter(
            min_silence_ms=self.min_silence_ms,
            min_speech_ms=self.min_speech_ms,
            max_speech_ms=self.max_segment_ms,
            speech_pad_ms=self.speech_pad_ms,
            hop_ms=self.vad_hop_ms,
            vad_model=self.vad_model,
            num_threads=1,
            provider=self.provider,
            models_dir=self.models_dir,
        )
        self._vad.load()

        self._recognizer = build_offline_recognizer(
            model_id=self.model_id,
            language=self.language,
            num_threads=self.num_threads,
            provider=self.provider,
            models_dir=self.models_dir,
            use_itn=self.use_itn,
        )

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
