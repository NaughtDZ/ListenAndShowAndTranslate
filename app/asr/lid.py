"""语种识别（LID）：判断当前在说哪种语言。

用 sherpa-onnx 的 ``SpokenLanguageIdentification``（底层是多语言 Whisper 编码器，
模型 103 MB）。只有用户把 ``asr.language`` 设为 ``auto`` 时才需要它。

⚠️ 实测边界：
- LID 是对**一段音频**做判断，不是逐帧。太短的音频（< 1 秒）判断不稳。
- 它只返回语言字符串，**不返回置信度**，所以"要不要切换语言"由调用方
  用"连续 N 次一致才切"的稳定器来兜（见 :class:`LanguageStabilizer`）。
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np
import sherpa_onnx

from app.asr.locate import require
from app.utils.log import get_logger

log = get_logger(__name__)

LID_MODEL_ID = "whisper-tiny-lid"
SAMPLE_RATE = 16000

# LID 训练用的语言标签 → 我们内部的语种代码
_LABEL_MAP = {
    "chinese": "zh", "mandarin": "zh", "zh": "zh",
    "english": "en", "en": "en",
    "japanese": "ja", "japanese (ja)": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "cantonese": "yue", "yue": "yue",
}


def normalize_language(label: str) -> str:
    """把 LID 返回值统一成内部语种代码。"""
    key = (label or "").strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    # 形如 "Language: zh" 或 "<|zh|>"
    cleaned = key.replace("<|", "").replace("|>", "").split(":")[-1].strip()
    return _LABEL_MAP.get(cleaned, cleaned or "auto")


class LanguageIdentifier:
    """包一层 SpokenLanguageIdentification。"""

    def __init__(
        self,
        model_id: str = LID_MODEL_ID,
        num_threads: int = 1,
        provider: str = "cpu",
        models_dir: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.num_threads = num_threads
        self.provider = provider
        self.models_dir = models_dir
        self._lid: sherpa_onnx.SpokenLanguageIdentification | None = None

    def load(self) -> None:
        paths = require(self.model_id, self.models_dir, roles=("encoder", "decoder"))
        cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(
            whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
                encoder=str(paths["encoder"]),
                decoder=str(paths["decoder"]),
            ),
            num_threads=self.num_threads,
            provider=self.provider,
        )
        self._lid = sherpa_onnx.SpokenLanguageIdentification(cfg)
        log.info("语种识别模型就绪（%s）", self.model_id)

    def identify(self, samples: np.ndarray) -> str:
        """识别一段音频的语种；失败时返回 ``auto``（不抛异常）。"""
        if self._lid is None:
            return "auto"
        try:
            stream = self._lid.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
            return normalize_language(self._lid.compute(stream))
        except Exception as exc:  # noqa: BLE001 - 语种识别失败不该中断流程
            log.warning("语种识别失败: %s", exc)
            return "auto"

    def close(self) -> None:
        self._lid = None


class LanguageStabilizer:
    """防抖：连续 N 次判断一致才真正切换语言。

    没有它，句与句之间 LID 偶尔报错就会让引擎来回重建，字幕会抖。
    """

    def __init__(self, window: int = 3, min_agree: int = 2) -> None:
        self.window = max(1, window)
        self.min_agree = max(1, min_agree)
        self._history: deque[str] = deque(maxlen=self.window)
        self._current = ""

    @property
    def current(self) -> str:
        return self._current

    def push(self, language: str, allowed: set[str] | None = None) -> str:
        """喂入一次判断结果，返回**稳定后**应该用的语言。"""
        lang = (language or "").strip()
        if not lang or lang == "auto":
            return self._current
        if allowed and lang not in allowed:
            # LID 报了一个我们根本没装模型的语言 → 忽略，等下次
            return self._current

        self._history.append(lang)
        counter = Counter(self._history)
        top, count = counter.most_common(1)[0]
        if count >= self.min_agree:
            self._current = top
        return self._current

    def force(self, language: str) -> None:
        self._current = language
        self._history.clear()

    def reset(self) -> None:
        self._history.clear()
        self._current = ""
