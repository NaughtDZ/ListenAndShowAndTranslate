"""离线识别：VAD 把音频切成句 → 每句整句识别 → 带**精确起止时间**的字幕。

⚠️ 这里刻意**不用** ``ASREvent`` 的时间戳（``audio_start_s`` / ``audio_end_s``）：
那两个字段填的是"上一次出结果的位置 → 本次喂完的位置"，**中间夹着静音**，
用来写字幕会让每一句都提前冒出来。因此直接读 ``SpeechSegment.start_sample``
和段长——那才是真实语音的起止。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import sherpa_onnx

from app.asr.postprocess import clean_text
from app.asr.sherpa_offline import build_offline_recognizer
from app.asr.vad import SAMPLE_RATE, VadSegmenter
from app.offline.cue import Cue, merge_fragments
from app.utils.log import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[float, float], None]
"""``(processed_seconds, total_seconds)``；total 为 0 表示未知时长。"""

LogCb = Callable[[str], None]

VAD_BLOCK_SECONDS = 0.1
"""喂给 VAD 的块大小——**这个值不能随便放大**。

实测（``ja_03.wav``，silero，同一段音频只改块大小）：

===========  ==========================================
块大小        结果
===========  ==========================================
100 ms       ✅ 正确切成 2 段（0.07~2.19 / 2.89~10.78）
200 ms       段边界开始漂移（每段多算 0.1s）
250 ms       漂移更明显
**320 ms**   ❌ 两段被并成一段（断句失效）
400~1000 ms  ❌ 并成一段
10 s         ❌ 只吐出末尾 1.7s，前面的全丢
===========  ==========================================

原因是 sherpa-onnx 的 ``accept_waveform`` 对"一次传入多长"敏感；
实时管线天然是 10~20ms 的回环块，所以从没踩到这个坑。
离线解码是按 10 秒一块出的，**必须再切小才能喂 VAD**。
"""


def _iter_slices(chunk: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, chunk.size, size):
        piece = chunk[start : start + size]
        if piece.size:
            yield piece


@dataclass
class TranscribeOptions:
    """离线识别参数。

    断句静音比实时的默认值大一些（句子更完整、翻译上下文更好），
    但 ⚠️ ``max_segment_ms`` **不能因为"离线可以等"就调很大**：

    VAD 正是靠它给"一直不停地说"强制断句。若它比最长连续语音还长，
    VAD 会认为"这一句始终没说完"，整段音频都挂在那儿不产出，
    直到 ``flush()`` 才吐出一小截（实测 11.4s 的连续日语音频、上限设 15s，
    最终只拿到末尾 1.7s、前面的全丢）。所以默认仍取 8s，与实时一致。
    """

    model_id: str = "sensevoice-int8"
    language: str = "auto"
    vad_model: str = "silero-vad"
    hop_ms: int = 1000

    min_silence_ms: int = 800
    """断句静音：静多久算一句话说完。

    **离线默认 800ms，比实时的 350ms 大**——离线不用跟手，而"句子被切碎"
    的代价比"字幕长一点"大得多（切碎会让翻译丢掉主语/谓语）。

    实测 ``real_ja_2.wav``（连续朗读）：

    =========  ======  ==================================================
    500ms      3 段    「日本語をちゃんと聞き取れ」「ているかどうかを」被切开
    800ms      3 段    靠碎片合并拼回来了
    **1200ms**  2 段    「…聞き取れているかどうかを図ります。」完整一句
    =========  ======  ==================================================

    但调太大也有风险：两人快速对话（停顿 0.8~1.2s）会被并成一句。
    建议 800 起步，分句还是碎就往上调，字幕长得离谱就往下调。
    """

    min_speech_ms: int = 250
    max_segment_ms: int = 8000
    speech_pad_ms: int = 120
    num_threads: int = 2
    provider: str = "cpu"
    use_itn: bool = True

    merge_fragments: bool = True
    """是否把"被 VAD 切出去的尾巴几个字"并回前一句。

    实测 ``real_ja_2.wav``：VAD 在「日本語をちゃんと聞き取れ」和「ているかどうか」
    之间切了一刀，后一段只有 1.19 秒——用户看到的就是"一句话最后几个字被单独
    打了个时间点"。合并后原文恢复完整，翻译也能拿到完整谓语。

    规则是"短（≤1.2s）+ 紧邻（≤0.6s）"，纯声学事实，不依赖语言。
    细节见 :func:`app.offline.cue.merge_fragments`。
    """

    diarize: bool = False
    """是否做**说话人分离**（区分角色）。

    ⚠️ 需要两个模型（pyannote-segmentation-3-0 约 5.7MB + campplus-sv 约 27MB），
    缺模型时自动跳过、不影响识别。

    成本：整段音频要多跑一遍 diarization（实测 RTF ≈ 0.018，一小时约 65 秒），
    并且**要把整段 PCM 留在内存里**（按 int16 存，一小时约 115MB）。

    产出的是 `S0 / S1 / S2` 这样的**匿名编号**（聚类结果，同一文件内稳定，
    跨文件不保证），不是角色名。细节见 :mod:`app.offline.diarize`。
    """


class Transcriber:
    """一个可复用的离线识别器（模型只加载一次，连续处理多个文件）。"""

    def __init__(
        self,
        options: TranscribeOptions | None = None,
        models_dir: Path | None = None,
        on_log: LogCb | None = None,
    ) -> None:
        self.options = options or TranscribeOptions()
        self.models_dir = models_dir
        self.on_log = on_log
        self._recognizer: sherpa_onnx.OfflineRecognizer | None = None
        self._vad: VadSegmenter | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        o = self.options
        self._recognizer = build_offline_recognizer(
            o.model_id,
            language=o.language,
            num_threads=o.num_threads,
            provider=o.provider,
            models_dir=self.models_dir,
            use_itn=o.use_itn,
        )
        self._vad = VadSegmenter(
            min_silence_ms=o.min_silence_ms,
            min_speech_ms=o.min_speech_ms,
            max_speech_ms=o.max_segment_ms,
            speech_pad_ms=o.speech_pad_ms,
            hop_ms=o.hop_ms,
            vad_model=o.vad_model,
            num_threads=1,
            provider=o.provider,
            models_dir=self.models_dir,
        )
        self._vad.load()
        self._log(f"识别器就绪：{o.model_id} / 语言 {o.language} / VAD {o.vad_model}")

    @property
    def ready(self) -> bool:
        return self._recognizer is not None and self._vad is not None

    # ------------------------------------------------------------------ #
    def run(
        self,
        chunks: Iterable[np.ndarray],
        total_seconds: float = 0.0,
        on_progress: ProgressCb | None = None,
        on_cue: Callable[[Cue], None] | None = None,
    ) -> list[Cue]:
        """把音频块流跑完，返回按时间排序的字幕。"""
        if not self.ready:
            self.load()
        assert self._vad is not None

        self._vad.reset()
        cues: list[Cue] = []
        processed = 0.0
        block = max(160, int(SAMPLE_RATE * VAD_BLOCK_SECONDS))

        # 做说话人分离时要把整段音频留住（diarization 是全局聚类的，不能流式）。
        # 按 int16 存：一小时约 115MB，float32 要 230MB。
        pcm_parts: list[np.ndarray] | None = [] if self.options.diarize else None

        for chunk in chunks:
            processed += chunk.size / SAMPLE_RATE
            if pcm_parts is not None:
                pcm_parts.append(
                    (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                )
            # ⚠️ 必须切小片：大块会让 silero 的断句直接失效（见 VAD_BLOCK_SECONDS 的实测表）
            for piece in _iter_slices(chunk, block):
                self._vad.accept(piece)
            for cue in self._take(on_cue):
                cues.append(cue)
            if on_progress:
                on_progress(processed, total_seconds)

        for cue in self._take(on_cue, flush=True):
            cues.append(cue)

        cues.sort(key=lambda c: c.start_s)
        if self.options.merge_fragments:
            before = len(cues)
            cues = merge_fragments(cues)
            if len(cues) < before:
                self._log(f"碎片合并：{before} → {len(cues)} 条（尾巴被切走的已并回）")
        if pcm_parts:
            self._assign_speakers(cues, pcm_parts)

        # 诊断：检出的语音占总时长的比例。
        # 轻声/耳语/ASMR 最常见的问题**不是识别错，而是压根没送进识别器**——
        # 默认 silero VAD 按正常音量训练，低电平人声会被判成静音。
        # 有个具体数字，用户就不用猜"是我耳朵灵还是模型瞎"。
        speech_s = sum(c.end_s - c.start_s for c in cues)
        ratio = (speech_s / processed) if processed > 0 else 0.0
        self._log(
            f"识别完成：{len(cues)} 句 / 音频 {processed:.1f}s"
            f"（检出语音 {speech_s:.0f}s，占 {ratio:.0%}）"
        )
        if processed >= 60 and ratio < 0.15:
            self._log(
                "  ⚠️ 检出的语音占比很低。如果素材是轻声/耳语/ASMR，"
                "请把「VAD 模型」换成 Whisper-VAD ASMR——"
                "默认 Silero 按正常音量训练，会把低电平人声判成静音，"
                "那些话根本没送进识别器（换 ASR 模型没用）。"
            )
        return cues

    # ------------------------------------------------------------------ #
    def _assign_speakers(self, cues: list[Cue], pcm_parts: list[np.ndarray]) -> None:
        """跑说话人分离并给每条字幕标角色。

        **任何失败都只是"没有角色"**，绝不影响已经识别出来的文本。
        """
        from app.offline.diarize import SpeakerDiarizer, assign_speakers, find_models

        if not cues:
            return
        if find_models(self.models_dir) is None:
            self._log(
                "说话人分离：模型缺失，已跳过"
                "（需要 data/models/pyannote-segmentation-3-0 + campplus-sv）"
            )
            return
        try:
            audio = np.concatenate(pcm_parts).astype(np.float32) / 32767.0
            diarizer = SpeakerDiarizer(
                models_dir=self.models_dir,
                num_threads=max(1, self.options.num_threads),
            )
            spans = diarizer.run(audio)
            diarizer.close()
            assign_speakers(cues, spans)
            n = len({c.speaker for c in cues if c.speaker >= 0})
            self._log(f"说话人分离：{len(spans)} 段 / 识别出 {n} 个角色")
        except Exception as exc:  # noqa: BLE001 - 分离失败不能影响字幕
            self._log(f"说话人分离失败（已跳过）：{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    def _take(
        self,
        on_cue: Callable[[Cue], None] | None,
        flush: bool = False,
    ) -> list[Cue]:
        assert self._vad is not None
        segments = self._vad.flush() if flush else self._vad.pop_ready()
        out: list[Cue] = []
        for seg in segments:
            text = self._recognize(seg.samples)
            if not text:
                continue
            start = seg.start_sample / SAMPLE_RATE
            end = start + seg.samples.size / SAMPLE_RATE
            cue = Cue(start_s=start, end_s=end, text=text)
            out.append(cue)
            if on_cue:
                on_cue(cue)
        return out

    def _recognize(self, samples: np.ndarray) -> str:
        if self._recognizer is None or samples.size == 0:
            return ""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        raw = (stream.result.text or "").strip()
        if not raw:
            return ""
        # 与实时管线用同一套后处理：日文词间多余空格、英文大小写、标点规整
        return clean_text(raw, self.options.language or "auto")

    def close(self) -> None:
        self._recognizer = None
        self._vad = None

    # ------------------------------------------------------------------ #
    def _log(self, message: str) -> None:
        log.info("%s", message)
        if self.on_log:
            try:
                self.on_log(message)
            except Exception:  # noqa: BLE001 - 日志回调不能影响主流程
                pass
