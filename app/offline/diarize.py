"""离线说话人分离（diarization）：判断"这几句是不是同一个人说的"。

## 它做什么、不做什么

- **做**：把音频切成"同一个人连续说话"的区间，给每个区间一个编号 S0 / S1 / S2。
- **不做**：不知道谁是"哥哥"、谁是"老师"。编号是**任意**的（同一段音频跑两次，
  谁排 S0 可能不同），要变成角色名得另做语义推断。

## 为什么不用 pyannote.audio

sherpa-onnx 1.13 已经把整套推理内置了（``OfflineSpeakerDiarization``），
只要两个 onnx 模型，**不需要 torch**——对一个纯 CPU 的桌面程序来说这是决定性的：

| 模型 | 体积 | 作用 |
|---|---|---|
| pyannote-segmentation-3-0 | 5.7 MB | 切出"有人在说话"的片段 |
| 3D-Speaker campplus | 27 MB | 每段算一个声纹向量 |

实测（本机 16.2s 双说话人拼接音频）：加载 0.32s，处理 0.29s，**RTF ≈ 0.018**——
一小时的片子约 65 秒，比 ASR 便宜得多。

## 编号稳定性

聚类是**全局**做的（整段音频一起算），所以同一个文件里的 S0 始终是同一个人。
但**跨文件不保证**：第 1 集的 S0 可能是第 2 集的 S1（聚类是各自独立跑的）。
所以按文件输出角色编号，不要跨文件对比。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.utils.log import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000

SEGMENTATION_DIR = "pyannote-segmentation-3-0"
SEGMENTATION_FILE = "model.onnx"
EMBEDDING_DIR = "campplus-sv"
EMBEDDING_FILE = "campplus.onnx"

DEFAULT_THRESHOLD = 0.5
"""聚类阈值：越大越"敢"把两段判成同一个人（人数变少），越小越容易分出新说话人。"""

MIN_DURATION_ON = 0.3
MIN_DURATION_OFF = 0.5


@dataclass
class SpeakerSpan:
    """一段"某个人在连续说话"的时间区间。"""

    start_s: float
    end_s: float
    speaker: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def find_models(models_dir: Path | None = None) -> tuple[Path, Path] | None:
    """定位两个模型；缺任何一个都返回 None（调用方据此跳过，不影响识别）。"""
    if models_dir is None:
        from app import paths

        models_dir = paths.MODELS_DIR
    models_dir = Path(models_dir)

    seg = models_dir / SEGMENTATION_DIR / SEGMENTATION_FILE
    emb = models_dir / EMBEDDING_DIR / EMBEDDING_FILE
    if seg.is_file() and emb.is_file():
        return seg, emb
    return None


def models_available(models_dir: Path | None = None) -> bool:
    return find_models(models_dir) is not None


class SpeakerDiarizer:
    """一个可复用的说话人分离器（模型只加载一次，连续处理多个文件）。"""

    def __init__(
        self,
        models_dir: Path | None = None,
        num_threads: int = 2,
        threshold: float = DEFAULT_THRESHOLD,
        num_clusters: int = -1,
    ) -> None:
        self.models_dir = models_dir
        self.num_threads = num_threads
        self.threshold = threshold
        self.num_clusters = num_clusters
        self._diarizer = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        import sherpa_onnx

        found = find_models(self.models_dir)
        if found is None:
            raise FileNotFoundError(
                f"说话人分离模型缺失：需要 data/models/{SEGMENTATION_DIR}/"
                f"{SEGMENTATION_FILE} 和 data/models/{EMBEDDING_DIR}/{EMBEDDING_FILE}"
            )
        seg_model, emb_model = found

        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model)
                ),
                num_threads=self.num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model), num_threads=self.num_threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=self.num_clusters, threshold=self.threshold
            ),
            min_duration_on=MIN_DURATION_ON,
            min_duration_off=MIN_DURATION_OFF,
        )
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    @property
    def ready(self) -> bool:
        return self._diarizer is not None

    # ------------------------------------------------------------------ #
    def run(self, audio: np.ndarray) -> list[SpeakerSpan]:
        """对整段 16k 单声道音频做分离。

        Args:
            audio: float32、一维、16kHz。

        Returns:
            按开始时间排序的区间列表；没人说话时返回空列表。
        """
        if self._diarizer is None:
            self.load()
        assert self._diarizer is not None

        samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []

        # 直接传 numpy：pybind11 的 Sequence[float] 接受 ndarray，
        # 不用 .tolist()——一小时的音频 tolist 要 ~2GB，直接传只要 230MB。
        result = self._diarizer.process(samples)
        spans = [
            SpeakerSpan(start_s=float(seg.start), end_s=float(seg.end), speaker=int(seg.speaker))
            for seg in result.sort_by_start_time()
        ]
        speakers = len({s.speaker for s in spans})
        log.info(
            "说话人分离：%d 段 / %d 人 / 音频 %.1fs",
            len(spans), speakers, samples.size / SAMPLE_RATE,
        )
        return spans

    def close(self) -> None:
        self._diarizer = None


# --------------------------------------------------------------------------- #
# 把说话人映射到字幕
# --------------------------------------------------------------------------- #
def assign_speakers(
    cues: list,
    spans: list[SpeakerSpan],
    min_overlap_ratio: float = 0.3,
) -> None:
    """给每条字幕标上说话人编号（就地修改 ``cue.speaker``）。

    规则：取**重叠时长最大**的那个区间。重叠不到该字幕时长的
    ``min_overlap_ratio`` 时标 -1（未知）——宁可不标，也不要标错人。

    为什么不用"字幕中心点落在哪个区间"：VAD 切出来的句子里可能有一半是
    前一个人说的（成交叠），按重叠时长算更稳。
    """
    if not spans:
        return

    for cue in cues:
        cue_len = max(1e-6, cue.end_s - cue.start_s)
        best_speaker = -1
        best_overlap = 0.0
        for span in spans:
            overlap = min(cue.end_s, span.end_s) - max(cue.start_s, span.start_s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = span.speaker
        cue.speaker = best_speaker if best_overlap / cue_len >= min_overlap_ratio else -1
