"""音频处理管线：WASAPI 原始流 → ASR 可直接吃的 16k 单声道。

输入（proc-tap 实测输出，见 docs/P1-实测记录.md 第 3.4 节）：
    48 000 Hz / 2 声道 / float32 交错排列 / 每块 480 帧（10 ms）

输出：
    16 000 Hz / 单声道 / float32 —— 绝大多数中文 ASR 模型的原生输入格式

设计要点：
- 用 ``soxr.ResampleStream`` 做**流式**重采样，不是每块独立重采样
  （每块独立重采样会在块边界产生不连续与咔哒声，并且丢失滤波器状态）
- 同时统计 RMS / 峰值 / 静音时长 —— 实测证明"目标没播声时数据照样有"，
  所以判断"到底有没有声音"只能靠这些能量指标，不能靠"有没有数据"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import soxr

from app.utils.log import get_logger

log = get_logger(__name__)

DEFAULT_IN_RATE = 48000
DEFAULT_IN_CHANNELS = 2
DEFAULT_OUT_RATE = 16000

# RMS 低于此值视为静音（float32 满幅为 1.0）。
#
# 实测教训（docs/P1-实测记录.md）：原先取 0.001（-60 dBFS）**太大**——
# 用户把小说音量调小时，真实语音的 RMS 只有 0.0005 左右，
# 会被误判成"目标没在放音"，UI 就会错误地提示用户去排查。
# 进程回环在目标不渲染音频时给出的是**精确的 0**，
# 因此阈值可以取得很低而不怕把底噪当人声：1e-4（-80 dBFS）。
SILENCE_RMS_THRESHOLD = 1e-4


@dataclass
class PipelineStats:
    """管线统计。UI 用它显示"目标程序是否正在出声"。"""

    input_chunks: int = 0
    input_samples: int = 0
    output_samples: int = 0
    peak: float = 0.0
    """整段采集以来的峰值（高水位，不衰减）。"""

    last_rms: float = 0.0
    last_peak: float = 0.0
    """**最近一块**的 RMS / 峰值。

    与 ``peak``（高水位）区分开：设置窗里的实时电平表要画"当前"的峰值，
    拿高水位会把 PEAK 条永久顶在最右边。
    """

    silent_chunks: int = 0
    """连续静音块数（每次遇到有声块就清零）。"""

    silent_seconds_total: float = 0.0

    silence_threshold: float = SILENCE_RMS_THRESHOLD

    def reset_peak(self) -> None:
        self.peak = 0.0

    @property
    def is_silent(self) -> bool:
        return self.last_rms < self.silence_threshold


class RingBuffer:
    """定长环形缓冲，满了丢弃**最旧**的数据。

    字幕是"实时消费品"：宁可丢过期音频，也不能无限堆内存（计划书第 3.2 节背压原则）。
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples 必须为正数")
        self.capacity = int(capacity_samples)
        self._buf: deque[np.ndarray] = deque()
        self._size = 0
        self.dropped_samples = 0

    def push(self, chunk: np.ndarray) -> int:
        """写入一块数据，返回本次因为溢出而丢弃的样本数。"""
        if chunk.size == 0:
            return 0
        self._buf.append(chunk)
        self._size += chunk.size

        dropped = 0
        while self._size > self.capacity and self._buf:
            oldest = self._buf[0]
            overflow = self._size - self.capacity
            if oldest.size <= overflow:
                self._buf.popleft()
                self._size -= oldest.size
                dropped += oldest.size
                self.dropped_samples += oldest.size
            else:
                # 只丢弃最旧块的前一部分
                self._buf[0] = oldest[overflow:]
                self._size -= overflow
                dropped += overflow
                self.dropped_samples += overflow
        return dropped

    def pop_all(self) -> np.ndarray:
        """取出全部数据并清空。"""
        if not self._buf:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(list(self._buf))
        self._buf.clear()
        self._size = 0
        return out

    def __len__(self) -> int:
        return self._size


class AudioPipeline:
    """48k 立体声 → 16k 单声道，带统计。"""

    def __init__(
        self,
        in_rate: int = DEFAULT_IN_RATE,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_rate: int = DEFAULT_OUT_RATE,
        downmix: bool = True,
        quality: str = "HQ",
        input_dtype: str = "float32",
        silence_threshold: float = SILENCE_RMS_THRESHOLD,
        gain: float = 1.0,
    ) -> None:
        self.in_rate = in_rate
        self.in_channels = in_channels
        self.out_rate = out_rate
        self.downmix = downmix
        self.input_dtype = input_dtype
        self.silence_threshold = silence_threshold
        self.gain = float(gain)
        """数字增益。实测采集发生在音量合成器之后，用户把音量调小时信号也变小，
        可用会话音量的倒数做补偿（见 process_list.suggest_gain）。"""

        out_channels = 1 if downmix else in_channels
        self._resampler = soxr.ResampleStream(
            in_rate, out_rate, out_channels, dtype="float32", quality=quality
        )
        self.stats = PipelineStats(silence_threshold=silence_threshold)
        self._chunk_seconds = 0.0

    # ------------------------------------------------------------------ #
    def process(self, pcm: bytes | np.ndarray) -> np.ndarray:
        """处理一块原始 PCM，返回 16k 单声道 float32。"""
        samples = self._to_float32(pcm)
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        if self.gain != 1.0:
            samples = np.clip(samples * self.gain, -1.0, 1.0)

        if self.in_channels > 1:
            if samples.size % self.in_channels != 0:
                # 不完整帧：丢掉尾部，避免 reshape 报错
                usable = samples.size - (samples.size % self.in_channels)
                samples = samples[:usable]
            if self.downmix:
                samples = samples.reshape(-1, self.in_channels).mean(axis=1)
            else:
                samples = samples.reshape(-1, self.in_channels)
        # 单声道时保持一维

        self.stats.input_chunks += 1
        self.stats.input_samples += int(samples.shape[0])
        self._update_energy(samples)

        out = self._resampler.resample_chunk(samples, last=False)
        out = np.ascontiguousarray(out, dtype=np.float32).reshape(-1)
        self.stats.output_samples += out.size
        return out

    def flush(self) -> np.ndarray:
        """冲掉重采样器内部残留（停止采集时调用，避免丢掉尾巴）。"""
        try:
            tail = self._resampler.resample_chunk(
                np.zeros(0, dtype=np.float32), last=True
            )
        except Exception as exc:  # noqa: BLE001 - 冲尾不该影响主流程
            log.debug("flush 重采样器失败: %s", exc)
            return np.zeros(0, dtype=np.float32)
        tail = np.ascontiguousarray(tail, dtype=np.float32).reshape(-1)
        self.stats.output_samples += tail.size
        return tail

    def reset(self) -> None:
        self._resampler.clear()
        self.stats = PipelineStats(silence_threshold=self.silence_threshold)

    def set_gain(self, gain: float) -> None:
        """调整数字增益（用于按会话音量做补偿）。"""
        self.gain = max(0.0, float(gain))

    def set_silence_threshold(self, threshold: float) -> None:
        """用户可随时调整"什么算静音"。"""
        self.silence_threshold = float(threshold)
        self.stats.silence_threshold = float(threshold)

    # ------------------------------------------------------------------ #
    def _to_float32(self, pcm: bytes | np.ndarray) -> np.ndarray:
        if isinstance(pcm, np.ndarray):
            return pcm.astype(np.float32, copy=False)
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        if self.input_dtype == "float32":
            return np.frombuffer(pcm, dtype=np.float32)
        if self.input_dtype == "int16":
            raw = np.frombuffer(pcm, dtype=np.int16)
            return raw.astype(np.float32) / 32768.0
        raise ValueError(f"不支持的 input_dtype: {self.input_dtype}")

    def _update_energy(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        peak = float(np.max(np.abs(samples)))
        self.stats.last_peak = peak
        if peak > self.stats.peak:
            self.stats.peak = peak

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        self.stats.last_rms = rms

        if rms < self.silence_threshold:
            self.stats.silent_chunks += 1
            self.stats.silent_seconds_total += self._block_seconds(samples)
        else:
            self.stats.silent_chunks = 0

    def _block_seconds(self, samples: np.ndarray) -> float:
        frames = samples.shape[0] if samples.ndim > 0 else 0
        if self.in_channels > 1 and not self.downmix:
            frames = frames  # 已是帧数
        return frames / self.in_rate if self.in_rate else 0.0


def pcm_duration_seconds(samples: np.ndarray, rate: int) -> float:
    """辅助函数：样本数换算时长。"""
    return samples.size / rate if rate else 0.0


@dataclass
class DownmixResample:
    """一次性（非流式）转换，供离线处理/测试使用。"""

    in_rate: int = DEFAULT_IN_RATE
    in_channels: int = DEFAULT_IN_CHANNELS
    out_rate: int = DEFAULT_OUT_RATE

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if self.in_channels > 1:
            x = x.reshape(-1, self.in_channels).mean(axis=1)
        return np.asarray(
            soxr.resample(x, self.in_rate, self.out_rate, quality="HQ"),
            dtype=np.float32,
        ).reshape(-1)
