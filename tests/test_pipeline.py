"""音频管线测试：降混、重采样、环形缓冲、能量统计。

全部为确定性计算，不依赖声卡/进程，任何机器都能跑。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.pipeline import (
    SILENCE_RMS_THRESHOLD,
    AudioPipeline,
    DownmixResample,
    RingBuffer,
    pcm_duration_seconds,
)

IN_RATE = 48000
OUT_RATE = 16000


def _sine(freq: float, seconds: float, rate: int = IN_RATE, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1).reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 降混
# --------------------------------------------------------------------------- #
def test_downmix_averages_channels():
    """L=1.0, R=0.0 → 单声道恒为 0.5。

    注意：不能在重采样**之后**判断——soxr 有 40ms 缓冲且边缘有 sinc 振铃。
    能量统计是在降混之后、重采样之前算的，正好用来断言降混结果。
    """
    stereo = np.tile(np.array([1.0, 0.0], dtype=np.float32), IN_RATE)  # 1 秒
    pipe = AudioPipeline()
    pipe.process(stereo)
    assert abs(pipe.stats.last_rms - 0.5) < 1e-3, pipe.stats.last_rms
    assert abs(pipe.stats.peak - 0.5) < 1e-3, pipe.stats.peak


def test_downmix_ignores_incomplete_frame():
    """单声道样本数是奇数时应丢掉尾部半帧，而不是抛异常。"""
    pipe = AudioPipeline()
    pipe.process(np.ones(4800 + 1, dtype=np.float32))
    assert pipe.stats.input_samples == 2400, "4800 个样本应降混成 2400 帧"


def test_mono_input_supported():
    pipe = AudioPipeline(in_channels=1)
    out = pipe.process(_sine(440, 0.1))
    assert out.size > 0


# --------------------------------------------------------------------------- #
# 重采样
# --------------------------------------------------------------------------- #
def test_resample_ratio_is_3x():
    """48k → 16k 应得到约 1/3 的样本数。"""
    pipe = AudioPipeline()
    pipe.process(_stereo(_sine(440, 1.0)))
    total = pipe.stats.output_samples
    assert abs(total - OUT_RATE) < OUT_RATE * 0.05, f"1 秒应约 {OUT_RATE} 样本，实得 {total}"


def test_resample_preserves_frequency():
    """重采样不能改变音高。"""
    pipe = AudioPipeline()
    out = pipe.process(_stereo(_sine(440, 1.0)))
    spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / OUT_RATE)
    peak = float(freqs[int(np.argmax(spectrum))])
    assert abs(peak - 440) < 15, f"峰值应在 440Hz 附近，实得 {peak}Hz"


def test_offline_downmix_resample_matches():
    conv = DownmixResample()
    out = conv(_stereo(_sine(1000, 0.5)))
    assert abs(out.size - OUT_RATE * 0.5) < OUT_RATE * 0.05


def test_flush_returns_array():
    pipe = AudioPipeline()
    pipe.process(_stereo(_sine(440, 0.1)))
    tail = pipe.flush()
    assert isinstance(tail, np.ndarray)


# --------------------------------------------------------------------------- #
# 流式重采样的缓冲行为（实测标定，见 docs/P1-实测记录.md）
# --------------------------------------------------------------------------- #
def test_streaming_resampler_buffers_before_first_output():
    """HQ 档首次输出固定出现在第 4 块 ⇒ 40ms 延迟。

    这是延迟预算的一部分，不是 bug：不能"修掉"，只能标定并接受。
    """
    pipe = AudioPipeline()
    block = _stereo(_sine(440, 0.01))  # 480 帧 = 10ms，与 proc-tap 真实分块一致

    first_block = None
    for i in range(1, 11):
        if pipe.process(block).size and first_block is None:
            first_block = i

    assert first_block == 4, f"HQ 应在第 4 块首次输出，实测 {first_block}"


def test_tiny_input_yields_no_output_but_no_error():
    """极小输入块不吐数据是正常缓冲行为（不能当错误处理）。"""
    pipe = AudioPipeline()
    out = pipe.process(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert out.size == 0


def test_streaming_output_converges_to_ratio():
    """长期看总输出量应回到 1/3 —— 缓冲只带来延迟，不丢数据。"""
    pipe = AudioPipeline()
    block = _stereo(_sine(440, 0.01))
    n_blocks = 300  # 3 秒

    for _ in range(n_blocks):
        pipe.process(block)

    expected = n_blocks * 480 / 3
    assert abs(pipe.stats.output_samples - expected) < expected * 0.02, (
        f"期望约 {expected:.0f} 样本，实得 {pipe.stats.output_samples}"
    )


# --------------------------------------------------------------------------- #
# 环形缓冲
# --------------------------------------------------------------------------- #
def test_ringbuffer_drops_oldest():
    ring = RingBuffer(capacity_samples=100)
    ring.push(np.arange(60, dtype=np.float32))
    dropped = ring.push(np.arange(60, dtype=np.float32) + 100)

    assert dropped == 20, "超出容量的 20 个样本应被丢弃"
    assert len(ring) == 100
    data = ring.pop_all()
    # 保留下来的应该是"新的那一半"：丢掉了最旧的 0..19
    assert data[0] == 20
    assert data[-1] == 159
    assert ring.dropped_samples == 20


def test_ringbuffer_pop_all_clears():
    ring = RingBuffer(capacity_samples=1000)
    ring.push(np.ones(10, dtype=np.float32))
    assert ring.pop_all().size == 10
    assert len(ring) == 0
    assert ring.pop_all().size == 0


def test_ringbuffer_rejects_bad_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


# --------------------------------------------------------------------------- #
# 能量统计（实测教训：静音时数据照样有，只能靠能量判断）
# --------------------------------------------------------------------------- #
def test_silence_is_detected():
    pipe = AudioPipeline()
    pipe.process(_stereo(np.zeros(4800, dtype=np.float32)))
    assert pipe.stats.last_rms < SILENCE_RMS_THRESHOLD
    assert pipe.stats.silent_chunks == 1
    assert pipe.stats.silent_seconds_total > 0


def test_loud_resets_silence_counter():
    pipe = AudioPipeline()
    pipe.process(_stereo(np.zeros(4800, dtype=np.float32)))
    assert pipe.stats.silent_chunks == 1
    pipe.process(_stereo(_sine(440, 0.1)))
    assert pipe.stats.silent_chunks == 0
    assert not pipe.stats.is_silent


def test_peak_tracks_maximum():
    pipe = AudioPipeline()
    pipe.process(_stereo(_sine(440, 0.1, amp=0.25)))
    pipe.process(_stereo(_sine(440, 0.1, amp=0.75)))
    assert abs(pipe.stats.peak - 0.75) < 0.01


def test_int16_input_dtype():
    pipe = AudioPipeline(input_dtype="int16")
    raw = (np.ones(IN_RATE, dtype=np.int16) * 16384).tobytes()  # 0.5 满幅，1 秒
    out = pipe.process(raw)
    assert out.size > 0
    assert abs(pipe.stats.peak - 0.5) < 0.01
    assert pipe.stats.peak <= 1.0


def test_unknown_dtype_raises():
    with pytest.raises(ValueError):
        AudioPipeline(input_dtype="float64").process(b"\x00" * 16)


def test_reset_clears_stats():
    pipe = AudioPipeline()
    pipe.process(_stereo(_sine(440, 0.1)))
    pipe.reset()
    assert pipe.stats.input_chunks == 0
    assert pipe.stats.peak == 0.0


def test_pcm_duration_helper():
    assert pcm_duration_seconds(np.zeros(16000, dtype=np.float32), 16000) == 1.0
    assert pcm_duration_seconds(np.zeros(0, dtype=np.float32), 16000) == 0.0
