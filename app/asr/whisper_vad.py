"""Whisper-VAD（耳语/轻声特化）的本地 ONNX 推理。

为什么要单独一个文件
--------------------
sherpa-onnx 的 ``VoiceActivityDetector`` 只认 **silero / ten-vad** 两种结构
（见 sherpa-onnx 文档的 VAD 章节），而
``TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`` 是
**whisper-base encoder + 2 层解码器**的判别式 VAD：输入 80 通道 log-mel、
输出 20ms 一帧的语音概率。它进不了 sherpa-onnx 的 VAD 接口，只能自己推理
（因此**依赖 onnxruntime**，且不需要 transformers / librosa —— mel 特征在这里自实现）。

为什么值得存在
--------------
Silero 是按正常音量训练的，**轻声/耳语经常被判成静音**，音频压根不进识别引擎——
这时换 ASR 模型没有意义，要换的是 VAD。这个模型用约 500 小时日语 ASMR 录音训练，
官方标签就是 ``whispered-speech`` / ``asmr``；判别式模型也不会像 Whisper 那样吐幻觉文本。

延迟与算力（必须知道的取舍）
----------------------------
- 模型固定吃 **30 秒**（80×3000 log-mel）→ 1500 帧输出（每帧 20ms）。
- 官方示例是"攒满 30 秒跑一次"，对字幕等于 30 秒延迟，**不可用**；
  所以这里做**滑动窗口**：每 ``hop_ms`` 拿"最近 30 秒"跑一次，
  只采信窗口靠前的部分（最右侧 ``MARGIN_MS`` 不采信：那里右侧上下文不足）。
- 代价：每次前向都要算满 30 秒窗口，所以 hop 越小 CPU 越高。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from app.asr.locate import require
from app.asr.vad import SAMPLE_RATE, SpeechSegment
from app.utils.log import get_logger

log = get_logger(__name__)

VAD_MODEL_ID = "whisper-vad-asmr"

# ---- 与模型绑定的常量（文件名、形状都来自仓库 model_metadata.json） ----
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000        # 320
WINDOW_FRAMES = 1500                                  # 模型**输出**帧数（30s / 20ms）
WINDOW_SAMPLES = WINDOW_FRAMES * FRAME_SAMPLES        # 480000

N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MEL_FRAMES = WINDOW_SAMPLES // HOP_LENGTH
"""log-mel 的时间步数（30s / 10ms = 3000），也就是模型输入最后一维。

⚠️ 别和 ``WINDOW_FRAMES``（输出 1500 帧）搞混：两者是 2:1 的关系，
截错就会只喂进 15 秒音频。"""
MARGIN_MS = 400
"""窗口最右侧不采信的长度：模型看不到右边时那里的判定不稳。"""

_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "auto": ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
}


# --------------------------------------------------------------------------- #
# whisper log-mel 特征（移植 librosa/WhisperFeatureExtractor 的默认参数）
# --------------------------------------------------------------------------- #
def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """HTK 之外的 Slaney 刻度（librosa 默认 htk=False）。"""
    freq = np.asarray(freq, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3.0
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        freq >= min_log_hz,
        min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3.0
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        mels >= min_log_mel,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        freqs,
    )


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """``(n_mels, 1 + n_fft // 2)`` 的 Slaney 归一化三角滤波器组。"""
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, 1 + N_FFT // 2)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2.0), N_MELS + 2)
    mel_f = _mel_to_hz(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fft_freqs[np.newaxis, :]

    weights = np.zeros((N_MELS, fft_freqs.size), dtype=np.float64)
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney 归一化：每个滤波器的面积归一到 2/(f[i+2]-f[i])
    enorm = 2.0 / (mel_f[2 : N_MELS + 2] - mel_f[:N_MELS])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def whisper_log_mel(audio: np.ndarray) -> np.ndarray:
    """把 16k 单声道音频转成 whisper 的 ``(80, 3000)`` log-mel。

    与 ``WhisperFeatureExtractor`` 一致的关键点（少一个就对不上）：
    ``n_fft=400`` / ``hop=160`` / **periodic** Hann 窗 / ``center=True`` 的 reflect padding /
    STFT 去掉最后一帧 / ``log10`` 下限 1e-10 / 动态范围压到 8 / ``(x+4)/4``。
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < WINDOW_SAMPLES:
        x = np.pad(x, (0, WINDOW_SAMPLES - x.size))
    else:
        x = x[:WINDOW_SAMPLES]

    pad = N_FFT // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]

    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic Hann
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2)[:-1]           # whisper 丢掉最后一帧

    mel = mel_filterbank() @ power.T                         # (80, T)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    if log_spec.shape[1] < MEL_FRAMES:                       # 理论上不会发生
        log_spec = np.pad(log_spec, ((0, 0), (0, MEL_FRAMES - log_spec.shape[1])))
    return log_spec[:, :MEL_FRAMES].astype(np.float32)


# --------------------------------------------------------------------------- #
# ONNX 模型
# --------------------------------------------------------------------------- #
class WhisperVadOnnx:
    """只做一件事：30 秒音频 → 1500 帧语音概率。"""

    def __init__(self, model_path: Path, num_threads: int = 1, provider: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.provider = provider
        self._session = None
        self._input_name = ""

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - 取决于用户环境
            raise RuntimeError(
                "耳语 VAD（Whisper-VAD ASMR）需要 onnxruntime，但当前环境没装。\n"
                "请运行: uv pip install --python .venv\\Scripts\\python.exe onnxruntime\n"
                "装好后在设置 → 识别 里重新选择；不想装就把 VAD 换回默认的 Silero。"
            ) from exc

        available = set(ort.get_available_providers())
        wanted = _PROVIDERS.get(self.provider, _PROVIDERS["cpu"])
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = self.num_threads
        opts.intra_op_num_threads = self.num_threads
        opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(self.model_path), providers=providers, sess_options=opts
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info(
            "耳语 VAD 就绪（provider=%s threads=%d）", providers[0], self.num_threads
        )

    def probabilities(self, window: np.ndarray) -> np.ndarray:
        """``window`` 为 30 秒音频 → 返回 1500 帧的语音概率。"""
        assert self._session is not None
        feats = whisper_log_mel(window)[np.newaxis, ...]      # (1, 80, 3000)
        outputs = self._session.run(None, {self._input_name: feats})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))                  # 模型输出是 logits

    @property
    def ready(self) -> bool:
        return self._session is not None


# --------------------------------------------------------------------------- #
# 流式分段器（接口与 ``app/asr/vad.py`` 的 VadSegmenter 保持一致）
# --------------------------------------------------------------------------- #
class WhisperVadSegmenter:
    """滑动窗口 + 双阈值滞回的流式 VAD。

    对外的四件事与 Silero 版完全一样：``load / accept / pop_ready / flush / reset``，
    所以 ``SherpaOfflineEngine`` 不需要知道到底用的是哪个 VAD。
    """

    def __init__(
        self,
        min_silence_ms: int = 350,
        min_speech_ms: int = 250,
        max_speech_ms: int = 8000,
        threshold: float = 0.5,
        speech_pad_ms: int = 120,
        hop_ms: int = 1000,
        num_threads: int = 1,
        provider: str = "cpu",
        models_dir: Path | None = None,
        buffer_seconds: float = 60.0,
    ) -> None:
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.threshold = threshold
        self.speech_pad_ms = speech_pad_ms
        self.hop_ms = max(200, hop_ms)
        self.num_threads = num_threads
        self.provider = provider
        self.models_dir = models_dir
        self.buffer_seconds = buffer_seconds

        self._model: WhisperVadOnnx | None = None
        self._hop_samples = max(FRAME_SAMPLES, SAMPLE_RATE * self.hop_ms // 1000)
        self._margin_frames = max(1, MARGIN_MS // FRAME_MS)
        self._min_speech_frames = max(1, self.min_speech_ms // FRAME_MS)
        self._min_silence_frames = max(1, self.min_silence_ms // FRAME_MS)
        self._max_speech_frames = max(1, self.max_speech_ms // FRAME_MS)
        self._pad_frames = max(0, self.speech_pad_ms // FRAME_MS)
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_frame0 = 0          # _audio[0] 对应的绝对帧号
        self._pending: list[np.ndarray] = []
        self._received = 0              # 累计 accept 的样本数（绝对）
        self._last_infer = 0
        self._probs: dict[int, float] = {}
        self._next_frame = 0            # 下一个待判定帧（绝对帧号）
        self._ready: list[SpeechSegment] = []
        self._triggered = False
        self._seg_start = 0
        self._temp_end = -1

    def load(self) -> None:
        paths = require(VAD_MODEL_ID, self.models_dir, roles=("model",))
        self._model = WhisperVadOnnx(
            paths["model"], num_threads=self.num_threads, provider=self.provider
        )
        self._model.load()
        self._reset_state()
        log.info(
            "VAD 就绪：Whisper-VAD ASMR（静音 %.0fms / 最短语音 %.0fms / 最长 %.0fs / 步进 %.0fms）",
            self.min_silence_ms, self.min_speech_ms, self.max_speech_ms / 1000, self.hop_ms,
        )

    # ------------------------------------------------------------------ #
    def accept(self, samples: np.ndarray) -> None:
        """喂入 16k 单声道 float32 音频。"""
        if self._model is None:
            return
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        self._pending.append(x)
        self._received += x.size
        if self._received - self._last_infer >= self._hop_samples:
            self._drain()
            self._infer()

    def _drain(self) -> None:
        """把待处理音频并进主缓冲（只在推理时机做，避免每 10ms 拷贝整个缓冲）。"""
        if not self._pending:
            return
        chunks = self._pending
        self._pending = []
        x = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        self._audio = x if self._audio.size == 0 else np.concatenate([self._audio, x])

        limit = int(self.buffer_seconds * SAMPLE_RATE)
        if self._audio.size > limit:
            drop = self._audio.size - limit
            drop -= drop % FRAME_SAMPLES          # 必须整帧丢弃，否则帧号会错位
            if drop > 0:
                self._audio = self._audio[drop:].copy()
                self._audio_frame0 += drop // FRAME_SAMPLES

    def _infer(self) -> None:
        assert self._model is not None
        avail = self._audio.size
        if avail < FRAME_SAMPLES:
            return

        win_len = min(avail, WINDOW_SAMPLES)
        start = avail - win_len
        start -= start % FRAME_SAMPLES
        window = self._audio[start : start + WINDOW_SAMPLES]
        n_frames = int(window.size // FRAME_SAMPLES)
        if window.size < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - window.size))

        probs = self._model.probabilities(window)

        usable = min(n_frames, WINDOW_FRAMES)
        if usable > self._margin_frames:
            usable -= self._margin_frames
        w0 = self._audio_frame0 + start // FRAME_SAMPLES
        for i in range(usable):
            self._probs[w0 + i] = float(probs[i])

        self._last_infer = self._received
        self._advance()

    # ------------------------------------------------------------------ #
    def _advance(self) -> None:
        """按顺序消费已经拿到概率的帧。"""
        while self._next_frame in self._probs:
            prob = self._probs.pop(self._next_frame)
            self._step(self._next_frame, prob)
            self._next_frame += 1

    def _step(self, frame: int, prob: float) -> None:
        """双阈值滞回（与 Silero / 官方 get_speech_timestamps 同思路，但是流式的）。"""
        neg = max(self.threshold - 0.15, 0.01)

        if not self._triggered:
            if prob >= self.threshold:
                self._triggered = True
                self._seg_start = frame
                self._temp_end = -1
            return

        if prob >= self.threshold:
            self._temp_end = -1
        elif prob < neg:
            if self._temp_end < 0:
                self._temp_end = frame
            if frame - self._temp_end >= self._min_silence_frames:
                self._emit(self._seg_start, self._temp_end)
                self._triggered = False
                self._temp_end = -1
                return

        if frame - self._seg_start >= self._max_speech_frames:
            self._emit(self._seg_start, frame)
            self._triggered = False
            self._temp_end = -1

    def _emit(self, f0: int, f1: int) -> None:
        f0 = max(0, f0 - self._pad_frames)
        f1 = f1 + self._pad_frames
        if f1 - f0 < self._min_speech_frames:
            return

        buf_start = self._audio_frame0 * FRAME_SAMPLES
        rel0 = max(0, f0 * FRAME_SAMPLES - buf_start)
        rel1 = min(self._audio.size, f1 * FRAME_SAMPLES - buf_start)
        if rel1 <= rel0:
            return
        self._ready.append(
            SpeechSegment(
                samples=self._audio[rel0:rel1].copy(),
                start_sample=buf_start + rel0,
            )
        )

    # ------------------------------------------------------------------ #
    def pop_ready(self) -> list[SpeechSegment]:
        out = self._ready
        self._ready = []
        return out

    def flush(self) -> list[SpeechSegment]:
        """音频结束：把尾巴上还没断句的语音也定稿。

        这里必须**强制再推理一次**：正常路径只在攒够一个 ``hop`` 时才判断，
        所以结尾不足一个 hop 的音频永远拿不到概率，最后那截语音会被整段丢掉
        （hop 调大时尤其明显，实测 hop=2000 会少掉 1.4 秒）。
        """
        self._drain()
        if self._model is not None and self._audio.size >= FRAME_SAMPLES:
            self._infer()
        if self._triggered:
            self._emit(self._seg_start, max(self._seg_start + 1, self._next_frame))
            self._triggered = False
            self._temp_end = -1
        return self.pop_ready()

    def reset(self) -> None:
        self._reset_state()

    @property
    def ready(self) -> bool:
        return self._model is not None
