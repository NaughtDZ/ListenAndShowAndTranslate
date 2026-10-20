"""耳语/轻声 VAD（Whisper-VAD ASMR）的测试。

119MB 的 ONNX 权重不进仓库，所以这里只测**不依赖权重**的部分：

1. log-mel 特征：形状必须是 ``(80, 3000)``（30s / 10ms），静音必须是 whisper 的 -1.5
   （截错成 1500 就会只喂进 15 秒音频——这个错真的犯过一次）；
2. 流式状态机：双阈值滞回、最短语音过滤、静音断句、最长憋句强制切；
3. 注册表/配置：它是 VAD 不是识别模型，而且**默认仍然是 Silero**（不能悄悄换掉）。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.asr.whisper_vad import (
    FRAME_SAMPLES,
    MEL_FRAMES,
    SAMPLE_RATE,
    WhisperVadSegmenter,
    mel_filterbank,
    whisper_log_mel,
)
from app.asr.whisper_vad import WINDOW_SAMPLES
from app.config import AppConfig
from app.models.registry import MODELS, pack_for_model, supports_language

# --------------------------------------------------------------------------- #
# 特征
# --------------------------------------------------------------------------- #
def test_log_mel_shape_is_30_seconds():
    feats = whisper_log_mel(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    assert feats.shape == (80, MEL_FRAMES) == (80, 3000)


def test_silence_is_whisper_constant():
    """whisper 对全静音的 log-mel 恒为 (log10(1e-10)+4)/4 = -1.5。"""
    feats = whisper_log_mel(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    assert np.allclose(feats, -1.5, atol=1e-4)


def test_mel_filterbank_is_monotonic_slaney():
    fb = mel_filterbank()
    assert fb.shape == (80, 201)
    peaks = [int(row.argmax()) for row in fb]
    assert all(a <= b for a, b in zip(peaks, peaks[1:])), "滤波器中心频率必须随序号递增"


def test_short_audio_is_padded_not_truncated():
    """不足 30 秒要补零，不能丢内容。"""
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
    feats = whisper_log_mel(tone)
    assert feats.shape == (80, 3000)
    assert feats[:, :10].max() > -1.5, "前段有声音，不该全是静音值"


# --------------------------------------------------------------------------- #
# 流式状态机（直接灌概率，绕开模型）
# --------------------------------------------------------------------------- #
def _segmenter(**kw) -> WhisperVadSegmenter:
    kw.setdefault("min_silence_ms", 200)     # 10 帧
    kw.setdefault("min_speech_ms", 100)      # 5 帧
    kw.setdefault("max_speech_ms", 4000)
    kw.setdefault("speech_pad_ms", 0)
    seg = WhisperVadSegmenter(threshold=0.5, **kw)
    seg._audio = np.full(FRAME_SAMPLES * 400, 0.1, dtype=np.float32)
    return seg


def _feed(seg: WhisperVadSegmenter, probs: list[float]) -> list:
    for frame, prob in enumerate(probs):
        seg._probs[frame] = prob
    seg._advance()
    return seg.pop_ready()


def test_splits_on_enough_silence():
    seg = _segmenter()
    probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 40
    segs = _feed(seg, probs)
    assert len(segs) == 1
    assert segs[0].start_sample == 5 * FRAME_SAMPLES
    assert segs[0].samples.size == 20 * FRAME_SAMPLES


def test_drops_segments_shorter_than_min_speech():
    seg = _segmenter(min_speech_ms=250)      # 13 帧
    probs = [0.0] * 5 + [0.9] * 4 + [0.0] * 40
    assert _feed(seg, probs) == []


def test_hysteresis_does_not_split_on_borderline_probability():
    """概率掉到 0.4（阈值与负阈值之间）时不该断句——那只是变轻了。"""
    seg = _segmenter()
    probs = [0.9] * 10 + [0.4] * 30 + [0.0] * 40
    segs = _feed(seg, probs)
    assert len(segs) == 1
    # 断句应发生在 0.4 结束（frame 40）之后，而不是进入 0.4 时
    assert segs[0].samples.size >= 30 * FRAME_SAMPLES


def test_force_cut_at_max_speech():
    """一直不停地说：到最长憋句时间必须强制切一刀。"""
    seg = _segmenter(max_speech_ms=200)      # 10 帧
    segs = _feed(seg, [0.9] * 30)
    assert len(segs) >= 2
    assert all(s.samples.size <= 10 * FRAME_SAMPLES for s in segs)


def test_flush_closes_open_segment():
    seg = _segmenter()
    _feed(seg, [0.0] * 5 + [0.9] * 10)       # 还没等到静音就结束
    assert seg.pop_ready() == []
    segs = seg.flush()
    assert len(segs) == 1
    assert segs[0].samples.size == 10 * FRAME_SAMPLES


# --------------------------------------------------------------------------- #
# 注册表与配置
# --------------------------------------------------------------------------- #
def test_registry_entry_is_a_vad_not_an_asr_model():
    spec = MODELS["whisper-vad-asmr"]
    assert spec.kind == "vad"
    assert spec.onnx_files, "必须有 onnx 文件"
    assert pack_for_model("whisper-vad-asmr") == "vad-asmr"
    # 关键：它绝不能出现在「识别模型」下拉框里（喂给它不会有字幕）
    assert supports_language("whisper-vad-asmr", "ja") is False


def test_default_vad_is_still_silero():
    """加新东西不能改变默认行为。"""
    assert AppConfig().asr.vad.model == "silero-vad"


def test_hop_defaults_and_bounds():
    """步进：默认 1000ms，范围受 Field 约束（太小会跟不上实时）。

    注意 pydantic v2 **赋值时不校验**，所以边界要用构造/加载来验
    （手改 config.json 走的正是 model_validate 这条路）。
    """
    from app.config import VADConfig

    vad = AppConfig().asr.vad
    assert vad.hop_ms == 1000
    vad.hop_ms = 2000
    assert vad.hop_ms == 2000

    assert VADConfig(hop_ms=500).hop_ms == 500
    assert VADConfig(hop_ms=5000).hop_ms == 5000
    with pytest.raises(Exception):
        VADConfig(hop_ms=100)      # 低于下限
    with pytest.raises(Exception):
        VADConfig(hop_ms=9000)     # 高于上限


def test_hop_reaches_the_whisper_segmenter():
    """配置里的 hop 必须真的落到 WhisperVAD 上（否则旋钮是假的）。"""
    fast = WhisperVadSegmenter(hop_ms=500)
    slow = WhisperVadSegmenter(hop_ms=2000)
    assert fast._hop_samples == SAMPLE_RATE * 500 // 1000
    assert slow._hop_samples == SAMPLE_RATE * 2000 // 1000
    # 再小也不会到 0（构造里对下限做了保护）
    assert WhisperVadSegmenter(hop_ms=1)._hop_samples >= FRAME_SAMPLES


def test_silero_backend_ignores_hop():
    """Silero 是逐帧判断的，hop 不该影响它（只被存下来、不参与加载）。"""
    from app.asr.vad import VadSegmenter

    seg = VadSegmenter(vad_model="silero-vad", hop_ms=2000)
    assert seg.hop_ms == 2000
    assert seg.vad_model == "silero-vad"


def test_vad_config_accepts_asmr_backend():
    assert AppConfig().asr.vad.model == "silero-vad"
    cfg = AppConfig().asr.vad
    cfg.model = "whisper-vad-asmr"
    assert cfg.model == "whisper-vad-asmr"


def test_missing_onnxruntime_gives_actionable_error(monkeypatch):
    """没装 onnxruntime 时要给出可执行的修复命令，而不是一个 ImportError 堆栈。"""
    from app.asr.whisper_vad import WhisperVadOnnx

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        WhisperVadOnnx("nope.onnx").load()


# --------------------------------------------------------------------------- #
# 设置界面（offscreen）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_settings_offers_vad_backends_and_saves_choice(qapp, monkeypatch):
    from app.models.downloader import ModelDownloader
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)

    win = SettingsWindow(AppConfig())
    try:
        offered = [win.vad_combo.itemData(i) for i in range(win.vad_combo.count())]
        assert offered == ["silero-vad", "whisper-vad-asmr"]

        win.vad_combo.setCurrentIndex(win.vad_combo.findData("whisper-vad-asmr"))
        win._save()
        assert win.config.asr.vad.model == "whisper-vad-asmr"

        # 换回默认也要能存回去
        win.vad_combo.setCurrentIndex(win.vad_combo.findData("silero-vad"))
        win._save()
        assert win.config.asr.vad.model == "silero-vad"
    finally:
        win.hide()
        win.deleteLater()


def test_settings_marks_undownloaded_vad(qapp, monkeypatch):
    from app.models.downloader import ModelDownloader
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "missing", raising=True)

    win = SettingsWindow(AppConfig())
    try:
        idx = win.vad_combo.findData("whisper-vad-asmr")
        assert "未下载" in win.vad_combo.itemText(idx)
        win.vad_combo.setCurrentIndex(idx)
        assert "还没下载" in win.model_hint.text()
        assert "vad-asmr" in win.model_hint.text()
    finally:
        win.hide()
        win.deleteLater()


def test_settings_hop_knob_writes_through(qapp, monkeypatch):
    """步进旋钮要真的写进配置，而不是只在界面上好看。"""
    from app.models.downloader import ModelDownloader
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)

    win = SettingsWindow(AppConfig())
    try:
        assert win.vad_hop.value() == 1000
        win.vad_hop.setValue(2000)
        win._save()
        assert win.config.asr.vad.hop_ms == 2000
    finally:
        win.hide()
        win.deleteLater()


def test_settings_hop_is_explained_and_greyed_for_silero(qapp, monkeypatch):
    """用户要的：这个调节是干嘛的，必须写在界面上。"""
    from app.models.downloader import ModelDownloader
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)

    win = SettingsWindow(AppConfig())
    try:
        win.vad_combo.setCurrentIndex(win.vad_combo.findData("silero-vad"))
        assert not win.vad_hop.isEnabled(), "Silero 不看步进，旋钮该置灰"
        silent_text = win.vad_hop_hint.text()
        assert "单核" in silent_text and "不生效" in silent_text

        win.vad_combo.setCurrentIndex(win.vad_combo.findData("whisper-vad-asmr"))
        assert win.vad_hop.isEnabled()
        active_text = win.vad_hop_hint.text()
        assert "单核" in active_text and "不生效" not in active_text

        # 拧到跟不上实时的档位要给出警告，而不是默默变卡
        win.vad_hop.setValue(500)
        assert "跟不上" in win.vad_hop_hint.text()
        win.vad_hop.setValue(5000)
        assert "跟不上" not in win.vad_hop_hint.text()
    finally:
        win.hide()
        win.deleteLater()
