"""电平表逻辑测试：RMS/峰值/峰值保持/静音阈值（用户可调）。"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.levels import (
    MIN_DB,
    LevelTracker,
    from_db,
    threshold_from_db,
    threshold_to_db,
    to_db,
)


def _sine(amp: float, n: int = 1600, freq: float = 440.0, rate: int = 16000) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / rate
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


# --------------------------------------------------------------------------- #
# dB 换算
# --------------------------------------------------------------------------- #
def test_to_db_reference_points():
    assert abs(to_db(1.0) - 0.0) < 1e-9
    assert abs(to_db(0.5) + 6.02) < 0.05
    assert abs(to_db(0.1) + 20.0) < 1e-9
    assert abs(to_db(1e-4) + 80.0) < 1e-9


def test_to_db_floor():
    assert to_db(0.0) == MIN_DB
    assert to_db(1e-30) == MIN_DB


def test_db_roundtrip():
    for db in (-80, -40, -6, 0):
        assert abs(threshold_to_db(from_db(db)) - db) < 1e-6


# --------------------------------------------------------------------------- #
# 静音阈值：核心是"用户能调"
# --------------------------------------------------------------------------- #
def test_threshold_is_respected_at_default():
    t = LevelTracker(silence_threshold=1e-4)
    t.update(_sine(0.00005))  # RMS ≈ 3.5e-5 < 1e-4
    assert t.snapshot().is_silent


def test_quiet_audio_not_silent_when_threshold_lowered():
    """同样的音频，阈值调低后就不算静音了——这就是"用户可调"的意义。"""
    quiet = _sine(0.00005)

    strict = LevelTracker(silence_threshold=1e-4)
    strict.update(quiet)
    assert strict.snapshot().is_silent

    lenient = LevelTracker(silence_threshold=1e-6)
    lenient.update(quiet)
    assert not lenient.snapshot().is_silent


def test_set_threshold_at_runtime():
    t = LevelTracker(silence_threshold=1e-6)
    t.update(_sine(0.00005))
    assert not t.snapshot().is_silent

    t.set_threshold(1e-3)  # 用户把滑杆往"更严格"拖
    assert t.snapshot().is_silent


def test_threshold_from_db_default():
    assert abs(threshold_from_db(-80.0) - 1e-4) < 1e-9


# --------------------------------------------------------------------------- #
# 弹道：攻击快、释放慢
# --------------------------------------------------------------------------- #
def test_attack_is_faster_than_release():
    t = LevelTracker(attack=0.9, release=0.05)

    t.update(_sine(0.5))
    after_rise = t.snapshot().rms

    t.update(np.zeros(1600, dtype=np.float32))
    after_fall = t.snapshot().rms

    # 上升幅度应明显大于下降幅度（同系数对比下）
    assert after_rise > 0.2, after_rise
    assert after_fall > after_rise * 0.5, "释放太快的表会看不见读数"


def test_rms_rises_toward_input():
    t = LevelTracker(attack=1.0)
    t.update(_sine(0.5))
    # amp=0.5 的正弦 RMS ≈ 0.354
    assert abs(t.snapshot().rms - 0.354) < 0.01


def test_peak_tracks_instantaneous_max():
    t = LevelTracker()
    t.update(_sine(0.8))
    assert abs(t.snapshot().peak - 0.8) < 0.01


# --------------------------------------------------------------------------- #
# 峰值保持
# --------------------------------------------------------------------------- #
def test_peak_hold_keeps_high_value_then_decays():
    t = LevelTracker(peak_hold_s=0.5, peak_hold_decay=3.0)
    t.update(_sine(0.9))
    assert abs(t.snapshot().peak_hold - 0.9) < 0.01

    # 静音 0.2 秒：还在保持期内，峰值不该掉
    for _ in range(10):
        t.update_with_dt(np.zeros(320, dtype=np.float32), dt=0.02)
    assert t.snapshot().peak_hold > 0.8, "保持期内峰值不应衰减"

    # 再静音 2 秒：保持期过了，应明显衰减
    for _ in range(100):
        t.update_with_dt(np.zeros(320, dtype=np.float32), dt=0.02)
    assert t.snapshot().peak_hold < 0.5, t.snapshot().peak_hold


def test_peak_hold_resets_on_new_peak():
    t = LevelTracker(peak_hold_s=0.1)
    t.update(_sine(0.2))
    for _ in range(20):
        t.update_with_dt(np.zeros(320, dtype=np.float32), dt=0.02)

    t.update(_sine(0.95))
    assert abs(t.snapshot().peak_hold - 0.95) < 0.01


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
def test_clipping_detected():
    t = LevelTracker()
    t.update(np.array([1.0, -1.0] * 100, dtype=np.float32))
    assert t.snapshot().clipping


def test_empty_input_does_not_move_level():
    t = LevelTracker()
    t.update(_sine(0.5))
    before = t.snapshot().rms
    t.update(np.zeros(0, dtype=np.float32))
    assert t.snapshot().rms == before
    t.update_with_dt(np.zeros(0, dtype=np.float32), 0.02)
    assert t.snapshot().rms == before


def test_zero_dt_is_ignored():
    t = LevelTracker()
    t.update_with_dt(_sine(0.5), 0.0)
    assert t.snapshot().rms == 0.0


def test_reset():
    t = LevelTracker()
    t.update(_sine(0.8))
    t.reset()
    s = t.snapshot()
    assert s.rms == 0.0 and s.peak == 0.0 and s.peak_hold == 0.0
    assert s.is_silent


def test_multichannel_input_is_flattened():
    t = LevelTracker()
    stereo = np.stack([_sine(0.5), _sine(0.5)], axis=1)
    t.update(stereo)
    assert abs(t.snapshot().peak - 0.5) < 0.01


def test_invalid_ballistics_rejected():
    with pytest.raises(ValueError):
        LevelTracker(attack=0.0)
    with pytest.raises(ValueError):
        LevelTracker(release=0.0)
