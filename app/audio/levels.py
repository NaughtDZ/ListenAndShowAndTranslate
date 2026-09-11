"""电平表逻辑：RMS / 峰值 / 峰值保持 / 静音判定。

纯计算，不依赖 Qt，便于单测（UI 只管画）。

弹道（ballistics）设计：
- **攻击快、释放慢**：真实电平表都是这样，否则数字会糊得看不清
- **峰值保持**：峰值条在 ``peak_hold_s`` 内钉住最高点，之后再缓慢回落，
  这样瞬间的爆音不会被肉眼漏掉

阈值不写死：``silence_threshold`` 由调用方（最终是用户配置）传入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 显示下限：低于此值统一按 -90 dB 显示，避免 log(0)
MIN_DB = -90.0


def to_db(amplitude: float) -> float:
    """线性幅度 → dBFS（满幅 1.0 = 0 dB）。"""
    if amplitude <= 0:
        return MIN_DB
    return max(MIN_DB, 20.0 * math.log10(amplitude))


def from_db(db: float) -> float:
    """dBFS → 线性幅度。"""
    return 10.0 ** (db / 20.0)


@dataclass
class LevelState:
    """一帧电平快照，供 UI 绘制。"""

    rms: float = 0.0
    peak: float = 0.0
    peak_hold: float = 0.0

    db_rms: float = MIN_DB
    db_peak: float = MIN_DB
    db_peak_hold: float = MIN_DB

    is_silent: bool = True
    """当前是否低于静音阈值（用户可调）。"""

    clipping: bool = False
    """是否接近满幅（≥0.999），提示可能削波。"""

    @property
    def silence_threshold_db(self) -> float:
        return MIN_DB


class LevelTracker:
    """把连续的音频块转成平滑、好看、可读的电平读数。"""

    def __init__(
        self,
        silence_threshold: float = 1e-4,
        attack: float = 0.6,
        release: float = 0.12,
        peak_hold_s: float = 1.5,
        peak_hold_decay: float = 1.5,
    ) -> None:
        """
        Args:
            silence_threshold: 静音阈值（线性幅度），用户可调
            attack: RMS 上升平滑系数 0~1，越大越跟手
            release: RMS 回落平滑系数 0~1，越小掉得越慢
            peak_hold_s: 峰值保持时长（秒）
            peak_hold_decay: 保持结束后峰值每秒衰减的系数
        """
        if not 0 < attack <= 1:
            raise ValueError("attack 必须在 (0, 1]")
        if not 0 < release <= 1:
            raise ValueError("release 必须在 (0, 1]")

        self.silence_threshold = silence_threshold
        self.attack = attack
        self.release = release
        self.peak_hold_s = peak_hold_s
        self.peak_hold_decay = peak_hold_decay

        self._rms = 0.0
        self._peak = 0.0
        self._peak_hold = 0.0
        self._hold_age = 0.0

    # ------------------------------------------------------------------ #
    def update(self, samples: np.ndarray) -> LevelState:
        """喂一块音频，假定块长约 20ms（proc-tap 实际 10ms，差别可忽略）。

        需要精确计时时用 :meth:`update_with_dt`。
        """
        return self.update_with_dt(samples, dt=0.02)

    def update_with_dt(self, samples: np.ndarray, dt: float) -> LevelState:
        """带时间步长的版本（推荐）：峰值保持按真实时间衰减。"""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0 or dt <= 0:
            return self.snapshot()

        inst_peak = float(np.max(np.abs(x)))
        inst_rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

        # 平滑系数按时间步长归一化（假定系数是按 20ms 一块标定的）
        ref = 0.02
        a = 1.0 - (1.0 - self.attack) ** (dt / ref)
        r = 1.0 - (1.0 - self.release) ** (dt / ref)
        coef = a if inst_rms >= self._rms else r
        self._rms += coef * (inst_rms - self._rms)

        self._peak = inst_peak

        if inst_peak >= self._peak_hold:
            self._peak_hold = inst_peak
            self._hold_age = 0.0
        else:
            self._hold_age += dt
            if self._hold_age > self.peak_hold_s:
                decay = math.exp(-self.peak_hold_decay * dt)
                self._peak_hold = max(inst_peak, self._peak_hold * decay)

        return self.snapshot()

    # ------------------------------------------------------------------ #
    def snapshot(self) -> LevelState:
        return LevelState(
            rms=self._rms,
            peak=self._peak,
            peak_hold=self._peak_hold,
            db_rms=to_db(self._rms),
            db_peak=to_db(self._peak),
            db_peak_hold=to_db(self._peak_hold),
            is_silent=self._rms < self.silence_threshold,
            clipping=self._peak >= 0.999,
        )

    def reset(self) -> None:
        self._rms = 0.0
        self._peak = 0.0
        self._peak_hold = 0.0
        self._hold_age = 0.0

    def set_threshold(self, threshold: float) -> None:
        """用户可随时调整"什么算静音"。"""
        self.silence_threshold = threshold


def threshold_from_db(db: float) -> float:
    """UI 里的 dB 滑杆 → 线性阈值。"""
    return from_db(db)


def threshold_to_db(threshold: float) -> float:
    """线性阈值 → 便于 UI 显示的 dB 值。"""
    return to_db(threshold)
