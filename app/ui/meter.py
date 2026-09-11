"""电平表悬浮窗 + 控制窗。

用户要求：**画一个音量条并显示 peak 或 rms**，且**阈值要能自己调**。

架构（也是 P3 悬浮字幕窗的预演）：
- ``LevelMeterWindow``：真正浮在游戏上的那条，无边框 / 透明 / 置顶 / 可点击穿透
- ``MeterControlWindow``：普通窗口，放阈值滑杆等控件。
  两窗分离是有意的——**一旦开启点击穿透，悬浮窗就点不动了**，
  所有控制必须放在另一个窗口里，否则用户会被自己锁死。
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import CaptureWorker, TargetSpec, resolve_target
from app.audio.levels import MIN_DB, LevelState, LevelTracker, threshold_from_db, to_db
from app.audio.process_list import get_session_volume, suggest_gain
from app.utils import win32
from app.utils.log import get_logger

log = get_logger(__name__)

# 电平表显示范围（dBFS）
DB_MIN = -70.0
DB_MAX = 0.0

# 阈值滑杆范围（dBFS）
THRESHOLD_DB_MIN = -90.0
THRESHOLD_DB_MAX = -20.0
DEFAULT_THRESHOLD_DB = -80.0

COLOR_BG = QColor(0, 0, 0, 165)
COLOR_TRACK = QColor(255, 255, 255, 38)
COLOR_TEXT = QColor(235, 235, 235)
COLOR_TEXT_DIM = QColor(170, 170, 170)
COLOR_WARN = QColor(255, 170, 60)
COLOR_THRESHOLD = QColor(255, 150, 40)
COLOR_HOLD = QColor(255, 255, 255, 230)
COLOR_CLIP = QColor(255, 70, 70)


def _level_color(db: float) -> QColor:
    """按电平高低给颜色：绿 → 黄 → 红。"""
    if db >= -3.0:
        return QColor(239, 68, 68)
    if db >= -12.0:
        return QColor(234, 179, 8)
    return QColor(34, 197, 94)


# 中文字体候选（Windows 上按优先级挑第一个存在的）
_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "微软雅黑",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)


def _pick_cjk_font() -> QFont:
    """挑一个能显示中文的字体。

    Qt6 不再自带字体，若系统字体解析不到，中文会变成方块——
    所以这里**显式**从候选里挑，而不是依赖默认字体。
    """
    try:
        from PySide6.QtGui import QFontDatabase

        families = set(QFontDatabase.families())
        for name in _CJK_FONT_CANDIDATES:
            if name in families:
                f = QFont(name)
                f.setHintingPreference(QFont.PreferFullHinting)
                return f
    except Exception as exc:  # noqa: BLE001
        log.debug("查询字体族失败: %s", exc)

    f = QFont(_CJK_FONT_CANDIDATES[0])
    f.setFamily(_CJK_FONT_CANDIDATES[0])
    return f


class LevelMeterWidget(QWidget):
    """纯绘制的电平条：RMS 主条 + PEAK 细条 + 峰值保持标记 + 用户阈值线。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.level = LevelState()
        self.threshold_linear = threshold_from_db(DEFAULT_THRESHOLD_DB)
        self.title = ""
        self.status = ""
        self.warning = ""
        self.setMinimumHeight(104)

        # 显式指定中文字体族：否则绘制文字用的 QFont() 不带族名，
        # 在缺少默认字体配置的环境下中文会渲染成方块（tofu）。
        self.setFont(_pick_cjk_font())
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def _font(self, size: int, bold: bool = False) -> QFont:
        """从控件自身字体派生，保证继承字体族（含中文支持）。"""
        f = QFont(self.font())
        f.setPointSize(size)
        f.setBold(bold)
        return f

    def set_level(self, level: LevelState) -> None:
        self.level = level
        self.update()

    def set_threshold(self, threshold_linear: float) -> None:
        self.threshold_linear = threshold_linear
        self.update()

    def set_texts(self, title: str, status: str, warning: str = "") -> None:
        self.title = title
        self.status = status
        self.warning = warning
        self.update()

    # ------------------------------------------------------------------ #
    def _x_for_db(self, db: float, x0: float, x1: float) -> float:
        db = max(DB_MIN, min(DB_MAX, db))
        return x0 + (db - DB_MIN) / (DB_MAX - DB_MIN) * (x1 - x0)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt 命名
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 先清成真透明：否则单独 grab() 时会叠在不透明底色上，半透明黑变成灰白块
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        w, h = self.width(), self.height()
        pad = 10.0
        x0, x1 = pad, max(pad + 10.0, w - pad)
        avail = int(x1 - x0)

        # 背景
        p.setPen(Qt.NoPen)
        p.setBrush(COLOR_BG)
        p.drawRoundedRect(0, 0, w - 1, h - 1, 8, 8)

        fm_title = QFontMetrics(self._font(9, bold=True))
        fm_small = QFontMetrics(self._font(8))

        # ---- ① 标题行：标题（左，超长省略）+ 状态（右）----
        p.setFont(self._font(9, bold=True))
        p.setPen(COLOR_TEXT)
        status = self.status or ""
        status_w = fm_title.horizontalAdvance(status) if status else 0
        title_room = max(60, avail - status_w - 12)
        p.drawText(
            int(x0), 3, title_room, 15, Qt.AlignLeft | Qt.AlignVCenter,
            fm_title.elidedText(self.title or "—", Qt.ElideMiddle, title_room),
        )
        if status:
            p.setFont(self._font(9))
            p.setPen(COLOR_TEXT_DIM)
            p.drawText(int(x1) - status_w - 2, 3, status_w + 2, 15,
                       Qt.AlignRight | Qt.AlignVCenter, status)

        # ---- ② RMS 主条 ----
        rms_y, rms_h = 22.0, 18.0
        self._draw_bar(p, x0, x1, rms_y, rms_h, self.level.db_rms)

        # ---- ③ PEAK 细条 ----
        pk_y, pk_h = 44.0, 7.0
        self._draw_bar(p, x0, x1, pk_y, pk_h, self.level.db_peak)

        # ---- 峰值保持标记 ----
        if self.level.peak_hold > 0:
            hx = self._x_for_db(self.level.db_peak_hold, x0, x1)
            p.setPen(QPen(COLOR_HOLD, 2))
            p.drawLine(int(hx), int(pk_y - 2), int(hx), int(pk_y + pk_h + 2))

        # ---- 阈值线（用户可调）----
        tdb = to_db(self.threshold_linear)
        tx = self._x_for_db(tdb, x0, x1)
        p.setPen(QPen(COLOR_THRESHOLD, 1.5, Qt.DashLine))
        p.drawLine(int(tx), int(rms_y - 3), int(tx), int(pk_y + pk_h + 3))

        # ---- ④ 刻度（单独一行，与数值读数分开，避免压字）----
        p.setPen(COLOR_TEXT_DIM)
        p.setFont(self._font(7))
        tick_y = pk_y + pk_h + 4
        for db in (-60, -40, -20, -12, -6, 0):
            x = self._x_for_db(db, x0, x1)
            p.drawLine(int(x), int(tick_y), int(x), int(tick_y + 3))
            p.drawText(int(x) - 12, int(tick_y + 4), 24, 10, Qt.AlignCenter, f"{db}")

        # ---- ⑤ 数值读数（独立一行，右对齐）----
        p.setFont(self._font(8))
        p.setPen(COLOR_TEXT)
        readout = f"RMS {self.level.db_rms:6.1f} dB    PEAK {self.level.db_peak:6.1f} dB"
        if self.level.peak_hold > 0:
            readout += f"    HOLD {self.level.db_peak_hold:6.1f} dB"
        readout = fm_small.elidedText(readout, Qt.ElideRight, avail)
        p.drawText(int(x0), int(tick_y + 16), avail, 13, Qt.AlignLeft | Qt.AlignVCenter, readout)

        # ---- ⑥ 警告（独占一行，否则会压住标题）----
        if self.warning:
            p.setFont(self._font(8))
            p.setPen(COLOR_WARN)
            p.drawText(
                int(x0), int(tick_y + 30), avail, 14, Qt.AlignLeft | Qt.AlignVCenter,
                fm_small.elidedText("⚠ " + self.warning, Qt.ElideRight, avail),
            )

    def _draw_bar(self, p: QPainter, x0: float, x1: float, y: float, height: float, db: float) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(COLOR_TRACK)
        p.drawRoundedRect(int(x0), int(y), int(x1 - x0), int(height), height / 2, height / 2)

        if db <= MIN_DB + 0.01:
            return
        fx = self._x_for_db(db, x0, x1)
        if fx <= x0:
            return
        p.setBrush(_level_color(db))
        p.drawRoundedRect(int(x0), int(y), int(fx - x0), int(height), height / 2, height / 2)


class _Bridge(QObject):
    """把采集线程的数据搬到 UI 线程（Qt 信号跨线程是队列投递，安全）。"""

    level = Signal(object)


class LevelMeterWindow(QWidget):
    """浮在游戏上的电平条。无边框 / 透明 / 置顶 / 可点击穿透。"""

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("听·显·译 电平表")
        self.resize(460, 108)

        self.meter = LevelMeterWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.meter)

        self._drag_from: QPoint | None = None
        self._click_through = False

        # 定时重申置顶：对抗游戏抢 Z 序（计划书第 2.4 节）
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._reassert_topmost)
        self._topmost_timer.start(2000)

    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self.apply_native_flags)

    def apply_native_flags(self) -> None:
        hwnd = win32.hwnd_of(self)
        if not hwnd:
            return
        win32.set_no_activate(hwnd, True)
        win32.set_taskbar_visible(hwnd, False)
        win32.set_click_through(hwnd, self._click_through)
        win32.reassert_topmost(hwnd)

    def _reassert_topmost(self) -> None:
        hwnd = win32.hwnd_of(self)
        if hwnd:
            win32.reassert_topmost(hwnd)

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = enabled
        hwnd = win32.hwnd_of(self)
        if hwnd:
            win32.set_click_through(hwnd, enabled)
        log.info("点击穿透: %s", "开" if enabled else "关")

    # 拖动（未开穿透时）
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        self._drag_from = None


class MeterControlWindow(QWidget):
    """阈值滑杆等控件。**故意独立于悬浮窗**，否则开了穿透就点不动。"""

    def __init__(self, meter_window: LevelMeterWindow) -> None:
        super().__init__(None)
        self.meter_window = meter_window
        self.setWindowTitle("听·显·译 — 电平表控制")
        self.resize(430, 250)

        self.target_label = QLabel("目标: 解析中…")
        self.target_label.setWordWrap(True)

        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(int(THRESHOLD_DB_MIN))
        self.threshold_slider.setMaximum(int(THRESHOLD_DB_MAX))
        self.threshold_slider.setValue(int(DEFAULT_THRESHOLD_DB))
        self.threshold_value = QLabel(f"{DEFAULT_THRESHOLD_DB:.0f} dBFS")

        self.auto_gain = QCheckBox("自动增益补偿（按会话音量放大，最高 +24 dB）")
        self.click_through = QCheckBox("点击穿透（鼠标穿透到游戏）")

        self.volume_label = QLabel("会话音量: —")

        self.quit_btn = QPushButton("关闭")

        root = QVBoxLayout(self)
        root.addWidget(self.target_label)
        root.addWidget(self.volume_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("静音阈值"))
        row.addWidget(self.threshold_slider, 1)
        row.addWidget(self.threshold_value)
        root.addLayout(row)

        hint = QLabel(
            "阈值＝低于此电平就算“没声音”。往左调更灵敏（小声也算有声），"
            "往右调更严格（能过滤底噪）。"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        root.addWidget(self.auto_gain)
        root.addWidget(self.click_through)
        root.addWidget(self.warn_label)
        root.addStretch(1)
        root.addWidget(self.quit_btn)

        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.click_through.toggled.connect(self.meter_window.set_click_through)
        self.quit_btn.clicked.connect(self.close)

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_value.setText(f"{value} dBFS")
        linear = threshold_from_db(float(value))
        self.meter_window.meter.set_threshold(linear)

    def threshold_linear(self) -> float:
        return threshold_from_db(float(self.threshold_slider.value()))

    def auto_gain_enabled(self) -> bool:
        return self.auto_gain.isChecked()


def run_meter(pid: int, threshold_db: float | None = None) -> int:
    """打开电平表（悬浮窗 + 控制窗），实时显示目标程序音频的 RMS/PEAK。"""
    from PySide6.QtWidgets import QApplication

    target = resolve_target(TargetSpec(pid=pid))
    if target is None:
        print(f"未找到 pid={pid} 的活跃音频会话，无法显示电平表。")
        print("提示：先运行 `python main.py --list-audio` 查看正在发声的进程与 PID。")
        return 2

    # 注意：**不要**在这里调用 win32.enable_dpi_awareness()。
    # 进程 DPI 感知只能设置一次；本模块在顶部就 import 了 PySide6，
    # Qt 初始化时会自己设成默认的 PerMonitorV2。我们先设一遍会让 Qt 设置失败
    # 并打印 "SetProcessDpiAwarenessContext() failed: 拒绝访问"。
    # Qt6 的默认值已经是我们想要的，交给它即可。
    app = QApplication.instance() or QApplication([])

    meter_window = LevelMeterWindow()
    control = MeterControlWindow(meter_window)
    if threshold_db is not None:
        control.threshold_slider.setValue(int(threshold_db))

    tracker = LevelTracker(silence_threshold=control.threshold_linear())
    bridge = _Bridge()
    bridge.level.connect(meter_window.meter.set_level)

    pipeline_holder: dict = {}

    def on_chunk(chunk) -> None:
        dt = float(chunk.size) / 16000.0
        state = tracker.update_with_dt(chunk, dt)
        bridge.level.emit(state)

    from app.audio.pipeline import AudioPipeline

    pipeline = AudioPipeline(out_rate=16000, silence_threshold=control.threshold_linear())
    pipeline_holder["pipeline"] = pipeline

    worker = CaptureWorker(
        TargetSpec(pid=pid), pipeline=pipeline, on_chunk=on_chunk, follow=True
    )

    control.target_label.setText(
        f"目标: {target.name} (PID {target.pid})\n{target.executable}"
    )

    started_at = time.time()

    def refresh_status() -> None:
        st = worker.snapshot()
        pipeline.set_silence_threshold(control.threshold_linear())
        tracker.set_threshold(control.threshold_linear())

        # 会话音量 / 自动增益（实测：采集幅度按会话音量线性缩放）
        vol = get_session_volume(pid)
        if vol is None:
            control.volume_label.setText("会话音量: 读取失败")
        else:
            volume, muted = vol
            control.volume_label.setText(
                f"会话音量: {volume * 100:.0f}%" + ("（已静音！）" if muted else "")
            )
            if control.auto_gain_enabled():
                gain = suggest_gain(volume)
                pipeline.set_gain(gain)
                control.volume_label.setText(
                    control.volume_label.text() + f"   自动增益 ×{gain:.2f}"
                )

        if not st.running:
            status = "未连接（目标没在放音？）"
        elif st.silent_seconds > 1.0:
            status = f"静音 {st.silent_seconds:.1f}s"
        else:
            status = "有声"

        warning = ""
        if vol is not None and vol[1]:
            warning = "该程序在音量合成器里被静音了，采不到声音"
        elif vol is not None and vol[0] < 0.15 and st.running and st.silent_seconds > 3:
            warning = f"会话音量仅 {vol[0] * 100:.0f}%，建议调大或勾选自动增益"

        meter_window.meter.set_texts(
            f"{target.name}  (PID {target.pid})   已运行 {time.time() - started_at:.0f}s",
            status,
            warning,
        )
        control.warn_label.setText(("⚠️ " + warning) if warning else "")

    status_timer = QTimer()
    status_timer.timeout.connect(refresh_status)
    status_timer.start(250)

    control.quit_btn.clicked.connect(app.quit)
    meter_window.show()
    control.show()
    worker.start()

    print(f"电平表已打开，目标: {target.name} (PID {target.pid})")
    print("控制窗里可调静音阈值、开点击穿透、开自动增益。关闭控制窗即退出。")

    try:
        code = app.exec()
    finally:
        worker.stop()
    return int(code)
