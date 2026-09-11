"""主程序：把字幕悬浮窗、控制窗和流水线接起来。

窗口布局刻意分成两个：

- **悬浮字幕窗**：浮在游戏上，可点击穿透 —— 一旦穿透就点不动了
- **控制窗**：普通窗口，放显示模式、点击穿透、暂停、统计

把控制项放在悬浮窗里是新手常犯的错：用户一开穿透就把自己锁死了。
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import TargetSpec, resolve_target
from app.config import AppConfig
from app.pipeline import SubtitlePipeline
from app.ui.subtitle_overlay import SubtitleOverlay
from app.utils import win32
from app.utils.log import get_logger

log = get_logger(__name__)


class SubtitleControlWindow(QWidget):
    """控制窗：所有可点的东西都放这里。"""

    def __init__(self, overlay: SubtitleOverlay, pipeline: SubtitlePipeline) -> None:
        super().__init__(None)
        self.overlay = overlay
        self.pipeline = pipeline
        self.setWindowTitle("听·显·译 — 控制")
        # 用布局的 sizeHint 决定初始尺寸，避免在高 DPI 屏上被压缩导致内容被裁
        self.setMinimumWidth(420)
        self.resize(480, 380)

        self.target_label = QLabel("目标: 解析中…")
        self.target_label.setWordWrap(True)

        self.status_label = QLabel("状态: 启动中…")
        self.status_label.setWordWrap(True)

        self.stats_label = QLabel("统计: —")
        self.stats_label.setWordWrap(True)

        self.hint_label = QLabel(
            "提示：开着点击穿透时本悬浮窗点不动，请在这里取消穿透。\n"
            "真·独占全屏游戏无法被普通窗口覆盖，请把游戏设为「无边框窗口全屏」。"
        )
        self.hint_label.setWordWrap(True)

        # 显示模式
        self.mode_row = QHBoxLayout()
        self.mode_row.addWidget(QLabel("显示:"))
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, label in (("source", "原文"), ("target", "译文"), ("bilingual", "双语")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(overlay.config.display_mode == mode)
            btn.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            self.mode_buttons[mode] = btn
            self.mode_row.addWidget(btn)
        self.mode_row.addStretch(1)

        # 滚动模式
        self.scroll_row = QHBoxLayout()
        self.scroll_row.addWidget(QLabel("滚动:"))
        self.scroll_buttons: dict[str, QPushButton] = {}
        for smode, label in (("accumulate", "累积"), ("replace", "单行")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(overlay.config.scroll_mode == smode)
            btn.clicked.connect(lambda _c=False, m=smode: self._set_scroll(m))
            self.scroll_buttons[smode] = btn
            self.scroll_row.addWidget(btn)
        self.scroll_row.addStretch(1)

        self.click_through = QCheckBox("点击穿透（鼠标穿透到游戏）")
        self.click_through.setChecked(overlay.config.click_through)
        self.click_through.toggled.connect(self.overlay.set_click_through)

        self.pause_btn = QPushButton("暂停字幕")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)

        self.quit_btn = QPushButton("退出")

        root = QVBoxLayout(self)
        root.addWidget(self.target_label)
        root.addWidget(self.status_label)
        root.addLayout(self.mode_row)
        root.addLayout(self.scroll_row)
        root.addWidget(self.click_through)
        root.addWidget(self.stats_label)
        root.addWidget(self.hint_label)
        root.addStretch(1)
        root.addWidget(self.pause_btn)
        root.addWidget(self.quit_btn)

        # 按内容自适应尺寸（高 DPI 下字体放大，固定尺寸会把按钮挤掉）
        self.adjustSize()

    # ------------------------------------------------------------------ #
    def _set_mode(self, mode: str) -> None:
        for m, b in self.mode_buttons.items():
            b.setChecked(m == mode)
        self.overlay.set_mode(mode)

    def _set_scroll(self, smode: str) -> None:
        for m, b in self.scroll_buttons.items():
            b.setChecked(m == smode)
        self.overlay.config.scroll_mode = smode  # type: ignore[assignment]
        self.overlay._relayout()
        self.overlay.update()

    def _toggle_pause(self, paused: bool) -> None:
        self.pause_btn.setText("继续字幕" if paused else "暂停字幕")
        self._paused = paused

    def is_paused(self) -> bool:
        return getattr(self, "_paused", False)

    def update_stats(self, payload: dict) -> None:
        hub = payload.get("hub", {})
        cache = payload.get("cache", {})
        self.stats_label.setText(
            f"语种={payload.get('language') or '判定中'} · 引擎={payload.get('engine') or '-'}\n"
            f"识别延迟 P50 {payload.get('asr_p50_ms', 0):.0f}ms / P90 {payload.get('asr_p90_ms', 0):.0f}ms · "
            f"翻译 P50 {payload.get('translate_p50_ms', 0):.0f}ms\n"
            f"字幕 {hub.get('segments', 0)} 条 · 缓存命中 {hub.get('cache_hits', 0)} · "
            f"失败 {hub.get('failures', 0)} · 切通道 {hub.get('fallbacks', 0)}\n"
            f"token {hub.get('prompt_tokens', 0)}+{hub.get('completion_tokens', 0)} · "
            f"缓存 {cache.get('entries', 0)} 条"
        )


def run_subtitles(pid: int | None = None, process_name: str = "", config: AppConfig | None = None) -> int:
    """启动字幕程序。返回进程退出码。"""
    cfg = config or AppConfig.load()
    spec = TargetSpec(pid=pid) if pid else TargetSpec(process_name=process_name)

    target = resolve_target(spec)
    if target is None:
        print(f"未找到活跃音频会话：{spec.describe()}")
        print("提示：先运行 `python main.py --list-audio` 查看正在发声的进程与 PID。")
        print("      注意：浏览器/Electron 应用要选**真正发声的那个子进程**。")
        return 2

    # Qt 自己会设置 DPI 感知，这里不要抢（见 app/utils/win32.py 的说明）
    app = QApplication.instance() or QApplication([])

    overlay = SubtitleOverlay(cfg.overlay)
    pipeline = SubtitlePipeline(cfg)
    control = SubtitleControlWindow(overlay, pipeline)

    # ---- 信号接线：状态只在 UI 线程被改 ----
    def on_partial(text: str, _lang: str) -> None:
        if control.is_paused():
            return
        pipeline.state.set_partial(text)
        overlay.update_state(pipeline.state)

    def on_final(line_id: int, text: str, language: str) -> None:
        if control.is_paused():
            return
        pipeline.state.add_final(text, language, line_id=line_id)
        overlay.update_state(pipeline.state)

    def on_translation(line_id: int, translation: str, error: str) -> None:
        if control.is_paused():
            return
        pipeline.state.set_translation(line_id, translation, error)
        overlay.update_state(pipeline.state)

    def on_status(text: str) -> None:
        control.status_label.setText(f"状态: {text}")
        overlay.set_status(text)

    def on_error(text: str) -> None:
        control.status_label.setText(f"⚠️ {text}")
        overlay.set_status(f"⚠️ {text}")

    pipeline.partialReceived.connect(on_partial)
    pipeline.finalReceived.connect(on_final)
    pipeline.translationReceived.connect(on_translation)
    pipeline.statusChanged.connect(on_status)
    pipeline.errorOccurred.connect(on_error)
    pipeline.statsChanged.connect(control.update_stats)

    control.target_label.setText(f"目标: {target.name} (PID {target.pid})\n{target.executable}")

    ok, note = pipeline.prepare()
    on_status(note or "就绪")

    control.quit_btn.clicked.connect(app.quit)
    overlay.show()
    control.show()

    if not pipeline.start(spec):
        control.status_label.setText("状态: 启动失败")

    # 字幕落定后滚动到底（累积模式下自动往下走）
    scroll_timer = QTimer()
    scroll_timer.timeout.connect(lambda: overlay.update())
    scroll_timer.start(500)

    print(f"字幕已启动，目标: {target.name} (PID {target.pid})")
    print("控制窗里可切显示模式/穿透/暂停。关掉控制窗即退出。")

    try:
        return int(app.exec())
    finally:
        pipeline.stop()
