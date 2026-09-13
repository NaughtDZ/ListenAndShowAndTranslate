"""主程序：把字幕悬浮窗、控制窗和流水线接起来。

窗口布局刻意分成两个：

- **悬浮字幕窗**：浮在游戏上，可点击穿透 —— 一旦穿透就点不动了
- **控制窗**：普通窗口，放显示模式、点击穿透、暂停、统计

把控制项放在悬浮窗里是新手常犯的错：用户一开穿透就把自己锁死了。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import TargetSpec, resolve_target
from app.config import AppConfig
from app.pipeline import SubtitlePipeline
from app.ui.lifecycle import configure_quit_policy
from app.ui.subtitle_overlay import SubtitleOverlay
from app.utils import win32
from app.utils.log import get_logger

if TYPE_CHECKING:  # 只为类型标注，运行时不导入（设置窗是懒加载的）
    from app.ui.settings import SettingsWindow

log = get_logger(__name__)


def capture_level(pipeline: SubtitlePipeline) -> tuple[float, float] | None:
    """设置窗里"实时电平"的读数来源：音源管线每块已经算好的 RMS / 峰值。

    为什么不用 ``capture.snapshot()``：那里的 ``peak`` 是**整段采集的高水位**，
    拿它画 PEAK 条会一直顶在最右边。这里读的是**最近一块**的 RMS / 峰值。

    进程模式与浏览器标签页模式共用同一套 AudioPipeline，所以这里统一走
    ``pipeline.pipeline_stats()``。返回 None 表示当前没有在采集。
    """
    stats = pipeline.pipeline_stats()
    if stats is None:
        return None
    return float(stats.last_rms), float(stats.last_peak)


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

        # 整体透明度：快速调到"看得清但不抢眼"
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(int(overlay.config.window_opacity * 100))
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("透明度"))
        op_row.addWidget(self.opacity_slider, 1)
        op_row.addWidget(self.opacity_label)

        self.lock_pos = QCheckBox("锁定位置（防止玩游戏时误拖）")
        self.lock_pos.setChecked(overlay.config.lock_position)
        self.lock_pos.toggled.connect(self._on_lock)
        # 锁定后就不能拖；而"能拖"的前提是没开点击穿透——在提示里说清
        self.drag_hint = QLabel(
            "取消勾选后<b>拖动字幕窗任意位置</b>可移动、拖<b>边缘或右下角</b>可缩放尺寸。"
            "注意：开了点击穿透就拖不动了。"
        )
        self.drag_hint.setWordWrap(True)

        self.pause_btn = QPushButton("暂停字幕")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)

        self.source_btn = QPushButton("换音频来源…")
        self.source_btn.setToolTip(
            "运行中换音源：换成别的程序，或换成「浏览器标签页」。\n"
            "换的时候识别/翻译引擎会重建，约 1~2 秒空档，字幕窗不会关。"
        )
        self.source_btn.clicked.connect(self._switch_source)

        self.tray_btn = QPushButton("最小化到托盘")
        self.settings_btn = QPushButton("设置…")

        self.quit_btn = QPushButton("退出")

        root = QVBoxLayout(self)
        root.addWidget(self.target_label)
        root.addWidget(self.status_label)
        root.addLayout(self.mode_row)
        root.addLayout(self.scroll_row)
        root.addWidget(self.click_through)
        root.addLayout(op_row)
        root.addWidget(self.lock_pos)
        root.addWidget(self.drag_hint)
        root.addWidget(self.stats_label)
        root.addWidget(self.hint_label)
        root.addStretch(1)
        root.addWidget(self.pause_btn)
        root.addWidget(self.source_btn)
        root.addWidget(self.tray_btn)
        root.addWidget(self.settings_btn)
        root.addWidget(self.quit_btn)

        self._setup_tray()

        # 按内容自适应尺寸（高 DPI 下字体放大，固定尺寸会把按钮挤掉）
        self.adjustSize()

    def _open_settings(self) -> "SettingsWindow":
        """打开设置窗，返回该窗口。**不能放在悬浮窗里**——开了点击穿透就点不动了。

        复用同一个实例：关掉设置窗只是 ``hide()``，再点「设置…」还是它；
        新建的话会堆出一打藏起来的窗口，而销毁旧窗口会带走还在跑的测试线程。
        """
        from app.ui.lifecycle import open_settings_window

        # 保存后立即应用，而不是让用户重启程序（用户明确反馈过这点）
        win = open_settings_window(
            self,
            self.pipeline.config,
            on_saved=self._apply_settings_live,
            # 设置窗里的电平表要有实时读数（用户反馈"启动时给了电平表，设置里不给"）
            options={"level_source": lambda: capture_level(self.pipeline)},
        )
        # 拉宽字幕窗时字号会跟着变大：把设置窗里的"字号"框接上，
        # 用户正开着设置窗时也能看到新值（否则要关掉再开才刷新）
        self.overlay._font_spin = win.font_size
        return win

    def _apply_settings_live(self) -> None:
        """设置保存后立刻生效。

        · 外观类（字号/透明度/显示模式/滚动/描边/宽度）→ 直接刷悬浮窗
        · 识别与翻译类 → 重建引擎（短暂空档），否则用户改了语言/通道却
          发现"设置没保存"（其实是没生效）
        """
        cfg = self.pipeline.config
        self.overlay.apply_config(cfg.overlay)
        self._sync_toggles()
        self.status_label.setText("设置已应用，正在按新配置重建识别/翻译…")
        ok = self.pipeline.reload()
        self.status_label.setText(
            "✅ 设置已生效（识别/翻译已重建）" if ok else "⚠️ 外观已生效，但引擎重建失败，请看日志"
        )

    def _switch_source(self) -> None:
        """运行中换音频来源：别的程序，或浏览器标签页。

        用户问过"为什么进了程序就调不了监听目标"——那只是历史包袱
        （启动窗口把 PID 写进命令行、子进程只有一个入口）。这里补上入口：
        停掉当前采集 → 重建引擎 → 按新音源开工，字幕窗全程不关。
        """
        from PySide6.QtWidgets import QDialog

        from app.ui.source_picker import SourcePickerDialog

        spec, tab_active = self.pipeline.current_source
        dlg = SourcePickerDialog(
            self,
            current_pid=spec.pid if spec else None,
            tab_active=bool(tab_active),
        )
        if dlg.exec() != QDialog.Accepted or dlg.choice is None:
            return

        kind, pid = dlg.choice
        self.status_label.setText("正在切换音频来源（约 1~2 秒）…")
        if kind == "tab":
            ok = self.pipeline.switch_source(tab_mode=True)
            desc = "浏览器标签页（在浏览器里点扩展图标 / Ctrl+Shift+U 才会开始送音频）"
        else:
            resolved = resolve_target(TargetSpec(pid=pid)) if pid else None
            desc = f"{resolved.name} (PID {resolved.pid})" if resolved else f"PID {pid}"
            ok = self.pipeline.switch_source(TargetSpec(pid=pid))

        # 旧字幕属于旧音源，留着会让人以为是新的
        self.pipeline.state.clear()
        self.overlay.update_state(self.pipeline.state)
        self.target_label.setText(f"目标: {desc}")
        self.status_label.setText(
            f"✅ 已切换到 {desc}" if ok else "⚠️ 切换失败，详见日志"
        )

    def _sync_toggles(self) -> None:
        """把配置里的状态同步回控制窗控件（不触发信号，避免回环）。"""
        cfg = self.pipeline.config
        for widget, value in (
            (self.click_through, cfg.overlay.click_through),
            (self.lock_pos, cfg.overlay.lock_position),
        ):
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(cfg.overlay.window_opacity * 100))
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.opacity_slider.blockSignals(False)

    # ------------------------------------------------------------------ #
    def _on_opacity(self, value: int) -> None:
        self.opacity_label.setText(f"{value}%")
        self.overlay.config.window_opacity = value / 100.0
        self.overlay._apply_opacity()

    def _on_lock(self, locked: bool) -> None:
        self.overlay.config.lock_position = locked

    # ------------------------------------------------------------------ #
    def _setup_tray(self) -> None:
        """托盘图标：开始采集后可以把控制窗收起来，不挡游戏。

        退出走托盘菜单或按钮都行；**托盘是唯一在隐藏后还能找回来的入口**，
        所以必须建好，否则用户一收起来就再也找不到控制窗了。
        """
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QMenu, QSystemTrayIcon

        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_btn.setEnabled(False)
            self.tray_btn.setToolTip("系统不支持托盘")
            return

        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(
            self.style().StandardPixmap.SP_ComputerIcon
        ))
        self.tray.setToolTip("听·显·译 — 字幕运行中")

        menu = QMenu()
        act_show = QAction("显示控制窗", self)
        act_show.triggered.connect(self._restore_from_tray)
        act_toggle = QAction("显示 / 隐藏字幕", self)
        act_toggle.triggered.connect(self._toggle_overlay_visible)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self._quit)
        menu.addAction(act_show)
        menu.addAction(act_toggle)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

        self.tray_btn.clicked.connect(self._to_tray)

    def _on_tray_activated(self, reason) -> None:
        from PySide6.QtWidgets import QSystemTrayIcon

        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self._restore_from_tray()

    def _to_tray(self) -> None:
        # 没有托盘就别藏：藏起来就再也找不回来了（托盘是唯一入口）
        if getattr(self, "tray", None) is None:
            return
        self.hide()
        self.tray.showMessage("听·显·译", "已最小化到托盘，双击图标可恢复。", 
                             self.tray.MessageIcon.Information, 3000)

    def _restore_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _toggle_overlay_visible(self) -> None:
        self.overlay.setVisible(not self.overlay.isVisible())

    def _quit(self) -> None:
        """显式退出：控制窗「退出」按钮、托盘菜单、右上角 ✕ 都走这里。"""
        from app.ui.lifecycle import quit_app, set_quitting

        if not set_quitting(self):
            return
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.hide()
        quit_app()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """右上角 ✕ = 退出程序（和从前一样）。

        注意：**只有这一条路是"关窗即退出"**。「最小化到托盘」走的是
        ``hide()``，不经过这里；设置窗关了也只关它自己——否则就会出现
        "把控制窗收进托盘后一关设置窗，整个程序没了"（用户实测反馈过）。
        """
        event.accept()
        self._quit()


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


def run_subtitles(
    pid: int | None = None,
    process_name: str = "",
    config: AppConfig | None = None,
    tab_mode: bool = False,
) -> int:
    """启动字幕程序。返回进程退出码。

    ``tab_mode=True`` 走浏览器标签页：音频由浏览器扩展经本机 WebSocket 送来，
    不需要 PID（``docs/浏览器标签页.md``）。
    """
    cfg = config or AppConfig.load()
    spec: TargetSpec | None = None
    target = None
    if not tab_mode:
        spec = TargetSpec(pid=pid) if pid else TargetSpec(process_name=process_name)
        target = resolve_target(spec)
        if target is None:
            # **不再"找不到就退出"**：用户完全可能先开字幕窗、再开播放器。
            # CaptureWorker 会每 2 秒重试一次，目标一开始出声就自动接上；
            # 同时控制窗里有「换音频来源…」可以立刻换一个。
            print(f"目标现在没有在输出音频：{spec.describe()}")
            print("提示：先开着字幕窗，等它开始播放会自动接上；")
            print("      也可以点控制窗里的「换音频来源…」立刻换一个。")

    # Qt 自己会设置 DPI 感知，这里不要抢（见 app/utils/win32.py 的说明）
    app = QApplication.instance() or QApplication([])
    # 退出必须显式：默认策略下"最后一个可见窗口关闭"会顺手退出整个程序，
    # 而悬浮窗是 Qt.Tool（不计数）、控制窗收进托盘后也不算可见窗口，
    # 于是"控制窗在托盘 + 关掉设置窗"= 程序自杀（用户实测反馈）。
    configure_quit_policy(app)

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

    if tab_mode:
        def on_tab_state(state: dict) -> None:
            title = state.get("tab_title") or "（等待你在浏览器里指定）"
            ext = state.get("extension_id") or "未连接"
            control.target_label.setText(
                f"目标: 浏览器标签页 — {title}\n扩展: {ext}"
            )
            if state.get("capturing"):
                control.status_label.setText(f"状态: 正在接收「{title}」的音频")
            elif state.get("connected"):
                control.status_label.setText("状态: 扩展已连接，等待开始采集")
            else:
                control.status_label.setText("状态: 等待浏览器扩展连接…")

        pipeline.tabStateChanged.connect(on_tab_state)
        control.target_label.setText("目标: 浏览器标签页（等待扩展连接…）")
        control.hint_label.setText(
            "浏览器标签页模式：<br>"
            "1. 在浏览器里切到你要字幕的那个标签页；<br>"
            "2. 点工具栏里的扩展图标（或按 Ctrl+Shift+U）。<br>"
            "浏览器规定必须由你亲手触发一次，程序代替不了——这是它的安全策略，不是 bug。<br>"
            "音频只走本机 127.0.0.1，不联网、不上传。"
        )
    else:
        if target is not None:
            control.target_label.setText(
                f"目标: {target.name} (PID {target.pid})\n{target.executable}"
            )
        else:
            control.target_label.setText(
                f"目标: {spec.describe() if spec else '未指定'}（现在没有在发声，等待中）\n"
                "可以点「换音频来源…」换一个"
            )

    ok, note = pipeline.prepare()
    on_status(note or "就绪")

    control.quit_btn.clicked.connect(control._quit)
    control.settings_btn.clicked.connect(control._open_settings)
    overlay.show()
    control.show()

    if tab_mode:
        ok = pipeline.start_tab_audio()
        if not ok:
            control.status_label.setText("状态: 启动失败（端口可能被占用，见日志）")
    elif not pipeline.start(spec):
        control.status_label.setText("状态: 启动失败")

    # 字幕落定后滚动到底（累积模式下自动往下走）
    scroll_timer = QTimer()
    scroll_timer.timeout.connect(lambda: overlay.update())
    scroll_timer.start(500)

    if tab_mode:
        print("字幕已启动（浏览器标签页模式）。")
        print("请在浏览器里切到要字幕的标签页，然后点扩展图标或按 Ctrl+Shift+U。")
    elif target is not None:
        print(f"字幕已启动，目标: {target.name} (PID {target.pid})")
    else:
        print("字幕已启动，正在等待目标开始播放…（也可以点「换音频来源…」换一个）")
    print("控制窗里可切显示模式/穿透/暂停，也可以「换音频来源…」换监听目标。")
    print("关掉控制窗即退出；想把控制窗收起来就点「最小化到托盘」。")

    try:
        return int(app.exec())
    finally:
        pipeline.stop()
