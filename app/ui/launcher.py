"""主窗口：挑一个正在发声的程序，开始字幕。

双击「启动.bat」后看到的就是这个窗口。为什么不直接开悬浮窗——
**因为必须先知道要听谁的声音**。这是整个程序唯一的必填项。

列表直接用 Windows 音频会话 API 的结果（`app/audio/process_list.py`），
也就是音量合成器里那些条目，所以"这里能看到的 = 音量合成器里能看到的"。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import TargetSpec
from app.audio.process_list import enumerate_audio_processes
from app.config import AppConfig
from app.utils.log import get_logger

log = get_logger(__name__)


class LauncherWindow(QWidget):
    """主窗口。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__(None)
        self.config = config or AppConfig.load()
        self.setWindowTitle("听·显·译 — 选择音频来源")
        self.resize(560, 420)

        self.hint = QLabel(
            "选一个正在播放的<b>小说软件</b>，点「开始字幕」。<br>"
            "游戏的声音不会被采集——程序只监听你选中的这一个进程。"
        )
        self.hint.setWordWrap(True)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemDoubleClicked.connect(lambda _i: self._start())
        self.empty_hint = QLabel("")
        self.empty_hint.setWordWrap(True)

        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh)
        self.start_btn = QPushButton("开始字幕")
        self.start_btn.clicked.connect(self._start)
        self.settings_btn = QPushButton("设置…")
        self.settings_btn.clicked.connect(self._open_settings)
        self.meter_btn = QPushButton("电平表")
        self.meter_btn.clicked.connect(self._open_meter)

        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(self.meter_btn)
        row.addWidget(self.settings_btn)
        row.addWidget(self.start_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.hint)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.empty_hint)
        lay.addLayout(row)

        # 每 3 秒自动刷新：用户经常是"先开程序再开小说"，手动刷新很烦
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        try:
            procs = enumerate_audio_processes(include_inactive=False)
        except Exception as exc:  # noqa: BLE001
            self.empty_hint.setText(f"❌ 枚举失败：{exc}")
            return

        current = self._selected_pid()
        self.list.clear()
        for p in procs:
            if p.is_system_sounds:
                continue
            item = QListWidgetItem(f"{p.name}    PID {p.pid}" + (f"    {p.title}" if p.title else ""))
            item.setData(Qt.UserRole, p.pid)
            item.setToolTip(p.executable or p.display_label)
            self.list.addItem(item)
            if p.pid == current:
                item.setSelected(True)

        if self.list.count() == 0:
            self.empty_hint.setText(
                "当前没有任何程序在输出音频。<br>"
                "请先让小说软件<b>开始播放</b>，再回来看列表——只有真正在发声的进程才会出现。"
            )
        else:
            self.empty_hint.setText(
                "提示：浏览器 / Electron 类应用有多个同名子进程，"
                "如果选错了会一直静音，换一个同名的试试。"
            )

    def _selected_pid(self) -> int | None:
        items = self.list.selectedItems()
        if not items:
            return None
        value = items[0].data(Qt.UserRole)
        return int(value) if value is not None else None

    # ------------------------------------------------------------------ #
    def _spawn_child(self, args: list[str]) -> None:
        """用**独立子进程**启动字幕/电平表。

        为什么必须是独立进程，而不是在本进程里再建窗口：

        1. 同一个 QApplication 里再调 ``app.exec()`` 会报
           ``QCoreApplication::exec: The event loop is already running``；
        2. 更糟的是 exec 失败后 ``run_subtitles`` 的 ``finally: pipeline.stop()``
           会**立刻执行**，日志表现为「开始采集 → 采集线程已停止」，
           用户看到的就是"点了没反应 / 采集不到音频"；
        3. 独立进程还能隔离崩溃：字幕进程挂了不会带走主窗口。

        优先用 ``pythonw.exe``（无控制台窗口）；找不到就退回 python.exe。
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        from app.paths import ROOT

        py = Path(sys.executable)
        pythonw = py.with_name("pythonw.exe")
        exe = pythonw if pythonw.exists() else py

        flags = 0
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [str(exe), str(ROOT / "main.py"), *args]
        log.info("启动子进程：%s", " ".join(cmd))
        try:
            subprocess.Popen(cmd, cwd=str(ROOT), creationflags=flags, close_fds=True)
        except Exception as exc:  # noqa: BLE001
            self.empty_hint.setText(f"❌ 启动失败：{exc}")

    def _start(self) -> None:
        pid = self._selected_pid()
        if pid is None:
            self.empty_hint.setText("请先在列表里选一个程序。")
            return
        self._spawn_child(["--run", str(pid)])
        # 字幕窗由子进程负责；主窗口先藏起来，免得两个窗口叠在一起
        self.hide()

    def _open_meter(self) -> None:
        pid = self._selected_pid()
        if pid is None:
            self.empty_hint.setText("请先选一个程序，再打开电平表。")
            return
        self._spawn_child(["--meter", str(pid)])

    def _open_settings(self) -> None:
        from app.ui.settings import SettingsWindow

        win = getattr(self, "_settings_win", None)
        if win is not None and win.isVisible():
            win.raise_()
            return
        self._settings_win = SettingsWindow(self.config)
        self._settings_win.show()


def run_launcher(config: AppConfig | None = None) -> int:
    """打开主窗口。返回退出码。"""
    from PySide6.QtWidgets import QApplication

    cfg = config or AppConfig.load()

    # 首次运行：先走向导，用户取消也不拦着（还能在设置里补）
    if not cfg.first_run_done:
        from app.ui.wizard import run_wizard

        run_wizard(cfg)
        cfg = AppConfig.load()

    app = QApplication.instance() or QApplication([])
    win = LauncherWindow(cfg)
    win.show()
    return int(app.exec())
