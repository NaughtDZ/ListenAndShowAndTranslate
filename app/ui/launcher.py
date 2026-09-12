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
from app.ui.lifecycle import (
    configure_quit_policy,
    open_settings_window,
    quit_app,
    set_quitting,
)
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
        self.wizard_btn = QPushButton("向导…")
        self.wizard_btn.setToolTip(
            "重新运行「首次运行向导」：改语言包（要下哪些模型）、重选档位、补下模型。\n"
            "会用你当前的配置预填，只改你想改的；已装好的模型会自动跳过下载。"
        )
        self.wizard_btn.clicked.connect(self._open_wizard)
        self.meter_btn = QPushButton("电平表")
        self.meter_btn.clicked.connect(self._open_meter)

        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(self.meter_btn)
        row.addWidget(self.wizard_btn)
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
    def _spawn_child(self, args: list[str]):
        """用**独立子进程**启动字幕/电平表。成功返回 Popen，失败返回 None。

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
            return subprocess.Popen(cmd, cwd=str(ROOT), creationflags=flags, close_fds=True)
        except Exception as exc:  # noqa: BLE001
            self.empty_hint.setText(f"❌ 启动失败：{exc}")
            return None

    def _start(self) -> None:
        pid = self._selected_pid()
        if pid is None:
            self.empty_hint.setText("请先在列表里选一个程序。")
            return
        proc = self._spawn_child(["--run", str(pid)])
        if proc is None:
            return

        # 选完音频，主窗口的任务就结束了。**不要只是 hide()**——它是普通窗口，
        # 藏起来就既看不见也点不到（`hide()` 不触发 closeEvent，程序会一直挂着），
        # 而且它又不是 Qt.Tool，托盘里还会多出一个图标。
        # 所以给它 1.5 秒自检：子进程活着就关掉自己，秒退就留在界面上报错
        # （pythonw 会把 traceback 吞掉，不这么做用户什么提示都看不到）。
        self._timer.stop()  # 别再无谓地每 3 秒枚举音频会话
        self.start_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.hint.setText(
            "字幕已启动，正在确认子进程是否正常…<br>"
            "稍后本窗口会自动关闭，字幕窗由子进程负责。"
        )
        QTimer.singleShot(1500, lambda: self._finish_start(proc))

    def _finish_start(self, proc) -> None:
        code = proc.poll()
        if code is None:
            log.info("字幕子进程 pid=%s 运行正常，关闭选择窗口", proc.pid)
            self._quit()
            return
        self.start_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self._timer.start(3000)
        self.hint.setText("选一个正在播放的<b>小说软件</b>，点「开始字幕」。")
        self.empty_hint.setText(
            f"❌ 字幕进程启动后立刻退出了（exit {code}）。<br>"
            "常见原因：这个进程其实没在发声（浏览器/Electron 选错子进程）、"
            "或者配置有误。日志见 data\\logs\\lst.log。"
        )

    def _open_meter(self) -> None:
        pid = self._selected_pid()
        if pid is None:
            self.empty_hint.setText("请先选一个程序，再打开电平表。")
            return
        self._spawn_child(["--meter", str(pid)])

    def _open_settings(self) -> None:
        """打开（复用）设置窗。**只有关这个窗不会退出程序**。"""
        open_settings_window(self, self.config)

    def _open_wizard(self) -> None:
        """重新跑首次运行向导（改语言包 / 补下载模型 / 重选档位）。

        用户用了一阵子之后想换模型，以前**没有入口**（只在 ``first_run_done``
        为假时自动跑一次），只能改配置文件——这条就是补那个入口。

        向导直接吃本窗口这份 config 对象，并且会按当前配置预填，所以：
        改完立刻生效、没改的东西不会被清回默认值。
        """
        from app.ui.wizard import run_wizard

        accepted = run_wizard(self.config) == 1
        if accepted:
            self.hint.setText(
                "✅ 向导已完成，配置已更新（模型按需下载）。"
                "现在选一个正在播放的程序，点「开始字幕」。"
            )
        self.refresh()

    def _quit(self) -> None:
        """显式退出：关主窗口 / 退出按钮都走这里。"""
        if not set_quitting(self):
            return
        quit_app()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """右上角 ✕ = 退出选择窗口（不是"只关掉这个窗"）。"""
        event.accept()
        self._quit()


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
    # 退出必须显式（见 app/ui/lifecycle.py）：不设这个的话，"关掉主窗口"
    # 会让还在看的设置窗连带失效，反过来"关掉设置窗"又会把主窗口带走。
    configure_quit_policy(app)
    win = LauncherWindow(cfg)
    win.show()
    return int(app.exec())
