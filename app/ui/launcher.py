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
    QRadioButton,
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

        # 音源模式：进程 / 浏览器标签页
        self.mode_process = QRadioButton("程序进程（小说软件 / 播放器）")
        self.mode_process.setChecked(True)
        self.mode_tab = QRadioButton("浏览器标签页（只听某一个标签页）")
        self.mode_tab.setToolTip(
            "浏览器把整个实例的音频混在一起，操作系统层分不出标签页。\n"
            "所以这条模式需要装一个很小的浏览器扩展，由它把目标标签页的音频\n"
            "经本机回环（127.0.0.1）送过来。详见「扩展与安装说明」。"
        )
        self.mode_process.toggled.connect(self._sync_mode)
        self.mode_tab.toggled.connect(self._sync_mode)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_process)
        mode_row.addWidget(self.mode_tab)
        mode_row.addStretch(1)

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
        self.guide_btn = QPushButton("扩展与安装说明…")
        self.guide_btn.setToolTip("浏览器标签页模式需要先装一个扩展；点这里看步骤")
        self.guide_btn.clicked.connect(self._open_extension_guide)
        self.guide_btn.hide()  # 只有标签页模式才需要

        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        row.addWidget(self.guide_btn)
        row.addWidget(self.meter_btn)
        row.addWidget(self.wizard_btn)
        row.addWidget(self.settings_btn)
        row.addWidget(self.start_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.hint)
        lay.addLayout(mode_row)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.empty_hint)
        lay.addLayout(row)

        # 每 3 秒自动刷新：用户经常是"先开程序再开小说"，手动刷新很烦
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)
        self.refresh()
        self._sync_mode()

    # ------------------------------------------------------------------ #
    def _sync_mode(self) -> None:
        """按音源模式切换界面：进程模式看列表，标签页模式看说明。"""
        tab_mode = self.mode_tab.isChecked()
        self.list.setVisible(not tab_mode)
        self.empty_hint.setVisible(not tab_mode)
        self.guide_btn.setVisible(tab_mode)
        self.meter_btn.setEnabled(not tab_mode)
        self.meter_btn.setToolTip(
            "电平表基于进程采集；标签页模式的电平请看字幕窗里的状态" if tab_mode else ""
        )
        self.refresh_btn.setEnabled(not tab_mode)
        if tab_mode:
            self.hint.setText(
                "浏览器标签页模式：<b>只字幕你指定的那一个标签页</b>，别的标签页照常有声。<br>"
                "① 先点「开始字幕」；② 在浏览器里切到要字幕的标签页；"
                "③ 点工具栏里的扩展图标（或按 Ctrl+Shift+U）。"
            )
            self.start_btn.setText("开始字幕（浏览器标签页）")
            self.empty_hint.setText("")
        else:
            self.hint.setText(
                "选一个正在播放的<b>小说软件</b>，点「开始字幕」。<br>"
                "游戏的声音不会被采集——程序只监听你选中的这一个进程。"
            )
            self.start_btn.setText("开始字幕")
            self.refresh()

    def refresh(self) -> None:
        if self.mode_tab.isChecked():
            return  # 标签页模式不看进程列表，别再无谓地枚举音频会话
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
        if self.mode_tab.isChecked():
            self._start_tab_mode()
            return
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

    def _start_tab_mode(self) -> None:
        """浏览器标签页模式：起字幕子进程，音频等扩展送来。"""
        proc = self._spawn_child(["--tab"])
        if proc is None:
            return
        self._timer.stop()
        self.start_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.mode_tab.setEnabled(False)
        self.mode_process.setEnabled(False)
        self.hint.setText(
            "字幕已启动（浏览器标签页模式）。<br>"
            "请在浏览器里切到要字幕的<b>那个标签页</b>，再点工具栏里的扩展图标"
            "（或按 <b>Ctrl+Shift+U</b>）——浏览器规定必须由你亲手触发一次。"
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
        self.mode_tab.setEnabled(True)
        self.mode_process.setEnabled(True)
        self._timer.start(3000)
        self._sync_mode()
        if self.mode_tab.isChecked():
            self.empty_hint.setText(
                f"❌ 字幕进程启动后立刻退出了（exit {code}）。<br>"
                "常见原因：端口 38991 被别的程序占用（设置里可改）、或配置有误。"
                "日志见 data\\logs\\lst.log。"
            )
            return
        self.hint.setText("选一个正在播放的<b>小说软件</b>，点「开始字幕」。")
        self.empty_hint.setText(
            f"❌ 字幕进程启动后立刻退出了（exit {code}）。<br>"
            "常见原因：这个进程其实没在发声（浏览器/Electron 选错子进程）、"
            "或者配置有误。日志见 data\\logs\\lst.log。"
        )

    def _open_extension_guide(self) -> None:
        """浏览器标签页模式的扩展安装说明。

        为什么必须有这一步：浏览器规定"取标签页音频"要由用户亲手触发过扩展
        （activeTab / kTabCaptureForTab 是按标签页授予的），程序代替不了；
        而扩展必须由用户自己侧载——我们能做的是把路径和步骤摆清楚。
        """
        import os
        from pathlib import Path

        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QMessageBox

        from app.paths import ROOT

        ext_dir = Path(ROOT) / "browser_extension"
        box = QMessageBox(self)
        box.setWindowTitle("浏览器标签页：扩展安装说明")
        box.setTextFormat(Qt.RichText)
        box.setText(
            "<b>为什么要装扩展：</b>浏览器把整个实例的音频混成一个流，"
            "操作系统层分不出标签页（实测两个标签页同时出声，音频会话仍只有 1 个）。"
            "标签页边界只有浏览器内部知道，所以由这个小扩展把目标标签页的音频"
            "经 <code>127.0.0.1</code> 送过来。<br><br>"
            "<b>步骤：</b><br>"
            "① 地址栏打开 <code>edge://extensions</code>（Chrome 是 <code>chrome://extensions</code>），"
            "打开左下角「开发人员模式」；<br>"
            "② 点「加载解压缩的扩展」，选中下面这个文件夹；<br>"
            f"<code>{ext_dir}</code><br>"
            "③ 回到本程序点「开始字幕」；<br>"
            "④ 在浏览器里切到要字幕的标签页，点工具栏里的扩展图标"
            "（或按 <b>Ctrl+Shift+U</b>）。<br><br>"
            "<b>注意：</b>第 ④ 步必须你亲手做——浏览器要求由用户触发一次"
            "（授权是按标签页给的），这是它的安全策略，不是本程序的 bug。"
            "换标签页需要再触发一次。<br>"
            "音频只走本机回环，不联网、不上传。"
        )
        open_btn = box.addButton("打开扩展文件夹", QMessageBox.ActionRole)
        copy_btn = box.addButton("复制 edge://extensions", QMessageBox.ActionRole)
        box.addButton("关闭", QMessageBox.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked is open_btn:
            try:
                if os.name == "nt":
                    os.startfile(str(ext_dir))  # noqa: S606 - 打开用户自己的目录
                else:
                    from PySide6.QtCore import QUrl
                    from PySide6.QtGui import QDesktopServices

                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(ext_dir)))
            except Exception as exc:  # noqa: BLE001
                self.empty_hint.setText(f"打不开目录：{exc}（路径：{ext_dir}）")
        elif clicked is copy_btn:
            QGuiApplication.clipboard().setText("edge://extensions")
            self.empty_hint.setText("已复制 edge://extensions，粘到地址栏即可。")

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
