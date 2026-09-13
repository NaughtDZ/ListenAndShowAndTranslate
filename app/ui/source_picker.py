"""运行中换音频来源的对话框。

用户问过："为什么只有启动时能选监听模式和目标，进了程序反而调不了？"
答案见 ``SubtitlePipeline.switch_source()`` 的注释——那只是历史包袱，不是限制。
这个对话框就是补上那个入口：控制窗里点「换音频来源…」即可，不用重启程序。

两种模式在同一张列表里：

- **浏览器标签页**：由浏览器扩展送来（需要侧载扩展，见 `docs/浏览器标签页.md`）
- **程序进程**：Windows 音频会话里**此刻正在发声**的那些（音量合成器同款）

只列"正在发声"的，和启动窗口口径一致：进程回环只对活跃音频会话有效，
列一个没在播放的程序只会让人以为坏了（它会在状态栏一直显示"等待开始播放"）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from app.audio.process_list import enumerate_audio_processes

TAB_LABEL = "浏览器标签页 · 只听某一个标签页（需要装扩展）"


class SourcePickerDialog(QDialog):
    """挑一个新的音频来源。``choice`` 给出结果：``("tab", None)`` 或 ``("process", pid)``。"""

    def __init__(
        self,
        parent=None,
        current_pid: int | None = None,
        tab_active: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("换音频来源")
        self.resize(520, 420)
        self._current_pid = current_pid
        self._tab_active = tab_active
        self._choice: tuple[str, int | None] | None = None

        self.hint = QLabel(
            "换音源会**重启识别与翻译引擎**（约 1~2 秒空档），字幕窗不会关。"
        )
        self.hint.setWordWrap(True)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemDoubleClicked.connect(lambda _i: self._accept_selected())

        self.empty = QLabel("")
        self.empty.setWordWrap(True)

        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh)

        buttons = QDialogButtonBox()
        self.ok_btn = buttons.addButton("开始监听这个来源", QDialogButtonBox.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self._accept_selected)
        buttons.rejected.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addWidget(self.hint)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.empty)
        lay.addLayout(row)
        lay.addWidget(buttons)

        self.refresh()

    # ------------------------------------------------------------------ #
    @property
    def choice(self) -> tuple[str, int | None] | None:
        return self._choice

    def refresh(self) -> None:
        """重列来源。只列**此刻正在发声**的程序（与启动窗口一致）。"""
        try:
            procs = [p for p in enumerate_audio_processes(include_inactive=False) if not p.is_system_sounds]
        except Exception as exc:  # noqa: BLE001 - 枚举失败不能让对话框打不开
            procs = []
            self.empty.setText(f"❌ 枚举音频会话失败：{exc}")

        self.list.clear()

        tab_item = QListWidgetItem(
            f"{TAB_LABEL}{'（当前）' if self._tab_active else ''}"
        )
        tab_item.setData(Qt.UserRole, "tab")
        tab_item.setToolTip(
            "音频由浏览器扩展经本机回环送来；别的标签页照常有声。\n"
            "要点：在浏览器里切到目标标签页，再点扩展图标（或按 Ctrl+Shift+U）。"
        )
        self.list.addItem(tab_item)

        for p in procs:
            mark = "（当前）" if (self._current_pid and p.pid == self._current_pid) else ""
            title = f"    {p.title}" if p.title else ""
            item = QListWidgetItem(f"{p.name}    PID {p.pid}{title}{mark}")
            item.setData(Qt.UserRole, ("process", p.pid))
            item.setToolTip(p.executable or p.display_label)
            self.list.addItem(item)

        # 默认选中"当前音源"，其次选中第一个程序
        target_row = 0
        for row in range(self.list.count()):
            data = self.list.item(row).data(Qt.UserRole)
            if self._tab_active and data == "tab":
                target_row = row
                break
            if data != "tab" and isinstance(data, tuple) and data[1] == self._current_pid:
                target_row = row
                break
        else:
            if not self._tab_active and self.list.count() > 1:
                target_row = 1
        self.list.setCurrentRow(target_row)

        if procs:
            self.empty.setText(
                "提示：浏览器 / Electron 类应用有多个同名子进程，选错了会一直静音，换一个同名的试试。"
            )
        elif not self.empty.text().startswith("❌"):
            self.empty.setText(
                "当前没有程序在输出音频。请先让要听的那个程序**开始播放**，再点「刷新列表」。"
            )

    # ------------------------------------------------------------------ #
    def _accept_selected(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if data == "tab":
            self._choice = ("tab", None)
        elif isinstance(data, tuple) and data[0] == "process":
            if data[1] == self._current_pid and not self._tab_active:
                # 选的就是当前音源：等于什么都没换，直接当取消，别白重启一次
                self.reject()
                return
            self._choice = ("process", int(data[1]))
        else:
            return
        self.hide()
        self.accept()
