"""窗口生命周期（退出策略 / 设置窗复用）的测试。

用 **offscreen** 平台跑：不需要显示器，也绝不弹窗打扰人。
这里验的不是"窗口好不好看"，而是**谁有权让程序退出**：

* 用户 2026-09-12 反馈 —— 控制窗收进托盘后，只关掉设置窗，整个程序就退出了。
  根因是 Qt 默认 ``quitOnLastWindowClosed=True``：悬浮窗是 ``Qt.Tool``（不计数）、
  控制窗隐藏后也不算，于是"关设置窗"被 Qt 当成"关掉最后一个窗口"。
* 另一个要守住的边界：控制窗自己右上角 ✕ 仍然必须退出程序（不能修好一个坏一个）。

本文件里的用例**有顺序要求**：``QApplication.exec()`` 只要被"自动退出"结束过一次，
同一进程里后续的 ``exec()`` 会立刻返回（Qt 会留下 quitNow 标记）。所以

1. 会提前结束循环的那条（验证"✕=退出"）放在**最后**；
2. 验证 Qt 默认策略的坑单独开子进程跑，避免污染本进程。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

# 必须在创建 QApplication **之前**设好（平台插件在构造时才确定）
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.pipeline import SubtitlePipeline  # noqa: E402
from app.ui.lifecycle import (  # noqa: E402
    configure_quit_policy,
    open_settings_window,
    set_quitting,
)
from app.ui.runner import SubtitleControlWindow  # noqa: E402
from app.ui.subtitle_overlay import SubtitleOverlay  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    configure_quit_policy(app)
    return app


def _control_window() -> SubtitleControlWindow:
    cfg = AppConfig()
    cfg.overlay.always_on_top = False  # 测试里不需要每 200ms 重申一次置顶
    overlay = SubtitleOverlay(cfg.overlay)
    return SubtitleControlWindow(overlay, SubtitlePipeline(cfg))


# --------------------------------------------------------------------------- #
# 策略本身
# --------------------------------------------------------------------------- #
def test_configure_quit_policy_disables_auto_quit(qapp):
    assert qapp.quitOnLastWindowClosed() is False


def test_qt_default_would_quit_on_settings_close():
    """把"坑"钉住：Qt 默认策略下，这个拓扑关掉设置窗就会退出程序。

    单独开子进程跑，因为一旦发生自动退出，本进程后续的 ``exec()`` 会立刻返回。
    如果哪天这条断言挂了：说明 Qt 改了默认行为（不是我们代码坏了），
    这时可以删掉本用例，同时更新 ``app/ui/lifecycle.py`` 的说明。
    """
    script = """
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication([])
assert app.quitOnLastWindowClosed() is True      # 不改策略 = Qt 默认

overlay = QWidget(); overlay.setWindowFlags(Qt.Tool); overlay.show()  # 悬浮窗：不计数
control = QWidget(); control.show(); control.hide()                   # 收进托盘
settings = QWidget(); settings.show()

events = []
QTimer.singleShot(20, settings.close)
QTimer.singleShot(400, lambda: (events.append("alive"), app.quit()))
app.exec()
print("ALIVE" if events else "QUIT")
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert proc.returncode == 0, proc.stderr
    assert "QUIT" in proc.stdout, (
        "Qt 的 quitOnLastWindowClosed 默认行为变了（见本用例开头的说明）："
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


# --------------------------------------------------------------------------- #
# 回归：控制窗在托盘 + 关掉设置窗 ≠ 退出程序
# --------------------------------------------------------------------------- #
def test_tray_hidden_control_survives_settings_close(qapp):
    """**核心回归用例**：控制窗收进托盘，关掉设置窗后程序必须还活着。"""
    control = _control_window()
    control.overlay.show()
    control.show()
    control.hide()  # 点「最小化到托盘」（测试环境没有托盘，直接 hide）
    assert control.isVisible() is False

    settings = open_settings_window(control, control.pipeline.config)
    assert settings.isVisible() is True

    events: list[str] = []
    QTimer.singleShot(30, settings.close)
    QTimer.singleShot(300, lambda: (events.append("alive"), qapp.quit()))
    qapp.exec()

    # 循环没有被 close 提前结束 → 程序还活着（这正是修好的那个 bug）
    assert events == ["alive"]
    assert settings.isVisible() is False
    assert control.isVisible() is False

    control.overlay.deleteLater()
    control.deleteLater()


def test_settings_window_is_reused_not_recreated(qapp):
    """关掉设置窗再打开：复用同一个实例，而不是再堆一个（避免 QThread 被销毁）。"""
    control = _control_window()
    first = getattr(control, "_settings_win", None)
    assert first is None

    win = control._open_settings()  # 控制窗自己的入口（真按钮走的就是它）
    assert win is not None
    assert control._settings_win is win
    assert win.isVisible() is True

    win.close()
    assert win.isVisible() is False

    again = control._open_settings()
    assert again is win  # 复用，不是新建
    assert again.isVisible() is True

    win.close()
    control.deleteLater()


class _FakeTray:
    """够用的假托盘：只实现 _to_tray/_quit 会碰到的那几个东西。"""

    class MessageIcon:  # noqa: N801 - 模仿 QSystemTrayIcon.MessageIcon
        Information = 1

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.hidden = 0

    def showMessage(self, _title, text, _icon, _ms) -> None:
        self.messages.append(text)

    def hide(self) -> None:
        self.hidden += 1


def test_minimize_to_tray_without_tray_keeps_window_visible(qapp):
    """系统没有托盘时不许隐藏：藏起来就再也找不回来了。"""
    control = _control_window()
    control.show()
    control._to_tray()
    assert control.isVisible() is True
    control.hide()
    control.deleteLater()


def test_minimize_to_tray_hides_window_and_keeps_app_alive(qapp):
    """有托盘：控制窗隐藏、提示一句，但**不退出程序**。"""
    control = _control_window()
    tray = _FakeTray()
    control.tray = tray
    control.show()
    control._to_tray()
    assert control.isVisible() is False
    assert tray.messages and "托盘" in tray.messages[0]
    assert getattr(control, "_quitting", False) is False
    control.deleteLater()


def test_open_settings_window_wires_saved_slot_only_once(qapp):
    """``saved`` 只接一次：接两次的话点一次「保存」会重建两遍识别/翻译引擎。"""

    class Owner:
        pass

    owner = Owner()
    cfg = AppConfig()
    win = open_settings_window(owner, cfg)
    hits: list[int] = []
    win.saved.connect(lambda: hits.append(1))
    win.saved.emit()
    assert hits == [1], "同一个槽被接了两遍"

    # 再次打开（复用）不应该重复接线
    open_settings_window(owner, cfg)
    win.saved.emit()
    assert hits == [1, 1]

    win.close()
    win.deleteLater()


def test_set_quitting_is_one_shot():
    class Owner:
        pass

    owner = Owner()
    assert set_quitting(owner) is True
    assert set_quitting(owner) is False  # 已在退出：别递归再退一次


# --------------------------------------------------------------------------- #
# 选择窗口：子进程自检（秒退要报错，活着才关掉自己）
# --------------------------------------------------------------------------- #
class _FakeProc:
    """假的 Popen：只用到 poll() 和 pid。"""

    pid = 4321

    def __init__(self, code: int | None) -> None:
        self._code = code

    def poll(self):
        return self._code


def _launcher_window() -> "LauncherWindow":
    from app.ui.launcher import LauncherWindow

    win = LauncherWindow(AppConfig.load())
    win._timer.stop()  # 别让 3 秒自动刷新在测试里乱跑
    return win


def test_launcher_reports_child_that_died_immediately(qapp):
    """子进程秒退（比如选错了没发声的进程）要留在界面上报错，而不是静默关闭。"""
    win = _launcher_window()
    try:
        assert win.start_btn.isEnabled() is True
        win._finish_start(_FakeProc(2))
        assert "立刻退出" in win.empty_hint.text()
        assert win.isVisible() is False  # 没 show，但也没有退出程序
        assert getattr(win, "_quitting", False) is False
        assert win.start_btn.isEnabled() is True
    finally:
        win._timer.stop()
        win.deleteLater()


def test_launcher_closes_itself_when_child_alive(qapp, monkeypatch):
    """子进程活着 → 选择窗口自己关了（不留看不见也点不到的窗口）。"""
    win = _launcher_window()
    quit_calls: list[bool] = []
    monkeypatch.setattr(win, "_quit", lambda: quit_calls.append(True))
    try:
        win._finish_start(_FakeProc(None))
        assert quit_calls == [True]
    finally:
        win._timer.stop()
        win.deleteLater()


# --------------------------------------------------------------------------- #
# 边界：控制窗自己的 ✕ 仍然必须退出（放在最后：这条会真的结束事件循环）
# --------------------------------------------------------------------------- #
def test_control_window_close_event_quits_app(qapp):
    """控制窗右上角 ✕ = 退出程序：老行为不能修坏。"""
    control = _control_window()
    control.overlay.show()
    control.show()

    events: list[str] = []
    QTimer.singleShot(30, control.close)
    QTimer.singleShot(400, lambda: (events.append("alive"), qapp.quit()))
    qapp.exec()

    assert events == []  # close → 显式退出，循环立刻结束
    assert getattr(control, "_quitting", False) is True
    # closeEvent 里已经调过 app.quit()，这里只清理控件
    control.overlay.deleteLater()
    control.deleteLater()
