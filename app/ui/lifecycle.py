"""窗口生命周期：退出策略 + 设置窗复用。

**为什么单独搞一个模块**：Qt 默认 ``quitOnLastWindowClosed = True``，规则是
"最后一个**参与计数**的可见窗口一关，整个 QApplication 就退出"。而本程序里
"谁参与计数"非常反直觉：

* 悬浮字幕窗用的是 ``Qt.Tool`` —— **不参与计数**（Qt 不给 Tool 窗口设
  ``WA_QuitOnClose``）；
* 控制窗点了「最小化到托盘」之后 ``hide()``，也不算可见窗口。

于是"控制窗已收进托盘 + 只关掉设置窗"在 Qt 眼里就是"最后一个窗口关了"，
整个程序（连音频采集线程）一起退出。用户 2026-09-12 反馈的正是这个：
**关掉设置窗，主程序也跟着没了**。

统一策略（别让每个窗口各写一套）：

1. :func:`configure_quit_policy` —— 关掉自动退出，**谁都不许"顺手"退出程序**；
2. 只有显式入口才真退出：控制窗「退出」按钮 / 托盘菜单「退出」/ 主窗口
   右上角 ✕（见各窗口的 ``_quit``），以及 :func:`quit_app`；
3. :func:`open_settings_window` —— 一个 owner 只养一个设置窗实例，关掉再打开是
   **复用**（重新 ``show()``），而不是再建一个。旧实例里可能还有正在跑的
   连通性测试线程（``settings._TestThread``），销毁它轻则
   ``QThread: Destroyed while thread is still running``，重则直接崩。
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "configure_quit_policy",
    "quit_app",
    "set_quitting",
    "open_settings_window",
]


def configure_quit_policy(app: Any) -> None:
    """关掉"最后一个窗口关闭就退出"：退出必须是显式动作。

    在创建任何窗口之前调用（QApplication 一建好就调，别等窗口关了才想起）。
    """
    app.setQuitOnLastWindowClosed(False)


def quit_app() -> None:
    """真正退出程序。所有"真的要退出"的地方都走这里，别直接 ``app.quit()``。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.quit()


def set_quitting(window: Any) -> bool:
    """标记窗口"正在退出"。返回 False 表示之前已经标记过。

    ``app.quit()`` 之后 Qt 还会挨个关窗、再次触发 ``closeEvent``，
    没有这个标志就会递归退出。
    """
    if getattr(window, "_quitting", False):
        return False
    window._quitting = True
    return True


def open_settings_window(
    owner: Any,
    config: Any,
    *,
    attr: str = "_settings_win",
    on_saved: Callable[[], None] | None = None,
) -> Any:
    """打开（或复用）设置窗，返回该窗口。

    复用而不是新建的理由见模块文档：旧实例里可能有正在跑的 QThread，
    销毁它会出事；而且反复开关会堆出一打藏起来的窗口。

    ``attr``：窗口挂在 owner 上的属性名，控制窗和主窗口各用各的。
    """
    from app.ui.settings import SettingsWindow

    win = getattr(owner, attr, None)
    if win is None:
        win = SettingsWindow(config)
        setattr(owner, attr, win)

    # 只接一次（窗口是复用的）：接两次的话，点一次「保存设置」会重建两遍引擎
    if on_saved is not None and not getattr(win, "_lst_saved_wired", False):
        win.saved.connect(on_saved)
        win._lst_saved_wired = True

    win.show()
    win.raise_()
    win.activateWindow()
    return win
