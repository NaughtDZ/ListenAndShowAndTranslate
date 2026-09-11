"""Win32 窗口操作（ctypes）：点击穿透、强制置顶、不抢焦点、DPI 感知。

这些是"悬浮字幕窗能盖在全屏游戏上"的技术地基（计划书第 2.4 节）。
先用电平表窗口把它跑通，P3 的字幕窗直接复用。

注意：所有函数都在非 Windows 或调用失败时**静默降级**，绝不抛异常打断 UI。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from app.utils.log import get_logger

log = get_logger(__name__)

IS_WINDOWS = sys.platform == "win32"

GWL_EXSTYLE = -20

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

_user32 = ctypes.windll.user32 if IS_WINDOWS else None


def _get_set_window_long():
    """64 位系统要用 *Ptr 版本，否则可能截断句柄。"""
    if not IS_WINDOWS:
        return None, None
    get = getattr(_user32, "GetWindowLongPtrW", None) or _user32.GetWindowLongW
    set_ = getattr(_user32, "SetWindowLongPtrW", None) or _user32.SetWindowLongW
    return get, set_


def hwnd_of(widget) -> int:
    """取 Qt 控件的原生窗口句柄。"""
    try:
        return int(widget.winId())
    except Exception as exc:  # noqa: BLE001
        log.debug("取 winId 失败: %s", exc)
        return 0


def set_click_through(hwnd: int, enabled: bool) -> bool:
    """点击穿透：鼠标事件直接穿到下面的窗口（游戏）。

    需要 WS_EX_LAYERED + WS_EX_TRANSPARENT。
    """
    if not IS_WINDOWS or not hwnd:
        return False
    get, set_ = _get_set_window_long()
    try:
        ex = get(hwnd, GWL_EXSTYLE)
        if enabled:
            ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT
        else:
            ex &= ~WS_EX_TRANSPARENT
        set_(hwnd, GWL_EXSTYLE, ex)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("设置点击穿透失败: %s", exc)
        return False


def set_no_activate(hwnd: int, enabled: bool = True) -> bool:
    """不抢焦点：点击本窗口不会把焦点从游戏抢走。"""
    if not IS_WINDOWS or not hwnd:
        return False
    get, set_ = _get_set_window_long()
    try:
        ex = get(hwnd, GWL_EXSTYLE)
        if enabled:
            ex |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        else:
            ex &= ~WS_EX_NOACTIVATE
        set_(hwnd, GWL_EXSTYLE, ex)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("设置不抢焦点失败: %s", exc)
        return False


def reassert_topmost(hwnd: int) -> bool:
    """重新把窗口钉到最前。

    游戏或某些全屏程序会抢走 Z 序，所以需要**定时重申**（计划书第 2.4 节）。
    """
    if not IS_WINDOWS or not hwnd:
        return False
    try:
        return bool(
            _user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("重申置顶失败: %s", exc)
        return False


def set_taskbar_visible(hwnd: int, visible: bool) -> bool:
    """控制是否出现在任务栏/Alt-Tab（用 WS_EX_TOOLWINDOW 切换）。"""
    if not IS_WINDOWS or not hwnd:
        return False
    get, set_ = _get_set_window_long()
    try:
        ex = get(hwnd, GWL_EXSTYLE)
        if visible:
            ex &= ~WS_EX_TOOLWINDOW
        else:
            ex |= WS_EX_TOOLWINDOW
        set_(hwnd, GWL_EXSTYLE, ex)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("设置任务栏可见性失败: %s", exc)
        return False


def enable_dpi_awareness() -> bool:
    """开启 per-monitor DPI 感知，否则多屏/缩放下悬浮窗会发虚或错位。

    必须在创建 QApplication 之前调用。
    """
    if not IS_WINDOWS:
        return False
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return True
    except Exception:  # noqa: BLE001
        try:
            _user32.SetProcessDPIAware()
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("设置 DPI 感知失败: %s", exc)
            return False


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """取窗口矩形 (left, top, right, bottom)。"""
    if not IS_WINDOWS or not hwnd:
        return None
    try:
        rect = wintypes.RECT()
        if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception as exc:  # noqa: BLE001
        log.debug("取窗口矩形失败: %s", exc)
    return None
