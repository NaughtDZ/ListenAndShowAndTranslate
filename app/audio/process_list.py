"""枚举"当前正在输出音频的进程"——UI 里"选择音频来源"下拉框的数据源。

原理：Windows 音频会话 API（IAudioSessionManager2 / IAudioSessionEnumerator，
即音量合成器背后的接口），通过 pycaw 调用。

⚠️ 重要澄清（实测结论，2026-02-21）：
    proc-tap 自带的 ``--list-audio-procs`` **不可用**——它在 Windows 上只是拿进程名
    去正则匹配 'audio|media|music|player|chrome|...'，或"列出有窗口的进程再按关键词
    过滤"，并未调用任何音频会话 API（尽管其 docstring 如此声称）。
    因此本模块自行实现枚举。

已知边界：
    ``AudioUtilities.GetAllSessions()`` 只枚举**默认播放设备**上的会话。
    若目标程序输出到非默认设备，需用 ``all_devices=True``（P1 待完善）。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass, field

from app.utils.log import get_logger

log = get_logger(__name__)

# WASAPI AUDIO_SESSION_STATE
SESSION_INACTIVE = 0
SESSION_ACTIVE = 1
SESSION_EXPIRED = 2

_STATE_NAMES = {0: "Inactive", 1: "Active", 2: "Expired"}

# 系统声音会话没有进程，PID 恒为 0
_SYSTEM_SOUNDS_PID = 0


@dataclass(frozen=True)
class AudioProcess:
    """一个正在（或曾经）输出音频的进程。"""

    pid: int
    name: str
    """进程名，如 "GenshinImpact.exe"。"""

    title: str = ""
    """主窗口标题（用于区分同一 exe 的多个实例）。"""

    state: int = SESSION_ACTIVE
    """WASAPI 会话状态：0=Inactive 1=Active 2=Expired。"""

    executable: str = ""
    is_system_sounds: bool = False
    session_count: int = 1

    @property
    def state_name(self) -> str:
        return _STATE_NAMES.get(self.state, str(self.state))

    @property
    def is_active(self) -> bool:
        return self.state == SESSION_ACTIVE

    @property
    def display_label(self) -> str:
        """给 UI 下拉框用的展示文本。"""
        if self.is_system_sounds:
            return "系统声音（System Sounds）"
        suffix = f"  —  {self.title}" if self.title else ""
        return f"{self.name}  (PID {self.pid}){suffix}"


# --------------------------------------------------------------------------- #
# 窗口标题（纯 ctypes，避免引入额外依赖）
# --------------------------------------------------------------------------- #
_user32 = ctypes.windll.user32

_EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)


def _window_titles_for_pid(pid: int, max_titles: int = 4) -> list[str]:
    """返回该进程所有可见顶层窗口的非空标题。"""
    titles: list[str] = []

    def _cb(hwnd: int, _lparam: int) -> bool:
        if len(titles) >= max_titles:
            return False
        if not _user32.IsWindowVisible(hwnd):
            return True
        wnd_pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wnd_pid))
        if wnd_pid.value != pid:
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        text = buf.value.strip()
        if text:
            titles.append(text)
        return True

    try:
        _user32.EnumWindows(_EnumWindowsProc(_cb), 0)
    except Exception as exc:  # noqa: BLE001 - 拿不到标题不该影响主流程
        log.debug("枚举窗口标题失败 pid=%s: %s", pid, exc)
    return titles


def _best_window_title(pid: int) -> str:
    titles = _window_titles_for_pid(pid)
    if not titles:
        return ""
    # 通常是"最长的那个"更可能是主窗口标题
    return max(titles, key=len)


# --------------------------------------------------------------------------- #
# 音频会话枚举
# --------------------------------------------------------------------------- #
def enumerate_audio_processes(
    include_inactive: bool = False,
    all_devices: bool = False,
    with_window_title: bool = True,
    exclude_pids: set[int] | None = None,
) -> list[AudioProcess]:
    """列出正在输出音频的进程。

    Args:
        include_inactive: 是否连"存在但当前没在播"的会话一起返回
            （UI 里可以让用户看到"已打开但静音中"的程序）。
        all_devices: 是否枚举所有播放设备（否则只枚举默认设备）。
        with_window_title: 是否解析窗口标题（略慢，但能区分同名多实例）。
        exclude_pids: 要排除的 PID。默认为 **本进程**——
            实测发现：我们自己对目标做进程回环采集时，**本进程也会在音频会话里
            出现并显示为 Active**（音量合成器里能看到它，标题是窗口标题）。
            不排除的话，它就会污染"选择音频来源"下拉框。

    Returns:
        去重（按 PID）后的列表，活跃的排在前面。
    """
    excluded = {os.getpid()} if exclude_pids is None else set(exclude_pids)

    sessions = _collect_sessions(all_devices)
    if sessions is None:
        return []

    merged: dict[int, AudioProcess] = {}
    for sess in sessions:
        pid = getattr(sess, "ProcessId", None)
        if pid is None or pid in excluded:
            continue
        state = int(getattr(sess, "State", SESSION_INACTIVE))
        if state == SESSION_EXPIRED:
            continue
        if not include_inactive and state != SESSION_ACTIVE:
            continue

        is_system = pid == _SYSTEM_SOUNDS_PID
        name = "系统声音" if is_system else _process_name(sess, pid)

        existing = merged.get(pid)
        if existing is not None:
            # 同一进程可能有多个会话：保留"更强的"状态并累加计数
            best_state = max(existing.state, state)
            merged[pid] = AudioProcess(
                pid=existing.pid,
                name=existing.name,
                title=existing.title,
                state=best_state,
                executable=existing.executable,
                is_system_sounds=existing.is_system_sounds,
                session_count=existing.session_count + 1,
            )
            continue

        merged[pid] = AudioProcess(
            pid=pid,
            name=name,
            title="" if is_system else (_best_window_title(pid) if with_window_title else ""),
            state=state,
            executable=_process_exe(sess, pid),
            is_system_sounds=is_system,
        )

    result = list(merged.values())
    # 活跃优先，然后按名字排序；系统声音永远排最后
    result.sort(key=lambda p: (p.is_system_sounds, not p.is_active, p.name.lower()))
    return result


def _collect_sessions(all_devices: bool):
    """取原始会话列表；失败时返回 None 而不是抛异常。"""
    try:
        if not all_devices:
            from pycaw.pycaw import AudioUtilities

            return list(AudioUtilities.GetAllSessions())

        # 枚举所有播放设备上的会话
        from pycaw.pycaw import AudioUtilities

        out = []
        for dev in AudioUtilities.GetAllDevices():
            try:
                mgr = AudioUtilities.GetAudioSessionManager(dev)
                enum = mgr.GetSessionEnumerator()
                out.extend(enum.GetSession(i) for i in range(enum.GetCount()))
            except Exception as exc:  # noqa: BLE001
                log.debug("枚举设备会话失败: %s", exc)
        return out
    except Exception as exc:  # noqa: BLE001 - 枚举失败必须降级，不能让 UI 崩
        log.warning("枚举音频会话失败（可能没有播放设备）: %s", exc)
        return None


def _process_name(sess, pid: int) -> str:
    proc = getattr(sess, "Process", None)
    if proc is not None:
        try:
            return proc.name()
        except Exception:  # noqa: BLE001 - 进程可能已退出
            pass
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001
        return f"<pid {pid}>"


def _process_exe(sess, pid: int) -> str:
    proc = getattr(sess, "Process", None)
    if proc is not None:
        try:
            return proc.exe()
        except Exception:  # noqa: BLE001
            pass
    try:
        import psutil

        return psutil.Process(pid).exe()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# 会话音量查询（P1 实测：采集幅度与会话音量成正比，可用于自动增益补偿）
# --------------------------------------------------------------------------- #
def get_session_volume(pid: int) -> tuple[float, bool] | None:
    """读取目标会话的主音量与静音状态。

    Returns:
        ``(volume, muted)``，拿不到时返回 None。
        volume 为 0.0~1.0。

    用途：实测证明**进程回环采集发生在音量合成器之后**——
    用户把小说音量调到 30%，我们收到的信号也只有约 32%。
    因此可以用这里的音量值做数字增益补偿，让用户把音量调轻（不打扰游戏）
    而识别端仍然拿到满幅信号。
    """
    try:
        from pycaw.pycaw import AudioUtilities

        for s in AudioUtilities.GetAllSessions():
            if getattr(s, "ProcessId", None) != pid:
                continue
            av = getattr(s, "SimpleAudioVolume", None)
            if av is None:
                return None
            return float(av.GetMasterVolume()), bool(av.GetMute())
    except Exception as exc:  # noqa: BLE001
        log.debug("读取会话音量失败 pid=%s: %s", pid, exc)
    return None


def suggest_gain(volume: float | None, max_gain_db: float = 24.0) -> float:
    """根据会话音量给出数字增益建议（线性倍数）。

    音量 1.0 → 1.0 倍；音量 0.2 → 最多 5 倍，但不超过 ``max_gain_db``。
    音量过低或为 0 时不做无限放大（那只会把底噪也放大），由上限兜住。
    """
    if volume is None or volume <= 0.001:
        return 1.0
    max_gain = 10.0 ** (max_gain_db / 20.0)
    return min(max_gain, 1.0 / volume)
