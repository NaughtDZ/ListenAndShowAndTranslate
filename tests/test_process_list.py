"""发声进程枚举测试。

真实枚举依赖系统音频会话，测试只做**契约检查**（不假定当前有程序在放音），
保证在任何机器/CI 上都能跑。
"""

from __future__ import annotations

from app.audio.process_list import (
    SESSION_ACTIVE,
    SESSION_EXPIRED,
    SESSION_INACTIVE,
    AudioProcess,
    enumerate_audio_processes,
)


def test_returns_list_and_never_raises():
    """核心契约：枚举失败也必须返回列表，绝不能让 UI 崩。"""
    result = enumerate_audio_processes()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, AudioProcess)


def test_include_inactive_superset():
    active = enumerate_audio_processes(include_inactive=False)
    all_sessions = enumerate_audio_processes(include_inactive=True)
    assert len(all_sessions) >= len(active), "包含非活跃会话时应不少于仅活跃会话"
    assert all(p.state == SESSION_ACTIVE for p in active), "默认只返回活跃会话"


def test_no_expired_sessions_returned():
    for p in enumerate_audio_processes(include_inactive=True):
        assert p.state != SESSION_EXPIRED, "已过期的会话不应出现在结果里"


def test_pids_are_unique():
    pids = [p.pid for p in enumerate_audio_processes(include_inactive=True)]
    assert len(pids) == len(set(pids)), "同一 PID 的多个会话应合并"


def test_system_sounds_sorted_last():
    result = enumerate_audio_processes(include_inactive=True)
    sys_idx = [i for i, p in enumerate(result) if p.is_system_sounds]
    if sys_idx and len(result) > 1:
        assert sys_idx == [len(result) - 1], "系统声音应排在最后"


def test_display_label_for_system_sounds():
    p = AudioProcess(pid=0, name="系统声音", is_system_sounds=True)
    assert "系统声音" in p.display_label


def test_state_name_and_is_active():
    assert AudioProcess(pid=1, name="a.exe", state=SESSION_ACTIVE).is_active
    assert AudioProcess(pid=1, name="a.exe", state=SESSION_INACTIVE).state_name == "Inactive"
    assert AudioProcess(pid=1, name="a.exe", state=SESSION_EXPIRED).state_name == "Expired"
