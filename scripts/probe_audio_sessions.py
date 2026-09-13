"""探针：**不按 PID 合并**地枚举音频会话的全部字段。

用途：判断"浏览器能否精确到标签页"。
    音频会话 API（音量合成器背后那套）里，一个进程可以有多个会话，
    会话身份由 ``GetSessionIdentifier`` / ``GetSessionInstanceIdentifier`` 区分。
    如果 Chromium 给每个标签页建了独立会话，那么 Identifier 应当逐个不同，
    且 DisplayName / IconPath 可能带标签页信息（标题 / favicon）。

    ⚠️ 注意：能"枚举到"不等于能"单独采集"——
    WASAPI 只有**进程级**回环（AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK），
    没有"会话级回环"。本探针回答的是"边界到底画在哪一层"。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_audio_sessions.py
    .venv\\Scripts\\python.exe scripts\\probe_audio_sessions.py --all-devices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _proc_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001
        return "?"


def _win_title(pid: int) -> str:
    try:
        from app.audio.process_list import _best_window_title

        return _best_window_title(pid)
    except Exception:  # noqa: BLE001
        return ""


def _short(text: str, n: int = 70) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def collect(all_devices: bool = False) -> list[tuple]:
    from pycaw.pycaw import AudioUtilities

    sessions = []
    if not all_devices:
        sessions = list(AudioUtilities.GetAllSessions())
    else:
        for dev in AudioUtilities.GetAllDevices():
            try:
                mgr = AudioUtilities.GetAudioSessionManager(dev)
                enum = mgr.GetSessionEnumerator()
                for i in range(enum.GetCount()):
                    sessions.append(enum.GetSession(i))
            except Exception as exc:  # noqa: BLE001
                print(f"  (设备会话枚举失败: {exc})")

    rows = []
    for s in sessions:
        pid = getattr(s, "ProcessId", None)
        identifier = getattr(s, "Identifier", "") or ""
        instance = getattr(s, "InstanceIdentifier", "") or ""
        rows.append(
            (
                pid,
                _proc_name(pid) if pid else "系统声音",
                int(getattr(s, "State", -1)),
                _short(identifier),
                _short(instance),
                _short(getattr(s, "DisplayName", "") or "", 50),
                _short(getattr(s, "IconPath", "") or "", 40),
                _win_title(pid) if pid else "",
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-devices", action="store_true")
    ap.add_argument("--pid", type=int, default=0, help="只看这个 PID")
    args = ap.parse_args()

    rows = collect(args.all_devices)
    if args.pid:
        rows = [r for r in rows if r[0] == args.pid]

    print(f"共 {len(rows)} 个会话（未按 PID 合并）\n")
    by_pid: dict[int, int] = {}
    for r in rows:
        by_pid[r[0]] = by_pid.get(r[0], 0) + 1

    print(f"{'PID':>7}  {'进程':<22} {'态':>2}  {'会话标识后 40 位':<42}  {'显示名':<24} 窗口标题")
    print("-" * 150)
    for pid, name, state, identifier, instance, display, icon, title in rows:
        tail = identifier[-40:] if len(identifier) > 40 else identifier
        print(f"{pid:>7}  {name:<22} {state:>2}  {tail:<42}  {display:<24} {_short(title, 34)}")

    multi = {p: c for p, c in by_pid.items() if c > 1}
    print("\n同一 PID 拥有多个会话的进程：")
    if not multi:
        print("  （无——说明本机当前没有「按会话细分的进程」）")
    for pid, count in sorted(multi.items(), key=lambda kv: -kv[1]):
        print(f"  pid={pid:<7} {_proc_name(pid):<22} {count} 个会话")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
