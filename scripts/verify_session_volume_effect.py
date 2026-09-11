"""P1 验收：会话音量/静音是否影响进程回环采集。

背景（计划书 P1 待验证清单 #3）：
    用户在 Windows 音量合成器里把目标程序调小或静音时，我们还能不能抓到声音？
    这直接决定 UI 上是否必须提示"别在音量合成器里静音它"。

设计：用 pycaw 直接控制目标会话的音量与静音（等价于用户拖音量合成器），
      每个状态各采一段音频，比较电平。全自动，无需人工操作。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_session_volume_effect.py --pid 23784
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from app.audio.capture import CaptureWorker, TargetSpec  # noqa: E402
from app.audio.process_list import enumerate_audio_processes  # noqa: E402

DEFAULT_PID = 0


def find_session(pid: int):
    """找到该 PID 的 pycaw 会话对象。"""
    from pycaw.pycaw import AudioUtilities

    for s in AudioUtilities.GetAllSessions():
        if s.ProcessId == pid:
            return s
    return None


def capture_metrics(pid: int, seconds: float) -> dict:
    """采集 N 秒音频，返回每块 RMS 的统计量。

    用"已收样本数"而非墙钟计时（采集启动约 0.6s，见 docs/P1-实测记录.md 6.2）。
    """
    rms_list: list[float] = []
    collected = [0]
    target = int(seconds * 16000)

    worker = CaptureWorker(TargetSpec(pid=pid), follow=False)
    worker.start()
    deadline = time.time() + seconds + 15

    try:
        while collected[0] < target and time.time() < deadline:
            st = worker.snapshot()
            rms_list.append(st.last_rms)
            collected[0] = st.samples_out
            time.sleep(0.05)
        if not worker.is_running and collected[0] == 0:
            time.sleep(0.3)
    finally:
        st = worker.snapshot()
        worker.stop()

    # 丢掉启动阶段的前 20% 采样（还没稳定）
    usable = rms_list[max(1, len(rms_list) // 5):] or rms_list or [0.0]
    return {
        "peak": st.peak,
        "median_rms": float(statistics.median(usable)),
        "p90_rms": float(np.percentile(usable, 90)),
        "samples": st.samples_out,
        "blocks": st.chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="会话音量/静音对采集的影响")
    ap.add_argument("--pid", type=int, required=True, help="目标进程 PID（必须是音频会话 PID）")
    ap.add_argument("--seconds", type=float, default=4.0, help="每个状态采集的音频秒数")
    args = ap.parse_args()

    pid = args.pid

    # 0) 先确认这个 PID 真的是音频会话持有者
    sessions = enumerate_audio_processes(include_inactive=True, with_window_title=False)
    match = [s for s in sessions if s.pid == pid]
    if not match:
        print(f"✗ PID {pid} 不在音频会话列表中，无法做本实验。")
        print("  当前会话：")
        for s in sessions:
            print(f"    pid={s.pid:<8} {s.name:<28} state={s.state_name}")
        print("\n  提示：浏览器/Electron 应用请用**活跃会话的那个 PID**，不是随便一个同名子进程。")
        return 2

    sess = match[0]
    print("=" * 74)
    print(f"目标: {sess.name} (PID {pid})  会话状态={sess.state_name}")
    print("=" * 74)

    ctl = find_session(pid)
    if ctl is None:
        print("✗ 拿不到 pycaw 会话控制对象，无法自动改音量")
        return 2

    av = getattr(ctl, "SimpleAudioVolume", None)
    if av is None:
        print("✗ 该会话没有 SimpleAudioVolume 接口")
        return 2

    orig_vol = av.GetMasterVolume()
    orig_mute = bool(av.GetMute())
    print(f"初始：音量={orig_vol:.2f}  静音={orig_mute}")

    # 关键：**绝不把音量调高于用户当前设置**。
    # 抬高音量会在用户听小说/打游戏时突然炸响，非常不礼貌也危险。
    quiet_vol = max(0.05, orig_vol * 0.3)
    print(f"本实验只会调小音量（降到 {quiet_vol:.2f}）与静音，最后恢复原状；不会调大。")

    def set_state(vol: float, mute: bool) -> None:
        av.SetMasterVolume(float(vol), None)
        av.SetMute(bool(mute), None)

    results: dict[str, dict] = {}
    try:
        print("\n[1/4] 基准（保持用户当前音量，不静音）")
        set_state(orig_vol, False)
        time.sleep(0.8)
        results["原始"] = capture_metrics(pid, args.seconds)
        print(f"      中位 RMS={results['原始']['median_rms']:.5f} "
              f"p90={results['原始']['p90_rms']:.5f} 峰值={results['原始']['peak']:.5f}")

        print(f"[2/4] 音量降到 {quiet_vol:.2f}")
        set_state(quiet_vol, False)
        time.sleep(0.8)
        results["降低音量"] = capture_metrics(pid, args.seconds)
        print(f"      中位 RMS={results['降低音量']['median_rms']:.5f} "
              f"p90={results['降低音量']['p90_rms']:.5f} 峰值={results['降低音量']['peak']:.5f}")

        print("[3/4] 会话静音")
        set_state(orig_vol, True)
        time.sleep(0.8)
        results["静音"] = capture_metrics(pid, args.seconds)
        print(f"      中位 RMS={results['静音']['median_rms']:.5f} "
              f"p90={results['静音']['p90_rms']:.5f} 峰值={results['静音']['peak']:.5f}")

    finally:
        print("[4/4] 恢复原状")
        set_state(orig_vol, orig_mute)
        time.sleep(0.5)

    # ---------------- 判定 ---------------- #
    print("\n" + "=" * 74)
    print("判定")
    print("=" * 74)
    base = results["原始"]["p90_rms"]
    quiet = results["降低音量"]["p90_rms"]
    muted = results["静音"]["p90_rms"]

    if base <= 5e-5:
        print(f"⚠️ 基准就接近静音（p90 RMS={base:.6f}）——目标当时没在放音，实验无效。")
        print("   请让目标程序播放一段有声音的内容后重试。")
        return 2

    mute_ratio = muted / base if base else 0
    vol_ratio = quiet / base if base else 0

    print(f"  原始p90={base:.5f}  降音量p90={quiet:.5f}（比值 {vol_ratio:.3f}）  "
          f"静音p90={muted:.5f}（比值 {mute_ratio:.3f}）")
    print()

    affected_volume = vol_ratio < 0.6
    affected_mute = mute_ratio < 0.1

    if affected_mute:
        print("  ✅ 结论：**会话静音会切断采集信号**（静音后几乎为零）")
        print('     ⇒ UI 必须提示用户：不要把目标程序在音量合成器里静音；')
        print('       检测到长时间静音时应给出"是否被静音了"的排查建议。')
    else:
        print(f"  ✅ 结论：**会话静音不影响采集**（静音后仍有 {mute_ratio:.2f} 倍信号）")
        print("     ⇒ 采集发生在音量合成器之前，用户静音不影响我们识别。")

    if affected_volume:
        print(f"  ✅ 结论：**音量大小会影响采集幅度**（降音量后为原来的 {vol_ratio:.2f} 倍）")
        print("     ⇒ 音量过小时应提示用户调大，否则 ASR 准确率下降。")
    else:
        print(f"  ✅ 结论：**音量大小不影响采集幅度**（降音量后仍为原来的 {vol_ratio:.2f} 倍）")

    # 退出码：0 = 实验完成（无论结论如何）
    print()
    print("实验完成。以上结论将写入 docs/P1-实测记录.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
