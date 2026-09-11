"""P1 端到端验收：CLI → 采集 → 降混 → 重采样 16k → WAV，并验证音高无损。

与 scripts/verify_process_isolation.py 的分工：
    - verify_process_isolation.py：验证"不会混入别人的声音"（隔离性）
    - verify_capture_e2e.py     ：验证"抓到的声音在整条管线里没被破坏"（保真性）

用法：
    .venv\\Scripts\\python.exe scripts\\verify_capture_e2e.py
    .venv\\Scripts\\python.exe scripts\\verify_capture_e2e.py --tone 441 --record-seconds 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)


def make_tone(path: Path, freq: float, seconds: float = 30.0, amp: float = 0.4) -> None:
    rate = 48000
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    sig = np.sin(2 * np.pi * freq * t) * amp
    stereo = np.stack([sig, sig], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())


_PLAYER = r"""
import os, sys, time, winsound
print("PID", os.getpid(), flush=True)
winsound.PlaySound(sys.argv[1], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
print("PLAYING", flush=True)
time.sleep(120)
"""


def spawn_player(wav: Path) -> tuple[subprocess.Popen, int]:
    """启动播放进程；真实 PID 由子进程自己上报（见 docs/P1-实测记录.md 3.1）。"""
    proc = subprocess.Popen(
        [str(PY), "-c", _PLAYER, str(wav)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", bufsize=1,
    )
    real_pid = -1
    for _ in range(200):
        line = (proc.stdout.readline() or "").strip()
        if not line:
            break
        if line.startswith("PID "):
            real_pid = int(line.split()[1])
        if line == "PLAYING":
            break
    return proc, real_pid


def main() -> int:
    ap = argparse.ArgumentParser(description="采集管线端到端验收")
    ap.add_argument("--tone", type=float, default=660.0, help="测试音频率")
    ap.add_argument("--live-seconds", type=float, default=4.0, help="实时电平演示时长")
    ap.add_argument("--record-seconds", type=float, default=3.0, help="录制音频时长")
    args = ap.parse_args()

    tmp = PROJECT / "data" / "selftest_audio"
    tmp.mkdir(parents=True, exist_ok=True)
    wav_in = tmp / "tone_e2e.wav"
    wav_out = tmp / "captured_e2e.wav"

    print("=" * 70)
    print(f"[1/5] 生成 {args.tone:.0f} Hz 测试音")
    make_tone(wav_in, args.tone)
    print(f"      {wav_in.name}")

    print("\n[2/5] 启动播放进程")
    proc, real_pid = spawn_player(wav_in)
    if real_pid <= 0:
        print("  ✗ 播放进程未能上报 PID")
        proc.terminate()
        return 2
    print(f"      启动器 pid={proc.pid}  真实播放 pid={real_pid}")

    try:
        print(f"\n[3/5] main.py --capture {real_pid} --seconds {args.live_seconds:.0f}（实时电平）")
        r1 = subprocess.run(
            [str(PY), "main.py", "--capture", str(real_pid), "--seconds", str(args.live_seconds)],
            cwd=str(PROJECT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        tail = [ln for ln in r1.stdout.splitlines() if ln.strip()][-4:]
        for ln in tail:
            print("      " + ln.strip())
        if r1.returncode != 0:
            print(f"  ✗ --capture 退出码 {r1.returncode}")
            return 1

        print(f"\n[4/5] main.py --capture --record（录 {args.record_seconds:.0f} 秒音频）")
        r2 = subprocess.run(
            [str(PY), "main.py", "--capture", str(real_pid),
             "--record", str(wav_out), "--seconds", str(args.record_seconds)],
            cwd=str(PROJECT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        for ln in [x for x in r2.stdout.splitlines() if x.strip()]:
            print("      " + ln.strip())
        if r2.returncode != 0:
            print(f"  ✗ --record 退出码 {r2.returncode}")
            return 1

        print("\n[5/5] 验证录出的 WAV")
        with wave.open(str(wav_out), "rb") as w:
            ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        print(f"      声道={ch} 位宽={sw * 8}bit 采样率={fr} 帧数={n} ({n / fr:.2f}s)")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        peak = float(np.max(np.abs(audio)))
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / fr)
        peak_hz = float(freqs[int(np.argmax(spectrum))])
        print(f"      峰值幅度={peak:.4f}  主频={peak_hz:.1f} Hz（期望 {args.tone:.0f} Hz）")

        problems = []
        if ch != 1:
            problems.append(f"应为单声道，实得 {ch}")
        if fr != 16000:
            problems.append(f"应为 16000 Hz，实得 {fr}")
        if abs(n - args.record_seconds * fr) > fr * 0.1:
            problems.append(f"时长偏差过大：期望约 {args.record_seconds * fr:.0f} 帧，实得 {n}")
        if abs(peak_hz - args.tone) > 15:
            problems.append(f"主频偏移：期望 {args.tone:.0f} Hz，实得 {peak_hz:.1f} Hz")
        if peak <= 0.05:
            problems.append(f"峰值过低：{peak:.4f}")

        print()
        if problems:
            for p in problems:
                print(f"  ✗ {p}")
            return 1
        print("  ✅ 通过：CLI → 采集 → 降混 → 重采样16k → WAV，音高与时长均正确")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
