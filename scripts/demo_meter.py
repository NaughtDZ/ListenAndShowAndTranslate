"""演示/调试电平表：没有真实音频时，用测试音把电平表跑起来。

生成一段幅度周期性起伏（0.03 → 0.95）的正弦音，由一个独立进程循环播放，
然后对着这个进程打开电平表。可以直观看到 RMS 条、PEAK 条、峰值保持标记
是如何随音量变化的，也能试阈值滑杆。

用法：
    .venv\\Scripts\\python.exe scripts\\demo_meter.py
    .venv\\Scripts\\python.exe scripts\\demo_meter.py --freq 220 --period 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

_PLAYER = r"""
import os, sys, time, winsound
print("PID", os.getpid(), flush=True)
winsound.PlaySound(sys.argv[1], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
print("PLAYING", flush=True)
time.sleep(3600)
"""


def make_sweeping_tone(path: Path, freq: float, period_s: float, seconds: float = 8.0) -> None:
    """幅度周期性起伏的音：电平表会明显上下动。"""
    rate = 48000
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    # 幅度包络 0.03 ~ 0.95 三角波
    phase = (t % period_s) / period_s
    envelope = 0.03 + (0.95 - 0.03) * (1 - np.abs(2 * phase - 1))
    sig = np.sin(2 * np.pi * freq * t) * envelope
    stereo = np.stack([sig, sig], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(stereo, -1, 1) * 32767).astype(np.int16).tobytes())


def spawn_player(wav: Path) -> tuple[subprocess.Popen, int]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _PLAYER, str(wav)],
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
    ap = argparse.ArgumentParser(description="电平表演示（测试音）")
    ap.add_argument("--freq", type=float, default=440.0)
    ap.add_argument("--period", type=float, default=4.0, help="幅度起伏周期（秒）")
    ap.add_argument("--threshold-db", type=float, default=None)
    args = ap.parse_args()

    tmp = PROJECT / "data" / "selftest_audio"
    tmp.mkdir(parents=True, exist_ok=True)
    wav = tmp / "tone_meter_demo.wav"
    make_sweeping_tone(wav, args.freq, args.period)
    print(f"测试音已生成: {wav}（幅度 {args.period:.0f} 秒一个起伏周期）")

    proc, real_pid = spawn_player(wav)
    if real_pid <= 0:
        proc.terminate()
        print("✗ 播放进程启动失败")
        return 2
    print(f"播放进程: 启动器 pid={proc.pid} → 真实 pid={real_pid}")
    time.sleep(1.0)

    try:
        from app.ui.meter import run_meter

        print("打开电平表…（关闭控制窗即退出）")
        return run_meter(real_pid, args.threshold_db)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
