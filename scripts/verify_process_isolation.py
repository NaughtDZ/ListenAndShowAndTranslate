"""P1 核心验收：验证"进程级音频隔离"是真的。

实验设计（不需要用户参与，全自动）：
    1. 生成两个频率不同的正弦音 WAV：A = 440 Hz，B = 1200 Hz
    2. 用两个独立进程分别循环播放（winsound，进程即播放器）
    3. 从音频会话 API 反查它们的 PID（并验证枚举确实能看到它们）
    4. 只对进程 A 做 WASAPI 进程回环采集 N 秒
    5. 对采集结果做频谱分析：
         - 必须存在 440 Hz 能量   → 证明"抓得到目标"
         - 必须没有 1200 Hz 能量  → 证明"没混入别人的声音"
    6. 若两条都满足，则"一个软件听小说、一个软件打游戏，只抓小说"的前提成立

用法：
    .venv\\Scripts\\python.exe scripts\\verify_process_isolation.py
    .venv\\Scripts\\python.exe scripts\\verify_process_isolation.py --seconds 6 --keep-files
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.process_list import enumerate_audio_processes  # noqa: E402

SAMPLE_RATE = 48000
TONE_A_HZ = 440.0
TONE_B_HZ = 1200.0
AMPLITUDE = 0.35


def make_tone(path: Path, freq: float, seconds: float = 30.0) -> None:
    """生成一段循环用的正弦音（16-bit PCM WAV，winsound 只认 PCM）。"""
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    # 加轻微淡入淡出，避免循环接缝爆音
    wave_data = np.sin(2 * np.pi * freq * t) * AMPLITUDE
    fade = int(SAMPLE_RATE * 0.01)
    wave_data[:fade] *= np.linspace(0, 1, fade)
    wave_data[-fade:] *= np.linspace(1, 0, fade)
    stereo = np.stack([wave_data, wave_data], axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


_PLAYER_CODE = r"""
import os, sys, time, winsound
path, seconds = sys.argv[1], float(sys.argv[2])
# 关键：上报"真正执行播放的进程"自己的 PID。
# 不能假定父进程 Popen 拿到的 p.pid 就是音频会话的拥有者——
# uv 的 .venv\Scripts\python.exe 是转发器，会拉起子进程，PID 并不相同。
print("PID", os.getpid(), flush=True)
try:
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
except Exception as exc:
    print("ERROR", repr(exc), flush=True)
    sys.exit(3)
print("PLAYING", flush=True)
time.sleep(seconds)
winsound.PlaySound(None, winsound.SND_PURGE)
"""


def spawn_player(wav: Path, seconds: float) -> tuple[subprocess.Popen, int]:
    """启动一个"进程即播放器"，返回 (进程句柄, 真实播放 PID)。"""
    proc = subprocess.Popen(
        [sys.executable, "-c", _PLAYER_CODE, str(wav), str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    real_pid = -1
    deadline = time.time() + 15
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            break
        line = line.strip()
        if line.startswith("PID "):
            real_pid = int(line.split()[1])
            break
        if line.startswith("ERROR"):
            print(f"      子进程播放失败: {line}")
            break
    return proc, real_pid


def wait_for_session(pid: int, timeout: float = 10.0) -> bool:
    """等待该 PID 出现在活跃音频会话里（说明它真的在放音）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in enumerate_audio_processes(include_inactive=True):
            if p.pid == pid and p.is_active:
                return True
        time.sleep(0.25)
    return False


def dominant_peaks(samples: np.ndarray, fs: int, top_n: int = 5) -> list[tuple[float, float]]:
    """返回频谱里最强的若干个 (频率 Hz, 归一化幅度)。"""
    if samples.size < fs // 2:
        return []
    mono = samples.astype(np.float64)
    mono = mono - mono.mean()
    window = np.hanning(mono.size)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / fs)
    # 只看 50 Hz ~ 4000 Hz（人声/音调范围）
    mask = (freqs >= 50) & (freqs <= 4000)
    spectrum, freqs = spectrum[mask], freqs[mask]
    if spectrum.size == 0 or spectrum.max() <= 0:
        return []
    norm = spectrum / spectrum.max()
    # 挑局部极大值
    idx = np.argsort(norm)[::-1]
    picked: list[tuple[float, float]] = []
    for i in idx:
        f = freqs[i]
        if all(abs(f - pf) > 50 for pf, _ in picked):
            picked.append((float(f), float(norm[i])))
        if len(picked) >= top_n:
            break
    return picked


def energy_at(samples: np.ndarray, fs: int, freq: float, bw: float = 30.0) -> float:
    """指定频率附近的能量占比（0~1），用于判定该频率是否存在。"""
    if samples.size < fs // 2:
        return 0.0
    mono = samples.astype(np.float64)
    mono = mono - mono.mean()
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / fs)
    band = (freqs >= freq - bw) & (freqs <= freq + bw)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(spectrum[band].sum() / total)


def main() -> int:
    ap = argparse.ArgumentParser(description="验证进程级音频隔离")
    ap.add_argument("--seconds", type=float, default=5.0, help="采集时长")
    ap.add_argument("--keep-files", action="store_true", help="保留生成的 WAV")
    ap.add_argument("--tone-freq", type=float, default=TONE_A_HZ, help="目标音频率")
    args = ap.parse_args()

    tmp = Path(__file__).resolve().parent.parent / "data" / "selftest_audio"
    tmp.mkdir(parents=True, exist_ok=True)
    wav_a = tmp / "tone_a.wav"
    wav_b = tmp / "tone_b.wav"

    print("=" * 74)
    print("进程级音频隔离验收实验")
    print("=" * 74)

    print(f"\n[1/6] 生成测试音：A={args.tone_freq:.0f}Hz  B={TONE_B_HZ:.0f}Hz")
    make_tone(wav_a, args.tone_freq)
    make_tone(wav_b, TONE_B_HZ)
    print(f"      {wav_a.name} / {wav_b.name}")

    players: list[subprocess.Popen] = []
    try:
        print("\n[2/6] 启动两个独立播放进程")
        pa, pid_a = spawn_player(wav_a, args.seconds + 12)
        pb, pid_b = spawn_player(wav_b, args.seconds + 12)
        players = [pa, pb]
        print(f"      进程A 启动器 pid={pa.pid} → 真实播放 pid={pid_a}（目标）")
        print(f"      进程B 启动器 pid={pb.pid} → 真实播放 pid={pid_b}（干扰源）")
        if pid_a <= 0 or pid_b <= 0:
            print("\n  ✗ 子进程未能上报真实 PID，实验无法继续")
            return 2

        print("\n[3/6] 通过音频会话 API 确认两者都在发声")
        ok_a = wait_for_session(pid_a)
        ok_b = wait_for_session(pid_b)
        print(f"      进程A 活跃会话: {'是' if ok_a else '否'}")
        print(f"      进程B 活跃会话: {'是' if ok_b else '否'}")
        if not (ok_a and ok_b):
            print("\n  ✗ 无法确认播放进程进入活跃音频会话，实验无法继续")
            print("    （可能原因：当前没有默认播放设备 / 会话枚举只覆盖默认设备）")
            return 2

        print("\n[4/6] 对进程A 做 WASAPI 进程回环采集")
        from proctap import ProcessAudioCapture

        cap = ProcessAudioCapture(pid_a)
        cap.start()
        fmt = cap.get_format()
        print(f"      实际格式: {fmt}")

        frames: list[np.ndarray] = []
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            data = cap.read(timeout=0.5)
            if data:
                frames.append(np.frombuffer(data, dtype=np.float32))
        cap.stop()
        cap.close()

        if not frames:
            print("\n  ✗ 采集不到任何数据")
            return 2

        samples = np.concatenate(frames)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        print(f"      采集 {samples.size} 样本（{samples.size / 2 / SAMPLE_RATE:.2f} 秒 @48k 立体声）")
        print(f"      峰值幅度: {peak:.4f}")

        print("\n[5/6] 频谱分析")
        mono = samples.reshape(-1, 2).mean(axis=1) if samples.size % 2 == 0 else samples
        peaks = dominant_peaks(mono, SAMPLE_RATE)
        print("      主要频率成分：")
        for f, a in peaks:
            print(f"        {f:8.1f} Hz   相对幅度 {a:.3f}")

        e_target = energy_at(mono, SAMPLE_RATE, args.tone_freq)
        e_interf = energy_at(mono, SAMPLE_RATE, TONE_B_HZ)
        print(f"\n      目标音 {args.tone_freq:.0f}Hz 能量占比: {e_target * 100:.2f}%")
        print(f"      干扰音 {TONE_B_HZ:.0f}Hz 能量占比: {e_interf * 100:.2f}%")

        print("\n[6/6] 判定")
        has_target = peak > 0.01 and e_target > 0.05
        # 干扰音能量必须显著低于目标音
        no_interference = e_interf < max(e_target * 0.2, 0.02)

        print(f"      {'✓' if has_target else '✗'} 抓到了目标进程的音频")
        print(f"      {'✓' if no_interference else '✗'} 未混入另一个进程的音频")

        if has_target and no_interference:
            print("\n  ✅ 通过：进程级音频隔离成立，项目前提得到验证\n")
            return 0
        print("\n  ❌ 未通过：需要进一步排查\n")
        return 1

    finally:
        for p in players:
            if p.poll() is None:
                p.terminate()
        if not args.keep_files:
            for f in (wav_a, wav_b):
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
