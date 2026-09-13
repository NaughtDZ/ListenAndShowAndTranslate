"""判决性实验二：**标签页级别的静音，会不会从进程回环采集里消失？**

背景：实测已知 Chromium 在 Windows 上"一个浏览器实例 = 一个音频会话"，
      所以 WASAPI 层面分不出标签页。但还有一条低成本路线：

          只采集整个 msedge 进程，同时把**其它标签页静音**，
          那么混音里就只剩目标标签页的声音。

      这条路线成立的前提是：标签页静音必须发生在**进程回环采样点之前**。
      Chromium 的标签页静音 = 把该标签页的音频流音量置 0（流还活着）。
      本实验用 WebAudio 的 GainNode 做等价操作（0.3 → 0 → 0.3 可逆切换），
      采集 msedge 进程回环，做频谱看对应频率有没有乖乖消失/回来。

时间线（页面内计时，从页面加载算起）：
  A 页（441Hz）：0–15s 出声 → 15–35s 增益归零（等价于标签页静音）→ 35s 后恢复
  B 页（883Hz）：全程出声

⚠️ 实测：`ProcessAudioCapture.start()` 到真正出数据要 ~6s，
   所以采集起点比页面起点晚 ~10s——**判定不依赖 offset**，
   相位是从能量时间线里自动识别出来的。

判定：若存在一段"441Hz ≈ 0 而 883Hz 基本不变"的时间段，
      且该段之后 441Hz 恢复 → 静音隔离成立，"静音其它标签页"路线可用。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_tab_isolation.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_browser_tabs import EDGE_CANDIDATES, PROFILE, find_edge  # noqa: E402

PAGES = ROOT / "data" / "tmp_browser_probe" / "pages"
REC = ROOT / "data" / "tmp_browser_probe" / "isolation.wav"

# 可逆音频时序页：gain 曲线由 JS 定时器切换
PAGE_TMPL = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>body{{font:16px sans-serif;padding:2em}} b{{font-size:1.6em}}</style>
<h1>{title}</h1>
<p>频率 <b>{freq} Hz</b>　增益曲线：<span id="s">-</span></p>
<script>
const FREQ = {freq};
const CURVE = {curve};   // [[时刻秒, 增益], ...]
(async () => {{
  const ctx = new AudioContext();
  await ctx.resume().catch(() => {{}});
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = FREQ;
  const g = ctx.createGain();
  g.gain.value = CURVE.length ? CURVE[0][1] : 0.006;   // 音量刻意压低：测试只需要频谱可辨
  osc.connect(g).connect(ctx.destination);
  osc.start();                      // 流全程存活，只改增益（等价于标签页静音）
  const t0 = performance.now() / 1000;
  setInterval(() => {{
    const t = performance.now() / 1000 - t0;
    let v = CURVE.length ? CURVE[0][1] : 0.3;
    for (const [at, gain] of CURVE) if (t >= at) v = gain;
    g.gain.value = v;
    document.getElementById("s").textContent =
      t.toFixed(1) + "s → gain " + v.toFixed(2) + (v > 0 ? " ♪" : "（静音）");
  }}, 200);
}})();
</script>
"""

CURVE_A = "[[0,0.006],[15,0],[35,0.006]]"  # 出声 → 归零（等价静音）→ 恢复
CURVE_B = "[[0,0.006]]"                   # 全程出声


def write_pages() -> tuple[Path, Path]:
    PAGES.mkdir(parents=True, exist_ok=True)
    a = PAGES / "curve_a.html"
    b = PAGES / "curve_b.html"
    a.write_text(PAGE_TMPL.format(title="CURVE-A-441Hz", freq=441, curve=CURVE_A), encoding="utf-8")
    b.write_text(PAGE_TMPL.format(title="CURVE-B-883Hz", freq=883, curve=CURVE_B), encoding="utf-8")
    return a, b


def instance_pids() -> list[int]:
    """临时实例的进程（Chromium 会把 --user-data-dir 传给所有子进程）。"""
    import psutil

    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (proc.info["name"] or "").lower() != "msedge.exe":
                continue
            if str(PROFILE).lower() in " ".join(proc.info["cmdline"] or []).lower():
                out.append(proc.info["pid"])
        except Exception:  # noqa: BLE001
            continue
    return out


def audio_pid_for_instance() -> int | None:
    from probe_audio_sessions import collect

    pids = set(instance_pids())
    for pid, name, *_ in collect():
        if pid in pids and "msedge" in (name or "").lower():
            return pid
    return None


def band_energy(sig: np.ndarray, rate: int, freq: float, half_hz: float = 25.0) -> float:
    """指定频点附近 ±half_hz 的幅度（单边谱）。"""
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win)) / (len(sig) / 2)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / rate)
    mask = (freqs >= freq - half_hz) & (freqs <= freq + half_hz)
    return float(spec[mask].max()) if mask.any() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    edge = find_edge()
    a, b = write_pages()
    print(f"Edge: {edge}\n")

    print("① 启动临时 Edge（关自动播放限制）+ 两个标签页 …")
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE}",
            "--no-first-run",
            "--no-default-browser-check",
            "--autoplay-policy=no-user-gesture-required",
            "--new-window",
            a.as_uri(),
        ],
        close_fds=True,
    )
    time.sleep(2.5)
    subprocess.Popen(
        [str(edge), f"--user-data-dir={PROFILE}", "--new-tab", b.as_uri()], close_fds=True
    )
    time.sleep(1.5)

    pid = audio_pid_for_instance()
    if pid is None:
        print("❌ 找不到临时实例的音频会话 PID")
        return 1
    print(f"   临时实例音频会话 PID = {pid}（该实例 {len(instance_pids())} 个进程）")

    print("② 采集该进程回环 45 秒（A 页时序：0–15s 出声，15–35s 增益归零，35s 后恢复）…")
    from app.audio.capture import CaptureWorker, TargetSpec

    frames: list[np.ndarray] = []

    def _collect(chunk) -> None:
        frames.append(np.asarray(chunk, dtype=np.float32).copy())

    worker = CaptureWorker(TargetSpec(pid=pid), on_chunk=_collect)
    t_started = time.time()
    worker.start()
    warm = None
    while time.time() - t_started < 45.0:
        if warm is None:
            snap = worker.snapshot()
            if snap.running and snap.chunks > 0:
                warm = time.time() - t_started
        time.sleep(0.2)
    worker.stop()
    if warm is not None:
        print(f"   ⏱ 采集预热（start → 第一块数据）= {warm:.1f}s")

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    rate = 16000
    dur = audio.size / rate
    print(f"   采集到 {audio.size} 采样 = {dur:.1f}s（{REC.name}）")
    if dur < 20:
        print("❌ 采集太短，实验无效")
        return 1

    with wave.open(str(REC), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    # ---- 逐 hop 能量，相位从数据里认，不依赖 offset --------------------------
    hop = 0.5
    timeline = []
    for k in range(int(dur / hop)):
        seg = audio[int(k * hop * rate) : int((k + 1) * hop * rate)]
        timeline.append((k * hop, band_energy(seg, rate, 441.0), band_energy(seg, rate, 883.0)))

    lvl441 = max(e1 for _, e1, _ in timeline) or 1e-9
    lvl883 = max(e2 for _, _, e2 in timeline) or 1e-9
    print("\n③ 能量时间线（采集时间，每 0.5s；柱长按各自最大值归一）")
    for t, e1, e2 in timeline:
        mark = "4" if e1 > 0.5 * lvl441 else ("." if e1 < 0.05 * lvl441 else "-")
        print(
            f"   {t:>6.1f}s  441Hz={e1:.5f} [{mark}] 883Hz={e2:.5f}"
            f"  {'#' * int(28 * e1 / lvl441)}|{'#' * int(28 * e2 / lvl883)}"
        )

    # 持续 ≥2s 的"441 归零但 883 还在"段 = 静音隔离段
    muted = [t for t, e1, e2 in timeline if e1 < 0.05 * lvl441 and e2 > 0.5 * lvl883]
    loud = [t for t, e1, _ in timeline if e1 > 0.5 * lvl441]

    def runs(ts: list[float]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for t in ts:
            if out and abs(t - out[-1][1]) <= hop + 1e-6:
                out[-1] = (out[-1][0], t)
            else:
                out.append((t, t))
        return [(a, b) for a, b in out if b - a >= 2.0]

    muted_runs = runs(muted)
    loud_runs = runs(loud)

    print("\n④ 自动识别出的相位")
    print(f"   441Hz 有声段：{[(round(a, 1), round(b, 1)) for a, b in loud_runs] or '无'}")
    print(f"   441Hz 归零段（883Hz 仍在）：{[(round(a, 1), round(b, 1)) for a, b in muted_runs] or '无'}")

    print("\n⑤ 判定")
    ok = False
    if muted_runs and loud_runs:
        m_start, m_end = max(muted_runs, key=lambda ab: ab[1] - ab[0])
        seg = audio[int(m_start * rate) : int(m_end * rate)]
        m441 = band_energy(seg, rate, 441.0)
        m883 = band_energy(seg, rate, 883.0)
        # 静音段之前的"有声"参考
        before = [t for t in loud if t < m_start]
        b441 = b883 = 0.0
        if before:
            bseg = audio[int(before[0] * rate) : int((before[-1] + hop) * rate)]
            b441 = band_energy(bseg, rate, 441.0)
            b883 = band_energy(bseg, rate, 883.0)
        after = [t for t in loud if t > m_end]
        a441 = 0.0
        if after:
            aseg = audio[int(after[0] * rate) : int((after[-1] + hop) * rate)]
            a441 = band_energy(aseg, rate, 441.0)

        drop = 20 * np.log10((m441 + 1e-12) / (b441 + 1e-12))
        keep = 20 * np.log10((m883 + 1e-12) / (b883 + 1e-12))
        ret = 20 * np.log10((a441 + 1e-12) / (b441 + 1e-12))
        print(f"   归零段 {m_start:.1f}–{m_end:.1f}s：441Hz {drop:+.1f} dB，883Hz {keep:+.1f} dB")
        print(f"   归零段之后 441Hz 恢复：{ret:+.1f} dB")
        ok = drop < -20 and keep > -3 and ret > -3
        print(
            "   ✅ 标签页静音能从进程回环里消失，且可逆、不影响别的标签页 → 「静音其它标签页」路线成立"
            if ok
            else "   ❌ 标签页静音没有从进程回环里消失（或不可逆 / 干扰了别的标签页）→ 该路线不成立"
        )
    else:
        print("   ❌ 没有识别到完整的「有声 → 归零 → 恢复」三段，实验无效（时序或自动播放被拦）")

    if not args.keep:
        from probe_browser_tabs import kill_probe_edge

        print(f"\n⑥ 已清理临时 Edge 进程 {kill_probe_edge()} 个")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
