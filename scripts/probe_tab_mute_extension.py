"""判决性实验三：**真实标签页静音能不能驱动进程回环的标签页级隔离。**

实验二证明了"标签页增益归零会从进程回环里消失"，但那是页面自己的 WebAudio 增益。
本实验用真正的 `chrome.tabs.update({muted:true})`，也就是浏览器标签页上的静音。

探针扩展（现场生成在 data/tmp_ext_probe/ext）按时间表动作：
    12s → 把**所有**正在发声的标签页静音 4 秒   ← 存活标记：两路信号同时归零，
                                                  页面常开曲线做不出这种事，藏不住
    16s → 全部取消静音
    22s → 静音**除目标标签页之外**所有正在发声的标签页
    30s → 全部取消静音
    38s → 再次静音其它标签页

（实测：采集真正开始比 service worker 启动晚约 6s，所以标记排在 12s。）

页面（都是常开恒定音量，没有任何自带曲线）：
    MTEST-441（目标）  441Hz
    MTEST-883（陪跑）  883Hz

判定（全部从数据自校准，不依赖任何 offset 假设）：
    1) 存在一段"441 与 883 同时归零"的标记窗口 → 扩展确实在动标签页
    2) 标记之后 883 消失、而 441 保留        → 标签页级隔离成立
    3) 随后 883 回来                         → 可逆
    4) 再随后 883 又消失                     → 可重复
    5) 除标记窗口外 441 从不掉电平           → 目标标签页全程未被影响

用法::

    .venv\\Scripts\\python.exe scripts\\probe_tab_mute_extension.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_browser_tabs import find_edge, kill_probe_edge  # noqa: E402
from probe_tab_isolation import PROFILE, band_energy, instance_pids  # noqa: E402

EXT = ROOT / "data" / "tmp_ext_probe" / "ext"
PAGES = ROOT / "data" / "tmp_ext_probe" / "pages"

TARGET_TITLE = "MTEST-441"

TONE_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>body{{font:16px sans-serif;padding:2em}}</style>
<h1>{title}</h1>
<p>听·显·译 标签页静音探针：{freq} Hz 常开，恒定增益</p>
<script>
(async () => {{
  const ctx = new AudioContext();
  await ctx.resume().catch(() => {{}});
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = {freq};
  const g = ctx.createGain();
  g.gain.value = 0.006;             // 音量刻意压到约 -44 dBFS：测试只需要频谱可辨，不该吵到人
  osc.connect(g).connect(ctx.destination);
  osc.start();
}})();
</script>
"""

MANIFEST = {
    "manifest_version": 3,
    "name": "LST Tab Mute Probe",
    "version": "0.1",
    "description": "一次性探针：验证真实标签页静音是否影响进程回环采集",
    "permissions": ["tabs"],
    "background": {"service_worker": "background.js"},
}

BACKGROUND = """// 一次性探针：按时间表动标签页，每个动作都会在音频采集里留下痕迹。
const TARGET_PREFIX = "MTEST-441";

function log(msg) { console.log("[lst-probe]", msg); }

async function reportTabs(tag) {
  const tabs = await chrome.tabs.query({});
  log(tag + " tabs=" + tabs.map(t => `${t.id}:${t.muted ? "MUTED" : "on"}:${t.audible ? "audible" : "-"}:${(t.title || "").slice(0, 14)}`).join(" | "));
}

// 存活标记：连目标一起静音 —— 两路信号同时归零，采集里一眼可见
async function muteAll(muted) {
  const tabs = await chrome.tabs.query({});
  let n = 0;
  for (const t of tabs) {
    if (!t.audible && !t.muted) continue;
    try { await chrome.tabs.update(t.id, { muted }); n++; } catch (e) { log("mute 失败: " + e); }
  }
  log(`muteAll(${muted}) 命中 ${n} 个`);
  await reportTabs(muted ? "全静音后" : "全恢复后");
}

async function muteOthers(muted) {
  const tabs = await chrome.tabs.query({});
  const target = tabs.find(t => (t.title || "").startsWith(TARGET_PREFIX));
  if (!target) { log("找不到目标标签页，放弃"); return; }
  let n = 0;
  for (const t of tabs) {
    if (t.id === target.id) continue;
    if (!t.audible) continue;
    try { await chrome.tabs.update(t.id, { muted }); n++; } catch (e) { log("mute 失败: " + e); }
  }
  log(`muteOthers(${muted}) 命中 ${n} 个（目标 tab=${target.id}）`);
  await reportTabs(muted ? "其它静音后" : "其它恢复后");
}

async function schedule() {
  log("探针启动");
  await reportTabs("初始");
  // 实测：浏览器冷启动耗时不稳定（采集真正开始比 SW 晚 6–16s），
  // 所以不做一次性时间表，改成 12s 周期的状态机，任何 offset 都必然覆盖到各相位：
  //   相位 0–3s  : 全部静音（存活标记，两路同时归零）
  //   相位 3–6s  : 全部恢复
  //   相位 6–12s : 只静音其它标签页（目标不受影响）
  const t0 = Date.now();
  let last = "";
  setInterval(async () => {
    const el = (Date.now() - t0) / 1000;
    const phase = el % 12;
    const want = phase < 3 ? "all-muted" : phase < 6 ? "all-on" : "others-muted";
    if (want === last) return;
    last = want;
    log(`相位 ${phase.toFixed(1)}s → ${want}`);
    if (want === "all-muted") await muteAll(true);
    else if (want === "all-on") await muteAll(false);
    else await muteOthers(true);
  }, 1000);
}

chrome.runtime.onInstalled.addListener(schedule);
chrome.runtime.onStartup.addListener(schedule);
schedule();
"""


def build_extension() -> tuple[Path, Path]:
    EXT.mkdir(parents=True, exist_ok=True)
    (EXT / "manifest.json").write_text(
        json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EXT / "background.js").write_text(BACKGROUND, encoding="utf-8")
    PAGES.mkdir(parents=True, exist_ok=True)
    a = PAGES / "mtest_a.html"
    b = PAGES / "mtest_b.html"
    a.write_text(TONE_PAGE.format(title=TARGET_TITLE, freq=441), encoding="utf-8")
    b.write_text(TONE_PAGE.format(title="MTEST-883", freq=883), encoding="utf-8")
    return a, b


def audio_pid_for_instance() -> int | None:
    from probe_audio_sessions import collect

    pids = set(instance_pids())
    for pid, name, *_ in collect():
        if pid in pids and "msedge" in (name or "").lower():
            return pid
    return None


def runs_of(flags: list[bool], times: list[float], min_len: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for t, f in zip(times, flags):
        if f:
            if out and abs(t - out[-1][1]) <= (times[1] - times[0]) + 1e-6:
                out[-1] = (out[-1][0], t)
            else:
                out.append((t, t))
    return [(a, b) for a, b in out if b - a >= min_len - 1e-6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--seconds", type=float, default=50.0)
    args = ap.parse_args()

    edge = find_edge()
    a, b = build_extension()
    print(f"Edge      : {edge}")
    print(f"探针扩展  : {EXT}\n")

    print("① 启动临时 Edge + 探针扩展 + 两个常开标签页 …")
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE}",
            f"--load-extension={EXT}",
            f"--disable-extensions-except={EXT}",
            "--no-first-run",
            "--no-default-browser-check",
            "--autoplay-policy=no-user-gesture-required",
            "--new-window",
            a.as_uri(),
        ],
        close_fds=True,
    )
    time.sleep(3.0)
    subprocess.Popen(
        [str(edge), f"--user-data-dir={PROFILE}", "--new-tab", b.as_uri()], close_fds=True
    )
    time.sleep(1.5)

    pid = audio_pid_for_instance()
    if pid is None:
        print("❌ 找不到临时实例的音频会话 PID")
        if not args.keep:
            kill_probe_edge()
        return 1
    print(f"   临时实例音频会话 PID = {pid}（{len(instance_pids())} 个进程）")

    print(f"② 采集该进程回环 {args.seconds:.0f} 秒 …")
    from app.audio.capture import CaptureWorker, TargetSpec

    frames: list[np.ndarray] = []

    def _collect(chunk) -> None:
        frames.append(np.asarray(chunk, dtype=np.float32).copy())

    worker = CaptureWorker(TargetSpec(pid=pid), on_chunk=_collect)
    t0 = time.time()
    worker.start()
    while time.time() - t0 < args.seconds:
        time.sleep(0.2)
    worker.stop()

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    rate = 16000
    dur = audio.size / rate
    print(f"   采集到 {dur:.1f}s")

    hop = 0.5
    times: list[float] = []
    e441s: list[float] = []
    e883s: list[float] = []
    for k in range(int(dur / hop)):
        seg = audio[int(k * hop * rate) : int((k + 1) * hop * rate)]
        times.append(k * hop)
        e441s.append(band_energy(seg, rate, 441.0))
        e883s.append(band_energy(seg, rate, 883.0))

    l441 = max(e441s) or 1e-9
    l883 = max(e883s) or 1e-9
    on441 = [v > 0.4 * l441 for v in e441s]
    on883 = [v > 0.4 * l883 for v in e883s]

    print("\n③ 能量时间线（采集时间，每 0.5s；4=441Hz 有声，8=883Hz 有声）")
    for t, e1, e2, a1, b1 in zip(times, e441s, e883s, on441, on883):
        print(
            f"   {t:>5.1f}s  441={e1:.4f}{'4' if a1 else '.'}  883={e2:.4f}{'8' if b1 else '.'}"
            f"   {'#' * int(20 * e1 / l441)}|{'#' * int(20 * e2 / l883)}"
        )

    both_off = [not a1 and not b1 for a1, b1 in zip(on441, on883)]
    both_on = [a1 and b1 for a1, b1 in zip(on441, on883)]
    only441 = [a1 and not b1 for a1, b1 in zip(on441, on883)]
    markers = runs_of(both_off, times, 2.0)
    restores = runs_of(both_on, times, 2.0)
    isolates = runs_of(only441, times, 4.0)

    print("\n④ 自动识别的相位（全部从数据来，不用任何 offset 假设）")
    print(f"   ①全部静音（标记）      {[(round(a,1), round(b,1)) for a, b in markers]}")
    print(f"   ②全部有声              {[(round(a,1), round(b,1)) for a, b in restores]}")
    print(f"   ③只有 441Hz（目标保留）{[(round(a,1), round(b,1)) for a, b in isolates]}")

    print("\n⑤ 判定")
    results: list[tuple[bool, str]] = []
    results.append((bool(markers), f"扩展在动标签页（出现「全部静音」标记窗口 {len(markers)} 段）"))
    results.append((bool(isolates), f"标签页级隔离（出现「仅目标有声、其它归零」{len(isolates)} 段，≥4s）"))
    results.append((bool(restores), f"可逆（标记/隔离之间回到全部有声 {len(restores)} 段）"))
    # 目标是否只在标记窗口里消失
    in_marker = [False] * len(times)
    for a, b in markers:
        for i, t in enumerate(times):
            if a <= t <= b:
                in_marker[i] = True
    bad = [times[i] for i, a1 in enumerate(on441) if not a1 and not in_marker[i]]
    results.append((not bad, f"目标 441Hz 只在标记窗口里消失（违规时刻：{[round(t,1) for t in bad][:8] or '无'}）"))
    # 标记以外 883 与 441 的相关性：≥2 段独立归零说明是"别的标签页"而不是"整机静音"
    results.append((len(isolates) >= 2, f"可重复（独立归零段数 = {len(isolates)}）"))

    for ok, label in results:
        print(f"   {'✅' if ok else '❌'} {label}")

    passed = sum(1 for ok, _ in results if ok)
    print(f"\n⑥ 结果：{passed}/{len(results)} 项通过")
    if all(ok for ok, _ in results):
        print("   ✅ 真实标签页静音 = 进程回环里的标签页级隔离（目标不受影响、可逆、可重复）")
        print("   → 「扩展只负责静音，采集/识别/翻译全用现有管线」这条路线成立")

    if not args.keep:
        print(f"\n⑦ 已清理临时 Edge 进程 {kill_probe_edge()} 个")
    return 0 if all(ok for ok, _ in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
