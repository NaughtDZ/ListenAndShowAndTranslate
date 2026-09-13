"""判决性实验：Chromium(Edge) 到底按"进程"还是按"标签页"建 Windows 音频会话？

做法：
  1. 用**临时 user-data-dir** 起一个 Edge 实例（不碰用户正在用的 Edge），
     关掉首个运行引导、允许无手势自动播放。
  2. 打开两个播放不同频率正弦波的标签页（音量极小，几乎听不见）。
  3. 用音频会话 API 枚举 msedge.exe 的所有会话，看是 1 个还是 2 个。
  4. 杀掉临时实例（只杀命令行里带该 user-data-dir 的进程），恢复现场。

结论含义：
  * 若两个标签页 → 1 个会话：**进程级回环就是天花板**，Windows 层面拿不到标签页边界，
    必须靠浏览器内部（扩展 / tabCapture / CDP）才能分标签页。
  * 若两个标签页 → 2 个会话且 Identifier 不同、DisplayName 是页面标题：
    则可以做"枚举标签页 + 静音其它会话"的本体方案（有副作用：会把别的标签页静音）。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_browser_tabs.py
    .venv\\Scripts\\python.exe scripts\\probe_browser_tabs.py --keep   # 不杀进程（调试用）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PROFILE = ROOT / "data" / "tmp_browser_probe" / "profile"
PAGES = ROOT / "data" / "tmp_browser_probe" / "pages"

EDGE_CANDIDATES = [
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
]

PAGE_TMPL = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>body{{font:16px sans-serif;padding:2em}}</style>
<h1>{title}</h1>
<p>这是听·显·译的标签页音频边界测试页，正在播放 {freq}Hz 正弦波（音量极小）。</p>
<script>
async function go() {{
  const ctx = new AudioContext();
  await ctx.resume().catch(() => {{}});
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = {freq};
  const gain = ctx.createGain();
  gain.gain.value = 0.008;          // 需要"有活跃流"，但不想吵到人
  osc.connect(gain).connect(ctx.destination);
  osc.start();
}}
go();
</script>
"""


def find_edge() -> Path:
    for p in EDGE_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit("找不到 msedge.exe")


def write_pages() -> tuple[Path, Path]:
    PAGES.mkdir(parents=True, exist_ok=True)
    a = PAGES / "tone_a.html"
    b = PAGES / "tone_b.html"
    a.write_text(PAGE_TMPL.format(title="TONE-A-441Hz", freq=441), encoding="utf-8")
    b.write_text(PAGE_TMPL.format(title="TONE-B-883Hz", freq=883), encoding="utf-8")
    return a, b


def launch(edge: Path, *args: str) -> None:
    subprocess.Popen([str(edge), f"--user-data-dir={PROFILE}", *args], close_fds=True)


def msedge_sessions() -> list[tuple]:
    from probe_audio_sessions import collect

    return [r for r in collect() if "msedge" in (r[1] or "").lower()]


def kill_probe_edge() -> int:
    """只杀命令行里带临时 profile 路径的 msedge 进程。"""
    import psutil

    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (proc.info["name"] or "").lower() != "msedge.exe":
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            if str(PROFILE).lower() in cmdline.lower():
                proc.kill()
                killed += 1
        except Exception:  # noqa: BLE001
            continue
    return killed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="结束后不杀 Edge")
    args = ap.parse_args()

    edge = find_edge()
    a, b = write_pages()
    print(f"Edge      : {edge}")
    print(f"临时配置  : {PROFILE}")
    print(f"测试页    : {a.name} / {b.name}\n")

    print("① 启动临时 Edge 实例 + 标签页 A …")
    launch(
        edge,
        "--no-first-run",
        "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required",
        "--new-window",
        a.as_uri(),
    )
    time.sleep(3.0)

    s1 = msedge_sessions()
    print(f"   此时 msedge 会话数 = {len(s1)}")

    print("② 复用同一实例再开标签页 B（应作为第二个标签页加入同一窗口）…")
    launch(edge, "--new-tab", b.as_uri())
    time.sleep(4.0)

    s2 = msedge_sessions()
    print(f"   两个标签页同时出声时 msedge 会话数 = {len(s2)}\n")

    for pid, name, state, identifier, instance, display, icon, title in s2:
        print(f"   pid={pid} state={state} display={display!r}")
        print(f"     identifier: …{identifier[-64:]}")
        print(f"     instance  : …{instance[-64:]}")
        print(f"     window    : {title!r}")

    # 判定
    print("\n③ 判定")
    if len(s2) >= 2:
        uniq = {r[3] for r in s2}
        print(f"   ✅ 同一 msedge 进程有 {len(s2)} 个会话，其中 {len(uniq)} 个不同 Identifier")
        print("   → 浏览器**在 Windows 层面按标签页分了会话**，"
              "可做「枚举标签页 + 静音其它会话」的本体方案")
    else:
        print("   ❌ 两个标签页仍然只有 1 个会话")
        print("   → Windows 侧（音频会话 / 进程回环）**拿不到标签页边界**，"
              "进程级回环就是天花板；要分标签页只能走浏览器内部（扩展 tabCapture / CDP）")

    if not args.keep:
        killed = kill_probe_edge()
        print(f"\n④ 已清理临时 Edge 进程 {killed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
