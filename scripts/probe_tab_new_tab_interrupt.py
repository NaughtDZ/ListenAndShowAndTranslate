"""两个真实现场的端到端验证（都来自用户实测反馈）。

现场 A：**新建一个会出声的标签页**时，正在进行的发送不能被搅乱。
    用户实测："如果新建了一个播放音频的标签页，那么原本正在进行的音频发送会暂停，
    还得手动再点一次。"
    实测根因（服务端日志抓到的）：用户日常浏览器里装的那份扩展也连上来（另一个
    Origin），它以 control 身份 report ``capturing=false``，于是
      ① 程序状态被改成"等待开始采集"（看起来像暂停了）；
      ② 旧的淘汰策略"最旧先踢"把**正在送音频的那条**踢掉了（音频真断了一下）。
    修复后要求：capturing 保持 True、目标标题不被别人覆盖、音频连接不被踢。

现场 B：**目标标签页的音频结束**（视频播完/暂停）时，应当自动恢复采集，
    不该再让用户手动点一次。修复后扩展会自动重试 startCapture（授权还在）。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_tab_new_tab_interrupt.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cdp_console import ConsoleTap  # noqa: E402
from probe_browser_tabs import find_edge, kill_probe_edge  # noqa: E402
from probe_tab_capture_e2e import EXT, PORT, PageServer, build_extension, launch, open_second_tab  # noqa: E402
from probe_tab_isolation import PROFILE  # noqa: E402

# 目标页：低音量正弦，且**音频可被外部控制**（用于"音频结束"现场）
PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>body{{font:16px sans-serif;padding:2em}}</style>
<h1>{title}</h1>
<p>{freq} Hz，可被脚本停止/重启（用于测试"音频结束"）</p>
<script>
const FREQ = {freq};
let ctx = null, osc = null, gain = null;
function start() {{
  ctx = ctx || new AudioContext();
  ctx.resume();
  if (!gain) {{ gain = ctx.createGain(); gain.gain.value = 0.006; gain.connect(ctx.destination); }}
  if (osc) {{ try {{ osc.stop(); }} catch (e) {{}} }}
  osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = FREQ;
  osc.connect(gain);
  osc.start();
}}
function stop() {{ if (osc) {{ try {{ osc.stop(); }} catch (e) {{}} osc = null; }} }}
window.__lstControl = {{ start, stop, state: () => (osc ? "playing" : "stopped") }};
start();
</script>
"""


def http_pages() -> tuple[PageServer, str, str]:
    """用 PageServer 的机制提供我们自己的可控页面。"""

    class CtlServer(PageServer):
        def __init__(self) -> None:  # noqa: D107 - 覆盖掉父类的固定页面
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            import threading

            pages = {
                "/a.html": PAGE.format(title="MTEST-441", freq=441),
                "/b.html": PAGE.format(title="MTEST-883", freq=883),
            }

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    body = pages.get(self.path.split("?")[0])
                    if body is None:
                        self.send_error(404)
                        return
                    data = body.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, *_a):
                    pass

            self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.port = self.httpd.server_address[1]
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    srv = CtlServer()
    return srv, srv.url("/a.html"), srv.url("/b.html")


def page_control(expr: str) -> str:
    """通过 CDP 在目标页里执行 JS（用于"让音频结束/恢复"）。"""
    import httpx

    from app.audio.ws_client import MiniWSClient

    with httpx.Client(trust_env=False, timeout=3.0) as c:
        targets = c.get("http://127.0.0.1:9333/json/list").json()
    page = next(
        (t for t in targets if t.get("type") == "page" and "a.html" in (t.get("url") or "")),
        None,
    )
    if page is None:
        return "（找不到目标页 target）"
    cli = MiniWSClient(page["webSocketDebuggerUrl"], origin="").connect()
    try:
        cli.send_json({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expr}})
        deadline = time.time() + 5
        while time.time() < deadline:
            kind, payload = cli.recv(timeout=2)
            if kind != "text":
                continue
            msg = json.loads(payload)
            if msg.get("id") == 1:
                return json.dumps(msg.get("result", {}).get("result", {}), ensure_ascii=False)
    finally:
        cli.close()
    return "（超时）"


def sw_eval(ext_id: str, expr: str) -> str:
    """在扩展的 Service Worker 里执行 JS（用来驱动"音轨结束→自动恢复"这条路径）。

    做法正当性说明：``ended`` 事件没法用纯浏览器操作稳定触发
    （实测停掉 WebAudio 振荡器并不会结束 tabCapture 音轨），
    所以这里直接调用 SW 里真实存在的函数，验证的是**同一条代码路径**。
    """
    import httpx

    from app.audio.ws_client import MiniWSClient

    with httpx.Client(trust_env=False, timeout=3.0) as c:
        targets = c.get("http://127.0.0.1:9333/json/list").json()
    sw = next(
        (
            t
            for t in targets
            if (t.get("url") or "") == f"chrome-extension://{ext_id}/background.js"
        ),
        None,
    )
    if sw is None:
        return "（找不到扩展 SW target）"
    cli = MiniWSClient(sw["webSocketDebuggerUrl"], origin="").connect()
    try:
        cli.send_json(
            {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expr, "awaitPromise": True, "returnByValue": True},
            }
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            kind, payload = cli.recv(timeout=3)
            if kind != "text":
                continue
            msg = json.loads(payload)
            if msg.get("id") == 1:
                res = msg.get("result", {})
                if res.get("exceptionDetails"):
                    return f"异常：{res['exceptionDetails'].get('text')}"
                return json.dumps(res.get("result", {}), ensure_ascii=False)[:200]
    finally:
        cli.close()
    return "（超时）"


def ext_targets_brief() -> str:
    """列出浏览器里的扩展 target（排错用：判断扩展到底加载没有）。"""
    import httpx

    try:
        with httpx.Client(trust_env=False, timeout=2.0) as c:
            targets = c.get("http://127.0.0.1:9333/json/list").json()
    except Exception as exc:  # noqa: BLE001
        return f"（查不到：{exc}）"
    ext = [
        f"{t.get('type')}:{(t.get('url') or '')[:80]}"
        for t in targets
        if (t.get("url") or "").startswith("chrome-extension://")
    ]
    return "; ".join(ext) or "（一个都没有）"


def kill_by_profile(profile: Path) -> int:
    """杀掉用指定 user-data-dir 起的 Edge（多个临时实例互不干扰）。"""
    import psutil

    killed = 0
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            if (proc.info["name"] or "").lower() != "msedge.exe":
                continue
            if str(profile).lower() in " ".join(proc.info["cmdline"] or []).lower():
                proc.kill()
                killed += 1
        except Exception:  # noqa: BLE001
            continue
    return killed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    edge = find_edge()
    build_extension()
    pages, url_a, url_b = http_pages()
    shutil.rmtree(PROFILE, ignore_errors=True)

    from app.audio.tab_audio import TabAudioServer
    from ext_id import unpacked_extension_id

    ext_id = unpacked_extension_id(EXT)
    server = TabAudioServer(port=PORT)
    server.start()
    tap = ConsoleTap(port=9333)
    tap.start()

    print(f"Edge: {edge}")
    print(f"扩展 ID（离线算得）: {ext_id}")
    print(f"标签页通道: ws://127.0.0.1:{server.port}/lst/tab\n")

    print("① 启动临时 Edge（带旁路开关：自动化里没人点扩展图标）")
    launch(edge, url_a, f"--allowlisted-extension-id={ext_id}")
    deadline = time.time() + 90
    while time.time() < deadline and not server.snapshot().capturing:
        time.sleep(0.3)
    snap = server.snapshot()
    print(f"   采集中={snap.capturing} 目标={snap.tab.title!r}")
    if not snap.capturing:
        print(f"   ⚠️ 没开始采集：connected={snap.connected} 连接数={snap.connections} "
              f"错误={snap.last_error!r}")
        print("   扩展控制台：")
        for line in tap.lines()[-15:]:
            print(f"     {line}")
        print(f"   /json/list 里的扩展 target：{ext_targets_brief()}")
        server.stop()
        pages.stop()
        tap.stop()
        kill_probe_edge()
        return 2

    time.sleep(3.0)
    base = server.snapshot()
    print(f"   基线：帧={base.frames}（{base.frames / 3:.0f} 帧/秒）")

    # ---------------- 现场 A ----------------
    print("\n② 现场 A：新建一个会出声的标签页（883Hz）")
    open_second_tab(edge, url_b)
    time.sleep(8.0)
    a = server.snapshot()
    grew = a.frames - base.frames
    print(f"   8 秒后：新增帧={grew} 采集中={a.capturing} 目标={a.tab.title!r}")
    print(f"   提示信息：{a.peer_note or '（无）'}")
    checks_a = [
        (grew > 300, f"音频没有中断（新增 {grew} 帧，正常约 400）"),
        (a.capturing is True, "状态仍是「正在采集」（不被另一个扩展实例改掉）"),
        (a.tab.title == "MTEST-441", f"目标标题没被覆盖（{a.tab.title!r}）"),
    ]

    # ---------------- 现场 C：另一个目录装的同一个扩展（= 用户日常浏览器那份） ----------------
    print("\n③ 现场 C：另一个目录装的同一个扩展（扩展 ID 不同）也连上来")
    foreign_ext = EXT.parent / "ext_foreign"
    shutil.rmtree(foreign_ext, ignore_errors=True)
    shutil.copytree(EXT, foreign_ext)  # 换个路径 → 扩展 ID 就不同（模拟用户日常那份）
    foreign_profile = PROFILE.parent / "profile_foreign"
    shutil.rmtree(foreign_profile, ignore_errors=True)
    print(f"   外来扩展 ID: {unpacked_extension_id(foreign_ext)}")
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={foreign_profile}",
            f"--load-extension={foreign_ext}",
            f"--disable-extensions-except={foreign_ext}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            url_b,
        ],
        close_fds=True,
    )
    deadline = time.time() + 60
    while time.time() < deadline and not server.snapshot().peer_note:
        time.sleep(0.5)
    time.sleep(3.0)
    c = server.snapshot()
    print(f"   提示信息：{c.peer_note or '（没检测到第二个实例）'}")
    print(f"   采集中={c.capturing} 目标={c.tab.title!r} 帧={c.frames}")
    checks_c = [
        (bool(c.peer_note), "检测到「另一个浏览器里的扩展也在连」并如实提示"),
        (c.capturing is True, "第二个实例没有把采集状态说成停了"),
        (c.tab.title == "MTEST-441", "目标标题仍是原来那个"),
        (c.frames > a.frames, f"音频没有被第二个实例挤断（{c.frames} > {a.frames}）"),
    ]
    kill_by_profile(foreign_profile)

    # ---------------- 现场 B：音轨结束 → 自动恢复 ----------------
    print("\n④ 现场 B：音轨结束 → 扩展自动恢复（不用手动再点）")
    tab_id = server.snapshot().tab.id
    print(f"   先模拟「音轨结束」：{sw_eval(ext_id, 'stopCapture(\"simulated-ended\")')}")
    time.sleep(2.0)
    before_resume = server.snapshot()
    print(f"   停之后：采集中={before_resume.capturing} 帧={before_resume.frames}")
    print(f"   触发自动恢复：{sw_eval(ext_id, f'resumeAfterTrackEnded({tab_id})')}")
    deadline = time.time() + 20
    while time.time() < deadline:
        s = server.snapshot()
        if s.capturing and s.frames > before_resume.frames + 50:
            break
        time.sleep(0.5)
    b2 = server.snapshot()
    print(f"   最终：采集中={b2.capturing} 帧={b2.frames}（恢复后新增 {b2.frames - before_resume.frames}）")
    checks_b = [
        (b2.capturing and b2.frames > before_resume.frames + 50,
         f"音轨结束后自动恢复采集（新增 {b2.frames - before_resume.frames} 帧，无需手动再点）"),
    ]

    print("\n⑤ 扩展控制台（关键行）")
    for line in tap.lines():
        if any(
            k in line
            for k in ("音轨结束", "自动尝试恢复", "已自动恢复", "停止采集", "音频通道断开",
                      "音频通道已连上", "开始采集")
        ):
            print(f"   {line}")

    print("\n⑥ 判定")
    ok_all = True
    for ok, label in checks_a + checks_c + checks_b:
        ok_all = ok_all and ok
        print(f"   {'✅' if ok else '❌'} {label}")

    server.stop()
    pages.stop()
    tap.stop()
    if not args.keep:
        print(f"\n⑥ 已清理临时 Edge 进程 {kill_probe_edge()} 个")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
