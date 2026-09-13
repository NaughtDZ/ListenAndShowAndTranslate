"""端到端实测：扩展送来的标签页音频**只有目标标签页那一路，并且没有副作用**。

要同时回答六件事：

1. 扩展能连上本程序并完成握手吗？
2. 浏览器会不会拒绝"用户没触发过"的采集请求？（这是安全策略，必须如实暴露）
3. 放行之后，扩展能真的把标签页音频送过来吗？
4. 送来的频谱里**只有目标标签页**吗？（另一个标签页同时在正常出声）
5. 另一个标签页的声音**有没有被影响**？（这条路线号称"零副作用"）
6. 目标标签页自己的声音**还在响吗**？（tabCapture 会把音频从默认输出摘走，
   扩展必须接回扬声器，否则用户就听不见了）

关于第 2 步与第 3 步之间的那道门——**Chromium 源码里的原话**（`tab_capture_api.cc`）::

    // Make sure either we have been granted permission to capture through an
    // extension icon click or our extension is allowlisted.
    if (!extension()->permissions_data()->HasAPIPermissionForTab(
            sessions::SessionTabHelper::IdForTab(target_contents).id(),
            mojom::APIPermissionID::kTabCaptureForTab) &&
        (GetAllowlistedExtensionID() != extension_id)) {
      return RespondNow(Error(kGrantError));
    }

也就是说：**授权是按标签页给的，只由"用户在该标签页上点扩展图标"产生**。
自动化测试里没人点图标，所以第二遍启动时用那个同样是官方的旁路开关
``--allowlisted-extension-id=<扩展ID>``（真实用户不需要、也不该用）。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_tab_capture_e2e.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_browser_tabs import find_edge, kill_probe_edge  # noqa: E402
from probe_tab_isolation import PROFILE, band_energy  # noqa: E402
from probe_tab_mute_extension import TONE_PAGE, audio_pid_for_instance  # noqa: E402

EXT_SRC = ROOT / "browser_extension"
WORK = ROOT / "data" / "tmp_tab_capture"
EXT = WORK / "ext"
PORT = 38991


class PageServer:
    """用 http://127.0.0.1 提供测试页（不用 file://，排除"浏览器内部页面不可采集"）。"""

    def __init__(self) -> None:
        pages = {
            "/a.html": TONE_PAGE.format(title="MTEST-441", freq=441),
            "/b.html": TONE_PAGE.format(title="MTEST-883", freq=883),
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的命名
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

            def log_message(self, *_args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        self.httpd.shutdown()


def build_extension() -> None:
    if EXT.exists():
        shutil.rmtree(EXT, ignore_errors=True)
    shutil.copytree(EXT_SRC, EXT)
    (EXT / "dev.json").write_text(
        json.dumps(
            {"autoStart": True, "autoStartTabTitle": "MTEST-441", "port": PORT},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def launch(edge: Path, url: str, *extra: str) -> None:
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE}",
            f"--load-extension={EXT}",
            f"--disable-extensions-except={EXT}",
            "--no-first-run",
            "--no-default-browser-check",
            "--autoplay-policy=no-user-gesture-required",
            "--remote-debugging-port=9333",
            *extra,
            "--new-window",
            url,
        ],
        close_fds=True,
    )


def open_second_tab(edge: Path, url: str) -> None:
    subprocess.Popen(
        [str(edge), f"--user-data-dir={PROFILE}", "--new-tab", url], close_fds=True
    )


def wait_for(predicate, timeout: float, step: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-console", action="store_true", help="不抓扩展控制台（默认抓，便于排错）")
    args = ap.parse_args()

    edge = find_edge()
    build_extension()
    pages = PageServer()
    shutil.rmtree(PROFILE, ignore_errors=True)

    from app.audio.capture import CaptureWorker, TargetSpec
    from app.audio.tab_audio import TabAudioServer
    from cdp_console import ConsoleTap

    tap = None
    if not args.no_console:
        tap = ConsoleTap(port=9333)
        tap.start()

    tab_chunks: list[np.ndarray] = []
    server = TabAudioServer(
        port=PORT,
        on_chunk=lambda c: tab_chunks.append(np.asarray(c, dtype=np.float32).copy()),
    )
    server.start()

    print(f"Edge         : {edge}")
    print(f"扩展（副本） : {EXT}")
    print(f"标签页通道   : ws://127.0.0.1:{server.port}/lst/tab")
    print(f"测试页       : {pages.url('/a.html')}(441Hz 目标) / {pages.url('/b.html')}(883Hz 陪跑)\n")

    # ---------------- 第一遍：如实暴露浏览器的拒绝 ----------------
    print("① 第一遍启动：不施加任何旁路，看浏览器会不会拒绝（预期会）…")
    launch(edge, pages.url("/b.html"))
    time.sleep(3.0)
    open_second_tab(edge, pages.url("/a.html"))

    connected = wait_for(lambda: server.snapshot().connected, 25)
    snap = server.snapshot()
    ext_id = snap.extension_id
    print(f"   扩展连上={connected} 扩展 ID={ext_id or '（没拿到）'}")
    time.sleep(4.0)
    snap = server.snapshot()
    print(f"   采集中={snap.capturing}")
    if snap.last_error:
        print(f"   浏览器拒绝原文：{snap.last_error}")

    if not ext_id:
        print("❌ 拿不到扩展 ID，后面的旁路实验做不了")
        server.stop()
        pages.stop()
        kill_probe_edge()
        return 2

    print("\n② 关掉这一遍（下面用 --allowlisted-extension-id 旁路掉「人手」这一步）…")
    kill_probe_edge()
    time.sleep(3.0)

    # ---------------- 第二遍：放行后验证音频通路 ----------------
    print("③ 第二遍启动：带 --allowlisted-extension-id=" + ext_id)
    launch(edge, pages.url("/b.html"), f"--allowlisted-extension-id={ext_id}")
    time.sleep(3.0)
    open_second_tab(edge, pages.url("/a.html"))

    started = wait_for(lambda: server.snapshot().capturing, 30)
    snap = server.snapshot()
    print(f"   连接={snap.connected} 采集中={snap.capturing} 目标={snap.tab.title!r}")
    if snap.tab.url:
        print(f"   来源 URL: {snap.tab.url}")
    if not started and snap.last_error:
        print(f"   ⚠️ 仍失败：{snap.last_error}")

    pid = audio_pid_for_instance()
    ref_chunks: list[np.ndarray] = []
    ref_worker = None
    if pid is not None:
        print(f"④ 同时采集整个 msedge 进程回环（PID={pid}）作为「用户实际听到的」对照…")
        ref_worker = CaptureWorker(
            TargetSpec(pid=pid),
            on_chunk=lambda c: ref_chunks.append(np.asarray(c, dtype=np.float32).copy()),
        )
        ref_worker.start()

    print(f"⑤ 采集 {args.seconds:.0f} 秒 …")
    time.sleep(args.seconds)
    if ref_worker is not None:
        ref_worker.stop()
    server.stop()
    pages.stop()
    snap = server.snapshot()  # 取最终计数（connected/capturing 会变 False，判定用前面的变量）

    rate = 16000
    tab_audio = np.concatenate(tab_chunks) if tab_chunks else np.zeros(0, dtype=np.float32)
    ref_audio = np.concatenate(ref_chunks) if ref_chunks else np.zeros(0, dtype=np.float32)
    print(f"   服务端计数：收帧={snap.frames} 收字节={snap.bytes_in} 输出样本={snap.samples_out}")
    print(f"   标签页通道 {tab_audio.size / rate:.1f}s；进程回环 {ref_audio.size / rate:.1f}s")
    if snap.frames and not tab_audio.size:
        print("   ⚠️ 帧到了但一个样本都没出管线——问题在程序侧的 AudioPipeline")
    WORK.mkdir(parents=True, exist_ok=True)
    if tab_audio.size:
        with wave.open(str(WORK / "tab_audio.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(tab_audio, -1, 1) * 32767).astype(np.int16).tobytes())

    def amp(audio: np.ndarray, freq: float) -> float:
        if audio.size < rate * 2:
            return 0.0
        seg = audio[int(audio.size * 0.15) : int(audio.size * 0.85)]
        return band_energy(seg, rate, freq)

    t441, t883 = amp(tab_audio, 441.0), amp(tab_audio, 883.0)
    r441, r883 = amp(ref_audio, 441.0), amp(ref_audio, 883.0)
    print("\n⑥ 频谱（幅度；测试音刻意压得很低，所以看的是**相对关系**而不是绝对值）")
    print(f"   标签页通道：441Hz={t441:.5f}  883Hz={t883:.5f}")
    print(f"   整个浏览器：441Hz={r441:.5f}  883Hz={r883:.5f}")

    floor = 2e-4
    results: list[tuple[bool, str]] = [
        (connected, "扩展能连上本程序并完成握手"),
        (bool(snap.extension_id), "能从 Origin 识别出扩展 ID"),
        (started, "放行后扩展真的开始送音频"),
        (t441 > floor and t441 > 5 * t883, f"目标标签页 441Hz 送到了本程序（{t441:.5f} vs 陪跑 {t883:.5f}）"),
        (t883 < 0.05 * max(t441, 1e-9), f"陪跑标签页 883Hz **没有**混进来（{t883:.5f}）"),
    ]
    if ref_audio.size:
        results.append(
            (r883 > 0.3 * r441 and r441 > floor, f"陪跑标签页照常出声、未被影响（{r883:.5f} vs {r441:.5f}）")
        )
        results.append((r441 > floor, f"目标标签页回放正常（tabCapture 没把声音吞掉）（{r441:.5f}）"))
    if ref_audio.size > rate * 4:
        hops = [
            band_energy(ref_audio[int(k * 0.5 * rate) : int((k + 1) * 0.5 * rate)], rate, 441.0)
            for k in range(int(ref_audio.size / rate / 0.5))
        ]
        results.append(
            (min(hops) > 0.3 * max(hops), f"目标回放全程未间断（{min(hops):.4f} / {max(hops):.4f}）")
        )

    print("\n⑦ 判定")
    for ok, label in results:
        print(f"   {'✅' if ok else '❌'} {label}")
    passed = sum(1 for ok, _ in results if ok)
    print(f"\n   结果：{passed}/{len(results)} 项通过")
    if passed == len(results):
        print("   ✅ 标签页级隔离成立，且零副作用（别的标签页照常有声、目标也照常有声）")

    if tap is not None:
        lines = tap.lines()
        print(f"\n⑧ 扩展控制台（{len(lines)} 行，最后 40 行）")
        for line in lines[-40:]:
            print(f"   {line}")
        tap.stop()

    if not args.keep:
        print(f"\n⑨ 已清理临时 Edge 进程 {kill_probe_edge()} 个")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
