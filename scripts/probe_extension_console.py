"""诊断：把浏览器扩展 Service Worker 的控制台/异常抓出来。

为什么要这个：扩展跑在浏览器里，``console.log`` 和未捕获异常**根本不会**出现在
我们的日志里（pythonw 还会吞掉自己的 traceback）。排查扩展问题时，唯一的真相
来源就是它自己的控制台，而 CDP 是拿到它的最直接办法。

做法：
  1. 用**全新**的临时配置启动 Edge，打开 `--remote-debugging-port`；
  2. ``GET /json/list`` 列出所有 target，挑出扩展的 service worker；
  3. 连上它的 webSocketDebuggerUrl，开 Runtime/Log，把事件打到 stdout。

（之所以能开远程调试端口：用的是临时 user-data-dir。Chrome/Edge 136+ 禁止对
**默认**配置文件开这个端口，那会连带暴露用户的登录态。）

用法::

    .venv\\Scripts\\python.exe scripts\\probe_extension_console.py
    .venv\\Scripts\\python.exe scripts\\probe_extension_console.py --extension 目录 --seconds 15
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

from probe_browser_tabs import find_edge, kill_probe_edge  # noqa: E402
from probe_tab_isolation import PROFILE  # noqa: E402

PORT = 9333


def targets(port: int) -> list[dict]:
    import httpx

    with httpx.Client(trust_env=False, timeout=3.0) as c:
        return c.get(f"http://127.0.0.1:{port}/json/list").json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extension", default=str(ROOT / "browser_extension"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fresh-profile", action="store_true", default=True)
    args = ap.parse_args()

    ext = Path(args.extension).resolve()
    if not ext.is_dir():
        print(f"找不到扩展目录：{ext}")
        return 2

    if args.fresh_profile:
        shutil.rmtree(PROFILE, ignore_errors=True)

    edge = find_edge()
    print(f"Edge     : {edge}")
    print(f"扩展目录 : {ext}")
    print(f"临时配置 : {PROFILE}\n")

    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE}",
            f"--load-extension={ext}",
            f"--disable-extensions-except={ext}",
            f"--remote-debugging-port={PORT}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        close_fds=True,
    )

    # 等调试端口起来，列出所有 target
    ext_targets: list[dict] = []
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            all_targets = targets(PORT)
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
            continue
        ext_targets = [
            t
            for t in all_targets
            if t.get("type") == "service_worker" and t.get("url", "").startswith("chrome-extension://")
        ]
        if ext_targets:
            break
        time.sleep(0.5)

    try:
        all_targets = targets(PORT)
    except Exception as exc:  # noqa: BLE001
        print(f"列 target 失败：{exc}")
        kill_probe_edge()
        return 1

    print("全部 target：")
    for t in all_targets:
        print(f"   [{t.get('type')}] {t.get('url')}")
    print()

    if not ext_targets:
        print("❌ 没有任何扩展 service worker（扩展可能没加载）")
        kill_probe_edge()
        return 1

    from app.audio.ws_client import MiniWSClient

    clients: list[tuple[str, MiniWSClient]] = []
    for t in ext_targets:
        try:
            cli = MiniWSClient(t["webSocketDebuggerUrl"], origin="").connect()
        except Exception as exc:  # noqa: BLE001
            print(f"（连不上 {t.get('url')}：{exc}）")
            continue
        cli.sock.settimeout(0.2)
        cli.send_json({"id": 1, "method": "Runtime.enable"})
        cli.send_json({"id": 2, "method": "Log.enable"})
        clients.append((t["url"], cli))
        print(f"已挂上：{t.get('url')}")

    if not clients:
        kill_probe_edge()
        return 1

    print(f"\n抓取 {args.seconds:.0f} 秒控制台输出…\n" + "-" * 70)
    end = time.time() + args.seconds
    while time.time() < end:
        for url, cli in clients:
            short = url.split("/")[2] if "/" in url else url
            try:
                kind, payload = cli.recv(timeout=0.2)
            except TimeoutError:
                continue
            except OSError:
                continue
            if kind != "text":
                continue
            msg = json.loads(payload)
            method = msg.get("method")
            if method == "Runtime.consoleAPICalled":
                parts = []
                for a in msg["params"].get("args", []):
                    parts.append(str(a.get("value") if "value" in a else a.get("description", "")))
                print(f"[{short}] {msg['params'].get('type')}: {' '.join(parts)}")
            elif method == "Runtime.exceptionThrown":
                detail = msg["params"].get("exceptionDetails", {})
                text = detail.get("exception", {}).get("description") or detail.get("text")
                print(f"[{short}] ❌ 异常: {text}")
            elif method == "Log.entryAdded":
                entry = msg["params"].get("entry", {})
                print(f"[{short}] log.{entry.get('level')}: {entry.get('text')}")

    print("-" * 70)
    for _url, cli in clients:
        cli.close()
    killed = kill_probe_edge()
    print(f"已清理临时 Edge 进程 {killed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
