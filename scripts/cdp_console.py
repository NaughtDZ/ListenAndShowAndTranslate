"""CDP 控制台抓取器：把浏览器扩展**各个页面/Service Worker** 的 console 抓出来。

为什么需要它：扩展跑在浏览器里，``console.log`` 与未捕获异常**不会**出现在我们的
日志里。排查扩展问题时唯一的真相来源就是它自己的控制台，而 CDP 是最直接的办法。

只依赖标准库 + httpx（``/json/list`` 拿 target），WebSocket 用本项目的
``app/audio/ws_client.MiniWSClient``。

用法::

    tap = ConsoleTap(port=9333)
    tap.start()          # 后台线程持续发现新 target 并挂上去
    ...
    for line in tap.lines():
        print(line)
    tap.stop()
"""

from __future__ import annotations

import json
import threading
import time
from urllib.parse import urlparse


class ConsoleTap:
    """持续抓取所有 ``chrome-extension://`` target 的控制台。"""

    def __init__(self, port: int = 9333, max_lines: int = 500) -> None:
        self.port = port
        self.max_lines = max_lines
        self._lines: list[str] = []
        self._clients: dict[str, object] = {}
        self._labels: dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="lst-cdp-tap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        for cli in list(self._clients.values()):
            try:
                cli.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    # ------------------------------------------------------------------ #
    def _targets(self) -> list[dict]:
        import httpx

        with httpx.Client(trust_env=False, timeout=2.0) as c:
            return c.get(f"http://127.0.0.1:{self.port}/json/list").json()

    def _short(self, url: str) -> str:
        if url.startswith("chrome-extension://"):
            parts = urlparse(url)
            name = parts.path.lstrip("/") or "(root)"
            return f"{parts.netloc[:8]}…/{name}"
        return url[:40]

    def _attach(self, target: dict) -> None:
        from app.audio.ws_client import MiniWSClient

        url = target.get("url", "")
        ws_url = target.get("webSocketDebuggerUrl")
        if not ws_url:
            return
        cli = MiniWSClient(ws_url, origin="", timeout=2.0).connect()
        cli.sock.settimeout(0.2)
        cli.send_json({"id": 1, "method": "Runtime.enable"})
        cli.send_json({"id": 2, "method": "Log.enable"})
        self._clients[target.get("id") or url] = cli
        self._labels[target.get("id") or url] = self._short(url)
        self._push(f"[{self._short(url)}] （已挂上控制台）")

    def _push(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self.max_lines:
                del self._lines[: len(self._lines) - self.max_lines]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for target in self._targets():
                    url = target.get("url", "")
                    if not url.startswith("chrome-extension://"):
                        continue
                    key = target.get("id") or url
                    if key in self._clients:
                        continue
                    try:
                        self._attach(target)
                    except Exception as exc:  # noqa: BLE001
                        self._push(f"[{self._short(url)}] 挂载失败：{exc}")
            except Exception as exc:  # noqa: BLE001 - 端口还没起来是常态
                self.errors.append(str(exc))

            for key, cli in list(self._clients.items()):
                try:
                    kind, payload = cli.recv(timeout=0.05)  # type: ignore[attr-defined]
                except TimeoutError:
                    continue
                except Exception:  # noqa: BLE001
                    continue
                if kind != "text":
                    continue
                try:
                    self._handle(self._labels.get(key, key[:12]), json.loads(payload))
                except Exception:  # noqa: BLE001
                    continue
            time.sleep(0.15)

    def _handle(self, label: str, msg: dict) -> None:
        method = msg.get("method")
        if method == "Runtime.consoleAPICalled":
            parts = []
            for a in msg["params"].get("args", []):
                parts.append(str(a.get("value") if "value" in a else a.get("description", "")))
            self._push(f"[{label}] {msg['params'].get('type')}: {' '.join(parts)}")
        elif method == "Runtime.exceptionThrown":
            detail = msg["params"].get("exceptionDetails", {})
            text = detail.get("exception", {}).get("description") or detail.get("text")
            self._push(f"[{label}] ❌ 异常: {text}")
        elif method == "Log.entryAdded":
            entry = msg["params"].get("entry", {})
            self._push(f"[{label}] log.{entry.get('level')}: {entry.get('text')}")
