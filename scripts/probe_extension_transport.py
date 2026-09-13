"""判决性实验四：**扩展能不能连上本机的控制通道（LNA / 权限策略）？**

为什么必须测：
    Chrome 142+ 开始推行 Local Network Access（`local-network-access` 权限策略），
    网页/组件连本机端口会被拦。这个项目要"扩展 ↔ 本程序"通信，
    最简单的方案是扩展直接连 `ws://127.0.0.1:<port>`（零安装、零注册表）。
    但 LNA 一类的策略随时可能把它掐掉——所以要拿本机 Edge 实测，
    不行就退回 **native messaging**（浏览器自己拉起本程序的中继进程，
    走 stdio + 命名管道，完全不经网络栈，也就没有 LNA 问题）。

怎么测（不需要 DevTools / 控制台）：
    Python 侧起一个裸 TCP 监听，手工完成最小的 WebSocket 握手；
    扩展侧 `new WebSocket("ws://127.0.0.1:8765/probe")` 并发一条 hello。
    **连接能不能到达、握手能不能完成、消息能不能收到** —— 就是答案。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_extension_transport.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_browser_tabs import find_edge, kill_probe_edge  # noqa: E402
from probe_tab_isolation import PROFILE  # noqa: E402

EXT = ROOT / "data" / "tmp_ext_probe" / "ws_ext"
PORT = 8765
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

MANIFEST = {
    "manifest_version": 3,
    "name": "LST Transport Probe",
    "version": "0.1",
    "description": "一次性探针：测试扩展能否连上本机 WebSocket",
    "permissions": ["tabs"],
    "background": {"service_worker": "background.js"},
}

BACKGROUND = """// 一次性探针：反复尝试连本机 WebSocket，成功就发一条 hello。
const URL_ = "ws://127.0.0.1:8765/probe";
let tries = 0;

function attempt() {
  tries += 1;
  let ws;
  try {
    ws = new WebSocket(URL_);
  } catch (e) {
    console.error("[lst-ws] 构造失败", e);
    if (tries < 20) setTimeout(attempt, 1500);
    return;
  }
  ws.onopen = () => {
    ws.send(JSON.stringify({
      hello: "from-extension",
      tries,
      ua: navigator.userAgent,
      ts: Date.now(),
    }));
    console.log("[lst-ws] 已连上");
  };
  ws.onerror = () => console.log("[lst-ws] 连接错误（第 " + tries + " 次）");
  ws.onclose = () => { if (tries < 20) setTimeout(attempt, 1500); };
}

attempt();
"""


def build_extension() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    (EXT / "manifest.json").write_text(
        json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EXT / "background.js").write_text(BACKGROUND, encoding="utf-8")


class Listener:
    """裸 TCP + 最小 WebSocket 握手，够用来判定"连得上/连不上"。"""

    def __init__(self, port: int) -> None:
        self.port = port
        self.events: list[str] = []
        self.hello: str = ""
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(8)
        srv.settimeout(0.5)
        self._sock = srv
        self._thread = threading.Thread(target=self._serve, name="lst-ws-probe", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            self.events.append(f"TCP 连接来自 {addr[0]}:{addr[1]}")
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(3.0)
            data = conn.recv(4096)
            text = data.decode("utf-8", "replace")
            first = text.split("\r\n", 1)[0]
            self.events.append(f"HTTP 请求行: {first}")
            origin = next(
                (l for l in text.split("\r\n") if l.lower().startswith("origin:")), ""
            )
            if origin:
                self.events.append(f"来源头: {origin}")
            key = next(
                (l.split(":", 1)[1].strip() for l in text.split("\r\n")
                 if l.lower().startswith("sec-websocket-key:")),
                "",
            )
            if not key:
                self.events.append("❌ 不是 WebSocket 升级请求（或没带 key）")
                return
            accept = base64.b64encode(
                hashlib.sha1((key + WS_GUID).encode()).digest()
            ).decode()
            conn.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode()
            )
            self.events.append("✅ WebSocket 握手完成（101）")
            payload = self._read_frame(conn)
            if payload:
                self.hello = payload
                self.events.append(f"✅ 收到扩展消息 {len(payload)} 字节")
            else:
                self.events.append("⚠️ 握手成功但没收到消息")
        except Exception as exc:  # noqa: BLE001
            self.events.append(f"处理连接出错: {exc}")
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _read_frame(conn: socket.socket) -> str:
        hdr = conn.recv(2)
        if len(hdr) < 2:
            return ""
        length = hdr[1] & 0x7F
        masked = bool(hdr[1] & 0x80)
        if length == 126:
            length = int.from_bytes(conn.recv(2), "big")
        elif length == 127:
            length = int.from_bytes(conn.recv(8), "big")
        mask = conn.recv(4) if masked else b""
        raw = b""
        while len(raw) < length:
            chunk = conn.recv(length - len(raw))
            if not chunk:
                break
            raw += chunk
        if masked and len(mask) == 4:
            raw = bytes(b ^ mask[i % 4] for i, b in enumerate(raw))
        return raw.decode("utf-8", "replace")

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--seconds", type=float, default=14.0)
    args = ap.parse_args()

    edge = find_edge()
    build_extension()
    listener = Listener(PORT)
    listener.start()
    print(f"Edge       : {edge}")
    print(f"探针扩展   : {EXT}")
    print(f"本机监听   : 127.0.0.1:{PORT}（裸 TCP + 最小 WS 握手）\n")

    print("① 启动临时 Edge 并加载探针扩展（扩展会反复尝试连本机）…")
    subprocess.Popen(
        [
            str(edge),
            f"--user-data-dir={PROFILE}",
            f"--load-extension={EXT}",
            f"--disable-extensions-except={EXT}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        close_fds=True,
    )
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.3)
        if listener.hello:
            break

    print("\n② 监听端记录")
    for line in listener.events or ["（什么也没收到）"]:
        print(f"   {line}")
    if listener.hello:
        print("\n③ 扩展发来的内容")
        try:
            payload = json.loads(listener.hello)
            for k, v in payload.items():
                print(f"   {k}: {v}")
        except Exception:  # noqa: BLE001
            print(f"   {listener.hello[:400]}")

    print("\n④ 判定")
    ok = bool(listener.hello)
    if ok:
        print("   ✅ 扩展可以直接连本机 WebSocket（LNA / 权限策略没有拦）")
        print("   → 控制通道用「扩展 ↔ ws://127.0.0.1」即可，不需要注册表/native messaging")
    elif listener.events:
        print("   ⚠️ 有连接到达但握手/消息没完成，见上表")
    else:
        print("   ❌ 完全连不上本机端口")
        print("   → 控制通道改用 native messaging（浏览器拉起本程序中继，走 stdio + 命名管道，不经网络栈）")

    if not args.keep:
        print(f"\n⑤ 已清理临时 Edge 进程 {kill_probe_edge()} 个")
    listener.stop()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
