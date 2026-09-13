"""极简 WebSocket 服务端（**纯标准库**，无第三方依赖）。

为什么不用 ``websockets`` / ``aiohttp``：
    本项目的依赖表一向克制（选 proc-tap 就是为了避免虚拟声卡驱动）。这里的
    协议两端**都由我们掌控**（浏览器扩展 ↔ 本程序，本机回环），需要的能力极少：

    · 单客户端（新连接替换旧连接）
    · 不拆分的文本帧 / 二进制帧
    · ping / pong / close 控制帧
    · 握手时校验路径与 Origin

    为此引入一个异步框架 + 一套版本漂移，收益不划算。所以自己写，
    并把它限制在这一个小文件里、配上单测（tests/test_ws.py）。

刻意不做的（用到再说，现在做只会引入没测过的分支）：
    · 分片消息（FIN=0）→ 直接按协议错误关闭（1003），并记日志
    · 扩展协商（permessage-deflate）→ 握手时忽略该头，不启用压缩
    · 多客户端广播、子协议

帧格式（RFC 6455）：
    客户端发来的帧**必须**带掩码（用 4 字节掩码逐字节异或）；服务端发出的帧
    **必须不带**掩码。这两条都在测试里验过。
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from typing import Callable

from app.utils.log import get_logger

log = get_logger(__name__)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

# 关闭码（RFC 6455 + 自定义）
CLOSE_NORMAL = 1000
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_POLICY_VIOLATION = 1008
CLOSE_REPLACED = 4001
"""本程序自定义：同一条通道上来了新连接，旧连接被替换。"""

MAX_HANDSHAKE_BYTES = 16 * 1024


class WSClient:
    """一条已握手的连接。带上层想挂的任何状态（用 ``meta``）。"""

    __slots__ = ("addr", "origin", "path", "meta", "connected_at", "closing", "greeted",
                 "_sock", "_send_lock")

    def __init__(self, addr: tuple[str, int], origin: str, path: str, sock: socket.socket) -> None:
        self.addr = addr
        self.origin = origin
        self.path = path
        self.meta: dict = {}
        self.connected_at = time.time()
        self.closing = False
        """请求关闭：读线程看到它就退出（不然被拒绝的连接会一直挂着，
        ``connected`` 永远为 True）。"""
        self.greeted = False
        """上层是否已经完成握手。淘汰连接时**优先踢没握手的**
        （它们多半是陌生连接/刚断线的僵尸，不能为了给它们腾地方把正在干活的踢掉）。"""
        self._sock = sock
        self._send_lock = threading.Lock()

    def send_text(self, text: str) -> bool:
        return self._send_frame(OP_TEXT, text.encode("utf-8"))

    def send_json(self, payload: dict) -> bool:
        return self.send_text(json.dumps(payload, ensure_ascii=False))

    def send_binary(self, data: bytes) -> bool:
        return self._send_frame(OP_BINARY, data)

    def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        payload = struct.pack("!H", code) + reason.encode("utf-8")[:120]
        self._send_frame(OP_CLOSE, payload)
        self.closing = True
        # 只关**写**方向：既保证已经排队的关闭帧真的发出去（SHUT_RDWR 在 Windows 上
        # 可能让出站缓冲被丢掉，对端就看不到关闭码了），又让读线程看到 EOF 自然退出。
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def _send_frame(self, opcode: int, payload: bytes) -> bool:
        """服务端帧：不加掩码。"""
        with self._send_lock:
            try:
                header = bytearray([0x80 | opcode])
                n = len(payload)
                if n < 126:
                    header.append(n)
                elif n < 65536:
                    header.append(126)
                    header += struct.pack("!H", n)
                else:
                    header.append(127)
                    header += struct.pack("!Q", n)
                self._sock.sendall(bytes(header) + payload)
                return True
            except OSError as exc:
                log.debug("发送帧失败（%s）: %s", self.addr, exc)
                return False


class WSServer:
    """线程版 WebSocket 服务端。

    Args:
        host: 绑定地址，**默认且推荐 127.0.0.1**（本机回环，不暴露到局域网）。
        port: 端口；传 0 让系统分配（测试用，实际端口见 :attr:`port`）。
        path: 只接受该路径的升级请求（``/`` 表示接受任意路径）。
        on_open: ``(client) -> bool``；返回 False 则拒绝该连接（用 1008 关闭）。
        on_text: ``(client, str) -> None``
        on_binary: ``(client, bytes) -> None``
        on_close: ``(client, code) -> None``
        ping_interval_s: 服务端主动 ping 的间隔；0 = 不发。
        ping_timeout_s: 多久没收到任何数据就判定死连接。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/",
        on_open: Callable[[WSClient], bool] | None = None,
        on_text: Callable[[WSClient, str], None] | None = None,
        on_binary: Callable[[WSClient, bytes], None] | None = None,
        on_close: Callable[[WSClient, int], None] | None = None,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 40.0,
        max_clients: int = 1,
    ) -> None:
        self.host = host
        self.path = path
        self.on_open = on_open
        self.on_text = on_text
        self.on_binary = on_binary
        self.on_close = on_close
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.max_clients = max(1, int(max_clients))
        """同时保留几条连接。超过就踢掉最旧的（默认 1 = 单客户端，新连接替换旧连接）。

        本程序用 2：扩展的 service worker 走控制（指令/状态），
        offscreen 文档另开一条走音频（PCM 直接二进制，不经扩展端口转发）。
        """

        self._srv: socket.socket | None = None
        self._port = int(port)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clients: list[WSClient] = []
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @property
    def port(self) -> int:
        """实际监听端口（``port=0`` 时由系统分配）。"""
        return self._port

    @property
    def client(self) -> WSClient | None:
        """最新一条连接（单客户端模式下就是唯一那条）。"""
        with self._client_lock:
            return self._clients[-1] if self._clients else None

    @property
    def clients(self) -> list[WSClient]:
        with self._client_lock:
            return list(self._clients)

    def drop_client(
        self, client: WSClient, code: int = CLOSE_REPLACED, reason: str = ""
    ) -> None:
        """把连接**立刻**从登记表摘掉并关闭。

        用于"同一角色的新连接替换旧的"：socket 关闭与读线程退出是异步的，
        但我们的簿记不该还挂着一个已经作废的连接（曾经因此把正在送音频的那条
        当成"僵尸"或反过来把它挤掉）。
        """
        with self._client_lock:
            if client in self._clients:
                self._clients.remove(client)
        client.close(code, reason)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self._port))
        srv.listen(4)
        srv.settimeout(0.5)
        self._port = int(srv.getsockname()[1])
        self._srv = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name="lst-ws", daemon=True)
        self._thread.start()
        log.info("WebSocket 服务端已监听 ws://%s:%d%s", self.host, self._port, self.path)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for client in self.clients:
            client.close(CLOSE_NORMAL, "server stopping")
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        srv = self._srv
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        self._srv = None
        log.info("WebSocket 服务端已停止")

    # ------------------------------------------------------------------ #
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            srv = self._srv
            if srv is None:
                break
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handshake_and_serve, args=(conn, addr), daemon=True
            ).start()

    def _handshake_and_serve(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        client: WSClient | None = None
        code = CLOSE_NORMAL
        try:
            request = self._read_handshake(conn)
            if request is None:
                return
            path, headers = request

            if self.path not in ("", "/") and not path.startswith(self.path):
                self._http_error(conn, 404, "Not Found")
                return
            key = headers.get("sec-websocket-key", "")
            if "websocket" not in headers.get("upgrade", "").lower() or not key:
                self._http_error(conn, 400, "Bad Request")
                return

            accept = base64.b64encode(
                hashlib.sha1((key + WS_GUID).encode()).digest()
            ).decode()
            conn.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode()
            )

            client = WSClient(addr, headers.get("origin", ""), path, conn)
            if self.on_open is not None and not self.on_open(client):
                client.close(CLOSE_POLICY_VIOLATION, "rejected")
                return
            self._install_client(client)
            code = self._read_loop(client)
        except (OSError, ValueError) as exc:
            log.debug("连接结束（%s）: %s", addr, exc)
        finally:
            with self._client_lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                conn.close()
            except OSError:
                pass
            if client is not None and self.on_close is not None:
                try:
                    self.on_close(client, code)
                except Exception as exc:  # noqa: BLE001 - 回调的锅不能让线程崩
                    log.error("on_close 回调抛异常: %s", exc)

    def _install_client(self, client: WSClient) -> None:
        """登记连接；超过 ``max_clients`` 就腾地方。

        **淘汰顺序很重要**：先踢"还没握手"的连接（陌生连接、刚断线的僵尸），
        再踢最旧的。以前一律"最旧的先踢"，结果实测踩到过：
        另一个浏览器配置文件里的同名扩展也在连（它还没握手/不在采集），
        新连接一到就把**正在送音频的那条**踢了——表现为"音频突然断一下"。
        """
        with self._client_lock:
            self._clients.append(client)
            victims: list[WSClient] = []
            while len(self._clients) > self.max_clients:
                others = [c for c in self._clients if c is not client]
                if not others:
                    break
                victim = next((c for c in others if not c.greeted), others[0])
                self._clients.remove(victim)
                victims.append(victim)
        for old in victims:
            log.info(
                "连接数超限，踢掉%s %s（新连接 %s）",
                "未握手的" if not old.greeted else "最旧的",
                old.addr,
                client.addr,
            )
            old.close(CLOSE_REPLACED, "replaced by a newer connection")

    def _read_handshake(self, conn: socket.socket) -> tuple[str, dict[str, str]] | None:
        conn.settimeout(5.0)
        data = b""
        while b"\r\n\r\n" not in data:
            if len(data) > MAX_HANDSHAKE_BYTES:
                return None
            try:
                chunk = conn.recv(4096)
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            data += chunk

        text = data.decode("latin-1")
        head, _, _rest = text.partition("\r\n\r\n")
        lines = head.split("\r\n")
        parts = lines[0].split()
        if len(parts) < 2 or parts[0].upper() != "GET":
            self._http_error(conn, 400, "Bad Request")
            return None
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            if name:
                headers[name.strip().lower()] = value.strip()
        conn.settimeout(None)
        return parts[1], headers

    @staticmethod
    def _http_error(conn: socket.socket, code: int, reason: str) -> None:
        try:
            conn.sendall(
                f"HTTP/1.1 {code} {reason}\r\nConnection: close\r\n"
                f"Content-Length: 0\r\n\r\n".encode()
            )
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def _read_loop(self, client: WSClient) -> int:
        """返回关闭码。"""
        conn = client._sock
        last_seen = time.time()
        next_ping = time.time() + self.ping_interval_s
        while not self._stop.is_set():
            if client.closing:
                return CLOSE_NORMAL
            timeout = 0.25
            conn.settimeout(timeout)
            try:
                frame = self._read_frame(conn)
            except socket.timeout:
                frame = None
            except _ProtocolError as exc:
                log.warning("协议错误（%s）: %s", client.addr, exc)
                client.close(CLOSE_PROTOCOL_ERROR, str(exc)[:100])
                return CLOSE_PROTOCOL_ERROR
            except OSError:
                return CLOSE_NORMAL

            now = time.time()
            if frame is not None:
                last_seen = now
                opcode, payload = frame
                if opcode == OP_CLOSE:
                    code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else CLOSE_NORMAL
                    try:
                        client._send_frame(OP_CLOSE, struct.pack("!H", CLOSE_NORMAL))
                    except Exception:  # noqa: BLE001
                        pass
                    return code
                if opcode == OP_PING:
                    client._send_frame(OP_PONG, payload)
                    continue
                if opcode == OP_PONG:
                    continue
                if opcode == OP_CONT:
                    raise _ProtocolError("不支持分片消息（FIN=0）")
                if opcode == OP_TEXT:
                    if self.on_text is not None:
                        self._safe(self.on_text, client, payload.decode("utf-8", "replace"))
                    continue
                if opcode == OP_BINARY:
                    if self.on_binary is not None:
                        self._safe(self.on_binary, client, payload)
                    continue
                raise _ProtocolError(f"未知 opcode {opcode}")

            if self.ping_interval_s and now >= next_ping:
                next_ping = now + self.ping_interval_s
                if not client._send_frame(OP_PING, b""):
                    return CLOSE_NORMAL
            if self.ping_timeout_s and now - last_seen > self.ping_timeout_s:
                log.info("连接 %s 超时无数据，关闭", client.addr)
                client.close(CLOSE_NORMAL, "timeout")
                return CLOSE_NORMAL
        return CLOSE_NORMAL

    def _read_frame(self, conn: socket.socket) -> tuple[int, bytes] | None:
        head = conn.recv(2)
        if not head:
            raise OSError("对端关闭")
        if len(head) < 2:
            return None

        fin = bool(head[0] & 0x80)
        opcode = head[0] & 0x0F
        if head[0] & 0x70:
            raise _ProtocolError("RSV 位不为 0（未启用扩展）")
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F
        if not masked:
            raise _ProtocolError("客户端帧必须带掩码")
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(conn, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(conn, 8))[0]
        mask = self._recv_exact(conn, 4)
        payload = self._recv_exact(conn, length) if length else b""
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

        if not fin:
            # 控制帧不允许分片；数据帧分片我们明确不支持
            raise _ProtocolError("不支持分片消息（FIN=0）")
        return opcode, payload

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise OSError("对端关闭")
            buf += chunk
        return buf

    @staticmethod
    def _safe(fn: Callable, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - 上层回调不该打死读线程
            log.error("WebSocket 回调抛异常: %s", exc)


class _ProtocolError(Exception):
    """帧格式不合法。"""
