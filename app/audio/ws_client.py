"""极简 WebSocket 客户端（**纯标准库**）：单测 / 模拟扩展 / `--tab-selftest` 共用。

它和 ``app/audio/ws.py`` 是一对：``ws.py`` 是服务端（本程序），本文件是客户端
（正常跑起来时是那个浏览器扩展；测试与自检时就是它自己）。
客户端帧**必须带掩码**，服务端帧**必须不带**——两边都按 RFC 6455 来，
所以它可以用来验证服务端的实现是否真的合规，而不只是"自己跟自己说话"。

只实现本协议需要的能力：握手、文本帧、二进制帧、ping/pong、close。
不支持分片与扩展协商（收到就报错）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time
from urllib.parse import urlparse

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class MiniWSClient:
    """够用就好的 WebSocket 客户端。"""

    def __init__(
        self,
        url: str,
        origin: str = "chrome-extension://lst-selftest",
        timeout: float = 5.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        u = urlparse(url)
        if u.scheme != "ws":
            raise ValueError(f"只支持 ws:// ：{url}")
        self.host = u.hostname or "127.0.0.1"
        self.port = int(u.port or 80)
        self.path = u.path or "/"
        if u.query:
            self.path += "?" + u.query
        self.origin = origin
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.sock: socket.socket | None = None
        self.closed = False

    # ------------------------------------------------------------------ #
    def connect(self) -> "MiniWSClient":
        key = base64.b64encode(os.urandom(16)).decode()
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        headers = [
            f"GET {self.path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        if self.origin:
            headers.append(f"Origin: {self.origin}")
        headers += [f"{k}: {v}" for k, v in self.extra_headers.items()]
        sock.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("latin-1"))

        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise OSError("握手过程中连接被关闭")
            data += chunk
        head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1")
        status = head.split("\r\n", 1)[0]
        if "101" not in status:
            raise OSError(f"握手失败：{status}")

        expected = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode()).digest()
        ).decode()
        got = ""
        for line in head.split("\r\n")[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-accept":
                got = value.strip()
        if got != expected:
            raise OSError("Sec-WebSocket-Accept 校验失败")

        self.sock = sock
        return self

    def __enter__(self) -> "MiniWSClient":
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.closed = True
        if self.sock is not None:
            try:
                self._send_frame(OP_CLOSE, struct.pack("!H", 1000))
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    @property
    def _sock(self) -> socket.socket:
        if self.sock is None:
            raise OSError("未连接")
        return self.sock

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def send_json(self, payload: dict) -> None:
        self.send_text(json.dumps(payload, ensure_ascii=False))

    def send_binary(self, data: bytes) -> None:
        self._send_frame(OP_BINARY, data)

    def recv(self, timeout: float | None = None) -> tuple[str, object]:
        """收一条消息，返回 ``("text", str)`` / ``("binary", bytes)`` / ``("close", code)``。

        收到 ping 会自动回 pong（并继续等真正的消息）。
        超时抛 ``TimeoutError``。
        """
        deadline = time.time() + (self.timeout if timeout is None else timeout)
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("等待消息超时")
            self._sock.settimeout(remain)
            opcode, payload = self._read_frame()
            if opcode == OP_PING:
                self._send_frame(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1000
                return "close", code
            if opcode == OP_TEXT:
                return "text", payload.decode("utf-8", "replace")
            if opcode == OP_BINARY:
                return "binary", payload
            raise OSError(f"未知 opcode {opcode}")

    def recv_json(self, timeout: float | None = None) -> dict:
        kind, payload = self.recv(timeout)
        if kind != "text":
            raise OSError(f"期待文本消息，收到 {kind}")
        return json.loads(payload)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)
        header += mask
        self._sock.sendall(bytes(header) + masked)

    def _read_frame(self) -> tuple[int, bytes]:
        head = self._recv_exact(2)
        fin = bool(head[0] & 0x80)
        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length) if length else b""
        if masked and len(mask) == 4:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if not fin:
            raise OSError("不支持分片消息")
        if masked:
            raise OSError("服务端帧不应带掩码")
        return opcode, payload

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise OSError("连接已关闭")
            buf += chunk
        return buf
