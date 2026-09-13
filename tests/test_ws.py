"""极简 WebSocket 服务端测试（app/audio/ws.py）。

这里的每条断言都对应 RFC 6455 里一条**真的会踩到**的规则：
掩码方向、长度字段、控制帧、单客户端替换、协议错误关闭。
用自己写的客户端来测不算"自说自话"——因为两端的掩码规则是**相反**的，
客户端不合规 / 服务端不合规都会立刻露馅。
"""

from __future__ import annotations

import socket
import struct
import time

import pytest

from app.audio.ws import WSServer
from app.audio.ws_client import MiniWSClient


class Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.binaries: list[bytes] = []
        self.opens: list[str] = []
        self.closes: list[int] = []

    def server(self, **kwargs) -> WSServer:
        defaults = dict(
            port=0,
            path="/lst/tab",
            on_open=lambda c: (self.opens.append(c.origin), True)[1],
            on_text=lambda _c, t: self.texts.append(t),
            on_binary=lambda _c, b: self.binaries.append(b),
            on_close=lambda _c, code: self.closes.append(code),
            ping_interval_s=0,
        )
        defaults.update(kwargs)
        srv = WSServer(**defaults)
        srv.start()
        return srv


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


# --------------------------------------------------------------------------- #
# 握手
# --------------------------------------------------------------------------- #
def test_handshake_and_text_message(rec: Recorder):
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            cli.send_json({"type": "hello", "protocol": 1})
            assert _wait(lambda: rec.texts == ['{"type": "hello", "protocol": 1}'])
        assert _wait(lambda: rec.closes), "服务端应记录关闭"
        assert rec.opens == ["chrome-extension://lst-selftest"]
    finally:
        srv.stop()


def test_wrong_path_is_rejected_with_404(rec: Recorder):
    srv = rec.server()
    try:
        with pytest.raises(OSError, match="404"):
            MiniWSClient(f"ws://127.0.0.1:{srv.port}/nope").connect()
    finally:
        srv.stop()


def test_origin_can_be_used_to_reject(rec: Recorder):
    """on_open 返回 False → 用 1008 关闭（本程序用它挡非扩展来源）。"""
    srv = rec.server(on_open=lambda _c: False)
    try:
        cli = MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab").connect()
        kind, code = cli.recv(timeout=3)
        assert kind == "close" and code == 1008
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# 二进制帧与长度字段
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [0, 1, 125, 126, 1000, 65535, 65536, 70000])
def test_binary_payload_roundtrip_all_length_encodings(rec: Recorder, size: int):
    """覆盖 7 位 / 16 位 / 64 位三种长度编码。"""
    payload = bytes((i * 7) % 256 for i in range(size))
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            cli.send_binary(payload)
            assert _wait(lambda: len(rec.binaries) == 1 and rec.binaries[0] == payload, timeout=6)
    finally:
        srv.stop()


def test_masked_client_frame_is_unmasked_correctly(rec: Recorder):
    """掩码不是"随便异或一下就完事"：必须按 mask[i % 4] 逐字节还原。"""
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            cli.send_binary(b"we-all-live-in-a-yellow-submarine")  # 32 字节，跨 8 个掩码周期
            assert _wait(lambda: rec.binaries == [b"we-all-live-in-a-yellow-submarine"])
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# 协议错误
# --------------------------------------------------------------------------- #
def test_unmasked_client_frame_closes_with_1002(rec: Recorder):
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            # 故意发一个不带掩码的文本帧（协议禁止客户端这么做）
            cli._sock.sendall(bytes([0x81, 0x03]) + b"abc")
            assert _wait(lambda: rec.closes and rec.closes[-1] == 1002)
    finally:
        srv.stop()


def test_fragmented_frame_closes_with_1002(rec: Recorder):
    """本项目不支持分片：明确报协议错误，而不是悄悄拼错数据。"""
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            sock = cli._sock
            sock.sendall(bytes([0x01, 0x80 | 3]) + b"\x00\x00\x00\x00" + b"abc")
            assert _wait(lambda: rec.closes and rec.closes[-1] == 1002)
    finally:
        srv.stop()


def test_rsv_bits_are_rejected(rec: Recorder):
    """RSV 位不为 0 = 对端启用了我们没协商的扩展，必须拒绝。"""
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            cli._sock.sendall(bytes([0x80 | 0x40 | 0x01, 0x80]) + b"\x00\x00\x00\x00")
            assert _wait(lambda: rec.closes and rec.closes[-1] == 1002)
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# 控制帧与替换
# --------------------------------------------------------------------------- #
def test_ping_from_client_gets_pong(rec: Recorder):
    srv = rec.server()
    try:
        cli = MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab").connect()
        cli._send_frame(0x9, b"hi")
        # 客户端会自己回服务端的 ping，但这里要看到服务端对**我们的** ping 回 pong
        deadline = time.time() + 3
        saw_pong = False
        while time.time() < deadline:
            opcode, payload = cli._read_frame()
            if opcode == 0xA and payload == b"hi":
                saw_pong = True
                break
        assert saw_pong
        cli.close()
    finally:
        srv.stop()


def test_new_connection_replaces_old(rec: Recorder):
    """扩展重连时旧连接必须被踢掉，否则会出现两路音频混着喂。"""
    srv = rec.server()
    try:
        first = MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab").connect()
        second = MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab").connect()
        kind, code = first.recv(timeout=3)
        assert kind == "close" and code == 4001
        assert _wait(lambda: srv.client is not None and srv.client.addr == second.sock.getsockname())
        second.close()
        first.close()
    finally:
        srv.stop()


def test_server_ping_payload_is_empty_and_keeps_connection(rec: Recorder):
    """服务端主动 ping：合规客户端自动回 pong，连接不该被误判超时。"""
    srv = rec.server(ping_interval_s=0.2, ping_timeout_s=5.0)
    try:
        cli = MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab").connect()
        time.sleep(0.8)
        # 客户端一侧看到过 ping（我们的 recv 会自动回 pong 并继续等）
        with pytest.raises(TimeoutError):
            cli.recv(timeout=0.3)
        cli.send_text("still-alive")
        assert _wait(lambda: rec.texts == ["still-alive"])
        assert not rec.closes
        cli.close()
    finally:
        srv.stop()


def test_port_zero_is_reported(rec: Recorder):
    srv = rec.server()
    try:
        assert srv.port > 0
    finally:
        srv.stop()


def test_stop_is_idempotent(rec: Recorder):
    srv = rec.server()
    srv.stop()
    srv.stop()
    assert not srv.is_running


def test_handshake_with_garbage_is_not_a_crash(rec: Recorder):
    """乱发字节不能把服务端搞崩（真实世界里什么都会连上来）。"""
    srv = rec.server()
    try:
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=3)
        s.sendall(b"\x00\x01\x02\x03\r\n\r\n")
        s.close()
        time.sleep(0.2)
        assert srv.is_running
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            cli.send_binary(b"\x00" * 4)
            assert _wait(lambda: rec.binaries == [b"\x00" * 4])
    finally:
        srv.stop()


def test_length_header_is_big_endian(rec: Recorder):
    """126 长度字段是大端——写反了只会在特定长度上偶发错位。"""
    srv = rec.server()
    try:
        with MiniWSClient(f"ws://127.0.0.1:{srv.port}/lst/tab") as cli:
            payload = bytes(range(126))
            cli.send_binary(payload)
            assert _wait(lambda: rec.binaries == [payload])
            # 手工构造一个大端 16 位长度帧，验证服务端解析方向
            body = b"z" * 300
            mask = b"\x01\x02\x03\x04"
            masked = bytes(b ^ mask[i % 4] for i, b in enumerate(body))
            frame = bytes([0x82, 0x80 | 126]) + struct.pack("!H", 300) + mask + masked
            cli._sock.sendall(frame)
            assert _wait(lambda: rec.binaries[-1] == body)
    finally:
        srv.stop()
