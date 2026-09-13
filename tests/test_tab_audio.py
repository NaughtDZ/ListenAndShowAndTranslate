"""标签页音频服务端测试（app/audio/tab_audio.py）。

这里验的是**协议与状态机**：握手、配对码、版本、格式协商、PCM 进管线、
静音判定、状态上报、断线、指令下发。全部用纯标准库的模拟扩展客户端驱动，
不需要浏览器、不需要声卡。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.audio.tab_audio import (
    AudioFormat,
    TabAudioServer,
    TabInfo,
    _tab_from_payload,
)
from app.audio import tab_audio as tab_audio_mod
from app.audio.ws_client import MiniWSClient

RATE = 48000
CHANNELS = 2


class Rig:
    """一个跑着的服务端 + 记录器。"""

    def __init__(self, **kwargs) -> None:
        self.chunks: list[np.ndarray] = []
        self.states: list = []
        defaults = dict(
            port=0,
            on_chunk=self.chunks.append,
            on_state=self.states.append,
        )
        defaults.update(kwargs)
        self.server = TabAudioServer(**defaults)
        self.server.start()

    @property
    def port(self) -> int:
        return self.server.port

    def client(self, **kwargs) -> MiniWSClient:
        url = kwargs.pop("url", f"ws://127.0.0.1:{self.port}/lst/tab")
        cli = MiniWSClient(url, **kwargs).connect()
        self.cli = cli
        return cli

    def hello(
        self,
        cli: MiniWSClient,
        *,
        token: str = "",
        protocol: int = 1,
        fmt: dict | None = None,
        tab: dict | None = None,
        capturing: bool = True,
        role: str = "both",
        legacy: bool = False,
    ) -> dict:
        payload = {
            "type": "hello",
            "protocol": protocol,
            "token": token,
            "format": fmt or {"rate": RATE, "channels": CHANNELS, "dtype": "float32"},
            "tab": tab if tab is not None else {"id": 7, "title": "测试标签页", "url": "https://x/y"},
            "browser": {"name": "Edge 153"},
            "capturing": capturing,
        }
        if not legacy:
            payload["role"] = role
        cli.send_json(payload)
        return cli.recv_json(timeout=3)

    def stop(self) -> None:
        try:
            getattr(self, "cli", None) and self.cli.close()
        except Exception:  # noqa: BLE001
            pass
        self.server.stop()


@pytest.fixture
def rig() -> Rig:
    r = Rig()
    yield r
    r.stop()


def _pcm(seconds: float = 0.02, freq: float = 440.0, rate: int = RATE, amp: float = 0.5) -> bytes:
    """生成一段 48k/立体声/float32 的 PCM（与 proc-tap 的格式一致）。"""
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    mono = (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
    return np.stack([mono, mono], axis=1).reshape(-1).astype(np.float32).tobytes()


# --------------------------------------------------------------------------- #
# 握手
# --------------------------------------------------------------------------- #
def test_hello_welcome_and_state(rig: Rig):
    cli = rig.client()
    reply = rig.hello(cli)
    assert reply["type"] == "welcome" and reply["ok"] is True
    assert reply["protocol"] == 1
    assert reply["format"] == {"rate": RATE, "channels": CHANNELS, "dtype": "float32"}

    snap = rig.server.snapshot()
    assert snap.connected and snap.capturing
    assert snap.tab.id == 7 and snap.tab.title == "测试标签页"
    assert snap.browser == "Edge 153"
    assert snap.format == AudioFormat()
    assert rig.states, "状态回调应被触发"


def test_token_mismatch_is_rejected(rig: Rig):
    rig.server.token = "正确配对码"
    cli = rig.client()
    cli.send_json({"type": "hello", "protocol": 1, "token": "错的"})
    msg = cli.recv_json(timeout=3)
    assert msg["type"] == "error" and msg["code"] == "token"
    kind, code = cli.recv(timeout=3)
    assert kind == "close" and code == 1008
    assert "配对码" in rig.server.snapshot().last_error


def test_token_match_is_accepted():
    rig = Rig(token="同一个码")
    try:
        cli = rig.client()
        reply = rig.hello(cli, token="同一个码")
        assert reply["type"] == "welcome"
    finally:
        rig.stop()


def test_protocol_version_mismatch_is_rejected(rig: Rig):
    cli = rig.client()
    cli.send_json({"type": "hello", "protocol": 99, "token": ""})
    msg = cli.recv_json(timeout=3)
    assert msg["type"] == "error" and msg["code"] == "protocol"
    kind, code = cli.recv(timeout=3)
    assert kind == "close" and code == 1008
    assert "99" in rig.server.snapshot().last_error


def test_non_extension_origin_is_refused():
    rig = Rig()
    try:
        cli = MiniWSClient(
            f"ws://127.0.0.1:{rig.port}/lst/tab", origin="https://evil.example"
        ).connect()
        kind, code = cli.recv(timeout=3)
        assert kind == "close" and code == 1008
        assert not rig.server.snapshot().connected
    finally:
        rig.stop()


def test_origin_check_can_be_disabled():
    rig = Rig(require_extension_origin=False)
    try:
        cli = MiniWSClient(f"ws://127.0.0.1:{rig.port}/lst/tab", origin="http://localhost:5173").connect()
        assert rig.hello(cli)["type"] == "welcome"
    finally:
        rig.stop()


def test_hello_timeout_closes_silent_connection(monkeypatch):
    monkeypatch.setattr(tab_audio_mod, "HELLO_TIMEOUT_S", 0.2)
    rig = Rig()
    try:
        cli = rig.client()
        kind, code = cli.recv(timeout=5)
        assert kind == "close" and code == 1008
    finally:
        rig.stop()


# --------------------------------------------------------------------------- #
# 音频
# --------------------------------------------------------------------------- #
def test_pcm_frames_reach_pipeline_and_callback(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    for _ in range(5):
        cli.send_binary(_pcm())

    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().frames < 5:
        time.sleep(0.02)

    snap = rig.server.snapshot()
    assert snap.frames == 5
    assert snap.bytes_in == 5 * len(_pcm())
    # 重采样器是**流式**的：实测 5 块 20ms（4800 帧）输出为 0/490/0/489/489，
    # 另有 132 样本留在它内部（flush 才吐）。所以既不能断言"每块都有输出"，
    # 也不能断言"样本数正好 1600"——只断言样本总量落在合理区间。
    assert 3 <= len(rig.chunks) <= 5
    total = sum(int(c.size) for c in rig.chunks)
    assert 1400 < total <= 1700
    assert all(c.dtype == np.float32 and c.ndim == 1 for c in rig.chunks)
    assert snap.peak > 0.4 and snap.last_rms > 0.1
    assert snap.likely_playing is True


def test_energy_matches_expected_level(rig: Rig):
    """幅度 0.5 的正弦波 RMS 应约 0.354（0.5/√2）——管线不能被改写幅度。"""
    cli = rig.client()
    rig.hello(cli)
    for _ in range(5):
        cli.send_binary(_pcm(amp=0.5))
    time.sleep(0.4)
    assert rig.server.snapshot().last_rms == pytest.approx(0.3536, abs=0.02)


def test_custom_format_is_honoured(rig: Rig):
    """扩展可以送 16k 单声道 int16；管线要按报告里的格式解释字节。"""
    cli = rig.client()
    rig.hello(
        cli,
        fmt={"rate": 16000, "channels": 1, "dtype": "int16"},
    )
    t = np.arange(1600, dtype=np.float64) / 16000
    mono = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    cli.send_binary(mono.tobytes())
    time.sleep(0.4)
    snap = rig.server.snapshot()
    assert snap.format.rate == 16000 and snap.format.channels == 1
    assert sum(int(c.size) for c in rig.chunks) == pytest.approx(1600, abs=40)
    assert snap.last_rms == pytest.approx(0.3536, abs=0.03)


def test_silence_is_detected(rig: Rig, monkeypatch):
    monkeypatch.setattr(tab_audio_mod, "STATE_STALE_SECONDS", 0.05)
    cli = rig.client()
    rig.hello(cli)
    cli.send_binary(_pcm(amp=0.5))
    time.sleep(0.3)
    assert rig.server.snapshot().silent_seconds == 0.0

    for _ in range(6):
        cli.send_binary(np.zeros(RATE // 50 * CHANNELS, dtype=np.float32).tobytes())
        time.sleep(0.05)
    snap = rig.server.snapshot()
    assert snap.silent_seconds > 0.1
    assert snap.likely_playing is False
    assert snap.total_silent_seconds > 0


def test_audio_before_hello_is_dropped(rig: Rig):
    """没握手就发音频：不能崩，也不能进管线。"""
    cli = rig.client()
    cli.send_binary(_pcm())
    time.sleep(0.2)
    assert rig.server.snapshot().frames == 0
    assert not rig.chunks


# --------------------------------------------------------------------------- #
# 状态与指令
# --------------------------------------------------------------------------- #
def test_status_updates_tab_and_capturing(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    cli.send_json(
        {
            "type": "status",
            "tab": {"id": 9, "title": "换了个标签页"},
            "capturing": False,
        }
    )
    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().tab.id != 9:
        time.sleep(0.02)
    snap = rig.server.snapshot()
    assert snap.tab.id == 9 and snap.tab.title == "换了个标签页"
    assert snap.capturing is False
    assert "扩展已连接" in snap.describe()


def test_error_message_from_extension_is_surfaced(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    cli.send_json({"type": "error", "message": "tabCapture 需要用户手势"})
    deadline = time.time() + 3
    while time.time() < deadline and "手势" not in rig.server.snapshot().last_error:
        time.sleep(0.02)
    assert "手势" in rig.server.snapshot().last_error


def test_request_capture_sends_command_with_format(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    assert rig.server.request_capture(tab_id=42) is True
    msg = cli.recv_json(timeout=3)
    assert msg["type"] == "capture" and msg["tabId"] == 42
    assert msg["format"] == {"rate": RATE, "channels": CHANNELS, "dtype": "float32"}


def test_request_stop_and_state_commands(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    assert rig.server.request_stop() is True
    assert cli.recv_json(timeout=3)["type"] == "stop"
    assert rig.server.request_state() is True
    assert cli.recv_json(timeout=3)["type"] == "state"


def test_commands_without_client_are_noops(rig: Rig):
    assert rig.server.request_capture(1) is False
    assert rig.server.request_stop() is False


def test_disconnect_updates_state(rig: Rig):
    cli = rig.client()
    rig.hello(cli)
    assert rig.server.snapshot().connected
    cli.close()
    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().connected:
        time.sleep(0.02)
    snap = rig.server.snapshot()
    assert not snap.connected and not snap.capturing
    assert "等待浏览器扩展连接" in snap.describe()


def test_reconnect_replaces_and_keeps_counting(rig: Rig):
    first = rig.client()
    rig.hello(first)
    second = rig.client()
    rig.hello(second)
    # 旧连接被替换
    kind, code = first.recv(timeout=3)
    assert kind == "close" and code == 4001
    assert rig.server.snapshot().connected
    assert rig.server.snapshot().connections == 2


def test_pipeline_reset_after_tab_switch(rig: Rig):
    """切标签页时重采样器要复位，否则会把两段音频缝在一起。"""
    cli = rig.client()
    rig.hello(cli)
    cli.send_binary(_pcm())
    time.sleep(0.2)
    before = rig.server.pipeline_stats().input_chunks
    rig.server.restart_pipeline()
    assert rig.server.pipeline_stats().input_chunks == 0
    assert before == 1


# --------------------------------------------------------------------------- #
# 双连接：控制（service worker）+ 音频（offscreen 文档）
# --------------------------------------------------------------------------- #
def _hello_role(cli: MiniWSClient, role: str, tab_id: int = 7) -> dict:
    cli.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "role": role,
            "token": "",
            "format": {"rate": RATE, "channels": CHANNELS, "dtype": "float32"},
            "tab": {"id": tab_id, "title": f"标签页 {tab_id}"},
            "browser": {"name": "Edge"},
            "capturing": role in ("audio", "both"),
        }
    )
    return cli.recv_json(timeout=3)


def test_control_and_audio_roles_coexist(rig: Rig):
    """真实扩展会开两条连接：SW 走控制、offscreen 走音频。两条必须互不顶掉。"""
    control = rig.client()
    reply = _hello_role(control, "control")
    assert reply["role"] == "control"

    audio = rig.client()
    reply = _hello_role(audio, "audio", tab_id=11)
    assert reply["role"] == "audio"

    # 两条都还在
    assert rig.server.snapshot().connected
    assert [c.meta.get("role") for c in rig.server._ws.clients] == ["control", "audio"]

    # 音频从 audio 连接来
    for _ in range(4):
        audio.send_binary(_pcm())
    time.sleep(0.4)
    assert rig.server.snapshot().frames == 4
    assert rig.chunks

    # 指令发给 control 连接
    assert rig.server.request_capture(tab_id=11) is True
    msg = control.recv_json(timeout=3)
    assert msg["type"] == "capture" and msg["tabId"] == 11


def test_replacing_same_role_keeps_the_other_role(rig: Rig):
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio")

    control2 = rig.client()
    _hello_role(control2, "control")
    kind, code = control.recv(timeout=3)
    assert kind == "close" and code == 4001
    # audio 那条不该被牵连
    assert [c.meta.get("role") for c in rig.server._ws.clients] == ["audio", "control"]
    audio.send_binary(_pcm())
    time.sleep(0.3)
    assert rig.server.snapshot().frames == 1


def test_audio_disconnect_keeps_control_connected(rig: Rig):
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio")

    audio.close()
    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().capturing:
        time.sleep(0.02)
    snap = rig.server.snapshot()
    assert snap.connected is True, "控制连接还在，不该显示成整体断开"
    assert snap.capturing is False, "送音频的那条断了，采集必然停了"


def test_audio_hello_sets_format(rig: Rig):
    """格式以音频连接为准（PCM 是从它来的）。"""
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    audio.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "role": "audio",
            "format": {"rate": 16000, "channels": 1, "dtype": "int16"},
            "tab": {"id": 3, "title": "x"},
            "capturing": True,
        }
    )
    audio.recv_json(timeout=3)
    snap = rig.server.snapshot()
    assert snap.format.rate == 16000 and snap.format.channels == 1


# --------------------------------------------------------------------------- #
# 多实例：另一个浏览器里也装了同一个扩展（实测真的会发生）
# --------------------------------------------------------------------------- #
def test_second_extension_cannot_pause_or_kick_the_live_one(rig: Rig):
    """实测现场（2026-09-13）：用户日常浏览器里那份扩展也连上来，

    它以 control 身份 report ``capturing=false``（因为它没在采集），
    结果程序状态被改成"等待开始采集"——看起来就像音频发送暂停了。
    它还会把正在送音频的那条挤掉（旧的淘汰策略是"最旧先踢"）。

    现在的规则：**别的浏览器里的扩展不能顶掉正在工作的那条**——
    新连接会被明确拒绝（busy），并如实告诉用户发生了什么。
    """
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio", tab_id=11)
    audio.send_binary(_pcm())
    time.sleep(0.3)
    assert rig.server.snapshot().capturing is True

    # 另一个扩展实例（不同 Origin）连上来，说自己没在采集、目标是别的标签页
    other = rig.client(origin="chrome-extension://another-extension-instance")
    other.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "role": "control",
            "format": {"rate": RATE, "channels": CHANNELS, "dtype": "float32"},
            "tab": {"id": 999, "title": "别人的标签页"},
            "browser": {"name": "Edge"},
            "capturing": False,
        }
    )
    msg = other.recv_json(timeout=3)
    assert msg["type"] == "error" and msg["code"] == "busy"
    kind, code = other.recv(timeout=3)
    assert kind == "close" and code == 1008

    snap = rig.server.snapshot()
    assert snap.capturing is True, "不能因为别人说自己没采集，就把我们的采集说成停了"
    assert snap.tab.id == 11, "目标标签页不能被别人的状态覆盖"
    assert "另一个浏览器扩展" in snap.peer_note and "被忽略" in snap.peer_note
    # 我们自己的两条连接一个都没掉
    assert [c.meta.get("role") for c in rig.server._ws.clients] == ["control", "audio"]
    # 音频连接还在，继续送帧依旧有效
    audio.send_binary(_pcm())
    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().frames < 2:
        time.sleep(0.02)
    assert rig.server.snapshot().frames >= 2
    assert rig.server.snapshot().capturing is True


def test_same_origin_reconnect_still_replaces(rig: Rig):
    """同一个浏览器（同源）重连是正常的——SW 会被回收，必须还能顶掉旧的。"""
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio", tab_id=11)

    control2 = rig.client()  # 同源（rig.client 默认同一个 origin）
    _hello_role(control2, "control")
    kind, code = control.recv(timeout=3)
    assert kind == "close" and code == 4001


def test_legacy_extension_is_flagged_with_actionable_hint(rig: Rig):
    """旧版扩展（hello 里没有 role 字段）会被当成 both 接受，但要提示用户重新加载。

    实测：用户日常浏览器里装的那份就是旧版（本程序加 role 之前装的），
    它一 hello 就把正在送音频的那条顶掉，表现为"音频发送暂停，还得手动再点"。
    """
    legacy = rig.client()
    legacy.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "format": {"rate": RATE, "channels": CHANNELS, "dtype": "float32"},
            "tab": {"id": 5, "title": "旧版扩展"},
            "browser": {"name": "Edge"},
            "capturing": False,
        }
    )
    assert legacy.recv_json(timeout=3)["type"] == "welcome"
    snap = rig.server.snapshot()
    assert "旧版扩展" in snap.peer_note
    assert "重新加载" in snap.peer_note


def test_foreign_status_message_is_ignored(rig: Rig):
    """非权威连接的 status 也不能改状态。"""
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio", tab_id=11)

    other = rig.client(origin="chrome-extension://another-extension-instance")
    other.send_json(
        {
            "type": "hello",
            "protocol": 1,
            "role": "control",
            "format": {"rate": RATE, "channels": CHANNELS, "dtype": "float32"},
            "tab": {"id": 999, "title": "别人的标签页"},
            "capturing": False,
        }
    )
    other.recv_json(timeout=3)
    other.send_json({"type": "status", "capturing": False, "tab": {"id": 999, "title": "改了"}})
    time.sleep(0.4)

    snap = rig.server.snapshot()
    assert snap.capturing is True
    assert snap.tab.id == 11


def test_eviction_prefers_unauthenticated_connections(rig: Rig):
    """连接数超限时先踢"还没握手"的，绝不能把正在工作的音频连接踢掉。"""
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio", tab_id=11)
    audio.send_binary(_pcm())
    time.sleep(0.2)

    # 塞两条陌生连接（不握手），把容量（4）占满
    strangers = [rig.client() for _ in range(2)]
    # 再来一条 → 超限，被踢的应是"还没握手"的那条
    late = rig.client()
    time.sleep(0.4)

    roles = [c.meta.get("role") for c in rig.server._ws.clients]
    assert "audio" in roles, "正在送音频的那条绝不能被踢"
    assert "control" in roles
    # 音频仍然可用
    audio.send_binary(_pcm())
    time.sleep(0.3)
    assert rig.server.snapshot().frames >= 2
    late.close()
    for s in strangers:
        s.close()


def test_audio_disconnect_falls_back_to_control_state(rig: Rig):
    """音频那条断了之后，采集状态以控制连接的说法为准。"""
    control = rig.client()
    _hello_role(control, "control")
    audio = rig.client()
    _hello_role(audio, "audio", tab_id=11)
    audio.close()
    deadline = time.time() + 3
    while time.time() < deadline and rig.server.snapshot().capturing:
        time.sleep(0.02)
    assert rig.server.snapshot().capturing is False
    # 控制连接 report 又开始采集 → 状态跟上
    control.send_json({"type": "status", "capturing": True, "tab": {"id": 12, "title": "新的"}})
    deadline = time.time() + 3
    while time.time() < deadline and not rig.server.snapshot().capturing:
        time.sleep(0.02)
    assert rig.server.snapshot().capturing is True


# --------------------------------------------------------------------------- #
# 纯逻辑
# --------------------------------------------------------------------------- #
def test_tab_from_payload_is_defensive():
    assert _tab_from_payload(None) == TabInfo()
    assert _tab_from_payload({"id": "abc", "title": None}) == TabInfo(id=0, title="")
    assert _tab_from_payload({"id": 3, "faviconUrl": "f.png"}).favicon_url == "f.png"
    assert _tab_from_payload({"id": 3, "favIconUrl": "g.png"}).favicon_url == "g.png"


def test_tab_short_title_truncates():
    tab = TabInfo(id=1, title="字" * 100)
    assert len(tab.short_title) == 60
    assert tab.describe().startswith("字" * 10)


def test_format_from_dict_defaults():
    assert AudioFormat.from_dict(None) == AudioFormat()
    # 0/缺失一律回落到默认（扩展不该送 0，但真送了也不能让管线去理解 0 声道）
    assert AudioFormat.from_dict({"channels": 0}).channels == 2
    # 非法负数被夹到 1
    assert AudioFormat.from_dict({"channels": -3}).channels == 1
    assert AudioFormat.from_dict({"rate": "48000"}).rate == 48000


def test_port_is_reported(rig: Rig):
    assert rig.port > 0
