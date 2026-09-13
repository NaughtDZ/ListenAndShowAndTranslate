"""浏览器标签页音频：扩展 → 本机 WebSocket → 与进程回环**同一条** AudioPipeline。

为什么需要这个模块（`docs/浏览器标签页.md` 有完整实测记录）：

1. Chromium 在 Windows 上**一个浏览器实例只建一个音频会话**——实测两个标签页
   同时出声，音频会话仍然只有 1 个。所以进程回环 / 音频会话 API 这两条路
   **物理上拿不到标签页边界**。
2. 唯一能拿到标签页边界的地方是浏览器内部（`chrome.tabCapture`）。
   于是让一个很小的浏览器扩展把**目标标签页**的音频取出来，
   用本机回环 WebSocket 送给本程序。
3. 扩展送来的 PCM 会被重新拼成 **48k / 立体声 / float32** ——和 proc-tap 的
   输出格式逐字节一致，因此下游（降混、重采样、VAD、语言路由、ASR、翻译）
   全部零改动。

协议（文本帧 = 控制，二进制帧 = PCM）::

    扩展 → 程序   {"type":"hello", "protocol":1, "token":"…",
                   "format":{...}, "tab":{...}, "capturing":true}
    程序 → 扩展   {"type":"welcome", "protocol":1, "ok":true, "format":{...}}
    程序 → 扩展   {"type":"capture", "tabId":12, "format":{...}}
    程序 → 扩展   {"type":"stop"}
    扩展 → 程序   {"type":"status", "tab":{...}, "audible":true, "muted":false}
    扩展 → 程序   {"type":"error", "message":"…"}
    扩展 → 程序   <二进制帧：裸 PCM，交错，小端>

单客户端：新连接替换旧连接（``CLOSE_REPLACED``），扩展自己负责重连。
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.audio.pipeline import AudioPipeline, PipelineStats, SILENCE_RMS_THRESHOLD
from app.audio.ws import CLOSE_REPLACED, WSClient, WSServer
from app.utils.log import get_logger

log = get_logger(__name__)

PROTOCOL_VERSION = 1

DEFAULT_PORT = 38991
"""默认端口。选它是因为它落在动态端口段之外、且不与常见服务冲突。"""

DEFAULT_PATH = "/lst/tab"

HELLO_TIMEOUT_S = 5.0
"""连上后多久没收到 hello 就断开（防止陌生连接空占通道）。"""

STATE_STALE_SECONDS = 5.0
"""多久没收到音频就提示"目标好像没在放音"（与进程回环的语义保持一致）。"""

BUSY_NOTE_SECONDS = 60.0
"""「另一个浏览器的扩展也在连、本条被忽略」这条提示保留多久。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TabInfo:
    """扩展报告的标签页信息。"""

    id: int = 0
    title: str = ""
    url: str = ""
    favicon_url: str = ""

    @property
    def short_title(self) -> str:
        title = (self.title or "").strip()
        return title if len(title) <= 60 else title[:59] + "…"

    def describe(self) -> str:
        if not self.id and not self.title:
            return "（未知标签页）"
        return f"{self.short_title or '未命名标签页'}（tab {self.id}）"


@dataclass(frozen=True)
class AudioFormat:
    """扩展送来的 PCM 格式。默认与 proc-tap 输出一致。"""

    rate: int = 48000
    channels: int = 2
    dtype: str = "float32"

    def as_dict(self) -> dict:
        return {"rate": self.rate, "channels": self.channels, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AudioFormat":
        raw = raw or {}
        return cls(
            rate=int(raw.get("rate") or 48000),
            channels=max(1, int(raw.get("channels") or 2)),
            dtype=str(raw.get("dtype") or "float32"),
        )


@dataclass
class TabAudioStats:
    """供 UI 显示的状态快照（字段含义与 ``CaptureStats`` 对齐）。"""

    running: bool = False
    connected: bool = False
    """扩展是否连上（连上≠正在送音）。"""

    capturing: bool = False
    """扩展是否正在采集标签页音频。"""

    tab: TabInfo = field(default_factory=TabInfo)
    browser: str = ""
    extension_id: str = ""
    """连上来的扩展 ID（从 Origin 里取）。诊断"是不是我们的扩展"用得上。"""
    format: AudioFormat = field(default_factory=AudioFormat)

    frames: int = 0
    bytes_in: int = 0
    samples_out: int = 0

    last_rms: float = 0.0
    peak: float = 0.0
    silent_seconds: float = 0.0
    total_silent_seconds: float = 0.0

    connections: int = 0
    last_error: str = ""
    connected_at: float = 0.0
    peer_note: str = ""
    """有多个扩展实例同时连上来时的提示（实测真的会发生：用户日常浏览器里装了扩展，
    临时跑测试的浏览器又装了一份，两边都连同一个端口）。"""

    @property
    def likely_playing(self) -> bool:
        return self.connected and self.capturing and self.silent_seconds < STATE_STALE_SECONDS

    def describe(self) -> str:
        if not self.connected:
            return "等待浏览器扩展连接…"
        if not self.capturing:
            return f"扩展已连接（{self.browser or '浏览器'}），等待开始采集…"
        tab = self.tab.short_title or f"tab {self.tab.id}"
        extra = "" if self.likely_playing else "（当前没在放音）"
        note = f" · {self.peer_note}" if self.peer_note else ""
        return f"标签页: {tab}{extra}{note}"


# --------------------------------------------------------------------------- #
# 服务端
# --------------------------------------------------------------------------- #
class TabAudioServer:
    """接收浏览器扩展送来的标签页音频，并喂给 AudioPipeline。

    用法::

        server = TabAudioServer(on_chunk=my_cb, on_state=my_state_cb)
        server.start()
        ...
        server.request_capture(tab_id=12)   # 让扩展开始送这个标签页
        ...
        server.stop()

    线程模型：``on_chunk`` / ``on_state`` 都在 WebSocket 读线程里被调用
    ——与 ``CaptureWorker`` 的回调语义一致，因此调用方可以直接复用
    "回调里只做拷贝/投递，不碰 Qt 控件"的既有约定。
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = "127.0.0.1",
        token: str = "",
        path: str = DEFAULT_PATH,
        on_chunk: Callable[[object], None] | None = None,
        on_state: Callable[[dict], None] | None = None,
        out_rate: int = 16000,
        silence_threshold: float = SILENCE_RMS_THRESHOLD,
        require_extension_origin: bool = True,
    ) -> None:
        self.host = host
        self.path = path
        self.token = token or ""
        self.out_rate = out_rate
        self.silence_threshold = silence_threshold
        self.require_extension_origin = require_extension_origin

        self._on_chunk = on_chunk
        self._on_state = on_state

        self.stats = TabAudioStats()
        self._pipeline: AudioPipeline | None = None
        self._lock = threading.Lock()
        self._last_audio_at = 0.0
        self._watchdog: threading.Thread | None = None
        self._control: WSClient | None = None
        self._audio: WSClient | None = None
        self._control_capturing = False
        """控制连接报告的"在采集"状态——**只在没有音频连接时才采信**。"""

        self._busy_note = ""
        self._busy_note_at = 0.0
        """最近一次"另一个浏览器的扩展被忽略"的提示（要留一会儿给用户看）。"""

        self._ws = WSServer(
            host=host,
            port=port,
            path=path,
            on_open=self._on_open,
            on_text=self._on_text,
            on_binary=self._on_binary,
            on_close=self._on_close,
            # 两条连接：扩展的 service worker 走控制，offscreen 文档走音频。
            # 音频**不走扩展端口转发**——实测 ArrayBuffer 经 chrome.runtime 端口的
            # 那一跳会变成 null（于是 ws.send(null) 发出去的是字符串 "[object Object]"），
            # 而且那样还要多一次拷贝。
            #
            # 上限给 4 而不是 2：实测用户日常浏览器里装的那份扩展也会连上来
            # （另一个配置文件/另一份拷贝），卡在 2 会把正在送音频的那条挤掉。
            max_clients=4,
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @property
    def port(self) -> int:
        return self._ws.port

    @property
    def is_running(self) -> bool:
        return self._ws.is_running

    @property
    def client(self) -> WSClient | None:
        return self._ws.client

    def start(self) -> None:
        self._ws.start()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="lst-tab-watchdog", daemon=True
        )
        self._watchdog.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._ws.stop(timeout=timeout)
        self._watchdog = None
        with self._lock:
            self.stats.running = False
            self.stats.connected = False
            self.stats.capturing = False

    def set_on_chunk(self, callback: Callable[[object], None] | None) -> None:
        self._on_chunk = callback

    def set_on_state(self, callback: Callable[[dict], None] | None) -> None:
        self._on_state = callback

    # ------------------------------------------------------------------ #
    # 指令下发
    # ------------------------------------------------------------------ #
    def request_capture(self, tab_id: int | None = None, fmt: AudioFormat | None = None) -> bool:
        """让扩展开始采集某个标签页（``tab_id=None`` = 由扩展用当前活动标签页）。"""
        payload: dict = {
            "type": "capture",
            "format": (fmt or AudioFormat()).as_dict(),
        }
        if tab_id:
            payload["tabId"] = int(tab_id)
        return self._send(payload)

    def request_stop(self) -> bool:
        return self._send({"type": "stop"})

    def request_state(self) -> bool:
        return self._send({"type": "state"})

    def _send(self, payload: dict) -> bool:
        client = self._control or self._audio or self._ws.client
        if client is None:
            log.debug("扩展未连接，指令丢弃: %s", payload.get("type"))
            return False
        return client.send_json(payload)

    # ------------------------------------------------------------------ #
    # WebSocket 回调
    # ------------------------------------------------------------------ #
    def _on_open(self, client: WSClient) -> bool:
        if self.require_extension_origin and not client.origin.startswith("chrome-extension://"):
            log.warning("拒绝非扩展来源的连接：%s（Origin=%r）", client.addr, client.origin)
            return False
        client.meta["hello_at"] = 0.0
        client.meta["deadline"] = time.time() + HELLO_TIMEOUT_S
        with self._lock:
            self.stats.connected = True
            self.stats.connections += 1
            self.stats.connected_at = time.time()
            self.stats.last_error = ""
            self.stats.extension_id = client.origin.removeprefix("chrome-extension://").rstrip("/")
            # ⚠️ 这里**刻意不动 ``capturing``**：新连接不代表采集停了。
            # 实测踩过：另一个扩展实例（用户日常浏览器里那份）一连上来就把
            # 状态重置成"没在采集"，界面上看起来就像音频发送暂停了。
            self._refresh_note_locked()
        log.info("浏览器扩展已连接：%s（Origin=%s）", client.addr, client.origin)
        self._emit_state()
        return True

    def _on_text(self, client: WSClient, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            log.warning("收到无法解析的控制消息：%r", text[:200])
            return
        kind = str(msg.get("type") or "")
        if kind == "hello":
            self._handle_hello(client, msg)
        elif kind == "format":
            self._handle_format(msg)
        elif kind == "status":
            self._handle_status(client, msg)
        elif kind == "log":
            log.info("[扩展] %s", msg.get("message") or "")
        elif kind == "error":
            message = str(msg.get("message") or "扩展报告了未知错误")
            with self._lock:
                self.stats.last_error = message
            log.warning("扩展报告错误：%s", message)
            self._emit_state()

    def _is_authoritative(self, client: WSClient) -> bool:
        """这条连接有没有资格改"当前音源"的状态。

        规则：**正在送音频的那条说了算**；没有音频连接时，才轮到控制连接。
        这样"另一个浏览器里也装了扩展、也连上来了"就不会把正在进行的采集搅乱
        （实测：它 report capturing=false，界面就显示成"等待开始采集"了）。
        """
        if client.meta.get("legacy"):
            # 旧版扩展没办法区分 SW/offscreen，谁报都算
            return True
        if self._audio is not None:
            return client is self._audio
        return client is self._control

    def _handle_hello(self, client: WSClient, msg: dict) -> None:
        version = int(msg.get("protocol") or 0)
        if version != PROTOCOL_VERSION:
            # 先落状态再关连接：对端一旦看到关闭帧，我们的状态就该已经是对的
            with self._lock:
                self.stats.last_error = f"扩展协议版本 {version} 与本程序 {PROTOCOL_VERSION} 不匹配"
            client.send_json(
                {"type": "error", "code": "protocol",
                 "message": f"协议版本不匹配：扩展 {version} / 程序 {PROTOCOL_VERSION}"}
            )
            client.close(1008, "protocol mismatch")
            self._emit_state()
            return

        token = str(msg.get("token") or "")
        # 注意：compare_digest 只接受 ASCII 字符串，中文配对码必须自己编码成 bytes，
        # 否则会抛 TypeError（而且是被读线程吞掉、表现为"发出去的消息石沉大海"）。
        if self.token and not hmac.compare_digest(
            token.encode("utf-8"), self.token.encode("utf-8")
        ):
            with self._lock:
                self.stats.last_error = "扩展送来的配对码不正确（设置里可以查看/修改）"
            client.send_json({"type": "error", "code": "token", "message": "配对码不正确"})
            client.close(1008, "bad token")
            log.warning("配对码校验失败")
            self._emit_state()
            return

        role_raw = msg.get("role")
        role = str(role_raw or "both").lower()
        if role not in ("control", "audio", "both"):
            role = "both"
        legacy = role_raw is None
        """没带 role 字段 = 旧版扩展（本程序加 role 之前的版本）。

        实测踩过：用户日常浏览器里装的是旧版，它一 hello 就被当成"两个角色都占"，
        把正在送音频的那条顶掉——用户看到的就是"音频发送暂停，得手动再点一次"。
        这里只标记出来并提示（协议本身向后兼容），真正防止顶掉的是下面的"同源才替换"。
        """
        client.meta["role"] = role
        client.meta["legacy"] = legacy
        client.meta["hello_at"] = time.time()
        client.greeted = True

        fmt = AudioFormat.from_dict(msg.get("format"))
        tab = _tab_from_payload(msg.get("tab"))
        browser = str((msg.get("browser") or {}).get("name") or "")

        # ---- 跨源保护：不许"另一个浏览器里的扩展"顶掉正在工作的连接 ----
        # 同一个源（同一个浏览器）重连是正常的（SW 会被回收），照旧替换；
        # 不同源且对方还活着 → 拒绝新来的，而不是把干活的踢掉。
        for slot_name, slot in (("audio", self._audio), ("control", self._control)):
            if (
                slot is not None
                and not slot.closing
                and slot.origin != client.origin
                and role in (slot_name, "both")
            ):
                msg_text = (
                    f"已有另一个浏览器扩展在当 {slot_name}"
                    f"（{slot.origin.removeprefix('chrome-extension://')[:8]}…），"
                    "本条连接被忽略"
                )
                log.warning("%s（新连接 %s）", msg_text, client.addr)
                with self._lock:
                    self.stats.last_error = msg_text
                    # 这条提示要活一会儿：被拒的连接随后会触发 _on_close，
                    # 而 _refresh_note_locked 会把"多实例"提示清掉——
                    # 用户还没来得及看见，提示就没了（实测就是这么丢的）。
                    self._busy_note = msg_text
                    self._busy_note_at = time.time()
                    self._refresh_note_locked()
                client.send_json({"type": "error", "code": "busy", "message": msg_text})
                self._ws.drop_client(client, 1008, "another extension is active")
                self._emit_state()
                return

        # 音频连接：格式以它为准（PCM 是从这条连接来的），状态也以它为准
        if legacy and any(
            c is not client and c.greeted and not c.closing and c.origin == client.origin
            for c in self._ws.clients
        ):
            # 旧版扩展的 SW 与 offscreen 都发不带 role 的 hello，两条长得一模一样，
            # 无从区分角色。若还按"同角色替换"处理，它们会互相顶成死循环；
            # 所以旧版连接一律**并存**：谁送音频都算，谁报状态都听
            # （等价于加 role 之前的历史行为）。
            with self._lock:
                self.stats.connected = True
                self.stats.format = fmt
                self._refresh_note_locked()
            client.send_json(
                {
                    "type": "welcome",
                    "protocol": PROTOCOL_VERSION,
                    "ok": True,
                    "role": "both",
                    "format": fmt.as_dict(),
                    "out_rate": self.out_rate,
                }
            )
            log.warning("旧版扩展又连了一条（hello 缺 role）——按并存处理；建议在 edge://extensions 重新加载")
            self._emit_state()
            return

        if role in ("audio", "both"):
            old = self._audio
            self._audio = client
            if old is not None and old is not client:
                self._ws.drop_client(old, CLOSE_REPLACED, "audio connection replaced")
            self._pipeline = AudioPipeline(
                in_rate=fmt.rate,
                in_channels=fmt.channels,
                input_dtype=fmt.dtype,
                out_rate=self.out_rate,
                silence_threshold=self.silence_threshold,
            )
        if role in ("control", "both"):
            old = self._control
            self._control = client
            if old is not None and old is not client:
                self._ws.drop_client(old, CLOSE_REPLACED, "control connection replaced")
            self._control_capturing = bool(msg.get("capturing"))

        audio_origin = self._audio.origin if self._audio is not None else None
        same_as_audio = audio_origin is not None and client.origin == audio_origin

        with self._lock:
            self.stats.connected = True
            if self._audio is not None:
                # 正在送音频，状态就由它说了算：只有音频连接本身、或与它同源的
                # 连接（同一个浏览器）才有资格改"目标标签页/浏览器"。
                # 否则另一个浏览器里的同名扩展一连上来就会把标题换掉。
                self.stats.capturing = True
                if role in ("audio", "both") or same_as_audio:
                    self.stats.tab = tab
                    self.stats.browser = browser
            elif role in ("control", "both"):
                self.stats.capturing = bool(msg.get("capturing"))
                self.stats.tab = tab
                self.stats.browser = browser
            self.stats.format = fmt
            self._refresh_note_locked()
        self._last_audio_at = time.time()

        client.send_json(
            {
                "type": "welcome",
                "protocol": PROTOCOL_VERSION,
                "ok": True,
                "role": role,
                "format": fmt.as_dict(),
                "out_rate": self.out_rate,
            }
        )
        log.info(
            "扩展握手完成（%s）：%s · %s · %dHz/%dch/%s · 正在采集=%s",
            role, browser or "浏览器", tab.describe(), fmt.rate, fmt.channels, fmt.dtype,
            self.stats.capturing,
        )
        self._emit_state()

    def _handle_format(self, msg: dict) -> None:
        """扩展报告**实际**格式（设备采样率可能与预期不同）。

        必须在开始送音频之前重建管线，否则会按错误的采样率解释字节——
        表现为"字幕全是乱码/识别不出"，而且很难查。
        """
        fmt = AudioFormat.from_dict(msg)
        with self._lock:
            same = fmt == self.stats.format
            self.stats.format = fmt
        if same:
            return
        self._pipeline = AudioPipeline(
            in_rate=fmt.rate,
            in_channels=fmt.channels,
            input_dtype=fmt.dtype,
            out_rate=self.out_rate,
            silence_threshold=self.silence_threshold,
        )
        log.info("按扩展报告的格式重建管线：%dHz/%dch/%s", fmt.rate, fmt.channels, fmt.dtype)
        self._emit_state()

    def _handle_status(self, client: WSClient, msg: dict) -> None:
        if not self._is_authoritative(client):
            log.debug("忽略非权威连接的状态消息（可能是另一个浏览器扩展）")
            return
        tab = _tab_from_payload(msg.get("tab")) if msg.get("tab") else None
        with self._lock:
            if tab is not None:
                self.stats.tab = tab
            if "capturing" in msg:
                # 权威连接自己报的状态要采信：
                # 它是音频连接（或没有音频连接时的控制连接），"我停了"就是真停了。
                self._control_capturing = bool(msg.get("capturing"))
                self.stats.capturing = bool(msg.get("capturing"))
            if msg.get("message"):
                self.stats.last_error = str(msg["message"])
        self._emit_state()

    def _on_binary(self, client: WSClient, data: bytes) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            log.debug("收到音频但尚未握手，丢弃 %d 字节", len(data))
            return

        out = pipeline.process(data)
        now = time.time()
        rms = pipeline.stats.last_rms

        with self._lock:
            self.stats.running = True
            # 有帧到 = 铁证"正在采集"。任何连接的状态消息都不该把它改回去。
            self.stats.capturing = True
            self.stats.frames += 1
            self.stats.bytes_in += len(data)
            self.stats.samples_out += int(out.size)
            self.stats.last_rms = rms
            self.stats.peak = pipeline.stats.peak
            if rms >= pipeline.silence_threshold:
                self._last_audio_at = now
                self.stats.silent_seconds = 0.0
            else:
                self.stats.silent_seconds = now - self._last_audio_at
                self.stats.total_silent_seconds = pipeline.stats.silent_seconds_total

        if out.size and self._on_chunk is not None:
            try:
                self._on_chunk(out)
            except Exception as exc:  # noqa: BLE001 - 回调的锅不能让读线程崩
                log.error("音频回调抛异常: %s", exc)

    def _on_close(self, client: WSClient, code: int) -> None:
        greeted = bool(client.meta.get("hello_at"))
        role = str(client.meta.get("role") or "")
        if client is self._control:
            self._control = None
        if client is self._audio:
            self._audio = None
        still_here = any(c.meta.get("hello_at") for c in self._ws.clients if c is not client)
        with self._lock:
            self.stats.connected = bool(still_here)
            if role in ("audio", "both") or client.meta.get("legacy"):
                # 送音频的那条断了：音频必然停了；但若控制连接还在，
                # 采集状态还要看它怎么说（可能正在重连）
                self.stats.running = False
                self.stats.capturing = self._control_capturing if still_here else False
            self._refresh_note_locked()
            if not greeted and not self.stats.last_error:
                self.stats.last_error = "连接在握手前断开（可能是扩展版本不匹配）"
        log.info("浏览器扩展连接断开（role=%s code=%s）", role or "?", code)
        self._emit_state()

    def _refresh_note_locked(self) -> None:
        """记录"是不是有多个扩展实例/旧版扩展连上来了"（调用方须已持锁）。"""
        clients = [c for c in self._ws.clients if c.meta.get("hello_at")]
        origins = {c.origin for c in clients if c.origin}
        legacy = [c for c in clients if c.meta.get("legacy")]
        notes: list[str] = []
        busy = bool(getattr(self, "_busy_note", "")) and (
            time.time() - getattr(self, "_busy_note_at", 0.0) < BUSY_NOTE_SECONDS
        )
        if busy:
            notes.append(self._busy_note)
        elif len(origins) > 1:
            # 与上面那条 busy 提示说的是同一件事，别重复刷屏
            notes.append(f"另有 {len(origins) - 1} 个浏览器扩展也连上来了（音频以正在送的那条为准）")
        if legacy:
            notes.append("检测到旧版扩展：请在 edge://extensions 里点【重新加载】")
        self.stats.peer_note = "；".join(notes)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def snapshot(self) -> TabAudioStats:
        with self._lock:
            return TabAudioStats(**vars(self.stats))

    def pipeline_stats(self) -> PipelineStats:
        pipeline = self._pipeline
        if pipeline is None:
            return PipelineStats(silence_threshold=self.silence_threshold)
        return pipeline.stats

    def restart_pipeline(self) -> None:
        """重采样器状态复位（例如目标标签页切换后，避免把两段音频缝在一起）。"""
        if self._pipeline is not None:
            self._pipeline.reset()

    def _emit_state(self) -> None:
        if self._on_state is None:
            return
        try:
            self._on_state(self.snapshot())
        except Exception as exc:  # noqa: BLE001
            log.error("状态回调抛异常: %s", exc)

    def _watchdog_loop(self) -> None:
        """把"连上但迟迟不握手"的连接踢掉。"""
        while self._ws.is_running:
            time.sleep(1.0)
            client = self._ws.client
            if client is None:
                continue
            deadline = float(client.meta.get("deadline") or 0.0)
            if deadline and not client.meta.get("hello_at") and time.time() > deadline:
                log.warning("连接 %s 在 %.0fs 内没有握手，断开", client.addr, HELLO_TIMEOUT_S)
                client.close(1008, "hello timeout")


def _tab_from_payload(raw: object) -> TabInfo:
    if not isinstance(raw, dict):
        return TabInfo()
    try:
        tab_id = int(raw.get("id") or 0)
    except (TypeError, ValueError):
        tab_id = 0
    return TabInfo(
        id=tab_id,
        title=str(raw.get("title") or ""),
        url=str(raw.get("url") or ""),
        favicon_url=str(raw.get("favIconUrl") or raw.get("faviconUrl") or ""),
    )
