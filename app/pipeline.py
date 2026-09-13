"""字幕流水线：采集 → 识别 → 翻译 → 字幕状态。

线程划分（这是本文件最需要注意的地方）：

    [采集线程]  CaptureWorker 回调 → LanguageRouter.feed() → ASR 事件
         │  emit Qt 信号（跨线程队列投递，安全）
         ▼
    [UI 线程]  收到信号 → 改 SubtitleState → 悬浮窗重绘
         │  put 待翻译字幕
         ▼
    [翻译线程]  攒一小批（或超时）→ TranslatorHub → emit 译文

**SubtitleState 只在 UI 线程被修改**——否则会出现"字幕串行/错位"这种极难查的问题。
识别在采集线程里做（模型推理本来就慢，不该占用 UI 线程）。

翻译做**小批合并**：实测批量比逐条省 70% token 且更快（docs/P4-翻译实测.md），
但也不能为了凑批让字幕等太久，所以是"凑够 batch 条 或 最多等 batch_wait_ms"。
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.asr.base import ASREvent, ASREventType
from app.asr.router import LanguageRouter
from app.audio.capture import CaptureWorker, TargetSpec, resolve_target
from app.config import AppConfig
from app.models.registry import human_size
from app.subtitle.model import SubtitleState
from app.translate.base import Segment
from app.translate.glossary import Glossary, load_glossary
from app.translate.hub import TranslatorHub
from app.utils.log import get_logger

log = get_logger(__name__)


class SubtitlePipeline(QObject):
    """把各模块接起来。所有对外信号都在 Qt 线程可用。"""

    partialReceived = Signal(str, str)
    """(未定稿原文, 语言)"""

    finalReceived = Signal(int, str, str)
    """(行 id, 定稿原文, 语言)"""

    translationReceived = Signal(int, str, str)
    """(行 id, 译文, 错误说明)"""

    statusChanged = Signal(str)
    errorOccurred = Signal(str)
    statsChanged = Signal(dict)
    tabStateChanged = Signal(dict)
    """浏览器标签页通道的状态（连接/断开/目标标签页变化）。参数是 TabAudioStats 的字段字典。"""

    def __init__(
        self,
        config: AppConfig,
        models_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.models_dir = models_dir
        self.state = SubtitleState()

        # 识别
        self.router: LanguageRouter | None = None
        # 采集
        self.capture: CaptureWorker | None = None
        self._target_spec: TargetSpec | None = None
        # 浏览器标签页音频（扩展 → 本机 WebSocket → 同一条 AudioPipeline）
        self.tab_server: "TabAudioServer | None" = None
        self._owns_tab_server = False
        self.tab_source = False
        """当前音源是不是"浏览器标签页"（restart 时用它决定怎么重新开工）。"""

        # 翻译
        self.hub: TranslatorHub | None = None
        self._glossary: Glossary = Glossary()
        self._queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
        self._trans_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_translate_activity = 0.0

        self.started_at = 0.0
        self.asr_latency_ms: list[float] = []
        self.translate_latency_ms: list[float] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 准备
    # ------------------------------------------------------------------ #
    def prepare(self) -> tuple[bool, str]:
        """加载术语表、构建翻译通道与识别路由器。返回 (成功, 说明)。"""
        cfg = self.config

        # 术语表
        glossary_path = (cfg.translate.glossary or {}).get("_path") if isinstance(cfg.translate.glossary, dict) else None
        if glossary_path:
            self._glossary = load_glossary(glossary_path)
        if cfg.translate.glossary and isinstance(cfg.translate.glossary, dict):
            # 配置里直接写的内联术语优先于文件
            inline = {k: v for k, v in cfg.translate.glossary.items() if not k.startswith("_")}
            self._glossary.update(inline)

        # 翻译通道
        self.hub = TranslatorHub(cfg.translate, glossary=self._glossary)
        notes: list[str] = []
        provider = cfg.translate.provider

        if provider == "llm" or provider.startswith("llm"):
            from app.translate.openai_compat import OpenAICompatTranslator

            llm = cfg.translate.llm
            if not llm.enabled:
                notes.append("LLM 通道未启用")
            else:
                t = OpenAICompatTranslator(
                    base_url=llm.base_url, api_key=llm.api_key, model=llm.model,
                    temperature=llm.temperature, max_tokens=llm.max_tokens,
                    timeout_s=llm.timeout_s, proxy=cfg.proxy,
                    verify_glossary=True, name="llm",
                    prompt_style=getattr(llm, "prompt_style", "") or "",
                    # 思考默认关（字幕翻译不需要思考，且会把预算烧光导致空译文）
                    disable_thinking=getattr(llm, "disable_thinking", True),
                )
                self.hub.register(t, priority=10)
                ok, msg = t.ping()
                notes.append(f"LLM({llm.model}): {msg}")
        elif provider != "none":
            from app.translate.traditional.providers import ALL_PROVIDERS, build_provider

            if provider not in ALL_PROVIDERS:
                notes.append(f"未知通道: {provider}")
            else:
                # web_* 是免 key 的网页版内部接口（非官方，可能失效）
                creds = cfg.translate.providers.get(provider, {})
                t = build_provider(
                    provider, creds, proxy=cfg.proxy, qps_limit=cfg.translate.qps_limit
                )
                if t is None:
                    notes.append(f"无法构造通道: {provider}")
                else:
                    self.hub.register(t, priority=20 if provider.startswith("web_") else 10)
                    ok, msg = t.ping()
                    tag = "（非官方接口，可能失效）" if provider.startswith("web_") else ""
                    notes.append(f"{provider}{tag}: {msg}")

        # 识别路由器
        self.router = LanguageRouter(
            cfg.asr, models_dir=self.models_dir,
            on_log=lambda m: log.info("[router] %s", m),
        )
        return True, "；".join(notes) if notes else "就绪"

    # ------------------------------------------------------------------ #
    # 启停
    # ------------------------------------------------------------------ #
    def start(self, target: TargetSpec) -> bool:
        if self.router is None or self.hub is None:
            self.prepare()

        self.tab_source = False
        resolved = resolve_target(target)
        if resolved is None:
            # **不因为"现在没在发声"就拒绝启动**：
            # 用户完全可能"先开字幕窗、再开播放器"，CaptureWorker 本来就会每
            # reconnect_interval_s 重试一次（follow=True），所以这里先跑起来、
            # 把状态如实报出去，等目标出声就自动接上。
            log.info("目标 %s 现在没有输出音频，先启动并等待", target.describe())
            self.statusChanged.emit(f"等待「{target.describe()}」开始播放…（会自动重连）")

        self._target_spec = target
        self._stop.clear()
        self.started_at = time.time()

        # 翻译线程
        self._trans_thread = threading.Thread(
            target=self._translate_loop, name="lst-translate", daemon=True
        )
        self._trans_thread.start()

        # 采集线程：识别就在这个线程里跑
        self.capture = CaptureWorker(
            target, on_chunk=self._on_audio, follow=True,
            reconnect_interval_s=self.config.audio.reconnect_interval_s,
        )
        self.capture.start()

        if resolved is not None:
            self.statusChanged.emit(
                f"已开始 · {resolved.name} (PID {resolved.pid}) · 语言={self.config.asr.language}"
            )
        return True

    def start_tab_audio(self, server=None) -> bool:
        """浏览器标签页模式：音频由扩展经本机 WebSocket 送来。

        与进程模式共用同一条 AudioPipeline（扩展送的就是 48k/立体声/float32，
        和 proc-tap 的输出格式一致），所以识别/翻译全链路零改动。

        Args:
            server: 复用外部已经跑着的 :class:`TabAudioServer`；不传就按配置新建一个。
        """
        from app.audio.tab_audio import TabAudioServer

        if self.router is None or self.hub is None:
            self.prepare()

        cfg = self.config.tab_audio
        self._stop.clear()
        self.started_at = time.time()

        # 翻译线程
        self._trans_thread = threading.Thread(
            target=self._translate_loop, name="lst-translate", daemon=True
        )
        self._trans_thread.start()

        if server is None:
            server = TabAudioServer(
                port=cfg.port,
                token=cfg.token,
                require_extension_origin=cfg.require_extension_origin,
                on_chunk=self._on_audio,
                on_state=self._on_tab_state,
            )
            self._owns_tab_server = True
            try:
                server.start()
            except OSError as exc:
                self.errorOccurred.emit(f"端口 {cfg.port} 被占用，无法监听：{exc}")
                return False
        else:
            self._owns_tab_server = False
            server.set_on_chunk(self._on_audio)
            server.set_on_state(self._on_tab_state)

        self.tab_server = server
        self.tab_source = True
        self.statusChanged.emit(
            f"等待浏览器扩展连接 · 本机端口 {server.port} · 语言={self.config.asr.language}"
        )
        return True

    def _on_tab_state(self, stats) -> None:
        """在 WebSocket 读线程里被调用——只 emit 信号，不碰 Qt 控件。"""
        try:
            payload = {
                "connected": stats.connected,
                "capturing": stats.capturing,
                "tab_title": stats.tab.title,
                "tab_id": stats.tab.id,
                "browser": stats.browser,
                "extension_id": stats.extension_id,
                "likely_playing": stats.likely_playing,
                "last_error": stats.last_error,
                "describe": stats.describe(),
            }
        except Exception as exc:  # noqa: BLE001
            log.debug("标签页状态转换失败: %s", exc)
            return
        self.tabStateChanged.emit(payload)
        if stats.last_error:
            self.errorOccurred.emit(stats.last_error)
        else:
            self.statusChanged.emit(stats.describe())

    def reload(self, timeout: float = 15.0) -> bool:
        """就地重建识别与翻译引擎，让"设置保存后立即生效"。

        用户反馈过"点了保存设置但功能没变化"——因为改完配置只写了文件，
        而正在跑的流水线用的还是内存里那份旧配置。
        这里停掉采集与翻译线程、重建 Hub 与 Router、再用同一个音源重启。

        代价：短暂（约 1~2 秒）采集空档，比"要重启整个程序"友好得多。
        """
        if self._target_spec is None and not self.tab_source:
            return False
        log.info("正在按新配置重建识别/翻译引擎…")
        ok = self.restart(timeout=timeout)
        log.info("引擎重建完成：%s", "成功" if ok else "启动失败")
        return ok

    def switch_source(
        self,
        spec: TargetSpec | None = None,
        tab_mode: bool = False,
        timeout: float = 15.0,
    ) -> bool:
        """**运行中换音频来源**（进程 ↔ 浏览器标签页），不需要重启程序。

        用户的疑问："为什么只有启动时能选监听模式和目标，进了程序反而调不了？"
        原因只是历史包袱（启动窗口把 PID 写进命令行、子进程只有一个入口），
        不是技术限制——换音源本质上就是"停掉当前采集 → 重建引擎 → 按新音源开工"，
        和"设置保存后立即生效"走的是同一条路（``restart``）。

        Args:
            spec: 进程模式的新目标。
            tab_mode: True = 换成浏览器标签页模式（忽略 ``spec``）。
        """
        self.tab_source = bool(tab_mode)
        self._target_spec = None if tab_mode else spec
        log.info(
            "切换音频来源：%s",
            "浏览器标签页" if tab_mode else (spec.describe() if spec else "（未指定）"),
        )
        ok = self.restart(timeout=timeout)
        if not ok:
            self.errorOccurred.emit("切换音源失败（见日志）")
        return ok

    def restart(self, timeout: float = 15.0) -> bool:
        """停掉当前音源与引擎，重建，再按**当前**音源开工。"""
        try:
            self.stop(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("重启前停止流水线出错（忽略并继续）: %s", exc)
        self._stop.clear()
        self._lang_reported = False  # 换了音源/配置，语言要重新报一次
        try:
            self.prepare()
        except Exception as exc:  # noqa: BLE001
            log.error("重建失败，仍继续启动: %s", exc)
        if self.tab_source:
            return self.start_tab_audio()
        if self._target_spec is None:
            return False
        return self.start(self._target_spec)

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        if self.tab_server is not None:
            try:
                # 先摘掉回调，避免停止过程中还有音频进来喂已经停了的识别器
                self.tab_server.set_on_chunk(None)
                self.tab_server.set_on_state(None)
                if self._owns_tab_server:
                    self.tab_server.stop()
            except Exception as exc:  # noqa: BLE001 - 停不干净也不能拦住退出
                log.warning("停止标签页通道出错（忽略）: %s", exc)
            finally:
                self.tab_server = None
        t = self._trans_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._trans_thread = None
        # 把最后一行未翻译的也冲出去
        if self.router is not None:
            for ev in self.router.finish():
                self._dispatch(ev)
        if self.hub is not None:
            self.hub.close()

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    @property
    def current_source(self) -> tuple[TargetSpec | None, bool]:
        """当前音源：``(进程目标, 是否浏览器标签页模式)``。UI 用来显示"当前"用。"""
        return (self._target_spec, self.tab_source)

    def pipeline_stats(self):
        """当前音源管线的统计（设置窗里的电平表用）。没在采集时返回 None。

        进程模式与标签页模式共用同一套 AudioPipeline，所以这里只是按音源转发一下。
        """
        if self.capture is not None:
            return self.capture.pipeline_stats()
        if self.tab_server is not None:
            return self.tab_server.pipeline_stats()
        return None

    # ------------------------------------------------------------------ #
    # 采集线程
    # ------------------------------------------------------------------ #
    def _on_audio(self, samples) -> None:
        """在采集线程里被调用——**这里不能碰 Qt 控件**，只能 emit 信号。"""
        if self.router is None or self._stop.is_set():
            return
        try:
            events = self.router.feed(samples)
        except Exception as exc:  # noqa: BLE001 - 识别异常不能中断采集
            log.error("识别失败: %s", exc)
            self.errorOccurred.emit(f"识别失败: {exc}")
            return
        for ev in events:
            self._dispatch(ev)
        # 语言落定后报一次状态
        if self.router.is_decided and not getattr(self, "_lang_reported", False):
            self._lang_reported = True
            self.statusChanged.emit(
                f"识别语种={self.router.language} · 引擎={self.router.engine_name}"
            )

    def _dispatch(self, ev: ASREvent) -> None:
        """把识别事件转成信号发给 UI 线程。"""
        if ev.type is ASREventType.ERROR:
            self.errorOccurred.emit(ev.error or "识别出错")
            return

        if ev.type is ASREventType.PARTIAL:
            self.partialReceived.emit(ev.text, ev.language)
            return

        if ev.type is ASREventType.FINAL and ev.text:
            line_id = self.state.new_id()
            if ev.latency_ms:
                with self._lock:
                    self.asr_latency_ms.append(ev.latency_ms)
            # 逐条记录：字幕程序出问题时，用户/开发者第一件事就是翻日志，
            # 没有这条记录就完全不知道它到底有没有在工作
            log.info("[字幕 %d] (%s) %s", line_id, ev.language or "?", ev.text)
            self.finalReceived.emit(line_id, ev.text, ev.language)
            # 送翻译
            try:
                self._queue.put_nowait((line_id, ev.text, ev.language))
            except queue.Full:
                log.warning("翻译队列已满，丢弃字幕 %s", line_id)

    # ------------------------------------------------------------------ #
    # 翻译线程
    # ------------------------------------------------------------------ #
    def _translate_loop(self) -> None:
        batch_size = max(1, min(8, self.config.translate.max_concurrency))
        wait_s = max(0.05, self.config.translate.stale_drop_s and 0.35)

        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.time() + wait_s
            while len(batch) < batch_size:
                remain = deadline - time.time()
                if remain <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remain))
                except queue.Empty:
                    break

            if self.hub is None:
                continue

            segs = [Segment(id=i, text=t, language=lang) for i, t, lang in batch]
            src_lang = segs[0].language or self.config.translate.source_language
            t0 = time.time()
            try:
                result = self.hub.translate(segs, src_lang)
            except Exception as exc:  # noqa: BLE001
                log.error("翻译失败: %s", exc)
                for i, _, _ in batch:
                    self.translationReceived.emit(i, "", f"翻译异常: {exc}")
                continue
            dt_ms = (time.time() - t0) * 1000
            with self._lock:
                self.translate_latency_ms.append(dt_ms)

            for i, _, _ in batch:
                if i in result.translations:
                    log.info("[译文 %d] %s", i, result.translations[i])
                    self.translationReceived.emit(i, result.translations[i], "")
                else:
                    reason = result.failures.get(i, "翻译失败")
                    log.warning("[译文 %d] 失败：%s", i, reason)
                    self.translationReceived.emit(i, "", reason)
            self._emit_stats()

    # ------------------------------------------------------------------ #
    def _emit_stats(self) -> None:
        with self._lock:
            asr_lat = list(self.asr_latency_ms)
            tr_lat = list(self.translate_latency_ms)
        payload = {
            "elapsed_s": time.time() - self.started_at if self.started_at else 0.0,
            "language": self.router.language if self.router else "",
            "engine": self.router.engine_name if self.router else "",
            "asr_p50_ms": _pct(asr_lat, 50),
            "asr_p90_ms": _pct(asr_lat, 90),
            "translate_p50_ms": _pct(tr_lat, 50),
            "translate_p90_ms": _pct(tr_lat, 90),
            "hub": self.hub.stats.as_dict() if self.hub else {},
            "cache": self.hub.cache_stats() if self.hub else {},
        }
        self.statsChanged.emit(payload)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))], 1)
