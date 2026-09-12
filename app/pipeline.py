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

        resolved = resolve_target(target)
        if resolved is None:
            self.errorOccurred.emit(f"未找到活跃音频会话：{target.describe()}")
            return False

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

        self.statusChanged.emit(
            f"已开始 · {resolved.name} (PID {resolved.pid}) · 语言={self.config.asr.language}"
        )
        return True

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
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
