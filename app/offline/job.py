"""一次完整的离线转写任务：解码 → 分段识别 →（可选翻译）→ 切行 → 导出。

多个文件共用一套已加载的模型（模型加载几百毫秒到几秒，批量跑时不该重复付）。

进度口径：每个文件占 ``1/N``，文件内部分三段——转写 70%、翻译 20%、导出 10%。
这是给进度条用的估算，不追求精确到字节。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.offline import decode
from app.offline.cue import Cue, split_all
from app.offline.export import FORMATS, write_subtitles
from app.offline.transcribe import TranscribeOptions, Transcriber
from app.utils.log import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[float, str], None]
"""``(0..1 总进度, 当前在干什么)``。"""

LogCb = Callable[[str], None]

TRANSCRIBE_SHARE = 0.70
TRANSLATE_SHARE = 0.20
EXPORT_SHARE = 0.10

MAX_TRANSLATE_BATCH = 400
"""单批最多几条字幕（与 ``openai_compat.MAX_BATCH_ITEMS`` 对齐）。

预算再大也不放开了：400 条的输出已经要 8000 token 量级，单次生成时间很长，
中途失败整批白干；而且模型漏条的概率随条数线性上升。
"""


@dataclass
class JobOptions:
    """离线字幕任务的参数（GUI 面板上的那一堆就是它）。"""

    transcribe: TranscribeOptions = field(default_factory=TranscribeOptions)

    formats: tuple[str, ...] = ("srt",)
    """要导出哪些格式（``srt`` / ``vtt`` / ``lrc``）。"""

    mode: str = "source"
    """输出内容：``source`` 原文 / ``bilingual`` 双语 / ``target`` 只要译文。"""

    max_line_width: float = 24.0
    """一行最多多少"全角字宽"；0 表示不切行（整句一条）。"""

    out_dir: str = ""
    """输出目录，空 = 与源文件同目录。"""

    translate: bool = False
    target_language: str = "zh"
    translate_batch: int = 8
    """一批送多少条去翻译（批量能显著省 token，见 docs/P4-翻译实测.md）。"""

    overwrite: bool = True


@dataclass
class FileResult:
    source: Path
    outputs: list[Path] = field(default_factory=list)
    cues: int = 0
    translated: int = 0
    failed: int = 0
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.outputs)


@dataclass
class JobResult:
    files: list[FileResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def outputs(self) -> list[Path]:
        return [path for item in self.files for path in item.outputs]

    @property
    def errors(self) -> list[str]:
        return [f"{item.source.name}: {item.error}" for item in self.files if item.error]


class SubtitleJob:
    """跑一批文件。UI 可以在后台线程里调 ``run``，通过 ``cancel`` 中止。"""

    def __init__(
        self,
        options: JobOptions | None = None,
        config=None,
        on_progress: ProgressCb | None = None,
        on_log: LogCb | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        self.options = options or JobOptions()
        self.config = config
        self.on_progress = on_progress
        self.on_log = on_log
        self.cancel = cancel or threading.Event()
        self._progress_value = 0.0

    # ------------------------------------------------------------------ #
    def run(self, paths: list[Path | str]) -> JobResult:
        started = time.perf_counter()
        files = [Path(p) for p in paths if str(p).strip()]
        total = len(files)
        result = JobResult()
        if total == 0:
            return result

        options = self.options
        transcriber = Transcriber(options.transcribe, on_log=self._log)
        transcriber.load()

        hub = None
        if options.translate:
            hub = self._build_hub()

        try:
            for index, path in enumerate(files):
                if self.cancel.is_set():
                    self._log("已取消")
                    break
                result.files.append(self._one_file(path, index, total, transcriber, hub))
        finally:
            transcriber.close()

        result.elapsed_s = time.perf_counter() - started
        self._progress(1.0, f"完成（{len(result.outputs)} 个文件）")
        return result

    # ------------------------------------------------------------------ #
    def _one_file(self, path: Path, index: int, total: int, transcriber: Transcriber, hub) -> FileResult:
        started = time.perf_counter()
        item = FileResult(source=path)
        base = index / total
        span = 1.0 / total

        try:
            info = decode.probe(path)
            self._log(
                f"[{index + 1}/{total}] {path.name} · {info.duration_label} · {info.backend or '未知解码器'}"
            )

            def on_audio(done: float, total_s: float) -> None:
                if total_s > 0:
                    fraction = min(1.0, done / total_s)
                else:
                    fraction = 0.0
                self._progress(
                    base + span * TRANSCRIBE_SHARE * fraction,
                    f"识别 {path.name}  {done:.0f}/{total_s:.0f}s" if total_s else f"识别 {path.name}",
                )

            cues = transcriber.run(
                decode.iter_pcm16k(path),
                total_seconds=info.duration_s,
                on_progress=on_audio,
            )
            item.cues = len(cues)

            if self.cancel.is_set():
                item.error = "已取消"
                return item

            if self.options.translate and cues:
                item.translated, item.failed = self._translate(
                    cues, hub, base + span * TRANSCRIBE_SHARE, span * TRANSLATE_SHARE
                )

            if self.options.max_line_width and self.options.max_line_width > 0:
                cues = split_all(cues, self.options.max_line_width)
                item.cues = len(cues)

            self._progress(base + span * (TRANSCRIBE_SHARE + TRANSLATE_SHARE), f"写出 {path.name}")
            item.outputs = self._export(path, cues)
            for output in item.outputs:
                self._log(f"  → {output}")
        except decode.DecodeError as exc:
            item.error = str(exc)
            self._log(f"  ✗ {exc}")
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不该中断整批
            item.error = f"{type(exc).__name__}: {exc}"
            log.exception("处理 %s 失败", path)

        item.elapsed_s = time.perf_counter() - started
        self._progress(base + span, f"完成 {path.name}")
        return item

    # ------------------------------------------------------------------ #
    def _build_hub(self):
        from app.config import AppConfig
        from app.translate.setup import build_translator_hub

        config = self.config
        if config is None:
            config = AppConfig.load()
            self.config = config
        hub, notes = build_translator_hub(config)
        for note in notes:
            self._log(f"翻译通道：{note}")
        return hub

    def _translate(self, cues: list[Cue], hub, base: float, span: float) -> tuple[int, int]:
        """按 token 预算分批翻译。

        以前是固定 8 条一批，和模型的真实窗口毫无关系。现在从
        ``config.translate.llm.input_budget_tokens`` 读预算（4096 / 8192 /
        16384 / 32768），按预算算出一批送多少条——8G 小显存约 100 条，
        5090 上 32k 预算能到 :data:`~app.translate.openai_compat.MAX_BATCH_ITEMS`。

        预算为 0（"不限"档）时退回旧的固定 ``translate_batch`` 条。
        """
        from app.translate.base import Segment
        from app.translate.budget import estimate_overhead_tokens, plan_batches

        if hub is None:
            return 0, len(cues)

        source_language = self.options.transcribe.language or "auto"
        target = self.options.target_language
        ok = failed = 0
        total = len(cues)

        segments = [
            Segment(id=i, text=cue.text, language=source_language)
            for i, cue in enumerate(cues)
        ]

        budget = 0
        glossary: dict[str, str] = {}
        cfg = self.config
        if cfg is not None:
            budget = int(getattr(cfg.translate.llm, "input_budget_tokens", 0) or 0)
            raw_glossary = cfg.translate.glossary if isinstance(cfg.translate.glossary, dict) else {}
            # 下划线开头的是元数据（_path 之类），不参与估值
            glossary = {k: v for k, v in raw_glossary.items() if not k.startswith("_")}

        if budget > 0:
            batches = plan_batches(
                segments, budget,
                overhead_tokens=estimate_overhead_tokens(glossary=glossary),
                max_items=MAX_TRANSLATE_BATCH,
            )
        else:
            size = max(1, int(self.options.translate_batch))
            batches = [segments[i:i + size] for i in range(0, total, size)]

        done = 0
        for batch in batches:
            if self.cancel.is_set():
                break
            try:
                result = hub.translate(
                    batch, source_language=source_language, target_language=target
                )
            except Exception as exc:  # noqa: BLE001 - hub 号称不抛，但别赌
                log.exception("翻译失败")
                result = None
                self._log(f"  ! 翻译出错：{exc}")

            for seg in batch:
                cue = cues[seg.id]
                text = result.translations.get(seg.id, "") if result else ""
                if text:
                    cue.translation = text
                    ok += 1
                else:
                    failed += 1
                    if result and seg.id in result.failures:
                        log.debug("翻译失败原因：%s", result.failures[seg.id])

            done = min(total, done + len(batch))
            self._progress(
                base + span * (done / total),
                f"翻译 {done}/{total}" + (f"（失败 {failed}）" if failed else ""),
            )

        if failed:
            self._log(f"  ! {failed} 条没拿到译文（其余照常写出原文）")
        return ok, failed

    # ------------------------------------------------------------------ #
    def _export(self, source: Path, cues: list[Cue]) -> list[Path]:
        options = self.options
        out_dir = Path(options.out_dir) if options.out_dir else source.parent
        suffix = ""
        if options.mode in ("bilingual", "target"):
            suffix = "." + (options.target_language or "out")

        written: list[Path] = []
        for fmt in options.formats:
            if fmt not in FORMATS:
                continue
            target = out_dir / f"{source.stem}{suffix}.{fmt}"
            if target.exists() and not options.overwrite:
                index = 1
                while True:
                    candidate = out_dir / f"{source.stem}{suffix}_{index}.{fmt}"
                    if not candidate.exists():
                        target = candidate
                        break
                    index += 1
            written.append(write_subtitles(target, cues, fmt, options.mode))
        return written

    # ------------------------------------------------------------------ #
    def _progress(self, fraction: float, message: str) -> None:
        # 进度条不允许倒退（多阶段估算会有小抖动）
        self._progress_value = max(self._progress_value, min(1.0, max(0.0, fraction)))
        if self.on_progress:
            try:
                self.on_progress(self._progress_value, message)
            except Exception:  # noqa: BLE001 - UI 回调不能影响任务
                pass

    def _log(self, message: str) -> None:
        log.info("%s", message)
        if self.on_log:
            try:
                self.on_log(message)
            except Exception:  # noqa: BLE001
                pass
