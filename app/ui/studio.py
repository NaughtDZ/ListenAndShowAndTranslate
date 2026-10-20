"""「字幕工坊」：拖入音视频 → 识别 → 导出 srt/vtt/lrc（可选 LLM 翻译）。

为什么独立成一个窗口而不是塞进设置窗：
这是**离线批处理**，和实时字幕（悬浮窗 + 控制窗）是两种完全不同的工作流——
参数（每行字数、输出格式、批量队列）只对文件有意义，
混进设置窗会让"实时用哪个模型"和"转写用哪个模型"互相干扰。

线程模型：识别在 ``QThread`` 里跑（模型也在那个线程里创建），UI 只收信号。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.models.registry import LANGUAGE_LABELS, MODELS
from app.offline.decode import find_ffmpeg
from app.offline.export import FORMATS, MODE_LABELS
from app.offline.job import JobOptions, JobResult, SubtitleJob
from app.offline.transcribe import TranscribeOptions
from app.utils.log import get_logger

log = get_logger(__name__)

AUDIO_EXTS = (".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".caf")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wmv")

TARGET_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("zh", "简体中文"),
    ("zh-tw", "繁體中文"),
    ("en", "English"),
    ("ja", "日本語"),
    ("ko", "한국어"),
)


# --------------------------------------------------------------------------- #
# 后台任务
# --------------------------------------------------------------------------- #
class _Worker(QThread):
    progressed = Signal(float, str)
    logged = Signal(str)
    done = Signal(object)

    def __init__(self, options: JobOptions, paths: list[Path], config, parent=None) -> None:
        super().__init__(parent)
        self.options = options
        self.paths = paths
        self.config = config
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:  # noqa: D401 - QThread 约定
        job = SubtitleJob(
            self.options,
            self.config,
            on_progress=lambda fraction, message: self.progressed.emit(fraction, message),
            on_log=lambda message: self.logged.emit(message),
            cancel=self._cancel,
        )
        try:
            result = job.run(self.paths)
        except Exception as exc:  # noqa: BLE001 - 后台线程里绝不能让异常吞掉界面
            self.logged.emit(f"✗ 任务异常：{type(exc).__name__}: {exc}")
            log.exception("字幕工坊任务异常")
            result = JobResult()
        self.done.emit(result)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class StudioWindow(QMainWindow):
    def __init__(self, config: AppConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self.config = config or AppConfig.load()
        self.worker: _Worker | None = None
        self._paths: set[str] = set()
        self._last_dir = ""
        self.setWindowTitle("听·显·译 · 字幕工坊（离线转写）")
        self.resize(1000, 800)
        self.setAcceptDrops(True)
        self._build_ui()

    # ------------------------------------------------------------------ #
    # 界面
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        intro = QLabel(
            "<b>把音频或视频拖进来，识别成字幕文件。</b><br>"
            "支持文件夹（自动挑出里面的音频/视频）。离线处理，不占用实时字幕的引擎。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ---- 1. 文件队列 ----
        file_box = QGroupBox("1. 要处理的文件")
        file_layout = QVBoxLayout(file_box)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMinimumHeight(140)
        file_layout.addWidget(self.file_list)

        row = QHBoxLayout()
        add_files = QPushButton("添加文件…")
        add_files.clicked.connect(self._pick_files)
        add_dir = QPushButton("添加文件夹…")
        add_dir.clicked.connect(self._pick_folder)
        remove = QPushButton("移除选中")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("清空")
        clear.clicked.connect(self._clear_files)
        for button in (add_files, add_dir, remove, clear):
            row.addWidget(button)
        row.addStretch(1)
        file_layout.addLayout(row)
        layout.addWidget(file_box)

        # ---- 2. 参数 ----
        param_box = QGroupBox("2. 参数")
        form = QFormLayout(param_box)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip("只列**离线分块**模型：它们不吃流式，整句识别最准。")
        self._fill_models()
        form.addRow("识别模型", self.model_combo)

        self.lang_combo = QComboBox()
        self.lang_combo.addItem("自动判断（较慢，但不用选对）", "auto")
        for code, label in sorted(LANGUAGE_LABELS.items(), key=lambda kv: kv[1]):
            if code == "*":
                continue
            self.lang_combo.addItem(f"{label}（{code}）", code)
        saved = getattr(self.config.asr, "language", "auto")
        index = self.lang_combo.findData(saved)
        self.lang_combo.setCurrentIndex(index if index >= 0 else 0)
        form.addRow("音频语言", self.lang_combo)

        self.vad_combo = QComboBox()
        self.vad_combo.addItem("Silero（默认，快）", "silero-vad")
        self.vad_combo.addItem("Whisper-VAD ASMR（轻声/耳语）", "whisper-vad-asmr")
        if getattr(self.config.asr.vad, "model", "") == "whisper-vad-asmr":
            self.vad_combo.setCurrentIndex(1)
        form.addRow("VAD 模型", self.vad_combo)

        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(200, 3000)
        self.silence_spin.setSingleStep(50)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(500)
        self.silence_spin.setToolTip("静多久算一句说完。调大 → 句子更完整、翻译上下文更好；调小 → 切得更碎。")
        form.addRow("断句静音", self.silence_spin)

        self.speech_spin = QSpinBox()
        self.speech_spin.setRange(50, 2000)
        self.speech_spin.setSingleStep(50)
        self.speech_spin.setSuffix(" ms")
        self.speech_spin.setValue(250)
        form.addRow("最短语音", self.speech_spin)

        self.maxseg_spin = QSpinBox()
        self.maxseg_spin.setRange(2000, 30000)
        self.maxseg_spin.setSingleStep(1000)
        self.maxseg_spin.setSuffix(" ms")
        self.maxseg_spin.setValue(8000)
        self.maxseg_spin.setToolTip(
            "一直不停地说时，最长憋到多久强制断一句。\n"
            "⚠️ 不要为了“句子完整”调得很大：超过最长连续语音时 VAD 会一直不出结果。"
        )
        form.addRow("最长成句", self.maxseg_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 120)
        self.width_spin.setSingleStep(2)
        self.width_spin.setSpecialValueText("不切（整句一条）")
        self.width_spin.setValue(24)
        self.width_spin.setToolTip(
            "一行最多多少个全角字。LRC 是歌词格式，一行 = 一屏，必须切；\n"
            "SRT/VTT 也建议切，否则一行太长不好读。0 = 不切。"
        )
        form.addRow("每行字数", self.width_spin)

        self.threads_spin = QSpinBox()
        default_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(default_threads)
        form.addRow("CPU 线程", self.threads_spin)

        layout.addWidget(param_box)

        # ---- 3. 输出 ----
        out_box = QGroupBox("3. 输出")
        out_form = QFormLayout(out_box)

        fmt_row = QHBoxLayout()
        self.fmt_checks: dict[str, QCheckBox] = {}
        for fmt in FORMATS:
            check = QCheckBox(fmt.upper())
            check.setChecked(fmt == "srt")
            self.fmt_checks[fmt] = check
            fmt_row.addWidget(check)
        fmt_row.addStretch(1)
        out_form.addRow("字幕格式", self._wrap(fmt_row))

        self.mode_combo = QComboBox()
        for mode, label in MODE_LABELS.items():
            self.mode_combo.addItem(label, mode)
        out_form.addRow("输出内容", self.mode_combo)

        self.translate_check = QCheckBox("调用 LLM 翻译字幕（用设置里已配置的翻译通道）")
        self.translate_check.setToolTip(
            "识别完再用「设置 → 翻译」里配置的通道逐句翻译。\n"
            "翻译会按批送、走缓存，失败的行会保留原文并在日志里报告。"
        )
        out_form.addRow("", self.translate_check)

        self.target_combo = QComboBox()
        for code, label in TARGET_LANGUAGES:
            self.target_combo.addItem(f"{label}（{code}）", code)
        lang = getattr(self.config.translate, "target_language", "zh") or "zh"
        index = self.target_combo.findData(lang)
        self.target_combo.setCurrentIndex(index if index >= 0 else 0)
        out_form.addRow("翻译成", self.target_combo)

        dir_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("留空 = 与源文件放在一起")
        browse = QPushButton("选择…")
        browse.clicked.connect(self._pick_out_dir)
        dir_row.addWidget(self.out_edit, 1)
        dir_row.addWidget(browse)
        out_form.addRow("输出目录", self._wrap(dir_row))

        self.overwrite_check = QCheckBox("同名文件直接覆盖（不勾则自动加 _1、_2）")
        self.overwrite_check.setChecked(True)
        out_form.addRow("", self.overwrite_check)

        layout.addWidget(out_box)

        # ---- 4. 执行 ----
        action_row = QHBoxLayout()
        self.start_btn = QPushButton("开始转写")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        action_row.addWidget(self.start_btn, 2)
        action_row.addWidget(self.cancel_btn, 1)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%  %v")
        layout.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        layout.addWidget(self.log_view, 1)

        if not find_ffmpeg():
            self._append_log(
                "⚠️ 没检测到 ffmpeg：只能处理 wav/flac/ogg/mp3 等，"
                "m4a/aac/mp4/mkv 会失败。装 ffmpeg 后重开本窗口即可。"
            )

    @staticmethod
    def _wrap(layout) -> QWidget:
        holder = QWidget()
        holder.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return holder

    def _fill_models(self) -> None:
        from app.models.downloader import ModelDownloader

        try:
            downloader = ModelDownloader()
        except Exception:  # noqa: BLE001
            downloader = None

        preferred = "sensevoice-int8"
        for spec in MODELS.values():
            if spec.kind != "asr_offline":
                continue
            installed = False
            if downloader is not None:
                try:
                    installed = downloader.status(spec.id) == "installed"
                except Exception:  # noqa: BLE001
                    installed = False
            self.model_combo.addItem(
                spec.display_name + ("" if installed else "（未下载）"), spec.id
            )
            self.model_combo.setItemData(
                self.model_combo.count() - 1,
                f"{spec.note}\n体积约 {spec.total_mb:.0f} MB",
                Qt.ToolTipRole,
            )
        index = self.model_combo.findData(preferred)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    # ------------------------------------------------------------------ #
    # 文件队列
    # ------------------------------------------------------------------ #
    def _media_exts(self) -> tuple[str, ...]:
        return AUDIO_EXTS + VIDEO_EXTS if find_ffmpeg() else AUDIO_EXTS

    def _add_file(self, path: Path) -> int:
        if not path.is_file():
            return 0
        if path.suffix.lower() not in self._media_exts():
            self._append_log(f"跳过（不是音频/视频）：{path.name}")
            return 0
        key = str(path.resolve())
        if key in self._paths:
            return 0
        self._paths.add(key)
        item = QListWidgetItem(f"{path.name}    —    {path.parent}")
        item.setData(Qt.UserRole, key)
        self.file_list.addItem(item)
        return 1

    def _add_path(self, path: Path) -> int:
        if path.is_dir():
            added = 0
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in self._media_exts():
                    added += self._add_file(child)
            return added
        return self._add_file(path)

    def _collect_paths(self) -> list[Path]:
        paths: list[Path] = []
        for row in range(self.file_list.count()):
            key = self.file_list.item(row).data(Qt.UserRole)
            if key:
                paths.append(Path(key))
        return paths

    def _pick_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in self._media_exts())
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音频或视频", self._last_dir, f"媒体文件 ({patterns})"
        )
        added = sum(self._add_file(Path(f)) for f in files)
        if files:
            self._last_dir = str(Path(files[0]).parent)
        if added:
            self._append_log(f"加入 {added} 个文件")

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", self._last_dir)
        if folder:
            self._last_dir = folder
            added = self._add_path(Path(folder))
            self._append_log(f"从文件夹加入 {added} 个文件")

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self._paths.discard(item.data(Qt.UserRole))
            self.file_list.takeItem(self.file_list.row(item))

    def _clear_files(self) -> None:
        self.file_list.clear()
        self._paths.clear()

    def _pick_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self._last_dir)
        if folder:
            self.out_edit.setText(folder)

    # ------------------------------------------------------------------ #
    # 拖放
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 约定
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        added = 0
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            added += self._add_path(Path(local))
        if added:
            self._append_log(f"加入 {added} 个文件")
        else:
            self._append_log("拖进来的东西里没有可处理的音频/视频")
        event.acceptProposedAction()

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def _collect_options(self) -> JobOptions:
        formats = tuple(fmt for fmt in FORMATS if self.fmt_checks[fmt].isChecked()) or ("srt",)
        return JobOptions(
            transcribe=TranscribeOptions(
                model_id=self.model_combo.currentData(),
                language=self.lang_combo.currentData() or "auto",
                vad_model=self.vad_combo.currentData(),
                min_silence_ms=self.silence_spin.value(),
                min_speech_ms=self.speech_spin.value(),
                max_segment_ms=self.maxseg_spin.value(),
                num_threads=self.threads_spin.value(),
            ),
            formats=formats,
            mode=self.mode_combo.currentData() or "source",
            max_line_width=float(self.width_spin.value()),
            out_dir=self.out_edit.text().strip(),
            translate=self.translate_check.isChecked(),
            target_language=self.target_combo.currentData() or "zh",
            overwrite=self.overwrite_check.isChecked(),
        )

    def _start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        paths = self._collect_paths()
        if not paths:
            QMessageBox.information(self, "还没有文件", "先把音频或视频拖进上面那个框。")
            return

        options = self._collect_options()
        if options.mode in ("bilingual", "target") and not options.translate:
            answer = QMessageBox.question(
                self, "没有译文",
                "你选了「双语」或「只要译文」，但没勾「调用 LLM 翻译」。\n"
                "这样导出会只有原文。要继续吗？",
            )
            if answer != QMessageBox.Yes:
                return

        self.log_view.clear()
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = _Worker(options, paths, self.config, self)
        self.worker.progressed.connect(self._on_progress)
        self.worker.logged.connect(self._append_log)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.cancel_btn.setEnabled(False)
            self._append_log("正在取消（当前这一步做完就停）…")
            self.worker.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.progress.setFormat(f"%p%  {message}")

    def _on_done(self, result: JobResult) -> None:
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        files = result.files
        ok = sum(1 for item in files if item.ok)
        translated = sum(item.translated for item in files)
        failed = sum(item.failed for item in files)
        cues = sum(item.cues for item in files)
        self._append_log(
            f"—— 完成：{ok}/{len(files)} 个文件 · {cues} 条字幕 · "
            f"{result.elapsed_s:.1f}s"
            + (f" · 翻译 {translated} 条" if translated else "")
            + (f"（{failed} 条失败）" if failed else "")
        )
        for error in result.errors:
            self._append_log(f"✗ {error}")
        if result.outputs:
            self.progress.setValue(1000)
        self.worker = None

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(event)


def run_studio(config: AppConfig | None = None) -> int:
    """入口：打开「字幕工坊」窗口（阻塞到关窗）。"""
    app = QApplication.instance() or QApplication([])
    window = StudioWindow(config or AppConfig.load())
    window.show()
    return int(app.exec())
