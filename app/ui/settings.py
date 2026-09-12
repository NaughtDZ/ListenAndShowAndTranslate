"""设置界面。

四个分页，都是"用户在真实使用中一定会要调"的东西：

  网络  —— 程序走哪个代理（模型下载、翻译通道共用），以及一键连通性测试
  翻译  —— 用哪个通道、凭据、上下文行数、提示词模板、术语表，以及一键测试
  识别  —— 语言、静音阈值（带**实时电平表 + 阈值线 + 实测参考线**）、
          延迟档位与五个延迟参数（每个都写清"调小/调大各会怎样"）
  外观  —— 原文/译文/双语、滚动方式、每屏行数、字号颜色、点击穿透

两个设计约束（都是踩过坑之后定的）：

1. **控制项不能放在悬浮窗里** —— 一旦开启点击穿透，悬浮窗就点不动了，
   用户会把自己锁死。所以所有设置都在这个普通窗口里。
2. **每个旋钮都要写清作用** —— 只写"延迟相关"等于没写。
   延迟参数的说明文字直接来自 ``app/config.py: LATENCY_KNOBS``，
   与 docs/延迟调节.md 同源，避免两处维护。
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.audio.levels import LevelState, LevelTracker, threshold_from_db
from app.config import LATENCY_KNOBS, LATENCY_PRESETS, AppConfig
from app.translate.prompts import list_templates
from app.translate.traditional.providers import ALL_PROVIDERS, WEB_PROVIDERS
from app.ui.meter import LevelMeterWidget
from app.utils.log import get_logger

log = get_logger(__name__)

LEVEL_POLL_MS = 100
"""设置窗里实时电平的刷新间隔（和独立电平表窗口一个量级）。"""

MEASURED_SPEECH_DB = -66.0
"""实测：把目标程序音量调小之后，真实语音的 RMS 大约在这个水平。

（见 docs/P1-实测记录.md：所以静音阈值不能取太高，默认 -80 是安全的。）
"""

LevelSource = Callable[[], "tuple[float, float] | None"]
"""实时电平来源：返回 ``(rms, peak)``（线性幅度），没有数据就返回 None。"""

# 通道 id → 展示名与需要的凭据字段
PROVIDER_META: dict[str, tuple[str, dict[str, str]]] = {
    "none": ("不翻译（只显示原文）", {}),
    "llm": ("本地/在线大模型（OpenAI 兼容）", {}),
    "web_google": ("谷歌网页版（免 key·需代理）", {}),
    "web_bing": ("必应网页版（免 key·需代理）", {}),
    "baidu": ("百度翻译 API", {"app_id": "APP ID", "secret_key": "密钥"}),
    "youdao": ("有道智云 API", {"app_key": "应用 ID", "app_secret": "应用密钥"}),
    "azure": ("微软 Azure 翻译", {"api_key": "密钥", "region": "区域（如 eastasia）"}),
    "google": ("谷歌云翻译 API", {"api_key": "API Key"}),
    "deepl": ("DeepL", {"api_key": "API Key"}),
}


class _TestThread(QThread):
    """在后台跑连通性测试，避免界面卡死（网络请求可能几秒）。"""

    done = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # noqa: D102
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # noqa: BLE001
            self.done.emit(f"✗ 测试异常: {type(exc).__name__}: {exc}")


class SettingsWindow(QWidget):
    """设置窗口。改动即时写回 config 对象，由调用方负责保存。"""

    saved = Signal()
    """保存成功后发出；控制窗据此**立即应用**，而不是等重启。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        level_source: LevelSource | None = None,
    ) -> None:
        super().__init__(None)
        self.config = config or AppConfig.load()
        self.setWindowTitle("听·显·译 — 设置")
        self.resize(720, 640)
        self.setMinimumWidth(560)

        self._threads: list[_TestThread] = []

        # 实时电平：来源由调用方给（字幕进程给采集线程的读数；启动窗口没有采集，
        # 就不给来源，电平表只显示阈值线和参考线）。
        self._level_source = level_source
        self._level_tracker = LevelTracker(
            silence_threshold=threshold_from_db(
                float(self.config.audio.silence_rms_threshold_db)
            )
        )
        self._level_status = ""
        self._level_idle = False
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(LEVEL_POLL_MS)
        self._level_timer.timeout.connect(self._poll_level)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_network_tab(), "网络")
        tabs.addTab(self._build_translate_tab(), "翻译")
        tabs.addTab(self._build_asr_tab(), "识别")
        tabs.addTab(self._build_appearance_tab(), "外观")

        self.status = QLabel("")
        self.status.setWordWrap(True)
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(save_btn)
        bottom.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.addWidget(tabs, 1)
        root.addLayout(bottom)

    # ------------------------------------------------------------------ #
    # 每次打开都同步一次"别处也能改"的字段
    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().showEvent(event)
        self._sync_live_fields()
        # 只在窗口可见时轮询电平：设置窗常年开着不该白烧 CPU
        if self._level_source is not None:
            self._level_timer.start()
            self._poll_level()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._level_timer.stop()
        super().hideEvent(event)

    def _on_silence_db_changed(self, value: float) -> None:
        """阈值数值一变，电平表上的橙色虚线和"算不算静音"立刻跟着变。"""
        linear = threshold_from_db(float(value))
        self.threshold_meter.set_threshold(linear)
        self._level_tracker.set_threshold(linear)

    def _poll_level(self) -> None:
        """把实时读数喂给电平表（采集线程算好的 RMS/峰值，读两个 float）。"""
        reading = None
        if self._level_source is not None:
            try:
                reading = self._level_source()
            except Exception as exc:  # noqa: BLE001 - 读数失败不该影响设置界面
                log.debug("读取实时电平失败: %s", exc)
                reading = None

        if reading is None:
            # 清一次就够：没有数据时不必 10Hz 重画
            if not self._level_idle:
                self._level_idle = True
                self.threshold_meter.set_level(LevelState())
            self._set_level_status(
                "未采集（字幕没在运行时没有实时数据；启动窗口的「电平表」可单独看）"
            )
            return

        self._level_idle = False
        rms, peak = reading
        state = self._level_tracker.update_values(
            rms, peak, dt=self._level_timer.interval() / 1000.0
        )
        self.threshold_meter.set_level(state)
        if state.clipping:
            self._set_level_status("⚠ 削波（音量太大，识别可能出错）")
        elif state.is_silent:
            self._set_level_status("静音（低于阈值，不会送去识别）")
        else:
            self._set_level_status("有声")

    def _set_level_status(self, text: str) -> None:
        """只在文案变化时重画，避免 10Hz 无意义刷新。"""
        if text == self._level_status:
            return
        self._level_status = text
        self.threshold_meter.set_texts("当前采集目标 · 实时电平", text)

    def _sync_live_fields(self) -> None:
        """从配置里回读那些**在别处也会被改**的值。

        字幕窗那边拖边框会改宽度/高度/字号，控制窗上有透明度滑杆和
        原文/译文/双语、累积/单行的按钮。以前这个设置窗只在构造时读一次值，
        用户拖完窗口再进来点一次「保存设置」，就会把拖出来的尺寸**覆盖回旧值**
        ——表现出来就是"设置保存不了 / 改完又变回去"（用户反馈过同类问题）。

        这里用 blockSignals 改控件，避免触发各种联动回调。
        """
        ov = self.config.overlay
        for widget, value in (
            (self.win_width, ov.window_width),
            (self.win_height, ov.window_height),
            (self.font_size, ov.font_size),
        ):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

        self.win_opacity.blockSignals(True)
        self.win_opacity.setValue(float(ov.window_opacity))
        self.win_opacity.blockSignals(False)

        for combo, value in (
            (self.mode_combo, ov.display_mode),
            (self.scroll_combo, ov.scroll_mode),
        ):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)

        # 静音阈值：独立电平表窗口里也有一根滑杆改的是同一个配置项，
        # 所以这里要回读，并且把电平表上的阈值线一起挪过去
        self.silence_db.blockSignals(True)
        self.silence_db.setValue(float(self.config.audio.silence_rms_threshold_db))
        self.silence_db.blockSignals(False)
        self._on_silence_db_changed(self.silence_db.value())

    # ------------------------------------------------------------------ #
    # 网络
    # ------------------------------------------------------------------ #
    def _build_network_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("网络代理")
        form = QFormLayout(box)

        self.proxy_edit = QLineEdit(self.config.proxy)
        self.proxy_edit.setPlaceholderText("如 http://127.0.0.1:2333，留空表示直连")
        form.addRow("代理地址", self.proxy_edit)

        hint = QLabel(
            "这个代理由<b>模型下载</b>、<b>翻译通道</b>共用。<br>"
            "· <b>本地地址</b>（127.0.0.1 / localhost，例如 LM Studio）自动不走代理<br>"
            "· 谷歌 / 必应网页版<b>必须走代理</b>（国内直连不通）<br>"
            "· 内置的 5 家官方翻译 API 大多可直连，但走代理也没问题"
        )
        hint.setWordWrap(True)
        form.addRow("", hint)

        row = QHBoxLayout()
        test_proxy = QPushButton("测试代理连通性")
        test_proxy.clicked.connect(lambda: self._run_test(self._test_proxy))
        test_all = QPushButton("测试全部通道")
        test_all.clicked.connect(lambda: self._run_test(self._test_all))
        row.addWidget(test_proxy)
        row.addWidget(test_all)
        row.addStretch(1)
        form.addRow("", row)

        outer.addWidget(box)

        self.net_log = QTextEdit()
        self.net_log.setReadOnly(True)
        self.net_log.setMinimumHeight(220)
        outer.addWidget(QLabel("测试结果"))
        outer.addWidget(self.net_log, 1)
        return page

    # ------------------------------------------------------------------ #
    # 翻译
    # ------------------------------------------------------------------ #
    def _build_translate_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        lay = QVBoxLayout(inner)

        # --- 通道 ---
        box = QGroupBox("翻译通道")
        form = QFormLayout(box)

        self.enable_translate = QCheckBox("启用翻译")
        self.enable_translate.setChecked(self.config.translate.enabled)
        form.addRow("", self.enable_translate)

        self.provider_combo = QComboBox()
        for pid, (label, _) in PROVIDER_META.items():
            tag = "  [非官方接口]" if pid in WEB_PROVIDERS else ""
            self.provider_combo.addItem(f"{label}{tag}", pid)
        idx = self.provider_combo.findData(self.config.translate.provider)
        self.provider_combo.setCurrentIndex(max(0, idx))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("通道", self.provider_combo)

        self.cred_fields: dict[str, QLineEdit] = {}
        self.cred_box = QGroupBox("凭据")
        self.cred_form = QFormLayout(self.cred_box)
        form.addRow(self.cred_box)

        # --- LLM 专用 ---
        self.llm_box = QGroupBox("大模型设置（OpenAI 兼容：LM Studio / Ollama / 云端）")
        lf = QFormLayout(self.llm_box)
        llm = self.config.translate.llm
        self.llm_base = QLineEdit(llm.base_url)
        self.llm_base.setPlaceholderText("http://127.0.0.1:1234/v1")
        self.llm_key = QLineEdit(llm.api_key)
        self.llm_key.setEchoMode(QLineEdit.Password)
        self.llm_key.setPlaceholderText("本地模型通常留空")
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)   # 可下拉选，也可手打
        self.llm_model.setMinimumWidth(260)
        if llm.model:
            self.llm_model.addItem(llm.model)
            self.llm_model.setCurrentText(llm.model)
        self.llm_model.lineEdit().setPlaceholderText("点右边「列出模型」从列表里选")
        self.llm_style = QComboBox()
        self.llm_style.addItem("指令模型（chat，走完整提示词）", "chat")
        self.llm_style.addItem("纯翻译模型（plain，只喂原文）", "plain")
        self.llm_style.setCurrentIndex(max(0, self.llm_style.findData(llm.prompt_style or "chat")))
        lf.addRow("接口地址", self.llm_base)
        lf.addRow("API Key", self.llm_key)
        lf.addRow("模型名", self.llm_model)
        lf.addRow("提示词风格", self.llm_style)
        model_hint = QLabel(
            "**模型由你自己选**，程序不替你决定——不同模型显存占用差别巨大"
            "（2B 约 2~4GB，27B 能吃到 20GB+）。<br>"
            "点「列出模型」会把 LM Studio / Ollama 里已有的模型填进下拉框。"
            "听小说建议用小模型（快、省显存），翻译质量不够再换大的。"
        )
        model_hint.setWordWrap(True)
        lf.addRow("", model_hint)
        style_hint = QLabel(
            "sakura-galtransl 这类<b>微调翻译模型</b>必须选 <b>plain</b>，"
            "否则它会把提示词当正文翻译回来（实测踩过）。"
        )
        style_hint.setWordWrap(True)
        lf.addRow("", style_hint)
        form.addRow(self.llm_box)

        # --- 通用 ---
        self.ctx_lines = QSpinBox()
        self.ctx_lines.setRange(0, 200)
        self.ctx_lines.setValue(self.config.translate.context_lines)
        self.ctx_lines.setSuffix(" 行")
        form.addRow("带入前文", self.ctx_lines)
        ctx_hint = QLabel("实测每行约 31 token；20 行仅占 8k 窗口的 9.7%，对译名一致帮助很大。")
        ctx_hint.setWordWrap(True)
        form.addRow("", ctx_hint)

        self.template_combo = QComboBox()
        for t in list_templates():
            self.template_combo.addItem(f"{t.name} — {t.description}", t.id)
        ti = self.template_combo.findData(self.config.translate.prompt_template)
        self.template_combo.setCurrentIndex(max(0, ti))
        form.addRow("提示词模板", self.template_combo)

        self.glossary_edit = QLineEdit(self._glossary_path())
        self.glossary_edit.setPlaceholderText("术语表文件（.tsv 或 .json），留空不用")
        form.addRow("术语表", self.glossary_edit)

        self.prompt_edit = QTextEdit(self.config.translate.custom_prompt)
        self.prompt_edit.setPlaceholderText("留空使用上面的内置模板；填了就完全覆盖系统提示词")
        self.prompt_edit.setMinimumHeight(70)
        form.addRow("自定义提示词", self.prompt_edit)

        test_btn = QPushButton("测试当前翻译通道")
        test_btn.clicked.connect(lambda: self._run_test(self._test_translate))
        form.addRow("", test_btn)

        lay.addWidget(box)

        # --- 术语表编辑 ---
        gbox = QGroupBox("术语表（原文 → 译法，每行一条）")
        gl = QVBoxLayout(gbox)
        self.glossary_edit_box = QTextEdit()
        self.glossary_edit_box.setPlaceholderText("リンファン\t林凡\n声堂\t青铜")
        self.glossary_edit_box.setMinimumHeight(120)
        self._load_glossary_into_box()
        gl.addWidget(self.glossary_edit_box)
        ghint = QLabel(
            "只注入<b>本条字幕里真正出现</b>的术语——整表注入既费 token 又会让模型硬套无关词。"
        )
        ghint.setWordWrap(True)
        gl.addWidget(ghint)
        lay.addWidget(gbox)

        self._on_provider_changed()
        return page

    # ------------------------------------------------------------------ #
    # 识别
    # ------------------------------------------------------------------ #
    def _build_asr_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        form_box = QGroupBox("识别设置")
        form = QFormLayout(form_box)

        self.lang_combo = QComboBox()
        for code, label in (
            ("auto", "自动识别（用 LID 判断语种）"),
            ("zh", "中文"), ("zh-en", "中英混说"), ("en", "英语"),
            ("ja", "日语"), ("ko", "韩语"), ("yue", "粤语"),
        ):
            self.lang_combo.addItem(label, code)
        li = self.lang_combo.findData(self.config.asr.language)
        self.lang_combo.setCurrentIndex(max(0, li))
        form.addRow("语言", self.lang_combo)

        lang_hint = QLabel(
            "⚠️ 官方<b>没有日语流式模型</b>，所以日语会走 SenseVoice 分块识别，"
            "延迟比中文高（约 1.0~1.8s vs 中文 0.4~0.7s）。这是模型可用性的限制，不是参数问题。"
        )
        lang_hint.setWordWrap(True)
        form.addRow("", lang_hint)

        # 静音阈值：用户要求"电平设置除了开始选程序时能调，设置里也要能调"，
        # 而且以前**根本没保存过**（关掉电平窗再开又回 -80）。
        self.silence_db = QDoubleSpinBox()
        self.silence_db.setRange(-100.0, -20.0)
        self.silence_db.setSingleStep(5.0)
        self.silence_db.setDecimals(0)
        self.silence_db.setSuffix(" dBFS")
        self.silence_db.setValue(self.config.audio.silence_rms_threshold_db)
        form.addRow("静音阈值", self.silence_db)

        # 电平表 + 阈值线：用户反馈"启动时给了电平表 UI，设置里反而不给"。
        # 这里直接复用独立电平表窗口那个控件（同一套观感），阈值线跟着上面的
        # 数值实时移动；字幕在跑时还能看到实时电平（level_source）。
        self.threshold_meter = LevelMeterWidget()
        self.threshold_meter.set_threshold(
            threshold_from_db(float(self.silence_db.value()))
        )
        self.threshold_meter.set_markers(
            [(MEASURED_SPEECH_DB, f"实测：音量调小后的真实语音 ≈ {MEASURED_SPEECH_DB:.0f} dBFS")]
        )
        self.threshold_meter.set_texts("当前采集目标 · 实时电平", "未采集")
        form.addRow("", self.threshold_meter)
        self.silence_db.valueChanged.connect(self._on_silence_db_changed)

        sth = QLabel(
            "低于此电平就当作「没有声音」。<b>越小越灵敏</b>（小声也算有声），"
            "越大越严格（能过滤底噪）——橙色虚线就是当前阈值，蓝色虚线是实测参考。<br>"
            "实测参考：进程回环在目标不播放时给的是精确的 0，"
            "而调小音量后的真实语音 RMS 约 -66 dBFS，所以默认取 -80。"
        )
        sth.setWordWrap(True)
        form.addRow("", sth)

        outer.addWidget(form_box)

        # --- 延迟档位 ---
        preset_box = QGroupBox("延迟档位")
        pv = QVBoxLayout(preset_box)
        row = QHBoxLayout()
        self.preset_buttons: dict[str, QPushButton] = {}
        for pid, label in (
            ("realtime", "最低延迟"), ("balanced", "平衡（默认）"), ("accurate", "最准"),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, p=pid: self._apply_preset(p))
            self.preset_buttons[pid] = b
            row.addWidget(b)
        row.addStretch(1)
        pv.addLayout(row)
        self.preset_label = QLabel("")
        pv.addWidget(self.preset_label)
        outer.addWidget(preset_box)

        # --- 五个延迟参数（说明文字与 docs/延迟调节.md 同源）---
        knob_box = QGroupBox("延迟参数（拖动后档位变为自定义）")
        kv = QVBoxLayout(knob_box)
        self.knob_sliders: dict[str, QSlider] = {}

        for knob in LATENCY_KNOBS:
            holder = QVBoxLayout()
            head = QHBoxLayout()
            name = QLabel(f"<b>{knob.label}</b>")
            value = QLabel("")
            head.addWidget(name)
            head.addStretch(1)
            head.addWidget(value)
            holder.addLayout(head)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(50)
            slider.setMaximum(15000 if knob.field != "partial_interval_ms" else 2000)
            slider.setValue(self._knob_value(knob))
            slider.valueChanged.connect(
                lambda v, k=knob, lb=value: self._on_knob_changed(k, v, lb)
            )
            holder.addWidget(slider)

            desc = QLabel(
                f"调小 → {knob.smaller}<br>调大 → {knob.larger}"
            )
            desc.setWordWrap(True)
            desc.setStyleSheet("color:#888;")
            holder.addWidget(desc)

            if knob.section == "asr" and self.config.asr.routing.get(
                self.config.asr.language, {}
            ):
                route = self.config.asr.routing.get(self.config.asr.language)
                if route is not None and not getattr(route, "streaming", True):
                    slider.setEnabled(False)
                    desc.setText(desc.text() + "<br><b>该语言使用分块识别，此项无效</b>")

            self.knob_sliders[knob.field] = slider
            value.setText(f"{slider.value()} ms")
            kv.addLayout(holder)
            kv.addSpacing(6)

        outer.addWidget(knob_box)
        outer.addStretch(1)
        self._refresh_preset_label()
        return page

    # ------------------------------------------------------------------ #
    # 外观
    # ------------------------------------------------------------------ #
    def _build_appearance_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        ov = self.config.overlay

        self.mode_combo = QComboBox()
        for code, label in (("source", "只有原文"), ("target", "只有译文"), ("bilingual", "双语")):
            self.mode_combo.addItem(label, code)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(ov.display_mode)))
        form.addRow("显示内容", self.mode_combo)

        self.scroll_combo = QComboBox()
        self.scroll_combo.addItem("累积上滚", "accumulate")
        self.scroll_combo.addItem("单行替换", "replace")
        self.scroll_combo.setCurrentIndex(max(0, self.scroll_combo.findData(ov.scroll_mode)))
        form.addRow("滚动方式", self.scroll_combo)

        self.max_lines = QSpinBox()
        self.max_lines.setRange(1, 20)
        self.max_lines.setValue(ov.max_lines)
        self.max_lines.setSuffix(" 条")
        form.addRow("每屏条数", self.max_lines)

        self.lines_per_sub = QSpinBox()
        self.lines_per_sub.setRange(1, 6)
        self.lines_per_sub.setValue(ov.lines_per_subtitle)
        self.lines_per_sub.setSuffix(" 行")
        form.addRow("每条最多换行", self.lines_per_sub)

        self.font_size = QSpinBox()
        self.font_size.setRange(8, 200)
        self.font_size.setValue(ov.font_size)
        self.font_size.setSuffix(" pt")
        form.addRow("字号", self.font_size)

        self.outline_width = QSpinBox()
        self.outline_width.setRange(0, 12)
        self.outline_width.setValue(ov.outline_width)
        form.addRow("描边粗细", self.outline_width)
        oh = QLabel("描边保证字幕在亮暗不定的游戏画面上都能看清；设为 0 则关闭描边。")
        oh.setWordWrap(True)
        form.addRow("", oh)

        self.win_width = QSpinBox()
        self.win_width.setRange(200, 6000)
        self.win_width.setValue(ov.window_width)
        self.win_width.setSuffix(" px")
        form.addRow("字幕窗宽度", self.win_width)

        # 高度：0 = 自动（正好装下当前内容，不留空行）；手动拖过就会写进这里
        self.win_height = QSpinBox()
        self.win_height.setRange(0, 2000)
        self.win_height.setSpecialValueText("自动（按内容）")
        self.win_height.setValue(ov.window_height)
        self.win_height.setSuffix(" px")
        form.addRow("字幕窗高度", self.win_height)
        whh = QLabel(
            "拖动字幕窗<b>上/下边缘</b>也会改这里。填 0 = 自动：窗口高度正好等于"
            "当前字幕实际占的行数（不留空行）。"
        )
        whh.setWordWrap(True)
        form.addRow("", whh)

        self.bg_opacity = QDoubleSpinBox()
        self.bg_opacity.setRange(0.0, 1.0)
        self.bg_opacity.setSingleStep(0.05)
        self.bg_opacity.setValue(ov.background_opacity)
        form.addRow("背景不透明度", self.bg_opacity)

        self.win_opacity = QDoubleSpinBox()
        self.win_opacity.setRange(0.2, 1.0)
        self.win_opacity.setSingleStep(0.05)
        self.win_opacity.setValue(ov.window_opacity)
        form.addRow("整体不透明度", self.win_opacity)
        woh = QLabel("整个字幕窗（含文字）的淡化程度，用于让它不那么抢眼。")
        woh.setWordWrap(True)
        form.addRow("", woh)

        self.resizable = QCheckBox("允许拖拽边缘缩放字幕窗")
        self.resizable.setChecked(ov.resizable)
        form.addRow("", self.resizable)

        self.auto_font = QCheckBox("缩放窗口时字号自动跟着放大/缩小")
        self.auto_font.setChecked(ov.auto_font_scale)
        form.addRow("", self.auto_font)
        afh = QLabel(
            "开启后，把字幕窗拉宽，字也会跟着变大——<b>不用再去改字号</b>。"
            "关掉则字号固定、只有窗口变宽。"
        )
        afh.setWordWrap(True)
        form.addRow("", afh)

        self.shrink_font = QCheckBox("放不下时自动缩小字号（长句过去后自动恢复）")
        self.shrink_font.setChecked(ov.auto_shrink_font)
        form.addRow("", self.shrink_font)
        sfh = QLabel(
            "窗口装不下当前字幕时（比如你把窗口拖矮了、或者一次来了好几条），"
            "把字号<b>动态</b>缩小到刚好放得下，最多缩到基准字号的 60%；"
            "内容一少，字号自动回到上面设置的基准值。"
        )
        sfh.setWordWrap(True)
        form.addRow("", sfh)

        self.always_on_top = QCheckBox("始终置顶（每 2 秒重申一次，对抗游戏抢 Z 序）")
        self.always_on_top.setChecked(ov.always_on_top)
        form.addRow("", self.always_on_top)

        # 穿透与锁定**故意不放在这里**：控制窗上已经有了。
        # 同一个开关两个入口，改了一处另一处不同步，用户会怀疑"设置没保存"（真实反馈）。
        dedupe_hint = QLabel(
            "「点击穿透」「锁定位置」在<b>控制窗</b>上直接切——那是玩游戏时要随手改的开关，"
            "放在那边更顺手；这里不重复提供，避免两处状态不一致。"
        )
        dedupe_hint.setWordWrap(True)
        form.addRow("", dedupe_hint)
        return page

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def _glossary_path(self) -> str:
        g = self.config.translate.glossary
        return g.get("_path", "") if isinstance(g, dict) else ""

    def _load_glossary_into_box(self) -> None:
        g = self.config.translate.glossary
        if not isinstance(g, dict):
            return
        lines = [f"{k}\t{v}" for k, v in g.items() if not k.startswith("_")]
        self.glossary_edit_box.setPlainText("\n".join(lines))

    def _on_provider_changed(self) -> None:
        pid = self.provider_combo.currentData() or "none"
        while self.cred_form.count():
            item = self.cred_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.cred_fields.clear()
        for key, label in PROVIDER_META.get(pid, ("", {}))[1].items():
            edit = QLineEdit(str(self.config.translate.providers.get(pid, {}).get(key, "")))
            if "secret" in key or "key" in key:
                edit.setEchoMode(QLineEdit.Password)
            self.cred_form.addRow(label, edit)
            self.cred_fields[key] = edit
        self.cred_box.setVisible(bool(self.cred_fields))
        self.llm_box.setVisible(pid == "llm")

    def _knob_value(self, knob) -> int:
        if knob.section == "asr":
            return int(getattr(self.config.asr, knob.field))
        return int(getattr(self.config.asr.vad, knob.field))

    def _on_knob_changed(self, knob, value: int, label: QLabel) -> None:
        label.setText(f"{value} ms")
        if knob.section == "asr":
            setattr(self.config.asr, knob.field, value)
        else:
            setattr(self.config.asr.vad, knob.field, value)
        self.config.asr.latency_preset = "custom"  # type: ignore[assignment]
        self._refresh_preset_label()

    def _apply_preset(self, preset: str) -> None:
        self.config.asr.apply_latency_preset(preset)
        for knob in LATENCY_KNOBS:
            s = self.knob_sliders.get(knob.field)
            if s is not None:
                s.blockSignals(True)
                s.setValue(self._knob_value(knob))
                s.blockSignals(False)
        self._refresh_preset_label()

    def _refresh_preset_label(self) -> None:
        cur = self.config.asr.detect_preset()
        for pid, b in self.preset_buttons.items():
            b.setChecked(pid == cur)
        labels = {"realtime": "最低延迟", "balanced": "平衡", "accurate": "最准", "custom": "自定义"}
        self.preset_label.setText(
            f"当前档位：<b>{labels.get(cur, cur)}</b>"
            + ("" if cur != "custom" else "（参数被手动改过）")
        )

    # ------------------------------------------------------------------ #
    def _run_test(self, fn) -> None:
        self.status.setText("测试中…（网络请求可能几秒）")
        t = _TestThread(fn)
        t.done.connect(self._on_test_done)
        self._threads.append(t)
        t.start()

    def _on_test_done(self, text: str) -> None:
        self.status.setText("测试完成")
        self.net_log.append(text)

    # --- 具体测试 ---
    def _test_proxy(self) -> str:
        import socket
        from urllib.parse import urlparse

        proxy = self.proxy_edit.text().strip()
        if not proxy:
            return "代理：未设置（直连）"
        u = urlparse(proxy)
        try:
            s = socket.socket()
            s.settimeout(2.0)
            s.connect((u.hostname or "", u.port or 0))
            s.close()
            return f"✅ 代理 {proxy} 端口可达"
        except Exception as exc:  # noqa: BLE001
            return f"❌ 代理 {proxy} 不可达：{exc}"

    def _test_all(self) -> str:
        out = [self._test_proxy()]
        proxy = self.proxy_edit.text().strip()
        for pid in ("web_google", "web_bing"):
            try:
                from app.translate.traditional.providers import build_provider

                t = build_provider(pid, {}, proxy=proxy, qps_limit=1.0)
                ok, msg = t.ping() if t else (False, "无法构造")
                out.append(f"{'✅' if ok else '❌'} {PROVIDER_META[pid][0]}：{msg}")
                if t:
                    t.close()
            except Exception as exc:  # noqa: BLE001
                out.append(f"❌ {pid}：{exc}")
        base = self.llm_base.text().strip()
        if base:
            try:
                from app.translate.openai_compat import probe_endpoint

                ok, msg, models = probe_endpoint(base, self.llm_key.text().strip(), proxy)
                out.append(f"{'✅' if ok else '❌'} 大模型 {base}：{msg}")
                if models:
                    self._fill_models(models)
                    out.append(f"     已把 {len(models)} 个模型填进上面的下拉框，"
                               "请自己挑一个（**不会再自动替你选**——"
                               "以前自动选列表第一个，可能直接加载一个 20GB+ 的大模型把显存吃满）")
            except Exception as exc:  # noqa: BLE001
                out.append(f"❌ 大模型：{exc}")
        return "\n".join(out)

    def _fill_models(self, models: list[str]) -> None:
        """把可用模型填进下拉框，但**不改用户当前的选择**。"""
        current = self.llm_model.currentText().strip()
        self.llm_model.blockSignals(True)
        self.llm_model.clear()
        for m in models:
            self.llm_model.addItem(m)
        if current:
            self.llm_model.setCurrentText(current)
        elif models:
            # 不自动选：清空当前项，让 placeholder 提示用户自己选
            self.llm_model.setCurrentIndex(-1)
            self.llm_model.setEditText("")
        self.llm_model.blockSignals(False)

    def _test_translate(self) -> str:
        pid = self.provider_combo.currentData() or "none"
        if pid == "none":
            return "当前选择的是「不翻译」"
        try:
            from app.translate.traditional.providers import build_provider

            if pid in ("llm",):
                from app.translate.openai_compat import OpenAICompatTranslator

                t = OpenAICompatTranslator(
                    base_url=self.llm_base.text().strip(),
                    api_key=self.llm_key.text().strip(),
                    model=self.llm_model.currentText().strip(),
                    proxy=self.proxy_edit.text().strip(),
                    prompt_style=self.llm_style.currentData() or "",
                )
            else:
                creds = {k: e.text().strip() for k, e in self.cred_fields.items()}
                t = build_provider(pid, creds, proxy=self.proxy_edit.text().strip())
            if t is None:
                return f"无法构造通道 {pid}"
            try:
                ok, msg = t.ping()
                return f"{'✅' if ok else '❌'} {PROVIDER_META.get(pid, (pid,))[0]}：{msg}"
            finally:
                t.close()
        except Exception as exc:  # noqa: BLE001
            return f"❌ 测试失败：{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        from pathlib import Path

        c = self.config
        c.proxy = self.proxy_edit.text().strip()
        c.translate.enabled = self.enable_translate.isChecked()
        c.translate.provider = self.provider_combo.currentData() or "none"
        c.translate.context_lines = self.ctx_lines.value()
        c.translate.prompt_template = self.template_combo.currentData() or "subtitle_direct"
        c.translate.custom_prompt = self.prompt_edit.toPlainText()  # QTextEdit 没有 text()！

        pid = c.translate.provider
        if self.cred_fields:
            bucket = dict(c.translate.providers.get(pid, {}))
            bucket.update({k: e.text().strip() for k, e in self.cred_fields.items()})
            c.translate.providers[pid] = bucket

        c.translate.llm.base_url = self.llm_base.text().strip()
        c.translate.llm.api_key = self.llm_key.text().strip()
        c.translate.llm.model = self.llm_model.currentText().strip()
        c.translate.llm.enabled = True
        c.translate.llm.prompt_style = self.llm_style.currentData() or ""

        # 术语表：内联优先，同时把路径记下来（便于用户下次继续用文件）
        glossary: dict[str, str] = {}
        path = self.glossary_edit.text().strip()
        if path:
            glossary["_path"] = path
        for line in self.glossary_edit_box.toPlainText().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("\t")] if "\t" in line else line.split(None, 1)
            if len(parts) >= 2 and parts[0] and parts[1]:
                glossary[parts[0]] = parts[1]
        c.translate.glossary = glossary

        c.asr.language = self.lang_combo.currentData() or "auto"
        # 静音阈值：以前只能在电平表窗口里调，而且**根本没存过**，
        # 关掉再开又回到 -80（用户反馈）。现在在这里也能调，并且会保存。
        c.audio.silence_rms_threshold_db = float(self.silence_db.value())
        c.overlay.display_mode = self.mode_combo.currentData() or "bilingual"
        c.overlay.scroll_mode = self.scroll_combo.currentData() or "accumulate"
        c.overlay.max_lines = self.max_lines.value()
        c.overlay.lines_per_subtitle = self.lines_per_sub.value()
        c.overlay.font_size = self.font_size.value()
        c.overlay.outline_width = self.outline_width.value()
        c.overlay.window_width = self.win_width.value()
        c.overlay.window_height = self.win_height.value()
        c.overlay.background_opacity = self.bg_opacity.value()
        c.overlay.window_opacity = self.win_opacity.value()
        c.overlay.resizable = self.resizable.isChecked()
        c.overlay.auto_font_scale = self.auto_font.isChecked()
        c.overlay.auto_shrink_font = self.shrink_font.isChecked()
        c.overlay.always_on_top = self.always_on_top.isChecked()
        # 穿透/锁定 的入口在控制窗（避免两处重复），这里不覆盖它们的值

        try:
            c.save()
            self.status.setText("✅ 已保存并即时生效（外观类立即反映；识别/翻译会重建）")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"❌ 保存失败：{exc}")
            return

        # 通知使用者立即应用（控制窗/字幕窗对接这个信号）
        self.saved.emit()

        # 术语表若填了路径，顺便落盘，方便用户用 Excel 维护
        if path:
            try:
                from app.translate.glossary import Glossary

                g = Glossary()
                for k, v in glossary.items():
                    if not k.startswith("_"):
                        g.add(k, v)
                g.to_file(Path(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("写术语表文件失败: %s", exc)
