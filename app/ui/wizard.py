"""首次运行向导（也能**重新运行**，用来改语言包 / 补下载模型 / 重选档位）。

目标：**让非开发者从零到能出字幕，全程不用看文档。**

五个步骤：
  1. 环境自检（解释器/依赖/日志目录）
  2. 硬件探测 → 推荐档位，并给出推荐语言包
  3. 下载模型（按语言包勾选，显示总体积与进度）
  4. 翻译通道（本地大模型 / 网页版免 key / 先不翻译）
  5. 完成，写入配置

设计上刻意做对的三件事：
- **下载放到后台线程**：模型 1.8GB，主线程卡住的话窗口会假死，用户以为程序坏了
- **每步都能看到"为什么"**：档位推荐要说明依据，体积要显示出来，
  测通道要给出可读结果——而不是让用户盲选
- **重跑时必须"以现状为默认值"**（``rerun``）：用户第二次打开向导是想改一项，
  不是想把代理、翻译通道、延迟档位全清回出厂设置。所以重跑时：
  档位取当前 ``asr.latency_preset``、语言包勾"已经装在本地的那几个"、
  代理与翻译通道用当前配置预填，向导表示不了的通道路由（百度/有道/…）
  会插一条"保持当前"。首次运行（``first_run_done`` 还是 False）才用硬件推荐值。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from app.config import AppConfig
from app.models.hardware import detect_hardware, tier_label
from app.models.registry import PACKS, human_size, total_bytes_for_packs
from app.utils.log import get_logger

log = get_logger(__name__)

# 档位 ⇄ 延迟预设 的对应关系（向导里选档位 = 设延迟预设）
TIER_TO_PRESET: dict[str, str] = {"low": "realtime", "mid": "balanced", "high": "accurate"}
PRESET_TO_TIER: dict[str, str] = {p: t for t, p in TIER_TO_PRESET.items()}


def installed_packs() -> list[str]:
    """已经**完整**下载到本地的语言包。

    重跑向导时用它当默认勾选：用户看到的是"我现在有什么"，而不是被重置成
    硬件推荐的那一套（那会让"只想补一个小包"变成"再下一堆东西"）。
    """
    try:
        from app.models.downloader import ModelDownloader

        downloader = ModelDownloader()
    except Exception as exc:  # noqa: BLE001 - 探测失败就按"都没装"处理，不拦着用户
        log.debug("初始化模型下载器失败：%s", exc)
        return []

    found: list[str] = []
    for pid, pack in PACKS.items():
        try:
            if all(downloader.status(m) == "installed" for m in pack.model_ids):
                found.append(pid)
        except Exception as exc:  # noqa: BLE001
            log.debug("检查语言包 %s 失败：%s", pid, exc)
    return found


# --------------------------------------------------------------------------- #
# 后台任务
# --------------------------------------------------------------------------- #
class _CheckThread(QThread):
    done = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:  # noqa: D102
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # noqa: BLE001
            self.done.emit(f"✗ {type(exc).__name__}: {exc}")


class _DownloadThread(QThread):
    progress = Signal(str, int)   # 文本, 百分比
    finished_all = Signal(bool, str)

    def __init__(self, model_ids: list[str], proxy: str) -> None:
        super().__init__()
        self.model_ids = model_ids
        self.proxy = proxy
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        from app.models.downloader import ModelDownloader

        dl = ModelDownloader(proxy=self.proxy)
        total_bytes = sum(
            (dl.model_dir(m).exists() and 0) or 0 for m in self.model_ids
        )
        ok_all = True
        for i, mid in enumerate(self.model_ids):
            if self._cancel:
                self.finished_all.emit(False, "已取消")
                return
            base_pct = int(i / max(1, len(self.model_ids)) * 100)

            def on_progress(p, _base=base_pct, _i=i) -> None:
                span = 100 / max(1, len(self.model_ids))
                pct = int(_base + p.fraction * span)
                self.progress.emit(p.describe(), pct)

            def on_log(msg, _b=base_pct, _i=i) -> None:
                self.progress.emit(msg, int((_i + 1) / max(1, len(self.model_ids)) * 100))

            try:
                ok = dl.install(mid, on_progress=on_progress, on_log=on_log)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self.progress.emit(f"✗ {mid} 失败：{exc}", base_pct)
            if not ok:
                ok_all = False

        self.finished_all.emit(ok_all, "全部完成" if ok_all else "有模型下载失败，可稍后重试")


# --------------------------------------------------------------------------- #
# 第 1 页：环境自检
# --------------------------------------------------------------------------- #
class WelcomePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("① 欢迎")
        self.setSubTitle("先确认运行环境是好的——这一步出问题，后面都不用谈。")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)

        self.again = QPushButton("重新检查")
        self.again.clicked.connect(self._run)

        self.intro = QLabel(
            "听·显·译：只监听你指定的那个程序的音频，实时生成字幕并翻译成中文。\n"
            "典型场景：一边听小说一边打游戏——只有小说声进识别，游戏声完全不理会。"
        )
        self.intro.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(self.intro)
        lay.addWidget(self.log, 1)
        lay.addWidget(self.again, 0, Qt.AlignRight)

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        if isinstance(wiz, FirstRunWizard) and wiz.rerun:
            self.setSubTitle("重新配置：每一步都已经填好你<b>当前</b>的设置，只改你想改的即可。")
            self.intro.setText(
                "这次是<b>重新配置</b>（不是从零开始）：语言包默认勾的是你<b>已经下好</b>的那些，"
                "代理与翻译通道也按现状预填。改完点「完成并保存」；"
                "不想改就点「稍后再设」直接退出，配置不会被动。"
            )
        self._run()

    def _run(self) -> None:
        self.log.setPlainText("检查中…")
        t = _CheckThread(self._check)
        t.done.connect(self.log.setPlainText)
        t.start()
        self._t = t

    @staticmethod
    def _check() -> str:
        import sys
        from pathlib import Path

        lines = []
        v = sys.version_info
        ok = (v.major, v.minor) == (3, 12)
        lines.append(f"{'✅' if ok else '❌'} Python {v.major}.{v.minor}.{v.micro}"
                     + ("" if ok else "（需要 3.12：proc-tap / sherpa-onnx 没有更高版本的轮子）"))
        lines.append(f"{'✅' if sys.prefix != sys.base_prefix else '❌'} 虚拟环境：{sys.prefix}")

        for mod, label in (("PySide6", "界面"), ("proctap", "音频采集"),
                           ("sherpa_onnx", "语音识别"), ("pycaw", "发声进程枚举"),
                           ("httpx", "网络")):
            try:
                __import__(mod)
                lines.append(f"✅ {label}（{mod}）")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"❌ {label}（{mod}）导入失败：{exc}")

        try:
            from app import paths

            paths.ensure_dirs()
            lines.append(f"✅ 数据目录：{paths.DATA_DIR}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"❌ 数据目录不可写：{exc}")

        if any(x.startswith("❌") for x in lines):
            lines.append("")
            lines.append("有项目没通过：请关闭本窗口，双击「首次安装.bat」重装依赖。")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 第 2 页：硬件与档位
# --------------------------------------------------------------------------- #
class HardwarePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("② 硬件与档位")
        self.setSubTitle("按你的机器推荐一个档位；不确定就照推荐来，之后可在设置里改。")
        self._installed: list[str] | None = None
        self._loading = False

        self.hw_log = QTextEdit()
        self.hw_log.setReadOnly(True)
        self.hw_log.setMinimumHeight(150)

        self.tier_buttons: dict[str, QRadioButton] = {}
        tier_box = QGroupBox("识别档位")
        tl = QVBoxLayout(tier_box)
        for tier, desc in (
            ("low", "轻量档：CPU 为主，只装中文/日语包，占用最小"),
            ("mid", "平衡档：中英日韩粤 + 语种识别，适合大多数机器"),
            ("high", "性能档：再加 99 语言兜底（多下 1GB），显存充足时最舒服"),
        ):
            rb = QRadioButton(f"{tier_label(tier)} —— {desc}")
            self.tier_buttons[tier] = rb
            rb.toggled.connect(self._refresh_packs)
            tl.addWidget(rb)

        self.pack_boxes: dict[str, QCheckBox] = {}
        pack_box = QGroupBox("语言包（可自行增删）")
        pl = QVBoxLayout(pack_box)
        for pack in PACKS.values():
            cb = QCheckBox(f"{pack.display_name}  [{human_size(pack.total_bytes)}] — {pack.description}")
            cb.stateChanged.connect(self._refresh_size)
            self.pack_boxes[pack.id] = cb
            pl.addWidget(cb)

        self.size_label = QLabel("")
        self.size_label.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(self.hw_log)
        lay.addWidget(tier_box)
        lay.addWidget(pack_box, 1)
        lay.addWidget(self.size_label)

    def initializePage(self) -> None:  # noqa: N802
        if getattr(self, "_done", False):
            return
        self._done = True
        hw = detect_hardware()
        self.hardware = hw
        lines = hw.summary_lines() + [""] + hw.notes
        self.hw_log.setPlainText("\n".join(lines))

        wiz = self.wizard()
        rerun = isinstance(wiz, FirstRunWizard) and wiz.rerun
        # 重跑时**不要**让"勾档位"顺手把语言包重置成推荐值：
        # 用户是来改一项的，不是来推倒重来的。
        self._installed = installed_packs() if rerun else []
        self._loading = True
        try:
            if rerun:
                tier = PRESET_TO_TIER.get(wiz.config.asr.latency_preset, "")
                if tier:
                    self.tier_buttons[tier].setChecked(True)
            else:
                self.tier_buttons[hw.recommended_tier].setChecked(True)
        finally:
            self._loading = False

        if rerun:
            packs = self._installed or self._tier_packs()
            for pid, cb in self.pack_boxes.items():
                cb.setChecked(pid in packs)
        else:
            self._refresh_packs()

    def _tier_packs(self) -> list[str]:
        hw = getattr(self, "hardware", None)
        tier = next(
            (t for t, rb in self.tier_buttons.items() if rb.isChecked()),
            getattr(hw, "recommended_tier", "mid"),
        )
        return self._packs_for_tier(tier, hw)

    @staticmethod
    def _packs_for_tier(tier: str, hw=None) -> list[str]:
        return {
            "low": ["core", "zh", "ja-ko-yue"],
            "mid": ["core", "zh", "zh-en", "ja-ko-yue", "lid"],
            "high": ["core", "zh", "zh-en", "en", "ja-ko-yue", "multilingual", "lid"],
        }.get(tier, list(getattr(hw, "recommended_packs", None) or ["core"]))

    def _refresh_packs(self) -> None:
        hw = getattr(self, "hardware", None)
        if hw is None:
            return
        if getattr(self, "_loading", False):
            self._refresh_size()  # 只是初始化勾选状态，别反过来覆盖用户已经装好的包
            return
        tier = next((t for t, rb in self.tier_buttons.items() if rb.isChecked()), hw.recommended_tier)
        want = self._packs_for_tier(tier, hw)
        for pid, cb in self.pack_boxes.items():
            cb.setChecked(pid in want)
        self._refresh_size()

    def _refresh_size(self) -> None:
        ids = self.selected_packs()
        total = total_bytes_for_packs(ids)
        if self._installed is None:
            self._installed = installed_packs()
        done = [pid for pid in ids if pid in self._installed]
        text = f"将下载 <b>{len(ids)}</b> 个语言包，合计 <b>{human_size(total)}</b>"
        if done:
            text += f"；其中 <b>{len(done)}</b> 个已经装好，会自动跳过"
        self.size_label.setText(text)

    def selected_packs(self) -> list[str]:
        return [pid for pid, cb in self.pack_boxes.items() if cb.isChecked()]


# --------------------------------------------------------------------------- #
# 第 3 页：模型下载
# --------------------------------------------------------------------------- #
class DownloadPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("③ 下载模型")
        self.setSubTitle("模型按需下载；中断了重跑本向导或 --models install 都会接着下。")

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(140)

        # 代理：**默认留空**，由用户按需填。
        # 以前这里预填了作者本机的代理，别人的机器上会直接连不上网，
        # 而且用户会以为是程序坏了。
        self.proxy_edit = QLineEdit("")
        self.proxy_edit.setPlaceholderText("例如 http://127.0.0.1:2333 —— 用不到就留空")
        proxy_row = QHBoxLayout()
        proxy_row.addWidget(QLabel("下载代理"))
        proxy_row.addWidget(self.proxy_edit, 1)
        proxy_hint = QLabel(
            "模型从 HuggingFace 下载，<b>国内直连通常很慢或失败</b>，这时才需要填代理；"
            "留空表示直连。以后也能在「设置 → 网络」里改。"
        )
        proxy_hint.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)

        self.start_btn = QPushButton("开始下载")
        self.start_btn.clicked.connect(self._start)
        self.skip_btn = QPushButton("跳过（稍后再下）")
        self.skip_btn.clicked.connect(self._skip)

        row = QHBoxLayout()
        row.addWidget(self.start_btn)
        row.addWidget(self.skip_btn)
        row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addWidget(self.info, 1)
        lay.addLayout(proxy_row)
        lay.addWidget(proxy_hint)
        lay.addWidget(self.bar)
        lay.addLayout(row)

    def initializePage(self) -> None:  # noqa: N802
        from app.models.registry import models_for_packs

        wiz = self.wizard()
        # 代理按**当前配置**预填一次（首次运行时配置里本来就是空的，不会预填出别人的地址）
        if isinstance(wiz, FirstRunWizard) and not getattr(self, "_proxy_inited", False):
            self._proxy_inited = True
            if wiz.config.proxy:
                self.proxy_edit.setText(wiz.config.proxy)

        packs = self._packs()
        self.model_ids = models_for_packs(packs)
        lines = ["将要下载："]
        for m in self.model_ids:
            from app.models.registry import MODELS

            spec = MODELS[m]
            lines.append(f"  · {spec.display_name}  [{human_size(spec.total_bytes)}]")
        lines.append(f"\n合计 {human_size(total_bytes_for_packs(packs))}")
        lines.append("\n已安装的会自动跳过；下载支持断点续传。")
        self.info.setPlainText("\n".join(lines))
        self.bar.setValue(0)

    def _packs(self) -> list[str]:
        wiz = self.wizard()
        if isinstance(wiz, FirstRunWizard):
            return wiz.selected_packs()
        return ["core"]

    def _start(self) -> None:
        if not getattr(self, "model_ids", None):
            self.info.append("没有需要下载的模型。")
            return
        self.start_btn.setEnabled(False)
        proxy = self.wizard().proxy() if isinstance(self.wizard(), FirstRunWizard) else ""
        t = _DownloadThread(self.model_ids, proxy)
        t.progress.connect(self._on_progress)
        t.finished_all.connect(self._on_done)
        self._t = t
        t.start()

    def _skip(self) -> None:
        self.info.append("\n已跳过。没有模型时程序无法识别，记得稍后补上。")
        self.wizard().next()

    def _on_progress(self, text: str, pct: int) -> None:
        self.bar.setValue(max(0, min(100, pct)))
        self.info.append(text)

    def _on_done(self, ok: bool, msg: str) -> None:
        self.bar.setValue(100 if ok else self.bar.value())
        self.info.append(f"\n{'✅' if ok else '⚠️'} {msg}")
        self.start_btn.setEnabled(True)
        if ok:
            self.wizard().next()


# --------------------------------------------------------------------------- #
# 第 4 页：翻译通道
# --------------------------------------------------------------------------- #
class TranslatePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("④ 翻译通道")
        self.setSubTitle("字幕本身就是外语的，翻译成中文才方便看。")

        self.provider = QComboBox()
        self.provider.addItem("本地大模型（LM Studio / Ollama，免费离线，推荐）", "llm")
        self.provider.addItem("谷歌网页版（免 key，必须有代理）", "web_google")
        self.provider.addItem("必应网页版（免 key，日译中质量好，必须有代理）", "web_bing")
        self.provider.addItem("先不翻译（只显示原文）", "none")
        self.provider.currentIndexChanged.connect(self._sync)

        self.base_url = QLineEdit("http://127.0.0.1:1234/v1")
        self.model = QLineEdit("")
        self.model.setPlaceholderText("点右侧「列出模型」自动获取")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("本地模型通常留空")
        self.style = QComboBox()
        self.style.addItem("指令模型（chat）", "chat")
        self.style.addItem("纯翻译模型（plain，如 sakura-galtransl）", "plain")

        self.llm_box = QGroupBox("大模型设置")
        lf = QFormLayout(self.llm_box)
        lf.addRow("接口地址", self.base_url)
        lf.addRow("模型名", self.model)
        lf.addRow("API Key", self.api_key)
        lf.addRow("提示词风格", self.style)

        list_btn = QPushButton("列出模型")
        list_btn.clicked.connect(self._list_models)
        test_btn = QPushButton("测试通道")
        test_btn.clicked.connect(self._test)
        row = QHBoxLayout()
        row.addWidget(list_btn)
        row.addWidget(test_btn)
        row.addStretch(1)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(110)

        hint = QLabel(
            "不知道选哪个？如果本机装了 LM Studio，选第一项并点「列出模型」即可；<br>"
            "没有本地模型就走网页版（要填代理），免 key 也能翻。"
        )
        hint.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("翻译方式"))
        lay.addWidget(self.provider)
        lay.addWidget(self.llm_box)
        lay.addLayout(row)
        lay.addWidget(self.result, 1)
        lay.addWidget(hint)

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        if isinstance(wiz, FirstRunWizard) and wiz.rerun and not getattr(self, "_inited", False):
            self._inited = True
            self._load_from_config(wiz.config)
        self._sync()

    def _load_from_config(self, config: AppConfig) -> None:
        """重跑向导时把**当前**翻译设置填进来，避免"改个模型顺手把翻译通道改了"。"""
        tr = config.translate
        idx = self.provider.findData(tr.provider)
        if idx < 0 and tr.provider not in ("", "none"):
            # 向导只提供 LLM / 网页版 / 不翻译，但设置里还有 5 家官方 API。
            # 这里插一条"保持当前"，否则用户一完成向导就被悄悄换成别的通道。
            label = self._provider_label(tr.provider)
            self.provider.insertItem(0, f"保持当前：{label}", tr.provider)
            idx = 0
        if idx >= 0:
            self.provider.setCurrentIndex(idx)

        llm = tr.llm
        self.base_url.setText(llm.base_url)
        self.model.setText(llm.model)
        self.api_key.setText(llm.api_key)
        style_idx = self.style.findData(llm.prompt_style)
        if style_idx >= 0:
            self.style.setCurrentIndex(style_idx)

    @staticmethod
    def _provider_label(pid: str) -> str:
        try:
            from app.ui.settings import PROVIDER_META

            return PROVIDER_META.get(pid, (pid, {}))[0]
        except Exception:  # noqa: BLE001 - 拿不到标签也不能拦着向导
            return pid

    def _sync(self) -> None:
        self.llm_box.setVisible(self.provider.currentData() == "llm")

    def _proxy(self) -> str:
        wiz = self.wizard()
        return wiz.proxy() if isinstance(wiz, FirstRunWizard) else ""

    def _list_models(self) -> None:
        from app.translate.openai_compat import probe_endpoint

        self.result.setPlainText("查询中…")
        base = self.base_url.text().strip()
        t = _CheckThread(lambda: self._do_list(base))
        t.done.connect(self.result.setPlainText)
        t.start()
        self._t = t

    def _do_list(self, base: str) -> str:
        ok, msg, models = probe_endpoint(base, self.api_key.text().strip(), self._proxy())
        if not ok:
            return f"❌ {msg}\n\n提示：LM Studio 需要在「Developer」里开启 Local Server。"
        self._models = models
        text = [f"✅ {msg}", "", "可用模型（复制到上面「模型名」）："]
        text += [f"  · {m}" for m in models[:20]]
        if len(models) > 20:
            text.append(f"  … 另有 {len(models) - 20} 个")
        return "\n".join(text)

    def _test(self) -> None:
        self.result.setPlainText("测试中…")
        t = _CheckThread(self._do_test)
        t.done.connect(self.result.setPlainText)
        t.start()
        self._t = t

    def _do_test(self) -> str:
        pid = self.provider.currentData()
        proxy = self._proxy()
        try:
            if pid == "none":
                return "已选择「不翻译」，跳过测试。"
            if pid == "llm":
                from app.translate.openai_compat import probe_endpoint

                ok, msg, models = probe_endpoint(
                    self.base_url.text().strip(), self.api_key.text().strip(), proxy
                )
                if models and not self.model.text().strip():
                    self.model.setText(models[0])
                return f"{'✅' if ok else '❌'} {msg}"
            from app.translate.traditional.providers import build_provider

            t = build_provider(pid, {}, proxy=proxy, qps_limit=1.0)
            if t is None:
                return f"❌ 无法构造通道 {pid}"
            try:
                ok, msg = t.ping()
            finally:
                t.close()
            return f"{'✅' if ok else '❌'} {msg}"
        except Exception as exc:  # noqa: BLE001
            return f"❌ {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 第 5 页：完成
# --------------------------------------------------------------------------- #
class FinishPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("⑤ 完成")
        self.setSubTitle("确认一下，然后就可以开始听了。")

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.summary)
        lay.addWidget(QLabel(
            "下一步：双击「启动.bat」，在窗口里选择正在播放小说的那个程序，"
            "字幕就会出现在屏幕底部。\n"
            "控制窗里有「设置…」可以改语言、延迟、外观和翻译通道。"
        ))

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        if not isinstance(wiz, FirstRunWizard):
            return
        packs = wiz.selected_packs()
        lines = [
            f"语言包：{'、'.join(packs) if packs else '（未选择）'}",
            f"合计体积：{human_size(total_bytes_for_packs(packs))}",
            f"翻译通道：{wiz.page4.provider.currentText()}",
        ]
        if wiz.page4.provider.currentData() == "llm":
            lines.append(f"  模型：{wiz.page4.model.text().strip() or '（未填）'}")
        lines.append(f"网络代理：{wiz.proxy() or '（直连）'}")
        lines.append("")
        lines.append("配置将保存到 data/config.json（不会进 Git）。")
        self.summary.setPlainText("\n".join(lines))


# --------------------------------------------------------------------------- #
# 向导本体
# --------------------------------------------------------------------------- #
class FirstRunWizard(QWizard):
    def __init__(self, config: AppConfig | None = None, *, rerun: bool | None = None) -> None:
        super().__init__(None)
        self.config = config or AppConfig.load()
        # 重跑 = 用它来改设置，而不是从零配一遍：各页会用**当前配置**预填。
        # 不传就按配置自己判断（first_run_done 为真说明以前配过了）。
        self.rerun = bool(self.config.first_run_done) if rerun is None else bool(rerun)
        self.setWindowTitle(
            "听·显·译 — 重新配置向导" if self.rerun else "听·显·译 — 首次运行向导"
        )
        self.resize(760, 620)
        self.setWizardStyle(QWizard.ClassicStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)

        self.page1 = WelcomePage()
        self.page2 = HardwarePage()
        self.page3 = DownloadPage()
        self.page4 = TranslatePage()
        self.page5 = FinishPage()
        for p in (self.page1, self.page2, self.page3, self.page4, self.page5):
            self.addPage(p)

        self.setButtonText(QWizard.NextButton, "下一步")
        self.setButtonText(QWizard.BackButton, "上一步")
        self.setButtonText(QWizard.FinishButton, "完成并保存")
        self.setButtonText(QWizard.CancelButton, "稍后再设")
        self.accepted.connect(self._apply)

    # ---- 向导内部共享 ----
    def proxy(self) -> str:
        """当前生效的代理。

        以向导里那个输入框为准（**默认空**），再回落到配置。
        不再预填作者本机的地址——别人的机器上不该有那个东西。
        """
        try:
            text = self.page3.proxy_edit.text().strip()
        except Exception:  # noqa: BLE001
            text = ""
        return text or self.config.proxy

    def set_proxy(self, value: str) -> None:
        self.config.proxy = value

    def selected_packs(self) -> list[str]:
        return self.page2.selected_packs()

    # ---- 收尾 ----
    def _apply(self) -> None:
        c = self.config
        c.proxy = self.page3.proxy_edit.text().strip()
        c.asr.preset = "custom"
        tier = next(
            (t for t, rb in self.page2.tier_buttons.items() if rb.isChecked()),
            self.page2.hardware.recommended_tier if hasattr(self.page2, "hardware") else "mid",
        )
        c.asr.latency_preset = TIER_TO_PRESET.get(
            tier, self.config.asr.latency_preset or "balanced"
        )
        try:
            c.asr.apply_latency_preset(c.asr.latency_preset)
        except Exception as exc:  # noqa: BLE001
            log.warning("应用延迟档位失败: %s", exc)

        pid = self.page4.provider.currentData() or "none"
        c.translate.provider = pid
        c.translate.enabled = pid != "none"
        if pid == "llm":
            c.translate.llm.enabled = True
            c.translate.llm.base_url = self.page4.base_url.text().strip()
            c.translate.llm.model = self.page4.model.text().strip()
            c.translate.llm.api_key = self.page4.api_key.text().strip()
            c.translate.llm.prompt_style = self.page4.style.currentData() or ""
        c.first_run_done = True
        try:
            c.save()
            log.info("%s完成，配置已保存", "重新配置向导" if self.rerun else "首次运行向导")
        except Exception as exc:  # noqa: BLE001
            log.error("保存配置失败: %s", exc)


def run_wizard(config: AppConfig | None = None, *, rerun: bool | None = None) -> int:
    """跑向导。返回 QDialog 结果码（1=完成并保存，0=取消）。

    ``rerun`` 留空时自动判断：``first_run_done`` 已为真就按"重新配置"跑
    （各页用当前配置预填），否则按首次运行跑。所以调用方**不用记**该传什么。
    """
    from PySide6.QtWidgets import QApplication

    from app.ui.lifecycle import configure_quit_policy, fit_window_to_screen

    app = QApplication.instance() or QApplication([])
    # 向导是这期间唯一的窗口：默认策略下向导一关就会调 QApplication.quit()，
    # 而下面主窗口的 app.exec() 还没开始跑，容易被这个"待退出"标记坑到。
    configure_quit_policy(app)
    wiz = FirstRunWizard(config, rerun=rerun)
    # 有的页面（语言包那一页）很长，先夹进屏幕，别让标题栏跑到屏幕外
    fit_window_to_screen(wiz)
    wiz.show()
    return int(wiz.exec())
