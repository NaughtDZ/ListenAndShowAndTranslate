"""配置模型与读写。

设计要点：
- 全部配置集中在 data/config.json（本机文件，.gitignore 红线）
- 用 pydantic 校验，损坏的配置不会导致程序崩溃（回退默认值并备份坏文件）
- ``redacted()`` 用于导出配置：**自动剔除 API Key / Secret / Token**
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app import paths
from app.utils.log import get_logger

log = get_logger(__name__)

CONFIG_VERSION = 1

# 键名里命中这些片段的，导出时一律替换为掩码
_SENSITIVE_HINTS = ("key", "secret", "token", "password", "passwd", "credential", "authorization")


# --------------------------------------------------------------------------- #
# 音频采集
# --------------------------------------------------------------------------- #
class AudioConfig(BaseModel):
    """音频来源配置（对应构想 1：只抓指定程序的音频）。"""

    source_mode: Literal["process", "device"] = "process"
    """process = WASAPI 进程回环（主路径）；device = 全设备回环（降级备选）。"""

    target_process_name: str = ""
    """目标进程名，如 "喜马拉雅.exe"。按名字跟随比按 PID 更稳（进程重启后 PID 会变）。"""

    target_pid: int | None = None
    """显式指定的 PID；None 表示按 target_process_name 动态解析。"""

    follow_process: bool = True
    """目标进程退出后是否自动等待并重连。"""

    reconnect_interval_s: float = Field(default=2.0, ge=0.5, le=60.0)

    buffer_ms: int = Field(default=200, ge=50, le=2000)
    """采集环形缓冲长度，越大越抗卡顿，但增加延迟。"""

    downmix_to_mono: bool = True
    target_sample_rate: int = 16000
    """ASR 输入采样率；16k 是绝大多数中文 ASR 模型的原生采样率。"""

    silence_rms_threshold_db: float = Field(default=-80.0, ge=-100.0, le=-20.0)
    """静音阈值（dBFS）——**由用户调整**，低于此电平视为"没有声音"。

    默认 -80 dBFS（线性 1e-4）。实测教训：定太严（如 -60 dBFS）会把
    "用户把音量调小后正在播放的语音"误判成静音（实测这种语音 RMS 可低至 5e-4）。
    进程回环在目标不渲染音频时给出的是精确的 0，所以默认可以放得很宽。
    """

    gain_db: float = Field(default=0.0, ge=-20.0, le=24.0)
    """数字增益（dB）。实测采集发生在音量合成器之后，可用它补偿被调小的音量。"""

    auto_gain: bool = False
    """按会话音量自动补偿增益（用户把小说音量调轻也能识别清楚）。"""

    max_auto_gain_db: float = Field(default=24.0, ge=0.0, le=48.0)


# --------------------------------------------------------------------------- #
# 语音识别
# --------------------------------------------------------------------------- #
class VADConfig(BaseModel):
    """语音活动检测（VAD）——**分块识别的延迟几乎完全由这几个参数决定**。

    每个参数"调大/调小各有什么后果"见 ``LATENCY_KNOBS`` 与 ``docs/延迟调节.md``。
    """

    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """VAD 判定"这是人声"的置信度阈值。调低更灵敏（小声也算），但可能把音乐/音效当人声。"""

    min_speech_ms: int = Field(default=250, ge=50, le=3000)
    """短于此长度的声音直接丢弃（滤掉咳嗽、键盘声等瞬态噪声）。"""

    min_silence_ms: int = Field(default=350, ge=100, le=5000)
    """**延迟最大的一颗旋钮**：说话停止后，要静多久才认定"这句话说完了"。

    调小 → 字幕更快出，但句子容易被切成碎片；
    调大 → 断句更完整、翻译上下文更好，但字幕延迟明显增加。
    """

    speech_pad_ms: int = Field(default=120, ge=0, le=1000)
    """在语音段前后各留一点余量，避免首尾字被切掉。调大会略微增加延迟与算力。"""

    max_segment_ms: int = Field(default=8000, ge=1000, le=60000)
    """**强制断句上限**：有人不停地说（或有持续音乐）时，最长憋到多久就强制切一刀。

    调小 → 长句延迟有上限，但可能在词中间被切断；
    调大 → 长句更完整，但连续说话时字幕会长时间不刷新。
    """


# --------------------------------------------------------------------------- #
# 延迟档位（用户可调，且必须让用户看懂每个参数的作用）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatencyKnob:
    """一个影响延迟的参数，附带"调它会发生什么"的说明。UI 直接展示这些文字。"""

    key: str
    label: str
    unit: str
    field: str
    smaller: str
    larger: str
    section: str = "vad"


LATENCY_KNOBS: tuple[LatencyKnob, ...] = (
    LatencyKnob(
        key="min_silence_ms",
        label="断句静音时长",
        unit="ms",
        field="min_silence_ms",
        smaller="字幕更快出来，但一句话容易被切成好几段（翻译会失去上下文）",
        larger="断句更完整、翻译更通顺，但字幕要等更久才出现",
    ),
    LatencyKnob(
        key="max_segment_ms",
        label="最长憋句时间",
        unit="ms",
        field="max_segment_ms",
        smaller="连续说话时字幕更新更勤，但可能在词中间被切断",
        larger="长句更完整，但一直不停说话时字幕会长时间不刷新",
    ),
    LatencyKnob(
        key="speech_pad_ms",
        label="语音段前后留白",
        unit="ms",
        field="speech_pad_ms",
        smaller="延迟略降，但句首句尾的字可能被切掉",
        larger="首尾更完整，延迟与算力略增",
    ),
    LatencyKnob(
        key="min_speech_ms",
        label="最短语音长度",
        unit="ms",
        field="min_speech_ms",
        smaller="能识别更短的气声与短词，但容易把噪声当人声",
        larger="能滤掉咳嗽/键盘等瞬态噪声，但很短的应答词会被丢掉",
    ),
    LatencyKnob(
        key="partial_interval_ms",
        label="中间结果刷新间隔",
        unit="ms",
        field="partial_interval_ms",
        section="asr",
        smaller="字幕更跟手、滚动更顺，但 CPU 占用略增（仅流式引擎有效）",
        larger="更省 CPU，但字幕会一跳一跳地更新",
    ),
)


LATENCY_PRESETS: dict[str, dict[str, int]] = {
    "realtime": {
        "min_silence_ms": 200,
        "max_segment_ms": 4000,
        "speech_pad_ms": 60,
        "min_speech_ms": 200,
        "partial_interval_ms": 150,
    },
    "balanced": {
        "min_silence_ms": 350,
        "max_segment_ms": 8000,
        "speech_pad_ms": 120,
        "min_speech_ms": 250,
        "partial_interval_ms": 200,
    },
    "accurate": {
        "min_silence_ms": 600,
        "max_segment_ms": 15000,
        "speech_pad_ms": 200,
        "min_speech_ms": 300,
        "partial_interval_ms": 300,
    },
}


# 支持的源语言（"auto" = 自动识别；"zh-en" = 中文里夹英文词）
SUPPORTED_LANGUAGES = (
    "auto", "zh", "zh-en", "en", "ja", "ko", "yue",
    "fr", "de", "es", "ru", "it", "pt", "ar", "th", "vi", "id", "tr",
)


class LanguageRoute(BaseModel):
    """某一种语言用哪个引擎、哪个模型。

    用户可覆盖——这是"多语言"能落地的前提：不同语言的可用模型差别很大
    （例如日语没有流式模型，只能走分块引擎，见计划书 12.3）。
    """

    engine: Literal[
        "sherpa_stream", "sherpa_offline", "whispercpp", "faster_whisper", "livecaptions"
    ] = "sherpa_stream"
    model: str = ""
    """模型标识（相对 data/models 的目录名或注册表键），留空表示用该引擎默认值。"""

    streaming: bool = True
    """该路线是否真流式。False 表示分块识别（延迟更高）。"""


def _default_routing() -> dict[str, LanguageRoute]:
    """默认语言路由（依据 sherpa-onnx 官方模型可用性，见计划书 12.2）。

    实测结论：官方**没有日语流式模型**，所以 ja 只能走分块引擎；
    韩语/粤语与日语共用同一个 SenseVoice 模型（一个模型覆盖 5 种语言）。
    """
    sense = "sense-voice-zh-en-ja-ko-yue-int8"
    return {
        "zh": LanguageRoute(engine="sherpa_stream", model="streaming-zipformer-zh-int8", streaming=True),
        "zh-en": LanguageRoute(engine="sherpa_stream", model="streaming-zipformer-bilingual-zh-en", streaming=True),
        "en": LanguageRoute(engine="sherpa_stream", model="streaming-zipformer-en", streaming=True),
        "ja": LanguageRoute(engine="sherpa_offline", model=sense, streaming=False),
        "ko": LanguageRoute(engine="sherpa_offline", model=sense, streaming=False),
        "yue": LanguageRoute(engine="sherpa_offline", model=sense, streaming=False),
        # 通配：小语种兜底走 Whisper（99 语言）
        "*": LanguageRoute(engine="whispercpp", model="whisper-turbo-int8", streaming=False),
    }


class ASRConfig(BaseModel):
    engine: Literal[
        "sherpa_stream",      # 流式：低延迟（中/英/韩/法有官方流式模型）
        "sherpa_offline",     # 分块：SenseVoice（中日英韩粤一个模型）/ Paraformer
        "whispercpp",         # 分块：Whisper 多语言兜底，Vulkan/CUDA
        "faster_whisper",     # 仅 NVIDIA
        "livecaptions",       # 零安装兜底
    ] = "sherpa_stream"

    preset: Literal["auto", "low", "mid", "high", "custom"] = "auto"

    language: str = "auto"
    """源语言：``auto`` 或 SUPPORTED_LANGUAGES 之一。

    设成具体语言会跳过语种识别（更快、更准）；``auto`` 会启用 LID。"""

    language_detection: bool = True
    """``language == "auto"`` 时是否启用语种识别（Whisper-tiny LID）。"""

    lid_model_dir: str = ""
    """语种识别模型目录，留空用内置默认位置。"""

    language_switch_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """LID 置信度低于此值时不切换语言，避免在句中断句处来回跳。"""

    routing: dict[str, LanguageRoute] = Field(default_factory=_default_routing)
    """语言 → 引擎/模型 路由表，用户可覆盖。键 ``"*"`` 为兜底。

    ⚠️ 注意：pydantic **不会**校验 ``routing["ja"] = {...}`` 这种**字典项赋值**，
    那样会留下一个裸 dict，之后访问 ``.engine`` 就会 AttributeError。
    要改路由请用 :meth:`set_route`，或整体用 ``model_validate`` 重新载入。
    """

    @field_validator("routing", mode="before")
    @classmethod
    def _coerce_routing(cls, value: Any) -> Any:
        """把 ``{"zh": {...}}`` 这种裸 dict 统一转成 LanguageRoute。"""
        if isinstance(value, dict):
            return {
                k: (v if isinstance(v, LanguageRoute) else LanguageRoute.model_validate(v))
                for k, v in value.items()
            }
        return value

    def set_route(self, language: str, route: LanguageRoute | dict[str, Any]) -> None:
        """安全地修改一条语言路由（会走校验）。"""
        self.routing[language] = (
            route if isinstance(route, LanguageRoute) else LanguageRoute.model_validate(route)
        )

    def route_for(self, language: str) -> LanguageRoute:
        """取某语言的路由，找不到就用 ``"*"`` 兜底。"""
        if language in self.routing:
            return self.routing[language]
        if "*" in self.routing:
            return self.routing["*"]
        return LanguageRoute(engine=self.engine, model="", streaming=(self.engine == "sherpa_stream"))

    fallback_engine: Literal["sherpa_offline", "whispercpp", "none"] = "sherpa_offline"
    """目标语言模型缺失/加载失败时的降级方向。"""

    model_dir: str = ""
    """留空 = 使用 data/models/<engine>/。"""

    num_threads: int = Field(default=0, ge=0, le=64)
    """0 = 自动（物理核数的一半，给游戏留资源）。"""

    provider: Literal["auto", "cpu", "cuda", "directml", "vulkan"] = "auto"
    partial_interval_ms: int = Field(default=200, ge=50, le=2000)
    """中间结果刷新间隔，越小字幕越"跳动"但越及时。"""

    latency_preset: Literal["realtime", "balanced", "accurate", "custom"] = "balanced"
    """延迟档位：``realtime`` 最低延迟 / ``balanced`` 平衡 / ``accurate`` 最准。

    选档会一次性写入 ``vad`` 与 ``partial_interval_ms``；
    用户手动改过任一参数后应设为 ``custom``（:meth:`apply_latency_preset` 会自动处理）。
    """

    def apply_latency_preset(self, preset: str) -> None:
        """把档位值写到具体参数上。"""
        if preset not in LATENCY_PRESETS:
            raise ValueError(f"未知延迟档位: {preset}（可选 {list(LATENCY_PRESETS)}）")
        values = LATENCY_PRESETS[preset]
        self.vad.min_silence_ms = values["min_silence_ms"]
        self.vad.max_segment_ms = values["max_segment_ms"]
        self.vad.speech_pad_ms = values["speech_pad_ms"]
        self.vad.min_speech_ms = values["min_speech_ms"]
        self.partial_interval_ms = values["partial_interval_ms"]
        self.latency_preset = preset  # type: ignore[assignment]

    def detect_preset(self) -> str:
        """反查当前参数更接近哪个档位（参数被手改过则返回 ``custom``）。"""
        for name, values in LATENCY_PRESETS.items():
            if (
                self.vad.min_silence_ms == values["min_silence_ms"]
                and self.vad.max_segment_ms == values["max_segment_ms"]
                and self.vad.speech_pad_ms == values["speech_pad_ms"]
                and self.vad.min_speech_ms == values["min_speech_ms"]
                and self.partial_interval_ms == values["partial_interval_ms"]
            ):
                return name
        return "custom"

    @staticmethod
    def latency_knobs() -> tuple[LatencyKnob, ...]:
        """供 UI 渲染"这个滑杆是干什么的"。"""
        return LATENCY_KNOBS

    vad: VADConfig = Field(default_factory=VADConfig)


# --------------------------------------------------------------------------- #
# 翻译
# --------------------------------------------------------------------------- #
class LLMConfig(BaseModel):
    """OpenAI 兼容通道——一套代码通吃 OpenAI / DeepSeek / 通义 / OpenRouter /
    Ollama / LM Studio / llama.cpp server。"""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:1234/v1"  # 默认指向本机 LM Studio
    api_key: str = ""
    model: str = ""
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    timeout_s: float = Field(default=60.0, gt=0)
    stream: bool = True


class TranslateConfig(BaseModel):
    enabled: bool = True
    provider: str = "none"
    """当前生效的翻译通道 id，如 "baidu" / "youdao" / "azure" / "google" /
    "deepl" / "llm" / "none"。"""

    source_language: str = "auto"
    """源语言。``auto`` 表示用 ASR 判定的语种（推荐）。

    多语言场景下必须把源语言传给传统翻译 API——
    百度/有道/微软/DeepL 都支持 ja→zh、ko→zh 等，但部分免费额度
    只接受 ``auto`` 或要求英中转，需在适配器里降级。"""

    target_language: str = "zh"
    display_mode: Literal["source", "target", "bilingual"] = "bilingual"
    prompt_template: str = "subtitle_direct"
    """内置模板名，见 assets/prompts/。"""

    custom_prompt: str = ""
    """非空时覆盖内置模板。"""

    context_lines: int = Field(default=3, ge=0, le=20)
    """把前 N 句原文+译文一起送进 prompt，保证称谓/语气连贯。"""

    glossary: dict[str, str] = Field(default_factory=dict)
    """术语强制映射：人名/地名/功法名。小说场景刚需。"""

    cache_enabled: bool = True
    max_concurrency: int = Field(default=4, ge=1, le=32)
    qps_limit: float = Field(default=5.0, gt=0)
    retry_times: int = Field(default=2, ge=0, le=10)
    stale_drop_s: float = Field(default=8.0, gt=0)
    """译文回来得太晚（已滚出屏幕）就丢弃，避免字幕错位。"""

    daily_char_limit: int = Field(default=0, ge=0)
    """0 = 不限制。超过后停止调用并提示，防止计费失控。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """各传统 API 的凭据与端点，形如 {"baidu": {"app_id": "...", "secret": "..."}}。"""


# --------------------------------------------------------------------------- #
# 悬浮字幕窗
# --------------------------------------------------------------------------- #
class OverlayConfig(BaseModel):
    font_family: str = "Microsoft YaHei UI"
    font_size: int = Field(default=34, ge=8, le=200)
    bold: bool = True

    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = Field(default=3, ge=0, le=12)
    """描边保证在亮/暗游戏画面上都可读。"""

    source_color: str = "#E6E6E6"
    target_color: str = "#FFE066"

    background_color: str = "#000000"
    background_opacity: float = Field(default=0.35, ge=0.0, le=1.0)
    margin_px: int = Field(default=24, ge=0, le=400)

    scroll_mode: Literal["accumulate", "replace", "typewriter", "karaoke"] = "accumulate"
    max_lines: int = Field(default=3, ge=1, le=20)
    line_spacing: float = Field(default=1.15, ge=0.8, le=3.0)
    fade_ms: int = Field(default=180, ge=0, le=2000)

    click_through: bool = True
    always_on_top: bool = True
    topmost_reassert_ms: int = Field(default=2000, ge=200, le=60000)
    """定时重申置顶，对抗游戏/其他软件抢置顶。"""

    show_in_taskbar: bool = False
    lock_position: bool = True
    screen_index: int = -1
    """-1 = 自动跟随游戏所在屏幕；>=0 = 指定第 N 块屏。"""

    position: Literal[
        "bottom-center", "bottom-left", "bottom-right",
        "top-center", "top-left", "top-right", "custom",
    ] = "bottom-center"
    custom_x: int = 0
    custom_y: int = 0

    window_width: int = Field(default=1200, ge=200, le=10000)


# --------------------------------------------------------------------------- #
# 电平表
# --------------------------------------------------------------------------- #
class MeterConfig(BaseModel):
    """电平表（音量条）显示设置。用户要求必须有可见的 RMS/PEAK 指示。"""

    enabled: bool = True
    show_rms: bool = True
    show_peak: bool = True
    show_peak_hold: bool = True
    show_threshold_line: bool = True

    db_min: float = Field(default=-70.0, ge=-120.0, le=-20.0)
    db_max: float = Field(default=0.0, ge=-40.0, le=12.0)

    peak_hold_s: float = Field(default=1.5, ge=0.0, le=10.0)
    attack: float = Field(default=0.6, gt=0.0, le=1.0)
    release: float = Field(default=0.12, gt=0.0, le=1.0)

    width: int = Field(default=460, ge=200, le=4000)
    height: int = Field(default=92, ge=40, le=600)
    opacity: float = Field(default=1.0, ge=0.1, le=1.0)
    click_through: bool = True


# --------------------------------------------------------------------------- #
# 顶层配置
# --------------------------------------------------------------------------- #
class AppConfig(BaseModel):
    version: int = CONFIG_VERSION
    proxy: str = "http://127.0.0.1:2333"
    """网络代理；留空表示直连。pip/模型下载/翻译 API 共用。"""

    first_run_done: bool = False
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    meter: MeterConfig = Field(default_factory=MeterConfig)

    # ---------------- 读写 ---------------- #
    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """读取配置。文件缺失/损坏时返回默认值，绝不抛异常打断启动。"""
        path = path or paths.CONFIG_FILE
        if not path.exists():
            log.info("配置文件不存在，使用默认配置：%s", path)
            return cls()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            backup = path.with_suffix(f".broken-{int(time.time())}.json")
            try:
                shutil.copy2(path, backup)
            except OSError:
                backup = None
            log.error("配置文件损坏（%s），已备份到 %s，使用默认配置", exc, backup)
            return cls()

        try:
            cfg = cls.model_validate(raw)
        except ValidationError as exc:
            log.error("配置校验失败，使用默认配置：%s", exc)
            return cls()

        if cfg.version != CONFIG_VERSION:
            cfg = _migrate(cfg, raw)
        return cfg

    def save(self, path: Path | None = None) -> None:
        """原子写入，避免写一半断电导致配置损坏。"""
        path = path or paths.CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        data = self.model_dump(mode="json")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ---------------- 导出 ---------------- #
    def redacted(self) -> dict[str, Any]:
        """导出用副本：递归把疑似密钥的字段替换为掩码。"""
        return _redact(self.model_dump(mode="json"))

    def export_to(self, path: Path) -> None:
        """导出可分享的配置（不含任何凭据）。"""
        path.write_text(
            json.dumps(self.redacted(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("已导出脱敏配置到 %s", path)


def _redact(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if any(h in k.lower() for h in _SENSITIVE_HINTS) and isinstance(v, str) and v:
                out[k] = "<redacted>"
            else:
                out[k] = _redact(v, k)
        return out
    if isinstance(value, list):
        return [_redact(v, parent_key) for v in value]
    return value


def _migrate(cfg: AppConfig, raw: dict[str, Any]) -> AppConfig:
    """配置版本迁移钩子。当前只有 v1，保留骨架以便日后升级。"""
    log.warning("配置版本 %s != %s，按最新结构重载", cfg.version, CONFIG_VERSION)
    cfg.version = CONFIG_VERSION
    return cfg
