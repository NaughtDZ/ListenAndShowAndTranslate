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
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

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
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, ge=50, le=3000)
    min_silence_ms: int = Field(default=300, ge=50, le=5000)
    """静音多久判定一句结束——直接决定字幕断句与延迟。"""

    speech_pad_ms: int = Field(default=120, ge=0, le=1000)


class ASRConfig(BaseModel):
    engine: Literal[
        "sherpa_stream",      # P2 首选：低延迟真流式
        "sherpa_offline",     # SenseVoice / Paraformer 分块
        "whispercpp",         # 高端档，Vulkan/CUDA
        "faster_whisper",     # 仅 NVIDIA
        "livecaptions",       # 零安装兜底
    ] = "sherpa_stream"

    preset: Literal["auto", "low", "mid", "high", "custom"] = "auto"
    language: Literal["auto", "zh", "en"] = "auto"
    model_dir: str = ""
    """留空 = 使用 data/models/<engine>/。"""

    num_threads: int = Field(default=0, ge=0, le=64)
    """0 = 自动（物理核数的一半，给游戏留资源）。"""

    provider: Literal["auto", "cpu", "cuda", "directml", "vulkan"] = "auto"
    partial_interval_ms: int = Field(default=200, ge=50, le=2000)
    """中间结果刷新间隔，越小字幕越"跳动"但越及时。"""

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
