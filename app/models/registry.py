"""模型注册表：本程序需要哪些模型、从哪下、多大、支持什么语言。

所有条目都用 **HF API 核实过仓库真实存在**，文件名为仓库里的真实文件名、
字节数为 API 返回的精确值（不是估算）——这样下载器才能做完整性与进度校验。

⚠️ 易踩的坑（已核实）：
  - ``sherpa-onnx-whisper-turbo`` 里的 ``turbo-encoder.onnx`` 只有 0.74 MB，
    真正的权重在外置 ``turbo-encoder.weights`` 里。
    所以下载器一律用**自包含的 int8 版本**，否则会拿到一个跑不起来的模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelKind = Literal["asr_stream", "asr_offline", "lid", "vad"]

HF_HOST = "https://huggingface.co"
HF_MIRROR = "https://hf-mirror.com"


@dataclass(frozen=True)
class ModelFile:
    """模型仓库里的单个文件。"""

    name: str
    size: int
    """精确字节数，用于校验下载完整性。"""

    direct_url: str = ""
    """非 HF 来源时直接给 URL（如 GitHub release）。"""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    kind: ModelKind
    engine: str
    """对应 ASRConfig.engine 的取值；lid / vad 不是识别引擎。"""

    factory: str = ""
    """加载这个模型用 sherpa-onnx 的哪个工厂函数（``from_<factory>``）。

    为什么单独一个字段：模型族（zipformer / SenseVoice / Dolphin / NeMo / Whisper…）
    的文件结构和加载函数都不一样，而「引擎大类」（流式还是分块）只有两种。
    用 ``engine`` 表达模型族会逼着 ``LanguageRoute.engine`` 的 Literal 一直加值，
    所以这里把"用哪个工厂"交给注册表，引擎按它分发。
    """

    languages: tuple[str, ...] = ()
    repo: str = ""
    files: tuple[ModelFile, ...] = ()
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6

    @property
    def is_hf(self) -> bool:
        return bool(self.repo)

    def url_for(self, filename: str, mirror: bool = False) -> str:
        """取某个文件的下载地址。"""
        for f in self.files:
            if f.name == filename:
                if f.direct_url:
                    return f.direct_url
                break
        host = HF_MIRROR if mirror else HF_HOST
        return f"{host}/{self.repo}/resolve/main/{filename}"

    def file_spec(self, filename: str) -> ModelFile | None:
        for f in self.files:
            if f.name == filename:
                return f
        return None

    @property
    def onnx_files(self) -> tuple[ModelFile, ...]:
        return tuple(f for f in self.files if f.name.endswith(".onnx"))

    @property
    def tokens_file(self) -> ModelFile | None:
        for f in self.files:
            if "token" in f.name.lower():
                return f
        return None


# --------------------------------------------------------------------------- #
# 模型清单（文件名与字节数均来自 HF API 实测）
# --------------------------------------------------------------------------- #
_HF = "csukuangfj/"

MODELS: dict[str, ModelSpec] = {
    # ---------------- 流式（低延迟，中/英） ----------------
    "zipformer-zh-int8": ModelSpec(
        id="zipformer-zh-int8",
        display_name="流式 zipformer 中文（int8）",
        kind="asr_stream",
        engine="sherpa_stream",
        factory="zipformer_transducer",
        languages=("zh",),
        repo=_HF + "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        files=(
            ModelFile("encoder.int8.onnx", 161141793),
            ModelFile("decoder.onnx", 5165083),
            ModelFile("joiner.int8.onnx", 1033416),
            ModelFile("tokens.txt", 20628),
        ),
        note="中文准确率最好、延迟最低（约 0.3~0.5s）",
    ),
    "zipformer-zh-en-int8": ModelSpec(
        id="zipformer-zh-en-int8",
        display_name="流式 zipformer 中英双语（int8）",
        kind="asr_stream",
        engine="sherpa_stream",
        factory="zipformer_transducer",
        languages=("zh", "zh-en", "en"),
        repo=_HF + "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        files=(
            ModelFile("encoder-epoch-99-avg-1.int8.onnx", 181895032),
            ModelFile("decoder-epoch-99-avg-1.int8.onnx", 13091040),
            ModelFile("joiner-epoch-99-avg-1.int8.onnx", 3228404),
            ModelFile("tokens.txt", 56317),
        ),
        note="中文里夹英文词（人名/术语）时比纯中文模型稳",
    ),
    "zipformer-en-int8": ModelSpec(
        id="zipformer-en-int8",
        display_name="流式 zipformer 英文（int8）",
        kind="asr_stream",
        engine="sherpa_stream",
        factory="zipformer_transducer",
        languages=("en",),
        repo=_HF + "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        files=(
            ModelFile("encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx", 71082637),
            ModelFile("decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx", 1307236),
            ModelFile("joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx", 259335),
            ModelFile("tokens.txt", 5048),
        ),
        note="英文有声书/播客",
    ),
    # ---------------- 分块（各语言专用，2025~2026 新模型） ----------------
    # 2026-09 按 ASR 榜单线索核实并**实测对拍**后加进来的候选（见 docs/P2-ASR实测.md）：
    # 结论是"新的不等于更好"——日语实测 Parakeet-ja（88.1% TTS / 79.7% 真实录音）
    # 反而不如原来的 SenseVoice（94.8% / 87.9%），所以默认路由仍留在 SenseVoice，
    # 这几个新模型作为**可在设置里切换的候选**保留。
    "sensevoice-int8": ModelSpec(
        id="sensevoice-int8",
        display_name="SenseVoice Small（中英日韩粤，int8）",
        kind="asr_offline",
        engine="sherpa_offline",
        factory="sense_voice",
        languages=("zh", "en", "ja", "ko", "yue"),
        repo=_HF + "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        files=(
            ModelFile("model.int8.onnx", 239233841),
            ModelFile("tokens.txt", 315894),
        ),
        note=(
            "日语/韩语/粤语的**实测最优**（2026-09 对拍：日语 TTS 94.8%、真实录音 87.9%，"
            "均高于 NVIDIA Parakeet-ja 与 Dolphin），一个模型覆盖 5 种语言、自带标点。"
            "⚠️ 不要换成 2025-09-09 那个版本：实测它对日语完全失效"
            "（输出恒为粤语解码），详见 docs/P2-ASR实测.md"
        ),
    ),
    "parakeet-ja-int8": ModelSpec(
        id="parakeet-ja-int8",
        display_name="NVIDIA Parakeet TDT-CTC 0.6B 日语（int8）",
        kind="asr_offline",
        engine="sherpa_offline",
        factory="nemo_ctc",
        languages=("ja",),
        repo=_HF + "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
        files=(
            ModelFile("model.int8.onnx", 655542604),
            ModelFile("tokens.txt", 28557),
        ),
        note=(
            "日语**专用**模型（NVIDIA NeMo，走 CTC 头），官方卡 CER 6.4~13.2%。"
            "比 SenseVoice 稳：不会出现「一个模型管五种语言」带来的语言串味"
            "（实测 SenseVoice 2025-09-09 版日语直接失效）。"
        ),
    ),
    "dolphin-base-ctc-int8": ModelSpec(
        id="dolphin-base-ctc-int8",
        display_name="Dolphin base（40 种亚洲语言 + 中国方言，int8）",
        kind="asr_offline",
        engine="sherpa_offline",
        factory="dolphin_ctc",
        languages=(
            "zh", "yue", "ja", "ko", "th", "vi", "id", "ms", "fil", "hi",
            "ta", "te", "ur", "ar", "tr", "fa",
        ),
        repo=_HF + "sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
        files=(
            ModelFile("model.int8.onnx", 103729802),
            ModelFile("tokens.txt", 504662),
        ),
        note=(
            "只有 99MB 却覆盖东亚/南亚/东南亚/中东 40 种语言 + 22 种中国方言"
            "（DataoceanAI Dolphin）。韩语/粤语/泰语/越南语这类「没有官方流式模型」的语言"
            "用它比原来的 SenseVoice 更合适。"
        ),
    ),
    "omnilingual-300m-ctc-int8": ModelSpec(
        id="omnilingual-300m-ctc-int8",
        display_name="Omnilingual ASR 300M（1600 语言，int8）",
        kind="asr_offline",
        engine="sherpa_offline",
        factory="omnilingual_asr_ctc",
        languages=("*",),
        repo=_HF + "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
        files=(
            ModelFile("model.int8.onnx", 365352120),
            ModelFile("tokens.txt", 86423),
        ),
        note=(
            "小语种兜底：**1600 种语言**、只有 348MB（Whisper turbo 是 1.0GB）。"
            "Whisper turbo 仍保留作最后一道兜底，但默认兜底换成它。"
        ),
    ),
    "fire-red-asr2-ctc-zh_en-int8": ModelSpec(
        id="fire-red-asr2-ctc-zh_en-int8",
        display_name="FireRedASR2 CTC（中英，int8）",
        kind="asr_offline",
        engine="sherpa_offline",
        factory="fire_red_asr_ctc",
        languages=("zh", "zh-en", "en"),
        repo="csukuangfj2/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
        files=(
            ModelFile("model.int8.onnx", 775861420),
            ModelFile("tokens.txt", 79172),
        ),
        note=(
            "中文「最准档」（2026-02 的 FireRedASR2 CTC 版）：分块识别，延迟比流式高，"
            "但中文/中英混说的准确率比流式模型更高。日常用流式 zipformer 就够。"
        ),
    ),
    # ---------------- 兜底（保留） ----------------
    "whisper-turbo-int8": ModelSpec(
        id="whisper-turbo-int8",
        display_name="Whisper large-v3-turbo（99 语言，int8）",
        kind="asr_offline",
        engine="whispercpp",
        factory="whisper",
        languages=("*",),
        repo=_HF + "sherpa-onnx-whisper-turbo",
        files=(
            ModelFile("turbo-encoder.int8.onnx", 674716297),
            ModelFile("turbo-decoder.int8.onnx", 361080764),
            ModelFile("turbo-tokens.txt", 816730),
        ),
        note=(
            "**保留的最后一道兜底**（用户要求）：99 语言、任何模型都加载失败时还有它。"
            "注意实测日语只有 75.5% 字准率，日常别拿它当主力。"
        ),
    ),
    # ---------------- 语种识别 ----------------
    "whisper-tiny-lid": ModelSpec(
        id="whisper-tiny-lid",
        display_name="语种识别（Whisper tiny，int8）",
        kind="lid",
        engine="lid",
        factory="lid",
        languages=("*",),
        repo=_HF + "sherpa-onnx-whisper-tiny",
        files=(
            ModelFile("tiny-encoder.int8.onnx", 12937772),
            ModelFile("tiny-decoder.int8.onnx", 89855401),
            ModelFile("tiny-tokens.txt", 816730),
        ),
        note="language=auto 时用来判断当前在说什么语言",
    ),
    # ---------------- VAD（必装） ----------------
    "silero-vad": ModelSpec(
        id="silero-vad",
        display_name="Silero VAD（语音活动检测）",
        kind="vad",
        engine="vad",
        languages=("*",),
        repo="",
        files=(
            ModelFile(
                "silero_vad.onnx",
                643854,
                direct_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "asr-models/silero_vad.onnx",
            ),
        ),
        note="必装：静音段跳过识别，低端机 CPU 占用能降一个量级",
    ),
}


# --------------------------------------------------------------------------- #
# 语言包：首次运行向导里给用户勾的粒度
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LanguagePack:
    id: str
    display_name: str
    description: str
    model_ids: tuple[str, ...]
    recommended: bool = False

    @property
    def total_bytes(self) -> int:
        return sum(MODELS[m].total_bytes for m in self.model_ids if m in MODELS)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6


PACKS: dict[str, LanguagePack] = {
    "core": LanguagePack(
        id="core",
        display_name="核心（必装）",
        description="VAD 语音活动检测——没有它静音段也会拿去识别，白白烧 CPU",
        model_ids=("silero-vad",),
        recommended=True,
    ),
    "zh": LanguagePack(
        id="zh",
        display_name="中文（推荐）",
        description="流式识别，延迟最低（实测 97.3% 字准率），日常听书就用它",
        model_ids=("zipformer-zh-int8",),
        recommended=True,
    ),
    "zh-en": LanguagePack(
        id="zh-en",
        display_name="中英双语（推荐）",
        description="中文里夹英文词（人名、术语、品牌）时更稳",
        model_ids=("zipformer-zh-en-int8",),
        recommended=True,
    ),
    "en": LanguagePack(
        id="en",
        display_name="英文",
        description="英文有声书 / 播客（流式，延迟最低）",
        model_ids=("zipformer-en-int8",),
    ),
    "ja-ko-yue": LanguagePack(
        id="ja-ko-yue",
        display_name="日 / 韩 / 粤（推荐）",
        description="SenseVoice：2026-09 实测**日语最优**（TTS 94.8% / 真实录音 87.9%）",
        model_ids=("sensevoice-int8",),
        recommended=True,
    ),
    "multilingual": LanguagePack(
        id="multilingual",
        display_name="小语种兜底（Whisper turbo，保留）",
        description="99 语言、1.0GB；默认兜底仍是它（用户要求保留；实测英文 97.8%）",
        model_ids=("whisper-turbo-int8",),
        recommended=True,
    ),
    # ---- 以下是 2026-09 新增候选：实测**没有赢过**默认模型，装了可在设置里切换 ----
    "ja-parakeet": LanguagePack(
        id="ja-parakeet",
        display_name="日语候选：NVIDIA Parakeet-ja（可选）",
        description="日语专用 CTC、626MB；实测 88.1% TTS / 79.7% 真实录音，低于 SenseVoice",
        model_ids=("parakeet-ja-int8",),
    ),
    "dolphin": LanguagePack(
        id="dolphin",
        display_name="亚洲语言候选：Dolphin（可选）",
        description="40 种亚洲语言 + 22 种中国方言、只有 99MB；实测日语 83.9% / 63.1%",
        model_ids=("dolphin-base-ctc-int8",),
    ),
    "omnilingual": LanguagePack(
        id="omnilingual",
        display_name="超多语言候选：Omnilingual 1600 语言（可选）",
        description="1600 种语言、348MB；本机没有小语种音频，暂未实测",
        model_ids=("omnilingual-300m-ctc-int8",),
    ),
    "zh-accurate": LanguagePack(
        id="zh-accurate",
        display_name="中文候选：FireRedASR2 CTC（可选）",
        description="740MB、分块；实测中文 96.2%（低于流式 97.6%），可拿来试方言/嘈杂音",
        model_ids=("fire-red-asr2-ctc-zh_en-int8",),
    ),
    "lid": LanguagePack(
        id="lid",
        display_name="语种自动识别",
        description="让程序自动判断当前语言（多语言合集中途换语言时有用）",
        model_ids=("whisper-tiny-lid",),
        recommended=True,
    ),
}


def default_pack_ids() -> list[str]:
    """首次运行向导的默认勾选。"""
    return [p.id for p in PACKS.values() if p.recommended]


def models_for_packs(pack_ids: list[str]) -> list[str]:
    """展开语言包 → 去重后的模型 id 列表（保持稳定顺序）。"""
    out: list[str] = []
    for pid in pack_ids:
        pack = PACKS.get(pid)
        if not pack:
            continue
        for mid in pack.model_ids:
            if mid not in out:
                out.append(mid)
    return out


def total_bytes_for_packs(pack_ids: list[str]) -> int:
    return sum(MODELS[m].total_bytes for m in models_for_packs(pack_ids) if m in MODELS)


def find_by_engine(engine: str, language: str = "") -> list[ModelSpec]:
    """按引擎（可选再按语言）筛选模型，供语言路由使用。"""
    result = [m for m in MODELS.values() if m.engine == engine]
    if language and language != "auto":
        result = [m for m in result if language in m.languages or "*" in m.languages]
    return result


# --------------------------------------------------------------------------- #
# 「哪些模型能识别哪些语言」—— 界面只让用户从这里选，避免乱下模型、乱用模型
# --------------------------------------------------------------------------- #
ASR_KINDS: tuple[str, ...] = ("asr_stream", "asr_offline")
"""真正能出识别结果的 kind；``lid``（语种识别）与 ``vad`` 不算。"""

LANGUAGE_LABELS: dict[str, str] = {
    "zh": "中文",
    "zh-en": "中英混说",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "yue": "粤语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "it": "意大利语",
    "pt": "葡萄牙语",
    "ar": "阿拉伯语",
    "th": "泰语",
    "vi": "越南语",
    "id": "印尼语",
    "tr": "土耳其语",
    "*": "其它语言（兜底）",
}


def supports_language(model_id: str, language: str) -> bool:
    """这个模型能不能识别这门语言。

    两个条件缺一不可：

    1. ``kind`` 必须是识别模型（``asr_stream`` / ``asr_offline``）——
       语种识别模型（Whisper tiny）与 VAD 虽然写了 ``languages=("*",)``，
       但**喂给它音频不会得到字幕**；
    2. ``languages`` 里有这门语言，或者模型声明了 ``"*"``（Whisper turbo 那种全语言）。

    界面拿它过滤下拉框，路由器拿它兜底——所以手改配置文件也塞不进乱模型。
    """
    spec = MODELS.get(model_id)
    if spec is None or spec.kind not in ASR_KINDS:
        return False
    return language in spec.languages or "*" in spec.languages


def models_for_language(language: str) -> list[ModelSpec]:
    """所有能识别该语言的模型，顺序同注册表（界面就按这个顺序列）。"""
    return [m for m in MODELS.values() if supports_language(m.id, language)]


def route_kwargs_for(model_id: str) -> dict[str, object]:
    """把注册表里的模型翻译成 ``LanguageRoute`` 的字段。

    **引擎类型必须以注册表为准**：流式模型塞进分块引擎会直接加载失败，
    用户在下拉框里选模型时不该还要自己判断「这个模型是流式的吗」。
    """
    spec = MODELS[model_id]
    return {
        "engine": spec.engine,
        "model": spec.id,
        "streaming": spec.kind == "asr_stream",
    }


def pack_for_model(model_id: str) -> str:
    """反查某模型属于哪个语言包（给"点哪个包能下到它"的提示用）。"""
    for pack in PACKS.values():
        if model_id in pack.model_ids:
            return pack.id
    return ""


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1000 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1000
    return f"{num_bytes:.1f} GB"
