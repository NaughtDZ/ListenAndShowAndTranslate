"""识别结果的文本后处理。

为什么必须有：不同引擎的输出格式差异很大，**直接上屏会很难看**。
实测（docs/P2-ASR实测.md）：

- 流式 zipformer 英文：**全大写且词间粘连** —— ``CHAPTER ONETHE NIGHT TRAIN``
- SenseVoice 日文：**字与字之间被插入空格** —— ``誰 も 失望 させ たり``
- 流式 zipformer 中文：无标点，一整片文字

这里做的是"能在不改动模型的前提下修好"的那部分。
修不了的（英文丢空格，如 ``ONETHE``）**不假装能修**——见模块末的说明。
"""

from __future__ import annotations

import re
import unicodedata

# CJK 统一表意文字、日文假名、全角标点
_CJK_RANGES = (
    (0x3000, 0x303F),   # CJK 标点
    (0x3040, 0x309F),   # 平假名
    (0x30A0, 0x30FF),   # 片假名
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xF900, 0xFAFF),   # CJK 兼容表意
    (0xFF00, 0xFFEF),   # 全角字符
)


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _has_lowercase(text: str) -> bool:
    return any(ch.islower() for ch in text)


def _letter_ratio_upper(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


# --------------------------------------------------------------------------- #
# 各语言的具体规则
# --------------------------------------------------------------------------- #
def remove_cjk_spaces(text: str) -> str:
    """删掉 CJK 字符之间的空格（日文/中文模型有时会插空格）。

    只删"两侧都是 CJK"的空格，保留中文与英文之间的空格。
    """
    chars = list(text)
    out: list[str] = []
    for i, ch in enumerate(chars):
        if ch in (" ", "\u3000"):
            prev_cjk = bool(out) and is_cjk(out[-1])
            next_cjk = i + 1 < len(chars) and is_cjk(chars[i + 1])
            if prev_cjk and next_cjk:
                continue  # 丢掉这个空格
        out.append(ch)
    return "".join(out)


# 这些全大写词是缩写/专名，不该被句子化改掉
_KEEP_UPPER = frozenset({
    "OK", "US", "UK", "EU", "AI", "TV", "PC", "CD", "DJ", "DNA", "RNA", "API", "SDK",
    "GPS", "USB", "HDMI", "CPU", "GPU", "RAM", "SSD", "ID", "PDF", "URL", "HTTP",
    "CEO", "GDP", "NBA", "FBI", "CIA", "NASA", "VR", "AR", "XP", "QQ", "RPG", "FPS",
    "MMO", "NPC", "HP", "MP", "BOSS", "CO", "LTD", "INC",
})


def restore_sentence_case(text: str) -> str:
    """把"全大写"转回正常的句子大小写。

    流式英文模型的输出是全大写。只在**几乎全是字母大写**时才动，
    并保留常见缩写（OK / US / AI …），避免把缩写改成 "Ok"。
    """
    if _has_lowercase(text):
        return text
    if _letter_ratio_upper(text) < 0.95:
        return text

    lowered = text.lower()
    # 句首、以及 . ! ? 之后的第一个字母大写
    def _cap(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2).upper()

    lowered = re.sub(r"(^|[.!?]\s+)([a-z])", _cap, lowered)
    # 单独的小写 i → I
    lowered = re.sub(r"\bi\b", "I", lowered)

    # 把缩写还原成全大写。注意必须在句子化**之后**做、且大小写不敏感——
    # 因为"OK"在句子化那一步已经先被改成了"Ok"。
    def _restore(m: re.Match[str]) -> str:
        word = m.group(0)
        return word.upper() if word.upper() in _KEEP_UPPER else word

    return re.sub(r"\b[A-Za-z]+\b", _restore, lowered)


_FULLWIDTH_TO_HALF = str.maketrans({
    "，": "，",  # 中文标点保持
})


def normalize_punctuation(text: str, language: str = "") -> str:
    """统一标点：压缩重复的句末标点、规整省略号、去掉空格乱插。"""
    text = text.replace("...", "…").replace("。。。", "。") if language.startswith("zh") else text
    text = re.sub(r"[。]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[！!]{2,}", "！", text)
    text = re.sub(r"[？?]{2,}", "？", text)
    text = re.sub(r"\s+([，。！？、；：])", r"\1", text)
    text = re.sub(r"([（({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([）)}\]]）)", r"\1", text)
    return text


def collapse_spaces(text: str) -> str:
    """多空格合一，并去掉首尾空白。"""
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def normalize_numbers(text: str, language: str = "") -> str:
    """把全角数字转半角（模型有时会输出全角）。"""
    return unicodedata.normalize("NFKC", text) if language.startswith(("zh", "ja")) else text


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def clean_text(text: str, language: str = "auto") -> str:
    """按语言清理一段识别结果。"""
    if not text:
        return ""
    lang = (language or "auto").lower()
    out = text

    if lang.startswith(("zh", "ja", "yue", "ko")):
        out = remove_cjk_spaces(out)
        out = normalize_numbers(out, lang)
    elif lang.startswith("en"):
        out = restore_sentence_case(out)

    out = collapse_spaces(out)
    out = normalize_punctuation(out, lang)
    return out.strip()


def needs_ending_punctuation(text: str, language: str = "auto") -> bool:
    """判断这句话结尾是不是缺标点（流式引擎几乎总是缺）。"""
    if not text:
        return False
    return text[-1] not in "。！？.!?…，、；："


def append_ending(text: str, language: str = "auto") -> str:
    """给缺标点的句子补一个句末标点（分块引擎一般自带，流式引擎需要）。"""
    if not text or not needs_ending_punctuation(text, language):
        return text
    lang = (language or "auto").lower()
    if lang.startswith("en"):
        return text + "."
    return text + "。"


# --------------------------------------------------------------------------- #
# 明确修不了的情况（不假装能修）
# --------------------------------------------------------------------------- #
KNOWN_LIMITATIONS = """
以下问题本模块**修不了**，不要试图在这里"兜底"：

1. 英文流式模型丢失词间空格（如 "ONETHE" = "ON THE"）。
   要修必须有英文词典做分词，那是另一个量级的工程；
   实务建议：英文优先用 SenseVoice（它有正确的空格与大小写）。

2. 专有名词同音字错（林凡 → 林繁）。
   这是识别层面的问题，要用 hotwords 热词偏置或翻译层术语表解决。

3. 中文流式模型完全没有标点。
   本模块只能"补句末标点"，句内的逗号分类补不出来；
   要完整标点需接标点模型（见 docs/P2-ASR实测.md 待办 #2）。
"""
