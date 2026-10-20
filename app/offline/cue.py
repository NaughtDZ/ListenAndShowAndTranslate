"""一条离线字幕，以及"把长句切成能看的行"。

为什么必须切行（这是格式要求，不是美化）：

- **LRC 是歌词格式**：一行 = 屏幕上一行。ASR 出来的是整句（十几到几十字），
  直接写进去会在播放器里溢出或被截断。
- SRT / VTT 虽然能放两行，但一行超过约 20 个全角字就影响阅读，翻译后还会变长。

切行策略：**先按句末标点 → 再按软标点 → 最后才硬切**；
时间按**显示宽度**比例线性分配（同一句里语速近似均匀）。

宽度而不是字符数：``こんにちは`` 是 5 个字，``hello`` 也是 5 个字符，
但屏占差一倍。所以按全角=1、半角=0.5 折算。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

SENTENCE_ENDS = "。！？!?…"
SOFT_BREAKS = "，、,；;：:"
CLOSERS = "」』）)】》〉"

# 一行最多多少"全角字宽"
DEFAULT_MAX_WIDTH = 24.0
MIN_WIDTH = 6.0
"""太短的行不收——避免切出一两个字一行。"""


@dataclass
class Cue:
    """一条字幕（一条 = 播放器里的一行/一块）。"""

    start_s: float
    end_s: float
    text: str
    translation: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def text_for(self, mode: str = "source") -> str:
        """按输出模式取文本：``source`` / ``target`` / ``bilingual``。

        没有译文时**一律回退原文**——空着一行比没有字幕更糟。
        """
        if mode == "target":
            return self.translation or self.text
        if mode == "bilingual":
            if self.translation and self.translation != self.text:
                return f"{self.text}\n{self.translation}"
            return self.text
        return self.text


# --------------------------------------------------------------------------- #
# 显示宽度
# --------------------------------------------------------------------------- #
def char_width(ch: str) -> float:
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5


def display_width(text: str) -> float:
    return sum(char_width(c) for c in text)


# --------------------------------------------------------------------------- #
# 切行
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    """第一层：按句末标点切，并把跟在后面的收尾括号带上（``…。」``）。"""
    chars = list(text)
    parts: list[str] = []
    buf = ""
    i = 0
    while i < len(chars):
        buf += chars[i]
        if chars[i] in SENTENCE_ENDS:
            while i + 1 < len(chars) and chars[i + 1] in CLOSERS:
                i += 1
                buf += chars[i]
            parts.append(buf)
            buf = ""
        i += 1
    if buf:
        parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def _soft_split(text: str, max_width: float) -> list[str]:
    """第二层：按软标点贪心切，仍超长就硬切（尽量不切断英文单词）。"""
    out: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        width = display_width(buf)
        if ch in SOFT_BREAKS and width >= max_width * 0.6:
            out.append(buf)
            buf = ""
        elif width >= max_width:
            # 硬切前如果正卡在英文单词中间，退到最近的空格
            if ch.isascii() and ch.isalnum() and " " in buf:
                head, _, tail = buf.rpartition(" ")
                if display_width(head) >= MIN_WIDTH:
                    out.append(head + " ")
                    buf = tail
                    continue
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return [p for p in (s.strip() for s in out) if p]


def split_text(text: str, max_width: float = DEFAULT_MAX_WIDTH) -> list[str]:
    """把一句话切成若干行。"""
    text = (text or "").strip()
    if not text:
        return []
    if display_width(text) <= max_width:
        return [text]

    chunks: list[str] = []
    for sentence in _split_sentences(text):
        if display_width(sentence) <= max_width:
            chunks.append(sentence)
        else:
            chunks.extend(_soft_split(sentence, max_width))
    return chunks or [text]


def slice_by_weights(text: str, weights: list[float]) -> list[str]:
    """按权重把一段文本切成若干份。

    用途：整句译文要跟着"切行后的原文"走——译文长度和原文不成比例，
    只能按原文各行的宽度占比，把译文顺序切开，保证第 i 行译文对应第 i 行原文。
    """
    if not text:
        return [""] * len(weights)
    if len(weights) <= 1:
        return [text]

    total = sum(weights) or 1.0
    out: list[str] = []
    pos = 0
    acc = 0.0
    for weight in weights[:-1]:
        acc += weight
        end = int(round(len(text) * acc / total))
        end = max(pos, min(len(text), end))
        out.append(text[pos:end].strip())
        pos = end
    out.append(text[pos:].strip())
    return out


def split_cue(cue: Cue, max_width: float = DEFAULT_MAX_WIDTH) -> list[Cue]:
    """把一条长字幕切成多条，时间按宽度比例分配，**译文也按同样比例切开**。

    最后一段强制对齐到原来的 ``end_s``，避免浮点累加把尾巴搓掉。
    """
    chunks = split_text(cue.text, max_width)
    if len(chunks) <= 1:
        return [cue]

    widths = [max(0.1, display_width(c)) for c in chunks]
    total = sum(widths)
    span = max(0.0, cue.end_s - cue.start_s)
    translations = slice_by_weights(cue.translation, widths)

    out: list[Cue] = []
    cursor = cue.start_s
    for idx, (chunk, width) in enumerate(zip(chunks, widths)):
        if idx == len(chunks) - 1:
            end = cue.end_s
        else:
            end = cursor + span * (width / total)
        if end <= cursor:
            end = cursor + 0.01
        out.append(Cue(cursor, end, chunk, translations[idx]))
        cursor = end
    return out


def split_all(cues: list[Cue], max_width: float = DEFAULT_MAX_WIDTH) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        out.extend(split_cue(cue, max_width))
    return out
