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

FRAGMENT_MAX_S = 1.2
"""短于这个时长的段落，会被当成"上一句的尾巴"。

实测（``real_ja_2.wav``，11.2s）：VAD 在「日本語をちゃんと聞き取れ」和
「ているかどうかを」之间切了一刀，后一段只有 **1.19 秒**——而它其实是
动词「聞き取れている」的延续部分。用户看到的就是"一句话最后几个字被
单独打了个时间点"。
"""

FRAGMENT_GAP_S = 0.6
"""碎片与前一句的间隔小于这个值才合并（间隔大说明真的是两个句子）。"""

FRAGMENT_LIMIT_S = 8.0
"""合并后总时长上限（和 VAD 的 max_segment_ms 对齐，别合出一个超长句）。"""

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
    speaker: int = -1
    """说话人编号（0 起）；``-1`` = 没做分离或没匹配上。

    编号是**聚类出来的**，同一文件内稳定（S0 始终是同一个人），
    但跨文件不保证（每集独立聚类）。
    """

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
        out.append(Cue(cursor, end, chunk, translations[idx], speaker=cue.speaker))
        cursor = end
    return out


def split_all(cues: list[Cue], max_width: float = DEFAULT_MAX_WIDTH) -> list[Cue]:
    out: list[Cue] = []
    for cue in cues:
        out.extend(split_cue(cue, max_width))
    return out


# --------------------------------------------------------------------------- #
# 碎片合并：把"被切出去的尾巴几个字"并回前一句
# --------------------------------------------------------------------------- #
def _join_text(left: str, right: str) -> str:
    """拼两段文本：中日文直接连（不加空格），英文之间补一个空格。"""
    if not left:
        return right
    if not right:
        return left
    a, b = left[-1], right[0]
    if a.isascii() and a.isalnum() and b.isascii() and b.isalnum():
        return f"{left} {right}"
    return left + right


def merge_fragments(
    cues: list[Cue],
    max_duration_s: float = FRAGMENT_MAX_S,
    max_gap_s: float = FRAGMENT_GAP_S,
    limit_s: float = FRAGMENT_LIMIT_S,
) -> list[Cue]:
    """把"很短 + 紧挨着前一句"的段落并回前一句。

    解决的是**VAD 在词中间切一刀**的问题：实测 ``real_ja_2.wav`` 里
    「日本語をちゃんと聞き取れ」和「ているかどうかを」（只有 1.19s、间隔
    0.41s）被拆成两条字幕——用户看到的就是"一句话最后几个字被单独打了个
    时间点"。合并后原文变回完整一句，翻译也能拿到完整谓语。

    为什么用"短 + 间隔小"而不是语法规则（比如"以て/た开头就并"）：

    * 日语「て/で/た」开头的独立句并不少见（「で、どうする？」「ただ…」）；
    * 反而"短 + 紧邻"是纯粹的声学事实，跨语言都成立，也不用维护词表。

    纯文本拼接：两段是分别识别的，拼起来就是完整句子（不需要重跑识别）。

    Args:
        cues: 按时间排好序的字幕。
        max_duration_s: 短于这个时长才算"碎片"。
        max_gap_s: 与前一句的间隔小于这个值才合并。
        limit_s: 合并后总时长上限，防止把一句话合到离谱长。

    Returns:
        新的列表（不修改入参）。
    """
    if not cues:
        return []

    out: list[Cue] = []
    for cue in cues:
        if out:
            prev = out[-1]
            is_fragment = cue.end_s - cue.start_s <= max_duration_s
            is_close = cue.start_s - prev.end_s <= max_gap_s
            fits = cue.end_s - prev.start_s <= limit_s
            if is_fragment and is_close and fits:
                out[-1] = Cue(
                    start_s=prev.start_s,
                    end_s=max(prev.end_s, cue.end_s),
                    text=_join_text(prev.text, cue.text),
                    translation=prev.translation,
                    speaker=prev.speaker,
                )
                continue
        out.append(cue)
    return out
