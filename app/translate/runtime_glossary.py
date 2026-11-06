"""单次任务的临时术语表（"边翻译边总结"）。

## 为什么需要

长字幕（一部电影 800~1500 条）会切成很多批送翻译。前文只能带几十到几百行，
**前面出现的人名一旦滚出前文，后面就可能换个译法**——同一个人一会儿叫"林凡"、
一会儿叫"林繁"。

用户已经可以手填术语表，但没人愿意为一部片子手工整理人名。所以这里做的是：
**任务进行中，从已经识别出的原文里自动找出"像是专名"的词，问模型一次译名，
再注入后续批次**。

## 为什么不是"每批都让模型顺便输出"

用户最初的设想是"翻译时顺带编制术语表"。实现上刻意分开成两次调用，原因：

1. **污染译文解析**：现在批量输出是「1. 译文」逐行编号。在同一个回答里再塞
   一段术语表，等于给解析器加一个必须容错的岔路——一旦模型把格式写歪，
   整批译文就废了。而译文是主产物，术语表只是锦上添花，不该让前者为后者冒险。
2. **成本差一个量级**：候选词靠**程序**从原文里统计（n-gram 频次，零 token），
   只有"这些词的译名"要问模型。每批都问 = 请求数翻倍；只在有新候选时问一次，
   整部片子通常也就 2~5 次。

所以流程是：**程序找候选 → 模型只给译名 → 注入后续批次**。

## 生命周期

**只在一次任务内存在**：`SubtitleJob._translate()` 里创建，函数返回即丢弃，
不落盘、不进缓存、不影响下一次任务。用户要的就是这个"临时"语义。

## 为什么不直接信任模型给的译名

猜错的译名比没有更糟——它会**污染后面所有批次**（这是项目里踩过的真实坑：
提示词垃圾进了上下文之后，后续请求全被带歪）。

所以有三道闸：

1. 候选必须**在原文里出现至少 min_count 次**（只出现一次的多半是噪音）；
2. 模型给回来的译名要过 :func:`_acceptable`（长度合理、不像解释、不像没翻译）；
3. **永不覆盖用户手填的术语表**，用户表优先。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.utils.log import get_logger

log = get_logger(__name__)

MIN_COUNT = 3
"""候选词至少出现这么多次才值得问（只出现一次的多半是识别噪音）。"""

MAX_CANDIDATES = 40
"""一次最多问多少个候选（问太多模型会开始瞎编）。"""

MAX_ENTRIES = 300
"""临时表最多存多少条（防止极端长片把内存和注入量撑爆）。"""

# ---- 候选提取规则 -------------------------------------------------------- #
#
# ⚠️ 这里**不能**用通用的 n-gram 频次统计（``Glossary.auto_extract_candidates``
# 那套，它是为中文新词发现写的）。实测在日语上会得到一堆碎片：
# ``ンファン`` / ``ファンは`` / ``の青銅の`` / ``鍵を強く`` ——因为日语人名后面
# 紧贴助词，按字符滑窗切必然跨词边界。把这些碎片注进术语表，模型会把
# 「ファンは」整体替换掉，直接破坏句子。
#
# 改用"按字符类别整段抓"：
#   * **片假名串**：日语的人名/外来语/招式几乎都是片假名，且平假名（助词）
#     不在这个区间里 —— 所以 `リンファンは` 天然被切成 `リンファン`。
#   * **英文专名**：连续的大写开头词（`Lin Fan` / `Bronze Key`），但排除冠词
#     与常见连词开头的那一段，免得把 `The` 也抓进来。
#
# ⚠️ **中文不做自动提取**。汉字没有"片假名=人名"这种字符级线索，2~3 字滑窗
# 一定会切出跨词碎片（实测：`林凡握`、`紧了手`、`里的青`、`林凡低`），
# 而碎片和真名的频次是同一个量级，靠阈值分不开。把这些灌进术语表会直接破坏
# 句子。中文场景请用手填术语表（设置 → 翻译 → 术语表），那里你能一眼看出对错。
#
# **日语可以抓汉字词**（`青銅`、`駅前`、`定食屋`）——因为日语里汉字串总被假名
# 隔开，边界是天然的。所以汉字规则只对"非中文源语言"启用（见 ``allow_han``）。
_KATAKANA_RE = re.compile(r"[\u30A0-\u30FF\u30FC]{2,12}")
_LATIN_RE = re.compile(
    r"\b(?!(?:The|A|An|And|But|Chapter)\b)(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\b"
)
_HAN_RE = re.compile(r"[\u4E00-\u9FFF]{2,4}")

_ZH_LANGUAGES = frozenset({"zh", "zh-en", "yue", "auto", ""})
"""这些源语言下**不启用**汉字候选（中文没有词边界；auto 时不知道语言，保守跳过）。"""

_LOOKS_LIKE_EXPLANATION = ("→", "->", "：", ":", "，", "。", "（", "(", "的", "是")
"""译名里出现这些，多半是模型在解释而不是给译名。"""


def _acceptable(source: str, target: str) -> bool:
    """这个「原文 → 译名」能不能收。

    宁可少收几条，也不要把错的收进来——错的会污染后面所有批次。
    """
    src = (source or "").strip()
    dst = (target or "").strip()
    if not src or not dst:
        return False
    if len(src) < 2 or len(src) > 20:
        return False
    if len(dst) < 1 or len(dst) > 40:
        return False
    if src == dst:
        return False          # 没翻
    if "\n" in dst or "\t" in dst:
        return False
    if any(mark in dst for mark in _LOOKS_LIKE_EXPLANATION):
        return False          # "林凡（人名）" 这种
    return True


@dataclass
class RuntimeGlossary:
    """一次任务里的临时术语表。**任务结束即丢，不落盘。**

    Attributes:
        entries: 自动学到的 原文 → 译名。
        asked: 已经问过译名的候选（避免重复问）。
        user_entries: 用户手填的术语表；**优先级永远高于自动学到的**。
    """

    entries: dict[str, str] = field(default_factory=dict)
    asked: set[str] = field(default_factory=set)
    user_entries: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.as_dict())

    def as_dict(self) -> dict[str, str]:
        """给注入用的合并表：**用户表优先**，自动学到的只补空缺。"""
        merged = dict(self.entries)
        merged.update(self.user_entries)
        return merged

    def has_user_entry(self, source: str) -> bool:
        return source in self.user_entries

    # ------------------------------------------------------------------ #
    def candidates(
        self,
        texts: list[str],
        min_count: int = MIN_COUNT,
        top: int = MAX_CANDIDATES,
        source_language: str = "",
    ) -> list[str]:
        """从已识别的原文里找"可能是专名"的词（**纯程序，不花 token**）。

        规则见模块顶部说明：按字符类别整段抓，不做滑窗，这样才不会把日语助词
        粘进人名里。

        Args:
            source_language: 源语言。**中文不启用汉字候选**（没有词边界，
                一定会切出跨词碎片）；日语则可以，因为汉字串天然被假名隔开。

        Returns:
            还没学过、也没问过、且用户表里没有的候选（按频次从高到低）。
        """
        allow_han = (source_language or "").strip().lower() not in _ZH_LANGUAGES

        counter: Counter[str] = Counter()
        for text in texts or []:
            if not text:
                continue
            for m in _KATAKANA_RE.finditer(text):
                counter[m.group()] += 1
            for m in _LATIN_RE.finditer(text):
                counter[m.group()] += 1
            if allow_han:
                for m in _HAN_RE.finditer(text):
                    counter[m.group()] += 1

        items = [(w, c) for w, c in counter.items() if c >= min_count]
        # 长的优先：`リンファン` 与 `リンファ` 同时出现时只留长的那个
        items.sort(key=lambda x: (-len(x[0]), -x[1]))
        kept: list[str] = []
        for word, _count in items:
            if any(word != k and word in k for k in kept):
                continue
            kept.append(word)

        # 再按"频次高→词长"排序，取前 top 个
        kept.sort(key=lambda w: (-counter[w], -len(w)))
        out: list[str] = []
        for word in kept:
            if word in self.entries or word in self.asked:
                continue
            if self.has_user_entry(word):
                continue
            out.append(word)
            if len(out) >= top:
                break
        return out

    # ------------------------------------------------------------------ #
    def learn(self, pairs: dict[str, str]) -> int:
        """把模型给出的译名收进来。

        Returns:
            实际收下的条数（被 :func:`_acceptable` 或用户表挡掉的不算）。
        """
        added = 0
        for src, dst in (pairs or {}).items():
            key = (src or "").strip()
            if not key:
                continue
            self.asked.add(key)
            if self.has_user_entry(key):
                continue          # 用户表优先，不覆盖
            if key in self.entries:
                continue          # 已经学过（不因为重复出现就改译名）
            if len(self.entries) >= MAX_ENTRIES:
                continue
            if not _acceptable(key, dst):
                log.debug("跳过不可信的专名候选：%r → %r", key, dst)
                continue
            self.entries[key] = (dst or "").strip()
            added += 1
        if added:
            log.info("临时术语表 +%d 条（累计 %d 条）", added, len(self.entries))
        return added

    def mark_asked(self, words: list[str]) -> None:
        """记下"这些词问过了但模型没给/给的不合格"，避免反复问。"""
        for w in words:
            self.asked.add(w)

    # ------------------------------------------------------------------ #
    def describe(self, limit: int = 8) -> str:
        """给日志/UI 看的摘要。"""
        if not self.entries:
            return "（无）"
        items = list(self.entries.items())[:limit]
        text = "、".join(f"{k}→{v}" for k, v in items)
        if len(self.entries) > limit:
            text += f" 等 {len(self.entries)} 条"
        return text
