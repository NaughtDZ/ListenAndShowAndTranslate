"""翻译的 token 预算：估算、分批、前文裁剪、输出上限推导。

**为什么需要这个模块**：以前"一批送几条"是写死的（离线 8 条、适配器 5 条），
和模型的真实窗口没有任何关系——8k 窗口的本地小模型和 32k 窗口的云端模型
送一样多的东西，前者可能爆窗口，后者在浪费。

现在改成**按 token 预算算**：用户选 4k / 8k / 16k，系统自己决定这一批送多少条、
带多少前文、给输出留多少额度。

## 估算为什么是启发式

不引入 tiktoken（要额外下载词表、给纯 Python 项目加几十 MB 依赖，不划算）。
这里按字符类型估：

* **CJK**（``ord > 0x2E80``，含日文假名、全角标点）：约 **1 token / 字**
* **ASCII**：约 **4 字符 / token**

对 DeepSeek / GPT / Qwen 这类 BPE 词表，这个比例的偏差在 ±20% 以内。
所以最后统一乘 ``SAFETY`` 留余量——**宁可少送几条，也不要因为估少了而爆窗口**。
"""

from __future__ import annotations

from app.translate.base import Segment

CJK_THRESHOLD = 0x2E80
"""超过这个码位的字符按"一个 token"算（CJK 统一表意文字从 0x4E00 起，但这里
把 0x2E80 以上的全角标点、假名都算进来——它们同样占一个 token 左右）。"""

ASCII_CHARS_PER_TOKEN = 4.0

SAFETY = 1.15
"""估算安全系数：乘上去之后，真实 token 数一般会低于估算值。"""

MIN_OUTPUT_TOKENS = 512
"""输出预算下限（再小连几条译文都放不下）。"""

MAX_OUTPUT_TOKENS = 32768
"""输出预算硬上限（大多数模型达不到，超了会被服务端拒）。"""

OUTPUT_RATIO = 0.8
"""输出预算 = 输入预算 × 这个比例。

为什么是 0.8：译文通常和原文差不多长（日→中会短些、中→日会长些），
再按编号/换行的格式开销留一点，0.8 是"够用又不至于闲置"的经验值。
"""


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（**偏保守**，见模块文档）。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if ord(ch) > CJK_THRESHOLD:
            cjk += 1
        else:
            other += 1
    return int(round((cjk + other / ASCII_CHARS_PER_TOKEN) * SAFETY))


def estimate_segments_tokens(segments: list[Segment]) -> int:
    """估算一批字幕**纯原文**的 token 数（不含提示词/前文/术语表）。"""
    return sum(estimate_tokens(seg.text) for seg in segments)


def estimate_context_tokens(context: list[tuple[str, str]] | None) -> int:
    """估算前文（原文+译文对）占用的 token。"""
    if not context:
        return 0
    return sum(
        estimate_tokens(src) + estimate_tokens(dst) + 4 for src, dst in context
    )


def estimate_overhead_tokens(
    template_chars: int = 0,
    glossary: dict[str, str] | None = None,
    system_chars: int = 400,
) -> int:
    """估算提示词骨架的开销（系统提示 + 模板 + 术语表）。

    默认 ``system_chars=400`` 是按内置模板的实际长度取的（``prompts.py``
    里那几段中文指令约 300~500 字）。
    """
    total = estimate_tokens("x" * max(0, system_chars))
    total += estimate_tokens("x" * max(0, template_chars))
    if glossary:
        # 术语表按"原文→译法"成对出现，粗略按每对 3 个 token
        total += 3 * len(glossary) + 16
    return total


def trim_context(
    context: list[tuple[str, str]], budget_tokens: int
) -> list[tuple[str, str]]:
    """按预算从**最新**往回保留前文。

    前文的作用是"保持译名/语气一致"，越近的越重要，所以从尾部往前装，
    装不下就停（而不是从开头切——那会留下最不相关的旧内容）。
    """
    if not context or budget_tokens <= 0:
        return []
    kept: list[tuple[str, str]] = []
    used = 0
    for src, dst in reversed(context):
        cost = estimate_tokens(src) + estimate_tokens(dst) + 4  # 4 是分隔符开销
        if kept and used + cost > budget_tokens:
            break
        kept.append((src, dst))
        used += cost
    kept.reverse()
    return kept


def plan_batches(
    segments: list[Segment],
    budget_tokens: int,
    overhead_tokens: int = 0,
    max_items: int = 0,
) -> list[list[Segment]]:
    """按 token 预算把字幕切成若干批。

    Args:
        segments: 待翻译的字幕（顺序即送出的顺序）。
        budget_tokens: 单次请求的输入预算；``<= 0`` 表示不限（此时只按
            ``max_items`` 切）。
        overhead_tokens: 每批都要付的固定开销（提示词 + 术语表 + 前文）。
            预算再小也要先扣掉它，否则会切出一堆"只有一条但已经爆窗口"的批。
        max_items: 单批条数上限；``<= 0`` 表示不限。

    Returns:
        批的列表。**不会返回空批**；单条超过预算时它自己独占一批
        （总比丢掉好——丢一条就是丢一句字幕）。
    """
    if not segments:
        return []

    cap_items = max_items if max_items > 0 else len(segments)
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    used = overhead_tokens

    for seg in segments:
        cost = estimate_tokens(seg.text)
        over_budget = budget_tokens > 0 and current and used + cost > budget_tokens
        over_items = len(current) >= cap_items
        if over_budget or over_items:
            batches.append(current)
            current = []
            used = overhead_tokens
        current.append(seg)
        used += cost

    if current:
        batches.append(current)
    return batches


def output_tokens_for(
    batch_tokens: int,
    cap: int = 0,
) -> int:
    """推导这一批该给多少输出 token。

    Args:
        batch_tokens: 这一批**输入**的估算 token（原文 + 前文）。
        cap: 硬上限（来自配置）；``<= 0`` 时用 :data:`MAX_OUTPUT_TOKENS`。

    Returns:
        输出 token 数。**最少也会给 256**——输出给得太小会让模型返回空串
        （预算全烧在思考或格式上），那比慢更糟。
    """
    limit = cap if cap > 0 else MAX_OUTPUT_TOKENS
    want = int(batch_tokens * OUTPUT_RATIO)
    return max(256, min(limit, max(MIN_OUTPUT_TOKENS, want)))
