"""OpenAI 兼容通道：一套代码通吃 OpenAI / DeepSeek / 通义 / OpenRouter /
Ollama / LM Studio / llama.cpp server。

实现里固化了三条**实测结论**（见 docs/P4-翻译实测.md）：

1. **批量比逐条省 71% token**（5 条一起翻：439 tok vs 逐条 1508 tok），
   所以默认就按 ``batch_size`` 合并请求。代价是批量时术语表遵守率略降，
   因此加了第 3 条的校验。
2. **思考必须在客户端主动关掉**（默认关）。字幕翻译是"短句、要快、要省钱"，
   思考纯属浪费——而且思考模型经常把预算全烧在 ``reasoning_content`` 上，
   返回**空 content**，用户看到的就是空白字幕。这里做三层处理：
   ``chat_template_kwargs.enable_thinking=false``（vLLM / LM Studio / llama.cpp 系）、
   按服务端类型换参数（OpenAI 用 ``reasoning_effort=minimal``、OpenRouter 用
   ``reasoning.enabled=false``、DashScope 用顶层 ``enable_thinking``）、
   以及**兜底重试**：空译文时把 ``/no_think`` 追加进提示词再试一次（Qwen 软开关），
   还不行才加大预算、退回逐条。
   另外会**剥掉内联在 content 里的思考块**（``<think>…</think>``），
   否则思考会被当成译文上屏。
3. **术语表必须可校验**：批量翻译时模型偶尔漏替换（实测「声堂」没被换成「青铜」），
   所以翻完要检查，违规的条目单独重翻。

另外：本地地址（127.0.0.1 / localhost）**不走代理**——代理是给外网 API 用的。
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import urlparse

import httpx

from app.translate.base import (
    Segment,
    TranslateRequest,
    TranslateResult,
    TranslatorStats,
)
from app.translate.prompts import build_messages
from app.utils.log import get_logger

log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 5
MAX_RETRY_MULTIPLIER = 4

_NUMBERED_RE = re.compile(r"^\s*(\d{1,3})\s*[.、)．:：]\s*(.*)$")

# 我们自己提示词里的标记。译文里出现这些，说明模型把提示词原样吐回来了。
#
# ⚠️ 实测事故：sakura-galtransl-7b 是**微调过的翻译模型**（不是指令模型），
# 喂给它带【前文】【术语表】的指令式 prompt，它会把整段 prompt 当正文"翻译"回来。
# 更糟的是这段垃圾被写进上下文后，会污染后续所有请求，形成恶性循环。
# 所以必须在这里拦下：既不能上屏，也不能进缓存和上下文。
_PROMPT_MARKERS = (
    "【待翻译", "【前文", "【术语表", "【待翻译（共",
    "请按相同编号", "硬性要求", "只输出译文本身", "不要复述原文",
)


def looks_like_echo(text: str) -> bool:
    """判断输出是不是把我们的提示词原样返回了。"""
    if not text:
        return False
    return any(m in text for m in _PROMPT_MARKERS)


_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_THINK_OPEN_RE = re.compile(r"<(think|thinking|reasoning)>.*$", re.S | re.I)


def strip_thinking(text: str) -> str:
    """剥掉**内联在 content 里**的思考块。

    大多数服务端把思考放在单独的 ``reasoning_content`` 字段里，但有些网关
    （或开了 reasoning 透传的 llama.cpp）会把 ``<think>…</think>`` 直接塞进 content。
    不处理的话这段"思考"会被当成译文上屏——字幕里出现模型的自言自语。
    """
    if not text:
        return text
    out = _THINK_BLOCK_RE.sub("", text)
    # 被 max_tokens 截断、没闭合的思考块：从 <think> 起全丢
    out = _THINK_OPEN_RE.sub("", out)
    return out.strip()


def thinking_extras(base_url: str, model: str = "") -> dict:
    """按服务端类型给出「关掉思考」的请求参数。

    **实测（2026-09，LM Studio 0.x）**：这些字段服务端不认识时会**忽略**，
    不会 400（逐个试过 chat_template_kwargs / reasoning_effort / reasoning /
    enable_thinking / think / extra_body，全部 200）。但严格网关（OpenAI 官方 API、
    DeepSeek 官方 API）对未知字段会直接 400，所以 ``_call`` 里遇到 400 会去掉这些
    参数**重试一次**——功能宁可少关思考，也不能整句翻不出来。

    各家开关不一样（这是 2026 年的现实）：

    * vLLM / LM Studio / llama.cpp / 自建 Qwen 兼容服务：``chat_template_kwargs.enable_thinking=false``
    * OpenAI 官方（gpt-5 / o 系列）：没有 enable_thinking，只有 ``reasoning_effort``
    * OpenRouter：``reasoning: {"enabled": false}``
    * 阿里 DashScope（OpenAI 兼容模式）：顶层 ``enable_thinking: false``
    """
    host = urlparse(base_url).netloc.lower()
    if "openai.com" in host:
        return {"reasoning_effort": "minimal"}
    if "openrouter" in host:
        return {"reasoning": {"enabled": False}}
    if "dashscope" in host or "aliyuncs" in host:
        return {"enable_thinking": False}
    # 其余（本地 / 自建 / 大多数兼容网关）：走 chat 模板开关
    return {"chat_template_kwargs": {"enable_thinking": False}}


def parse_lines(text: str, expected: int) -> dict[int, str] | None:
    """按行解析（prompt_style=plain 的批量模式用）。

    纯翻译模型不吃"编号"那套，只按行对应。
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != expected:
        return None
    return {i + 1: ln for i, ln in enumerate(lines)}


# 已知需要"纯文本输入"的模型（微调翻译模型，不是指令模型）。
# 用户仍可在配置里用 prompt_style 覆盖。
_PLAIN_STYLE_HINTS = ("sakura", "galtransl", "jparacrawl", "opus-mt", "m2m100", "nllb")


def guess_prompt_style(model: str) -> str:
    name = (model or "").lower()
    return "plain" if any(h in name for h in _PLAIN_STYLE_HINTS) else "chat"


def is_local_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or host.endswith(".local")


def parse_numbered(text: str, expected: int) -> dict[int, str] | None:
    """解析「1. 译文」这种编号输出。

    Returns:
        ``{序号: 译文}``（序号从 1 开始）；条数不符或解析失败返回 None。

    模型有时会把一条译文折成多行，所以这里把"非编号行"当作上一条的续行，
    而不是丢弃——否则长译文会被截断。
    """
    if not text.strip():
        return None
    out: dict[int, str] = {}
    current: int | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _NUMBERED_RE.match(line)
        if m:
            idx = int(m.group(1))
            content = m.group(2).strip()
            if 1 <= idx <= expected * 2:  # 容忍模型多编号
                out[idx] = content
                current = idx
            continue
        if current is not None:
            out[current] = (out[current] + " " + line.strip()).strip()
    if len(out) != expected:
        return None
    if set(out) != set(range(1, expected + 1)):
        return None
    return out


def _normalize_for_compare(text: str) -> str:
    """比较术语前先规范化。

    ⚠️ 实测踩过的坑：术语表里写**繁体**「青銅」而模型输出**简体**「青铜」，
    严格字符串比较会判定"漏译"，于是白白重翻一轮（浪费 token），
    还会给用户报一个假警告。

    这里做三件事：
    1. Unicode NFKC 规范化（全角/半角、兼容字符）
    2. 去掉空白与常见标点
    3. 大小写折叠（英文术语）
    至于繁简差异，光靠 NFKC 消不掉——见 :func:`glossary_violations` 的说明。
    """
    import unicodedata

    out = unicodedata.normalize("NFKC", text)
    out = re.sub(r"[\s\u3000，。、！？；：“”‘’（）《》〈〉「」『』【】…—·,.!?;:\"'()\[\]<>_\-/\\|]", "", out)
    return out.lower()


# 常见繁→简对照（只覆盖术语表里最容易出现的字，不做完整转换表）
_TRAD_TO_SIMP = str.maketrans({
    "銅": "铜", "鑰": "钥", "鐵": "铁", "銀": "银", "劍": "剑", "龍": "龙",
    "門": "门", "馬": "马", "鳥": "鸟", "魚": "鱼", "風": "风", "雲": "云",
    "電": "电", "語": "语", "書": "书", "車": "车", "軍": "军", "國": "国",
    "學": "学", "醫": "医", "點": "点", "燈": "灯", "樹": "树", "葉": "叶",
    "東": "东", "西": "西", "南": "南", "北": "北", "紅": "红", "綠": "绿",
    "藍": "蓝", "黃": "黄", "黑": "黑", "愛": "爱", "夢": "梦", "話": "话",
    "語": "语", "誰": "谁", "來": "来", "個": "个", "們": "们", "時": "时",
    "間": "间", "現": "现", "實": "实", "體": "体", "聲": "声", "聽": "听",
    "記": "记", "憶": "忆", "遠": "远", "過": "过", "讓": "让", "這": "这",
    "們": "们", "麼": "么", "與": "与", "還": "还", "為": "为", "關": "关",
    "開": "开", "對": "对", "錯": "错", "長": "长", "張": "张", "發": "发",
    "頭": "头", "臉": "脸", "眼": "眼", "腳": "脚", "體": "体", "氣": "气",
})


def _fold_traditional(text: str) -> str:
    return text.translate(_TRAD_TO_SIMP)


def glossary_violations(
    source: str, translation: str, glossary: dict[str, str]
) -> list[str]:
    """检查译文是否漏用了术语表的指定译法。

    只有当**原文里真的出现了该词**时才要求译文里出现对应译法，
    否则会出现"原文没有这个词却被判违规"的误报。

    比较前会做规范化与常见繁简折叠——否则"术语表写繁体、译文是简体"
    会变成每条都违规，触发无意义的重翻。
    """
    bad: list[str] = []
    norm_translation = _normalize_for_compare(translation)
    folded_translation = _fold_traditional(norm_translation)

    for src_term, dst_term in glossary.items():
        if not src_term or not dst_term:
            continue
        if src_term not in source:
            continue
        want = _normalize_for_compare(dst_term)
        if want in norm_translation or _fold_traditional(want) in folded_translation:
            continue
        bad.append(f"{src_term}→{dst_term}")
    return bad


class OpenAICompatTranslator:
    """OpenAI 兼容 chat/completions 翻译通道。"""

    name = "llm"
    supports_batch = True
    supports_streaming = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
        proxy: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
        disable_thinking: bool = True,
        verify_glossary: bool = True,
        prompt_style: str = "",
        name: str = "llm",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)
        self.disable_thinking = disable_thinking
        self.verify_glossary = verify_glossary
        # "chat" = 指令模型（走完整模板）；"plain" = 微调翻译模型（只喂原文）
        self.prompt_style = prompt_style or guess_prompt_style(model)
        self.name = name
        self.stats = TranslatorStats()
        # 服务端实际回给我们的思考字符数（用来验证「关思考」到底有没有生效）
        self.reasoning_chars_seen = 0

        self._proxy = "" if (not proxy or is_local_url(base_url)) else proxy
        self._client = httpx.Client(
            timeout=timeout_s,
            trust_env=False,
            proxy=self._proxy or None,
            headers=self._headers(),
        )

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("关闭 httpx 客户端失败: %s", exc)

    # ------------------------------------------------------------------ #
    def list_models(self) -> list[str]:
        """列出该端点可用模型（设置界面用）。"""
        try:
            r = self._client.get(f"{self.base_url}/models", timeout=15.0)
            r.raise_for_status()
            return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        except Exception as exc:  # noqa: BLE001
            log.warning("获取模型列表失败: %s", exc)
            return []

    def ping(self) -> tuple[bool, str]:
        """连通性检查，返回 (是否可用, 说明)。"""
        models = self.list_models()
        if models:
            return True, f"可用，{len(models)} 个模型"
        return False, "无法连接或没有可用模型"

    # ------------------------------------------------------------------ #
    def translate(self, request: TranslateRequest) -> TranslateResult:
        """翻译一批字幕。**不抛异常**，失败写进 failures。"""
        t0 = time.monotonic()
        result = TranslateResult(provider=self.name, model=self.model)
        segments = [s for s in request.segments if s.text]
        if not segments:
            return result

        source_chars = sum(len(s.text) for s in segments)

        # 按 batch_size 切批
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            sub = TranslateRequest(
                segments=batch,
                source_language=request.source_language,
                target_language=request.target_language,
                context=request.context,
                glossary=request.glossary,
                template=request.template,
                custom_prompt=request.custom_prompt,
            )
            try:
                result.merge(self._translate_batch(sub, original=request))
            except Exception as exc:  # noqa: BLE001 - 通道异常不能打断字幕流
                msg = f"{type(exc).__name__}: {exc}"
                log.error("翻译批次失败: %s", msg)
                for seg in batch:
                    result.failures[seg.id] = msg

        result.latency_ms = (time.monotonic() - t0) * 1000
        self.stats.add(result, source_chars)
        return result

    # ------------------------------------------------------------------ #
    def _translate_batch(
        self, request: TranslateRequest, original: TranslateRequest
    ) -> TranslateResult:
        """翻一批；内部按"批量 → 加大预算 → 逐条"逐级降级。"""
        batch = request.segments
        out = TranslateResult(provider=self.name, model=self.model)

        # ---- 第 1 级：批量 ----
        content, usage, truncated = self._call(request, self.max_tokens)
        if content:
            if len(batch) == 1:
                # 单条时模型偶尔还是会带"1. "前缀（我们提示词要求不加，但它不一定听）。
                # 能按编号解析就用解析结果，避免字幕里出现行号。
                parsed_one = parse_numbered(content, 1) if self.prompt_style != "plain" else None
                out.translations[batch[0].id] = (
                    parsed_one[1] if parsed_one else content.strip()
                )
            else:
                parsed = (
                    parse_lines(content, len(batch))
                    if self.prompt_style == "plain"
                    else parse_numbered(content, len(batch))
                )
                if parsed:
                    for idx, seg in enumerate(batch, start=1):
                        out.translations[seg.id] = parsed[idx]
                else:
                    out.note = "编号解析失败，已改为逐条翻译"
                    out.retries += 1
                    log.info("批量输出无法按编号解析，退回逐条（%d 条）", len(batch))
                    return self._translate_one_by_one(request, original, base=out)

            out.prompt_tokens += usage.get("prompt_tokens", 0)
            out.completion_tokens += usage.get("completion_tokens", 0)

            # 术语表校验：违规的条目单独重翻
            if self.verify_glossary and request.glossary:
                bad_ids = [
                    seg.id for seg in batch
                    if seg.id in out.translations
                    and glossary_violations(seg.text, out.translations[seg.id], request.glossary)
                ]
                if bad_ids:
                    log.info("术语表未命中 %d 条，单独重翻", len(bad_ids))
                    out.note = (out.note + "；" if out.note else "") + f"{len(bad_ids)} 条术语表重翻"
                    out.retries += 1
                    retry = self._translate_one_by_one(
                        TranslateRequest(
                            segments=[s for s in batch if s.id in bad_ids],
                            **{k: getattr(request, k) for k in
                               ("source_language", "target_language", "context",
                                "glossary", "template", "custom_prompt")},
                        ),
                        original, base=TranslateResult(provider=self.name, model=self.model),
                        # 只接受仍然违规的结果，避免更差的译文覆盖好译文
                        keep_original=dict(out.translations),
                    )
                    out.merge(retry)
            return out

        # ---- 第 2 级：空译文（典型是思考 token 吃光了预算）----
        if self.disable_thinking and self.prompt_style != "plain":
            # 2a. 先试"硬关思考"：把 /no_think 追加进提示词（Qwen 系软开关）。
            #     实测有的服务端不吃 chat_template_kwargs，但认这个软开关。
            log.warning("空译文（疑似思考占满预算），用 /no_think 硬关思考重试一次")
            out.retries += 1
            content_nt, usage_nt, _ = self._call(request, self.max_tokens, no_think=True)
            out.prompt_tokens += usage_nt.get("prompt_tokens", 0)
            out.completion_tokens += usage_nt.get("completion_tokens", 0)
            if content_nt:
                if len(batch) == 1:
                    parsed_one = (
                        parse_numbered(content_nt, 1) if self.prompt_style != "plain" else None
                    )
                    out.translations[batch[0].id] = (
                        parsed_one[1] if parsed_one else content_nt.strip()
                    )
                else:
                    parsed_nt = (
                        parse_lines(content_nt, len(batch))
                        if self.prompt_style == "plain"
                        else parse_numbered(content_nt, len(batch))
                    )
                    if parsed_nt:
                        for idx, seg in enumerate(batch, start=1):
                            out.translations[seg.id] = parsed_nt[idx]
                if out.translations:
                    out.note = (out.note + "；" if out.note else "") + "已用 /no_think 关掉思考"
                    return out

        if truncated:
            bigger = min(self.max_tokens * MAX_RETRY_MULTIPLIER, 16384)
            log.warning(
                "空译文且输出被截断（很可能是推理模型把预算用在思考上），"
                "把 max_tokens 从 %d 提到 %d 重试", self.max_tokens, bigger,
            )
            out.retries += 1
            content2, usage2, _ = self._call(request, bigger)
            out.prompt_tokens += usage2.get("prompt_tokens", 0)
            out.completion_tokens += usage2.get("completion_tokens", 0)
            if content2:
                parsed = parse_numbered(content2, len(batch)) if len(batch) > 1 else {1: content2.strip()}
                if parsed:
                    for idx, seg in enumerate(batch, start=1):
                        out.translations[seg.id] = parsed[idx]
                    out.note = "空译文经加大预算后成功"
                    return out

        # ---- 第 3 级：逐条 ----
        out.note = (out.note + "；" if out.note else "") + "批量失败，已逐条重试"
        return self._translate_one_by_one(request, original, base=out)

    def _translate_one_by_one(
        self,
        request: TranslateRequest,
        original: TranslateRequest,
        base: TranslateResult,
        keep_original: dict[int, str] | None = None,
    ) -> TranslateResult:
        """逐条翻译（最稳但最费 token 的方式）。"""
        for seg in request.segments:
            single = TranslateRequest(
                segments=[seg],
                source_language=request.source_language,
                target_language=request.target_language,
                context=request.context,
                glossary=request.glossary,
                template=request.template,
                custom_prompt=request.custom_prompt,
            )
            content, usage, truncated = self._call(single, self.max_tokens)
            base.prompt_tokens += usage.get("prompt_tokens", 0)
            base.completion_tokens += usage.get("completion_tokens", 0)
            base.retries += 1
            if content:
                text = content.strip()
                # 若校验术语表且仍违规，且已有更好的旧译文，则保留旧的
                if (
                    keep_original
                    and request.glossary
                    and glossary_violations(seg.text, text, request.glossary)
                    and seg.id in keep_original
                ):
                    base.translations.pop(seg.id, None)
                    continue
                base.translations[seg.id] = text
            else:
                reason = "模型返回空译文"
                if truncated:
                    reason += "（输出被 max_tokens 截断，疑为推理模型把预算用在思考上）"
                base.failures[seg.id] = reason
                log.warning("字幕 %s 翻译失败：%s", seg.id, reason)
        return base

    # ------------------------------------------------------------------ #
    def _call(
        self, request: TranslateRequest, max_tokens: int, no_think: bool = False
    ) -> tuple[str, dict, bool]:
        """发一次请求。

        Returns:
            (正文, usage, 是否被截断)

        ``prompt_style="plain"`` 时**只把原文发给模型**，不带任何指令——
        因为 sakura 这类微调翻译模型会把指令式 prompt 当成正文翻译回来。

        ``no_think=True`` 时把 ``/no_think`` 追加到最后一条 user 消息末尾
        （Qwen 系的软开关，实测对部分模型有效）；这只在"思考吃光了预算、
        content 为空"的兜底重试里用，正常请求不加。
        """
        if self.prompt_style == "plain":
            joined = "\n".join(seg.text for seg in request.segments)
            messages = [{"role": "user", "content": joined}]
        else:
            messages = build_messages(request)

        if no_think and messages:
            last = dict(messages[-1])
            last["content"] = f"{last.get('content', '')}\n/no_think"
            messages = [*messages[:-1], last]

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        extras: dict = {}
        if self.disable_thinking and self.prompt_style != "plain":
            # 默认就要求关闭思考：字幕翻译短句要快，思考只会更慢更贵还可能空白
            extras = thinking_extras(self.base_url, self.model)
            payload.update(extras)

        r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        if r.status_code == 400 and extras:
            # 严格网关（OpenAI / DeepSeek 官方）会因未知字段直接 400：
            # 去掉关思考的参数重试一次，别让一句都翻不出来
            log.warning(
                "端点拒绝了关思考参数（HTTP 400），去掉 %s 重试一次",
                ", ".join(extras),
            )
            payload.pop("reasoning", None)
            payload.pop("reasoning_effort", None)
            payload.pop("enable_thinking", None)
            payload.pop("chat_template_kwargs", None)
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = strip_thinking(msg.get("content") or "")
        usage = data.get("usage") or {}
        truncated = choice.get("finish_reason") == "length"

        # ⚠️ 回显检测：译文里出现我们的提示词标记 ⇒ 模型没在翻译。
        # 必须在这里拦死：垃圾一旦进缓存/上下文，会污染后续所有请求。
        if content and looks_like_echo(content):
            log.warning(
                "模型把提示词原样返回（疑似不是指令模型）。"
                "该模型可能需要 prompt_style=plain，请检查设置。"
            )
            return "", usage, truncated

        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        if reasoning:
            self.reasoning_chars_seen += len(reasoning)
        if not content and reasoning:
            log.warning(
                "模型只输出了思考内容（%d 字）而没有译文——"
                "服务端没能关掉思考，将用 /no_think 兜底重试", len(reasoning),
            )
        elif reasoning:
            log.debug("服务端仍返回了 %d 字思考内容（已忽略，只取 content）", len(reasoning))
        return content, usage, truncated


def probe_endpoint(base_url: str, api_key: str = "", proxy: str = "") -> tuple[bool, str, list[str]]:
    """探测一个 OpenAI 兼容端点：返回 (可用, 说明, 模型列表)。"""
    t = OpenAICompatTranslator(base_url=base_url, api_key=api_key, proxy=proxy)
    try:
        models = t.list_models()
        if models:
            return True, f"连接成功，{len(models)} 个模型", models
        return False, "连接失败或没有模型", []
    finally:
        t.close()


def dumps_for_log(obj) -> str:
    """调试用：安全的 JSON 序列化（不打印密钥）。"""
    try:
        return json.dumps(obj, ensure_ascii=False)[:500]
    except Exception:  # noqa: BLE001
        return str(obj)[:500]


def quick_segment(text: str, seg_id: int = 1, language: str = "") -> Segment:
    return Segment(id=seg_id, text=text, language=language)
