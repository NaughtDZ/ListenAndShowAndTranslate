"""OpenAI 兼容通道：一套代码通吃 OpenAI / DeepSeek / 通义 / OpenRouter /
Ollama / LM Studio / llama.cpp server。

实现里固化了三条**实测结论**（见 docs/P4-翻译实测.md）：

1. **批量比逐条省 71% token**（5 条一起翻：439 tok vs 逐条 1508 tok），
   所以默认就按 ``batch_size`` 合并请求。代价是批量时术语表遵守率略降，
   因此加了第 3 条的校验。
2. **推理模型的思考开关在 API 侧关不掉**（``enable_thinking=false``、``/no_think``
   实测全部无效），会导致 ``content`` 为空字符串而 ``reasoning_content`` 有一大堆内容。
   不处理的话用户看到的就是**空白字幕**。这里会检测并自动加大预算重试、再退回单条。
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
        self.name = name
        self.stats = TranslatorStats()

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
                out.translations[batch[0].id] = content.strip()
            else:
                parsed = parse_numbered(content, len(batch))
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
        self, request: TranslateRequest, max_tokens: int
    ) -> tuple[str, dict, bool]:
        """发一次请求。

        Returns:
            (正文, usage, 是否被截断)
        """
        messages = build_messages(request)
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.disable_thinking:
            # 实测：对 LM Studio 里的 qwen3.8 无效，但对其他服务端可能有效，
            # 所以照发；真正的兜底是上面的"空译文重试"逻辑。
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        r = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        usage = data.get("usage") or {}
        truncated = choice.get("finish_reason") == "length"

        if not content:
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reasoning:
                log.warning(
                    "模型只输出了思考内容（%d 字）而没有译文——"
                    "推理模型的思考需要在服务端关闭", len(reasoning),
                )
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
