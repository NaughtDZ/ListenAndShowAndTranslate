"""翻译调度中心：把"一批字幕"变成"译文"，负责缓存、通道选择、熔断、限流与成本护栏。

它在整条流水线里的位置：

    字幕段 → [Hub] → 缓存命中就直接返回；未命中才调通道
                     ├─ 选健康通道（优先本地 LLM，失败切下一个）
                     ├─ 只注入本条命中的术语
                     ├─ 带上前 N 句上下文保持译名一致
                     └─ 成功后写缓存 + 记录用量（成本护栏）

**熔断**是基于实测加的：有的模型（如思考关不掉的推理模型）会持续返回空译文，
不能让它每次都白白拖慢字幕；连续失败到阈值就临时禁用，切到别的通道。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import TranslateConfig
from app.translate.base import Segment, TranslateRequest, TranslateResult, Translator
from app.translate.cache import TranslationCache, cache_key
from app.translate.glossary import Glossary
from app.utils.log import get_logger

log = get_logger(__name__)

# 连续失败多少次就熔断该通道
FAILURE_THRESHOLD = 3
# 熔断后多久再试（秒）
COOLDOWN_S = 60.0


@dataclass
class ProviderSlot:
    """一个已注册的通道及其健康状态。"""

    translator: Translator
    priority: int = 100
    healthy: bool = True
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_error: str = ""
    disabled_until: float = 0.0
    last_call_at: float = 0.0

    @property
    def is_available(self) -> bool:
        if time.time() < self.disabled_until:
            return False
        return self.healthy or self.consecutive_failures < FAILURE_THRESHOLD

    def note_success(self) -> None:
        self.consecutive_failures = 0
        self.healthy = True
        self.last_error = ""

    def note_failure(self, error: str) -> bool:
        """记录一次失败，返回是否因此被熔断。"""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_error = error
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            self.healthy = False
            self.disabled_until = time.time() + COOLDOWN_S
            return True
        return False


@dataclass
class HubStats:
    requests: int = 0
    segments: int = 0
    cache_hits: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    characters: int = 0
    fallbacks: int = 0
    """因为通道失败而切换通道的次数。"""

    retries: int = 0
    """通道内部重试次数（如空译文加大预算重试、术语表重翻）。"""

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "segments": self.segments,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "characters": self.characters,
            "fallbacks": self.fallbacks,
            "retries": self.retries,
        }


class TranslatorHub:
    """翻译调度：缓存优先 → 选通道 → 失败切换 → 写缓存。"""

    def __init__(
        self,
        config: TranslateConfig,
        glossary: Glossary | None = None,
        cache: TranslationCache | None = None,
    ) -> None:
        self.config = config
        self.glossary = glossary or Glossary()
        self.cache = cache if config.cache_enabled else None
        self.stats = HubStats()

        self._providers: dict[str, ProviderSlot] = {}
        self._lock = threading.Lock()
        self._context: deque[tuple[str, str]] = deque(maxlen=max(0, config.context_lines))
        self._day = time.strftime("%Y-%m-%d")
        self._chars_today = 0

    # ------------------------------------------------------------------ #
    # 通道注册
    # ------------------------------------------------------------------ #
    def register(self, translator: Translator, priority: int = 100) -> None:
        self._providers[translator.name] = ProviderSlot(translator=translator, priority=priority)
        log.info("注册翻译通道: %s（优先级 %d）", translator.name, priority)

    def unregister(self, name: str) -> None:
        slot = self._providers.pop(name, None)
        if slot is not None:
            try:
                slot.translator.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("关闭通道 %s 失败: %s", name, exc)

    def active_providers(self) -> list[str]:
        return [
            name for name, slot in sorted(self._providers.items(), key=lambda kv: kv[1].priority)
            if slot.is_available
        ]

    def status(self) -> list[dict]:
        """通道状态（设置界面显示"哪个通道挂了、为什么"）。"""
        out = []
        for name, slot in sorted(self._providers.items(), key=lambda kv: kv[1].priority):
            cooling = max(0.0, slot.disabled_until - time.time())
            out.append({
                "name": name,
                "priority": slot.priority,
                "available": slot.is_available,
                "cooldown_s": round(cooling, 1),
                "requests": slot.total_requests,
                "failures": slot.total_failures,
                "consecutive_failures": slot.consecutive_failures,
                "last_error": slot.last_error[:200],
            })
        return out

    # ------------------------------------------------------------------ #
    # 上下文
    # ------------------------------------------------------------------ #
    def set_context(self, pairs: list[tuple[str, str]]) -> None:
        self._context.clear()
        for p in pairs[-self._context.maxlen:] if self._context.maxlen else []:
            self._context.append(p)

    def add_context(self, source: str, translation: str) -> None:
        if self._context.maxlen and source and translation:
            self._context.append((source, translation))

    def recent_context(self) -> list[tuple[str, str]]:
        return list(self._context)

    # ------------------------------------------------------------------ #
    # 成本护栏
    # ------------------------------------------------------------------ #
    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._chars_today = 0

    def remaining_char_budget(self) -> int | None:
        """返回今日剩余字符额度；None 表示不限制。"""
        limit = self.config.daily_char_limit
        if limit <= 0:
            return None
        self._roll_day()
        return max(0, limit - self._chars_today)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def translate(
        self,
        segments: list[Segment],
        source_language: str = "auto",
        target_language: str = "",
    ) -> TranslateResult:
        """翻译一批字幕。**不抛异常**：失败写进 failures 并附上可读原因。"""
        target = target_language or self.config.target_language
        clean = [s for s in segments if s.text]
        result = TranslateResult(provider="hub")

        if not clean:
            return result
        if not self.config.enabled:
            for s in clean:
                result.failures[s.id] = "翻译功能已关闭"
            return result

        # ---- 成本护栏 ----
        budget = self.remaining_char_budget()
        if budget is not None:
            self._roll_day()
            total_chars = sum(len(s.text) for s in clean)
            if total_chars > budget:
                msg = (f"已达今日字符上限（{self.config.daily_char_limit}），"
                       f"今日剩余 {budget} 字。可在设置里调高或明天再听。")
                for s in clean:
                    result.failures[s.id] = msg
                log.warning(msg)
                return result

        # ---- 缓存优先 ----
        pending = list(clean)
        keys: dict[int, str] = {}
        if self.cache is not None:
            provider_hint = ""  # 缓存键里不绑通道，换通道也复用（由术语+模板+语言决定译文）
            for s in clean:
                sub = self.glossary.subset_for(s.text)
                keys[s.id] = cache_key(
                    s.text, source_language, target,
                    provider=provider_hint, model="",
                    template=self.config.prompt_template,
                    extra=self.glossary.subset_hash(sub),
                )
            cached = self.cache.get_many(list(keys.values()))
            hit_ids = [sid for sid, k in keys.items() if k in cached]
            if hit_ids:
                by_id = {s.id: s for s in clean}
                for sid in hit_ids:
                    result.translations[sid] = cached[keys[sid]]
                    self.add_context(by_id[sid].text, cached[keys[sid]])
                result.from_cache = len(hit_ids)
                self.stats.cache_hits += len(hit_ids)
                pending = [s for s in clean if s.id not in set(hit_ids)]
                log.debug("缓存命中 %d/%d 条", len(hit_ids), len(clean))

        if not pending:
            result.provider = "cache"
            return result

        # ---- 选通道并翻译 ----
        order = self.active_providers()
        if not order:
            for s in pending:
                result.failures[s.id] = "没有可用的翻译通道（请在设置里配置并测试连通性）"
            return result

        last_error = ""
        for idx, name in enumerate(order):
            slot = self._providers[name]
            req = TranslateRequest(
                segments=pending,
                source_language=source_language,
                target_language=target,
                context=self.recent_context(),
                glossary=self._union_glossary(pending),
                template=self.config.prompt_template,
                custom_prompt=self.config.custom_prompt,
            )
            try:
                self._throttle(slot)
                sub_result = slot.translator.translate(req)
            except Exception as exc:  # noqa: BLE001 - 通道异常必须自己扛住
                last_error = f"{type(exc).__name__}: {exc}"
                just_banned = slot.note_failure(last_error)
                log.error("通道 %s 抛异常: %s%s", name, last_error,
                          "（已熔断，暂时切换）" if just_banned else "")
                continue

            slot.total_requests += 1
            self.stats.prompt_tokens += sub_result.prompt_tokens
            self.stats.completion_tokens += sub_result.completion_tokens
            self.stats.retries += sub_result.retries

            if sub_result.ok_count:
                slot.note_success()
                result.merge(sub_result)
                result.provider = name
                # 写缓存 + 记录上下文 + 用量
                if self.cache is not None:
                    self._store_cache(req, sub_result, source_language, target, keys)
                for s in pending:
                    if s.id in sub_result.translations:
                        self.add_context(s.text, sub_result.translations[s.id])
                self._chars_today += sum(len(s.text) for s in pending)
                self.stats.characters += sum(len(s.text) for s in pending)

                if sub_result.fail_count:
                    # 部分失败：不切换通道，剩下的让上层决定（通常已明确报错）
                    log.warning("通道 %s 部分失败 %d 条", name, sub_result.fail_count)
                self._finish_stats(result)
                return result

            # 全部失败 → 熔断并换下一个
            err = next(iter(sub_result.failures.values()), "未知失败")
            just_banned = slot.note_failure(err)
            last_error = err
            log.warning("通道 %s 全部失败（%s）%s", name, err,
                        "，已熔断" if just_banned else "，尝试下一个通道")
            self.stats.fallbacks += 1

        # 所有通道都失败
        msg = f"所有翻译通道都失败。最后原因: {last_error[:200]}" if last_error else "翻译失败"
        for s in pending:
            result.failures[s.id] = msg
        self._finish_stats(result)
        return result

    # ------------------------------------------------------------------ #
    def _finish_stats(self, result: TranslateResult) -> None:
        self.stats.requests += 1
        self.stats.segments += result.ok_count
        self.stats.failures += result.fail_count

    def _throttle(self, slot: ProviderSlot) -> None:
        """按 qps_limit 做简单节流，避免把免费额度打爆。"""
        qps = getattr(self.config, "qps_limit", 0) or 0
        if qps <= 0:
            return
        min_interval = 1.0 / qps
        elapsed = time.time() - slot.last_call_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        slot.last_call_at = time.time()

    def _union_glossary(self, segments: list[Segment]) -> dict[str, str]:
        """把这一批里命中的术语取并集。

        只注入命中项：整表注入既浪费 token，又会让模型把无关词硬套上去。
        """
        if not self.glossary:
            return {}
        merged: dict[str, str] = {}
        for s in segments:
            merged.update(self.glossary.subset_for(s.text))
        return merged

    def _store_cache(
        self,
        req: TranslateRequest,
        result: TranslateResult,
        source_language: str,
        target: str,
        keys: dict[int, str],
    ) -> None:
        assert self.cache is not None
        rows = []
        for seg in req.segments:
            text = result.translations.get(seg.id)
            if not text:
                continue
            key = keys.get(seg.id)
            if key is None:
                sub = self.glossary.subset_for(seg.text)
                key = cache_key(
                    seg.text, source_language, target, provider="", model="",
                    template=self.config.prompt_template,
                    extra=self.glossary.subset_hash(sub),
                )
            rows.append((key, seg.text, text, source_language, target, result.provider, result.model))
        try:
            self.cache.put_many(rows)
        except Exception as exc:  # noqa: BLE001 - 缓存写失败不能影响字幕
            log.warning("写翻译缓存失败: %s", exc)

    # ------------------------------------------------------------------ #
    def cache_stats(self) -> dict:
        return self.cache.stats() if self.cache else {"entries": 0, "total_hits": 0}

    def close(self) -> None:
        for slot in self._providers.values():
            try:
                slot.translator.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("关闭通道失败: %s", exc)
        self._providers.clear()
        if self.cache is not None:
            self.cache.close()
