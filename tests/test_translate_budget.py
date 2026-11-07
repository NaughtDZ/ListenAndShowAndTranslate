"""翻译 token 预算：估算、分批、前文裁剪、输出上限、失败二分重试。

背景：以前"一批送几条"是写死的（离线 8 条 / 适配器 5 条），和模型的真实
上下文窗口毫无关系。现在由 ``input_budget_tokens`` 决定（4k / 8k / 16k / 32k），
输出上限按这批的实际大小自动推导。
"""

from __future__ import annotations

import pytest

from app.offline.cue import Cue
from app.translate.base import Segment, TranslateRequest, TranslateResult
from app.translate.budget import (
    MIN_OUTPUT_TOKENS,
    OUTPUT_RATIO,
    estimate_context_tokens,
    estimate_tokens,
    output_tokens_for,
    plan_batches,
    trim_context,
)


# --------------------------------------------------------------------------- #
# 估算
# --------------------------------------------------------------------------- #
def test_cjk_is_about_one_token_per_char():
    """中文/日文约 1 token / 字（乘了安全系数，所以略微大于字数）。"""
    text = "夜の列車に乗って"          # 8 个字
    est = estimate_tokens(text)
    assert 8 <= est <= 12, est


def test_ascii_is_about_four_chars_per_token():
    text = "the quick brown fox " * 5   # 100 字符
    est = estimate_tokens(text)
    assert 24 <= est <= 34, est


def test_estimate_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("短") < estimate_tokens("更长的句子")


def test_context_estimate_grows_with_pairs():
    one = [("原文", "译文")]
    two = one * 2
    assert estimate_context_tokens(two) > estimate_context_tokens(one)
    assert estimate_context_tokens([]) == 0


# --------------------------------------------------------------------------- #
# 分批
# --------------------------------------------------------------------------- #
def _segs(n: int, text: str = "这是一句大约二十个字的字幕文本内容。") -> list[Segment]:
    return [Segment(id=i, text=text) for i in range(n)]


def test_budget_splits_into_fewer_bigger_batches():
    """4k 预算下，100 条短字幕应该只切成 1~2 批（旧行为是 8 条一批 → 13 批）。"""
    segs = _segs(100)
    batches = plan_batches(segs, 4096, overhead_tokens=500)
    assert 1 <= len(batches) <= 2, [len(b) for b in batches]
    assert sum(len(b) for b in batches) == 100
    # 顺序不能乱
    assert [s.id for b in batches for s in b] == list(range(100))


def test_max_items_caps_even_with_huge_budget():
    """预算再大也不许把几百条塞成一批（输出会爆、失败代价太高）。"""
    segs = _segs(500)
    batches = plan_batches(segs, 32768, overhead_tokens=500, max_items=200)
    assert all(len(b) <= 200 for b in batches)
    assert sum(len(b) for b in batches) == 500


def test_budget_zero_means_single_batch():
    """预算 0 表示"不限"，此时只受 max_items 约束。"""
    segs = _segs(30)
    batches = plan_batches(segs, 0, max_items=200)
    assert len(batches) == 1
    assert len(batches[0]) == 30


def test_oversized_single_segment_gets_its_own_batch():
    """单条就超预算时让它独占一批——总比丢掉一条字幕好。"""
    long_text = "很长的句子。" * 2000       # 远超 4k
    segs = [Segment(id=0, text="短句"), Segment(id=1, text=long_text), Segment(id=2, text="短句")]
    batches = plan_batches(segs, 1024, overhead_tokens=100)
    assert any(len(b) == 1 and b[0].id == 1 for b in batches)
    assert sum(len(b) for b in batches) == 3


def test_plan_batches_never_returns_empty():
    assert plan_batches([], 4096) == []


# --------------------------------------------------------------------------- #
# 前文裁剪
# --------------------------------------------------------------------------- #
def test_trim_context_keeps_newest():
    """前文按预算从**最新**往回留：越近越重要。"""
    pairs = [(f"原文{i}", f"译文{i}") for i in range(50)]
    kept = trim_context(pairs, 200)
    assert 0 < len(kept) < 50
    # 保留的必须是末尾那几对（最新的）
    assert kept[-1] == ("原文49", "译文49")
    # 而且是从尾部连续拿的一段
    start = 50 - len(kept)
    assert kept == [(f"原文{i}", f"译文{i}") for i in range(start, 50)]


def test_trim_context_zero_budget_drops_everything():
    assert trim_context([("a", "b")], 0) == []


# --------------------------------------------------------------------------- #
# 输出上限
# --------------------------------------------------------------------------- #
def test_output_tokens_follow_input_budget():
    small = output_tokens_for(2000)
    big = output_tokens_for(16000)
    assert small < big
    assert big == int(16000 * OUTPUT_RATIO)


def test_output_tokens_have_floor_and_cap():
    """再小的批也要给够 512：给太少会让模型返回空串（预算烧在格式上）。"""
    assert output_tokens_for(10) >= MIN_OUTPUT_TOKENS
    assert output_tokens_for(10**7, cap=4096) == 4096


# --------------------------------------------------------------------------- #
# 适配器：按预算切批 + 失败先二分
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, content: str) -> None:
        self.status_code = 200
        self._body = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._body


class _RecordingClient:
    def __init__(self, responder) -> None:
        self.payloads: list[dict] = []
        self._responder = responder

    def post(self, url, json=None, **_kw):  # noqa: A002 - httpx 签名就叫 json
        self.payloads.append(dict(json or {}))
        return _Resp(self._responder(len(self.payloads), json))

    def close(self) -> None:
        pass


def _translator(**kwargs):
    from app.translate.openai_compat import OpenAICompatTranslator

    return OpenAICompatTranslator(
        base_url="http://127.0.0.1:1234/v1", model="test-model", **kwargs
    )


def test_translator_uses_one_request_for_large_budget():
    """预算足够时，30 条应该一次请求搞定（旧代码是 6 次）。

    只断言"请求次数"，不关心模型说什么——这才是预算有没有生效的直接证据。
    """
    t = _translator(input_budget_tokens=8192, max_tokens=0)
    segs = [Segment(id=i, text="短小的日文字幕。") for i in range(30)]
    client = _RecordingClient(
        lambda n, payload: "\n".join(f"{i}. 译{i}" for i in range(1, 31))
    )
    t._client = client

    res = t.translate(
        TranslateRequest(segments=segs, source_language="ja", target_language="zh")
    )

    assert len(client.payloads) == 1, f"发了 {len(client.payloads)} 次请求"
    assert len(res.translations) == 30


def test_translator_splits_when_budget_is_tight():
    """预算很小（刚够几条）时会切成多批。"""
    t = _translator(input_budget_tokens=64, max_tokens=0, batch_size=1)
    segs = [Segment(id=i, text="这是一句大约二十个字的字幕文本内容。") for i in range(10)]
    client = _RecordingClient(lambda n, payload: "1. 译文")
    t._client = client

    t.translate(TranslateRequest(segments=segs, source_language="zh", target_language="zh"))
    assert len(client.payloads) > 1


def test_parse_failure_splits_in_half_instead_of_going_one_by_one():
    """编号解析失败时先**二分**，不是直接逐条。

    这是这一版的关键改进：一次送 100 条时只要模型漏一条，逐条就是 100 次
    请求（几分钟 + 100 倍 token）。二分 10 条只需要 1 + 2 = 3 次。
    """
    t = _translator(input_budget_tokens=8192, max_tokens=0)
    segs = [Segment(id=i, text="测试字幕。") for i in range(10)]

    def responder(n, payload):
        if n == 1:
            return "这段输出没法按编号解析"          # 逼它触发二分
        return "\n".join(f"{i}. 译{i}" for i in range(1, 6))   # 每半批 5 条

    client = _RecordingClient(responder)
    t._client = client

    res = t.translate(
        TranslateRequest(segments=segs, source_language="ja", target_language="zh")
    )

    assert len(client.payloads) == 3, (
        f"发了 {len(client.payloads)} 次请求；二分应该是 1+2=3，逐条会是 11"
    )
    assert len(res.translations) == 10


def test_output_tokens_auto_when_zero():
    """``max_tokens=0`` 时按这批实际大小推导，不是写死 512。"""
    t = _translator(max_tokens=0)
    small = TranslateRequest(segments=[Segment(id=0, text="短句")])
    big = TranslateRequest(segments=[Segment(id=i, text="较长的一句话。" * 20) for i in range(20)])
    assert t._out_tokens(big) > t._out_tokens(small)
    assert t._out_tokens(small) >= MIN_OUTPUT_TOKENS


def test_output_tokens_respect_explicit_cap():
    t = _translator(max_tokens=256)
    req = TranslateRequest(segments=[Segment(id=i, text="较长的一句话。" * 20) for i in range(50)])
    assert t._out_tokens(req) == 256


# --------------------------------------------------------------------------- #
# 离线任务：按预算分批（替掉旧的固定 8 条）
# --------------------------------------------------------------------------- #
def test_offline_job_batches_by_budget():
    """100 条 cue + 4k 预算 → 应当 1~2 批；旧的固定 8 条会是 13 批。"""
    from app.config import AppConfig
    from app.offline.cue import Cue
    from app.offline.job import JobOptions, SubtitleJob

    class _FakeHub:
        def __init__(self) -> None:
            self.batches: list[int] = []

        def translate(self, segments, **_kw):
            self.batches.append(len(segments))
            res = TranslateResult(provider="fake")
            for seg in segments:
                res.translations[seg.id] = f"译{seg.id}"
            return res

    cfg = AppConfig()
    cfg.translate.llm.input_budget_tokens = 4096

    job = SubtitleJob(JobOptions(translate=True, target_language="zh"), config=cfg)
    hub = _FakeHub()
    cues = [Cue(start_s=i, end_s=i + 1, text="这是一句大约二十个字的字幕。") for i in range(100)]

    ok, failed = job._translate(cues, hub, base=0.0, span=1.0)

    assert ok == 100 and failed == 0
    assert 1 <= len(hub.batches) <= 2, hub.batches
    assert sum(hub.batches) == 100
    assert all(c.translation for c in cues)


def test_offline_job_falls_back_to_fixed_size_without_budget():
    """预算 0（"不限"档之外的老配置）= 退回固定 translate_batch 条。"""
    from app.config import AppConfig
    from app.offline.cue import Cue
    from app.offline.job import JobOptions, SubtitleJob

    class _FakeHub:
        def __init__(self) -> None:
            self.batches: list[int] = []

        def translate(self, segments, **_kw):
            self.batches.append(len(segments))
            res = TranslateResult(provider="fake")
            for seg in segments:
                res.translations[seg.id] = "x"
            return res

    cfg = AppConfig()
    cfg.translate.llm.input_budget_tokens = 0

    job = SubtitleJob(JobOptions(translate=True, translate_batch=8), config=cfg)
    hub = _FakeHub()
    cues = [Cue(start_s=i, end_s=i + 1, text="短句。") for i in range(20)]

    job._translate(cues, hub, base=0.0, span=1.0)

    assert hub.batches == [8, 8, 4]


# --------------------------------------------------------------------------- #
# 回归：v0.4.0 的"第二个文件起翻译全失败"
# --------------------------------------------------------------------------- #
def test_overhead_is_estimated_as_chinese_not_ascii():
    """提示词是中文，1 字 ≈ 1 token；按 ASCII 的 4 字符/token 会低估 4 倍。

    v0.4.0 就是这样：``estimate_tokens("x" * 400)`` 算出 115 token，
    而真实的中文提示词约 460 token——每次请求凭空多出 350 token。
    """
    from app.translate.budget import estimate_overhead_tokens, estimate_tokens

    ascii_guess = estimate_tokens("x" * 400)      # 旧算法的错误答案
    real_guess = estimate_overhead_tokens()        # 按中文估

    assert real_guess > ascii_guess * 3, (
        f"中文提示词估算 {real_guess} 与 ASCII 估算 {ascii_guess} 差得太少，"
        "说明又退回按英文估了"
    )


def test_batches_plus_context_reserve_stay_within_budget():
    """**核心回归**：正文 + 提示词 + 前文预留 ≤ 单次预算。

    v0.4.0 的 bug：job 只把提示词/术语表算进 overhead，前文由 hub 另按
    1/3 预算加进去，实际请求 = 预算 × 1.33。症状是第一个文件（还没有前文）
    正常，第二个文件起整批失败。
    """
    from app.translate.base import Segment
    from app.translate.budget import (
        CONTEXT_BUDGET_SHARE,
        estimate_overhead_tokens,
        estimate_tokens,
        plan_batches,
    )

    budget = 4096
    reserve = int(budget * CONTEXT_BUDGET_SHARE)
    fixed = estimate_overhead_tokens() + reserve

    segments = [Segment(id=i, text="あ" * 20) for i in range(500)]
    batches = plan_batches(segments, budget, overhead_tokens=fixed, max_items=400)

    assert len(batches) > 1, "这个规模应该切出多批"
    for idx, batch in enumerate(batches):
        if len(batch) == 1:
            continue          # 单条超预算时允许独占一批
        body = sum(estimate_tokens(s.text) for s in batch)
        assert body + fixed <= budget, (
            f"第 {idx + 1} 批超预算：正文 {body} + 固定 {fixed} = {body + fixed} > {budget}"
        )


def test_job_reserves_context_budget(monkeypatch):
    """job 调 plan_batches 时，overhead 必须包含前文预留。"""
    from app.config import AppConfig
    from app.offline.job import JobOptions, SubtitleJob
    from app.translate import budget as budget_mod
    from app.translate.base import TranslateResult

    captured: dict = {}

    real_plan = budget_mod.plan_batches

    def spy(segments, budget_tokens, overhead_tokens=0, max_items=0):
        captured["budget"] = budget_tokens
        captured["overhead"] = overhead_tokens
        return real_plan(segments, budget_tokens, overhead_tokens, max_items)

    monkeypatch.setattr(budget_mod, "plan_batches", spy)

    class _Hub:
        def translate(self, segments, **kw):
            out = TranslateResult()
            for seg in segments:
                out.translations[seg.id] = "译"
            return out

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    cfg.translate.llm.input_budget_tokens = 4096
    job = SubtitleJob(JobOptions(translate=True), config=cfg)
    cues = [Cue(start_s=i, end_s=i + 1, text="短句。") for i in range(20)]
    job._translate(cues, _Hub(), base=0.0, span=1.0)

    reserve = int(4096 * budget_mod.CONTEXT_BUDGET_SHARE)
    assert captured["overhead"] >= reserve, captured
    assert captured["overhead"] + 0 < 4096


def test_hub_context_never_exceeds_its_share():
    """hub 裁出来的前文，实际 token 不超过 0.33 × 预算。"""
    from app.config import TranslateConfig
    from app.translate.budget import CONTEXT_BUDGET_SHARE, estimate_context_tokens
    from app.translate.glossary import Glossary
    from app.translate.hub import TranslatorHub

    cfg = TranslateConfig(provider="llm", cache_enabled=False)
    cfg.llm.input_budget_tokens = 4096
    hub = TranslatorHub(cfg, Glossary(), None)
    for i in range(200):
        hub.add_context(f"这是一句二十个字左右的日语原文第{i}句。", f"这是二十个字左右的中文译文第{i}句。")

    ctx = hub._budgeted_context()
    used = estimate_context_tokens(ctx)
    assert used <= int(4096 * CONTEXT_BUDGET_SHARE) * 1.05, (used, len(ctx))
    hub.close()


def test_translate_reports_failure_reasons():
    """失败原因必须打出来——用户看到"301 条失败"却不知道为啥，无从下手。"""
    from app.config import AppConfig
    from app.offline.job import JobOptions, SubtitleJob
    from app.translate.base import TranslateResult

    logs: list[str] = []

    class _BrokenHub:
        def translate(self, segments, **kw):
            out = TranslateResult()
            for seg in segments:
                out.failures[seg.id] = "Error 400: context length exceeded"
            return out

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    cfg.translate.llm.input_budget_tokens = 4096
    job = SubtitleJob(
        JobOptions(translate=True), config=cfg, on_log=logs.append
    )
    cues = [Cue(start_s=i, end_s=i + 1, text="短句。") for i in range(10)]
    ok, failed = job._translate(cues, _BrokenHub(), base=0.0, span=1.0)

    assert ok == 0 and failed == 10
    joined = "\n".join(logs)
    assert "context length exceeded" in joined, logs
    assert "单次预算" in joined, "整批失败时要提示用户调小预算"
