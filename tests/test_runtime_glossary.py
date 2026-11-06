"""边翻译边总结：临时专名表。

用户要的是"翻译长文件时，让模型顺带编制一个人名/专名表，供后续分段翻译用；
只存在于这一次任务，结束就删"。

实现上刻意拆成两步（见 ``runtime_glossary.py`` 模块文档）：
**程序找候选（零 token）→ 模型只给译名（一次小请求）→ 注入后续批次**。
不把术语表塞进翻译输出里，是因为那会给译文解析加一条必须容错的岔路。
"""

from __future__ import annotations

import pytest

from app.translate.base import Segment, TranslateRequest, TranslateResult
from app.translate.runtime_glossary import RuntimeGlossary


# --------------------------------------------------------------------------- #
# 候选发现（纯程序，不花 token）
# --------------------------------------------------------------------------- #
def test_candidates_find_repeated_names():
    """反复出现的名字要被找出来；只出现一次的不算（多半是噪音）。"""
    g = RuntimeGlossary()
    texts = (
        ["リンファンは走った。"] * 3
        + ["青銅の鍵を握った。"] * 3
        + ["一度だけ出てくる珍しい単語。"]
    )
    cands = g.candidates(texts)
    joined = "|".join(cands)
    assert "リンファン" in joined or "ファン" in joined, cands


def test_candidates_skip_known_and_user_entries():
    """已经学过的、用户表里有的、已经问过的，都不再当候选。"""
    g = RuntimeGlossary(
        entries={"リンファン": "林凡"},
        asked={"青銅"},
        user_entries={"駅前": "车站前"},
    )
    texts = ["リンファンと青銅と駅前の話"] * 5
    cands = g.candidates(texts)
    for word in ("リンファン", "青銅", "駅前"):
        assert word not in cands, f"{word} 本该被过滤掉：{cands}"


def test_no_candidates_for_short_text():
    assert RuntimeGlossary().candidates(["短い。"]) == []


def test_candidates_skip_han_for_chinese():
    """**中文不做汉字候选**：2~3 字滑窗一定会切出跨词碎片。

    实测中文会得到 `林凡握` / `紧了手` / `里的青` / `林凡低` 这种垃圾，
    而碎片和真名的频次是同一量级，靠阈值分不开。灌进术语表会破坏句子。
    """
    zh = [
        "林凡握紧了手里的青铜钥匙。",
        "林凡低声说道。",
        "青铜钥匙发光了。",
        "林凡走了。",
    ] * 4
    assert RuntimeGlossary().candidates(zh, source_language="zh") == []


def test_candidates_allow_han_for_japanese():
    """日语可以抓汉字词——日语里汉字串天然被假名隔开，边界是干净的。"""
    ja = ["青銅の鍵がかすかに光った。"] * 4
    assert "青銅" in RuntimeGlossary().candidates(ja, source_language="ja")
    # auto 时不知道语言，保守跳过汉字（宁可少抓）
    assert RuntimeGlossary().candidates(ja, source_language="auto") == []


def test_candidates_do_not_glue_japanese_particles():
    """日语助词不能被粘进人名里（n-gram 滑窗的老毛病）。"""
    ja = ["リンファンは走った。リンファンが叫んだ。"] * 4
    cands = RuntimeGlossary().candidates(ja, source_language="ja")
    assert "リンファン" in cands, cands
    for bad in ("ンファン", "ファンは", "リンファ"):
        assert bad not in cands, f"{bad} 是滑窗碎片，不该出现：{cands}"


# --------------------------------------------------------------------------- #
# 学习 + 三道闸
# --------------------------------------------------------------------------- #
def test_learn_accepts_normal_pairs():
    g = RuntimeGlossary()
    added = g.learn({"リンファン": "林凡", "青銅の鍵": "青铜钥匙"})
    assert added == 2
    assert g.as_dict()["リンファン"] == "林凡"


def test_learn_rejects_explanations():
    """「林凡（人名）」这种是在解释，不是译名——收进来会污染后续所有批次。"""
    g = RuntimeGlossary()
    added = g.learn({
        "リンファン": "林凡（人名）",
        "青銅": "青铜：一种金属",
        "よい": "好的→语气词",
    })
    assert added == 0, g.entries


def test_learn_rejects_identity_and_junk():
    g = RuntimeGlossary()
    assert g.learn({"リンファン": "リンファン"}) == 0     # 没翻
    assert g.learn({"リンファン": ""}) == 0               # 空
    assert g.learn({"リンファン": "林\n凡"}) == 0         # 含换行
    assert g.learn({"单": "单"}) == 0                     # 太短


def test_learn_never_overrides_user_glossary():
    """用户手填的永远优先——这是最关键的一道闸。"""
    g = RuntimeGlossary(user_entries={"リンファン": "林凡"})
    added = g.learn({"リンファン": "林繁"})       # 模型给了个不同译名
    assert added == 0
    assert g.as_dict()["リンファン"] == "林凡"
    assert "リンファン" in g.asked, "问过就要记下，避免反复问"


def test_learn_does_not_change_existing_translation():
    """已经学过的词不因为重复出现就改译名（否则又是一次漂移）。"""
    g = RuntimeGlossary()
    g.learn({"青銅": "青铜"})
    g.learn({"青銅": "青铜色"})
    assert g.as_dict()["青銅"] == "青铜"


def test_as_dict_user_wins():
    g = RuntimeGlossary(entries={"a": "自动", "b": "自动b"}, user_entries={"a": "用户"})
    merged = g.as_dict()
    assert merged["a"] == "用户"
    assert merged["b"] == "自动b"


def test_mark_asked_prevents_repeat_questions():
    g = RuntimeGlossary()
    g.mark_asked(["甲", "乙"])
    assert g.candidates(["甲と乙と丙の話"] * 5) == ["丙"] or "甲" not in g.candidates(
        ["甲と乙と丙の話"] * 5
    )


def test_describe_is_readable():
    g = RuntimeGlossary()
    assert g.describe() == "（无）"
    g.learn({"リンファン": "林凡"})
    assert "リンファン→林凡" in g.describe()


# --------------------------------------------------------------------------- #
# 专名表解析（模型输出不一定合规）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expect",
    [
        ("リンファン\t林凡", {"リンファン": "林凡"}),
        ("1. リンファン\t林凡\n2. 青銅\t青铜", {"リンファン": "林凡", "青銅": "青铜"}),
        ("リンファン→林凡", {"リンファン": "林凡"}),
        ("リンファン -> 林凡", {"リンファン": "林凡"}),
        ("リンファン 林凡", {"リンファン": "林凡"}),
        ("リンファン　林凡", {"リンファン": "林凡"}),
        ("没有分隔符的一行", {}),
        ("", {}),
        ("【候选词】\nリンファン\t林凡", {"リンファン": "林凡"}),
    ],
)
def test_parse_name_pairs(text, expect):
    from app.translate.openai_compat import _parse_name_pairs

    assert _parse_name_pairs(text) == expect


# --------------------------------------------------------------------------- #
# 适配器：extract_names 走一次真请求
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, content: str) -> None:
        self.status_code = 200
        self._body = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._body


class _Client:
    def __init__(self, content: str) -> None:
        self.content = content
        self.payloads: list[dict] = []

    def post(self, url, json=None, **_kw):  # noqa: A002
        self.payloads.append(dict(json or {}))
        return _Resp(self.content)

    def close(self) -> None:
        pass


def _translator(**kw):
    from app.translate.openai_compat import OpenAICompatTranslator

    return OpenAICompatTranslator(
        base_url="http://127.0.0.1:1234/v1", model="test-model", **kw
    )


def test_extract_names_sends_one_request_and_parses():
    t = _translator()
    client = _Client("リンファン\t林凡\n青銅の鍵\t青铜钥匙")
    t._client = client

    pairs = t.extract_names(
        ["リンファン", "青銅の鍵"],
        [("リンファンは走った。", "林凡跑了。")],
    )

    assert pairs == {"リンファン": "林凡", "青銅の鍵": "青铜钥匙"}
    assert len(client.payloads) == 1
    # 提示词里要能看到候选词
    joined = str(client.payloads[0])
    assert "リンファン" in joined


def test_extract_names_returns_empty_on_garbage():
    t = _translator()
    t._client = _Client("抱歉，我不能……")
    assert t.extract_names(["甲"], []) == {}


def test_extract_names_skips_plain_style():
    """纯翻译模型（sakura 之类）不吃指令，问了只会把提示词吐回来。"""
    t = _translator(prompt_style="plain")
    client = _Client("随便什么")
    t._client = client
    assert t.extract_names(["甲"], []) == {}
    assert client.payloads == []


def test_extract_names_survives_request_exception():
    """提取失败绝不能影响翻译主流程。"""

    class _Boom:
        def post(self, *a, **kw):
            raise RuntimeError("网络炸了")

        def close(self):
            pass

    t = _translator()
    t._client = _Boom()
    assert t.extract_names(["甲"], []) == {}


# --------------------------------------------------------------------------- #
# 注入：hub.translate(extra_glossary=...)
# --------------------------------------------------------------------------- #
def test_hub_injects_extra_glossary_but_user_wins():
    from app.config import TranslateConfig
    from app.translate.glossary import Glossary
    from app.translate.hub import TranslatorHub

    class _Recorder:
        name = "rec"
        supports_batch = True
        supports_streaming = False

        def __init__(self) -> None:
            self.seen: list[dict] = []

        def translate(self, req):
            self.seen.append(dict(req.glossary))
            out = TranslateResult()
            for seg in req.segments:
                out.translations[seg.id] = "x"
            return out

        def close(self):
            pass

    user = Glossary()
    user.add("リンファン", "林凡")          # 用户手填

    cfg = TranslateConfig(provider="llm", cache_enabled=False)
    cfg.auto_glossary = True
    hub = TranslatorHub(cfg, glossary=user, cache=None)
    rec = _Recorder()
    hub.register(rec, priority=1)

    hub.translate(
        [Segment(id=1, text="リンファンと青銅の鍵")],
        source_language="ja",
        target_language="zh",
        extra_glossary={"リンファン": "林繁", "青銅の鍵": "青铜钥匙"},
    )

    assert rec.seen[0]["リンファン"] == "林凡", "用户表的译名必须优先"
    assert rec.seen[0]["青銅の鍵"] == "青铜钥匙", "运行时表只补空缺"


def test_hub_extract_names_returns_empty_without_support():
    from app.config import TranslateConfig
    from app.translate.glossary import Glossary
    from app.translate.hub import TranslatorHub

    class _Plain:
        name = "plain"
        supports_batch = True
        supports_streaming = False

        def translate(self, req):
            return TranslateResult()

        def close(self):
            pass

    hub = TranslatorHub(TranslateConfig(provider="baidu", cache_enabled=False), Glossary(), None)
    hub.register(_Plain(), priority=1)
    assert hub.extract_names(["甲"], []) == {}


# --------------------------------------------------------------------------- #
# 任务层：只在 LLM 通道开、任务结束就丢
# --------------------------------------------------------------------------- #
def _job_with_provider(provider: str, auto: bool = True):
    from app.config import AppConfig
    from app.offline.job import JobOptions, SubtitleJob

    cfg = AppConfig()
    cfg.translate.provider = provider
    cfg.translate.auto_glossary = auto
    return SubtitleJob(JobOptions(translate=True), config=cfg), cfg


def test_runtime_glossary_only_for_llm_channel():
    job, _ = _job_with_provider("llm")
    assert job._make_runtime_glossary({}) is not None

    job2, _ = _job_with_provider("baidu")
    assert job2._make_runtime_glossary({}) is None, "传统翻译 API 没有这个能力"


def test_runtime_glossary_respects_switch():
    job, _ = _job_with_provider("llm", auto=False)
    assert job._make_runtime_glossary({}) is None


def test_runtime_glossary_is_not_persisted():
    """任务结束就丢：不落盘、不进缓存、下一次任务是一张新表。"""
    job, _ = _job_with_provider("llm")
    first = job._make_runtime_glossary({})
    first.learn({"甲": "甲某"})
    second = job._make_runtime_glossary({})
    assert second is not first
    assert second.entries == {}, "上一次任务的词不能带到下一次"


def test_job_refreshes_glossary_and_injects_it():
    """端到端：翻完第一批后学一批专名，第二批要带着它去翻。"""
    from app.config import AppConfig
    from app.offline.cue import Cue
    from app.offline.job import JobOptions, SubtitleJob

    class _FakeHub:
        def __init__(self) -> None:
            self.glossaries: list[dict] = []
            self.extract_calls = 0

        def translate(self, segments, **kw):
            self.glossaries.append(dict(kw.get("extra_glossary") or {}))
            out = TranslateResult()
            for seg in segments:
                out.translations[seg.id] = "译"
            return out

        def extract_names(self, candidates, samples):
            self.extract_calls += 1
            return {candidates[0]: "译名"}

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    cfg.translate.llm.input_budget_tokens = 0     # 用固定小批，保证有多批
    cfg.translate.auto_glossary = True

    job = SubtitleJob(JobOptions(translate=True, translate_batch=5), config=cfg)
    hub = _FakeHub()

    # 同一句人名反复出现，保证 n-gram 能统计到
    cues = [
        Cue(start_s=i, end_s=i + 1, text="リンファンは走った。") for i in range(30)
    ]
    job._translate(cues, hub, base=0.0, span=1.0)

    assert hub.extract_calls >= 1, "应该至少问过一次译名"
    # 第一批时还没有可学的，所以第一批注入为空；后面某批必须带上
    assert any(g for g in hub.glossaries), hub.glossaries


# --------------------------------------------------------------------------- #
# 批量任务：专名表要不要跨文件
# --------------------------------------------------------------------------- #
def test_translate_reuses_shared_runtime():
    """传入共享表时，`_translate` 直接用它，且它学到的词会注入后续批。"""
    from app.config import AppConfig
    from app.offline.cue import Cue
    from app.offline.job import JobOptions, SubtitleJob

    class _FakeHub:
        def __init__(self) -> None:
            self.glossaries: list[dict] = []

        def translate(self, segments, **kw):
            self.glossaries.append(dict(kw.get("extra_glossary") or {}))
            out = TranslateResult()
            for seg in segments:
                out.translations[seg.id] = "译"
            return out

        def extract_names(self, candidates, samples):
            return {}

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    cfg.translate.llm.input_budget_tokens = 0
    job = SubtitleJob(JobOptions(translate=True, translate_batch=5), config=cfg)

    shared = job._make_runtime_glossary({})
    shared.learn({"リンファン": "林凡"})      # 模拟"上一个文件"学到的

    hub = _FakeHub()
    cues = [Cue(start_s=i, end_s=i + 1, text="リンファンは走った。") for i in range(10)]
    job._translate(cues, hub, base=0.0, span=1.0, runtime=shared)

    assert any(g.get("リンファン") == "林凡" for g in hub.glossaries), hub.glossaries


def test_translate_creates_private_runtime_when_not_shared():
    """不传共享表 → 本文件新建一张，**外面拿不到**（各文件互不影响）。"""
    from app.config import AppConfig
    from app.offline.cue import Cue
    from app.offline.job import JobOptions, SubtitleJob

    class _FakeHub:
        def __init__(self) -> None:
            self.glossaries: list[dict] = []

        def translate(self, segments, **kw):
            self.glossaries.append(dict(kw.get("extra_glossary") or {}))
            out = TranslateResult()
            for seg in segments:
                out.translations[seg.id] = "译"
            return out

        def extract_names(self, candidates, samples):
            return {}

    cfg = AppConfig()
    cfg.translate.provider = "llm"
    cfg.translate.llm.input_budget_tokens = 0
    job = SubtitleJob(JobOptions(translate=True, translate_batch=5), config=cfg)

    hub = _FakeHub()
    cues = [Cue(start_s=i, end_s=i + 1, text="短い。") for i in range(5)]
    job._translate(cues, hub, base=0.0, span=1.0)      # 不传 runtime

    # 私有表里没学到东西，所以注入为空；关键是调用没炸、也没往外泄漏
    assert hub.glossaries == [{}]


class _FakeTranscriber:
    def __init__(self, *a, **kw) -> None:
        pass

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass


def _run_with_fake_transcriber(monkeypatch, job, paths):
    """跑 `run()`，但把识别器换掉、把 `_one_file` 换成记录器。"""
    from app.offline.job import FileResult

    seen: list = []

    def fake_one_file(path, index, total, transcriber, hub, shared_glossary=None):
        seen.append(shared_glossary)
        return FileResult(source=path)

    monkeypatch.setattr("app.offline.job.Transcriber", _FakeTranscriber)
    monkeypatch.setattr(job, "_build_hub", lambda: object())
    monkeypatch.setattr(job, "_one_file", fake_one_file)
    job.run(paths)
    return seen


def test_run_shares_one_glossary_for_whole_batch(monkeypatch):
    """勾上「跨文件保持」：整批任务共用**同一个**表对象。

    同一部剧的分集人名当然一样——每集重新学既浪费请求，又会出现
    "第 1 集叫林凡、第 2 集叫林繁"。
    """
    from app.offline.job import JobOptions, SubtitleJob

    job, _cfg = _job_with_provider("llm")
    job.options = JobOptions(translate=True, keep_glossary=True)

    seen = _run_with_fake_transcriber(monkeypatch, job, ["a.mp3", "b.mp3", "c.mp3"])

    assert len(seen) == 3
    assert seen[0] is not None
    assert seen[0] is seen[1] is seen[2], "三集必须是同一张表"


def test_run_does_not_share_when_unchecked(monkeypatch):
    """不勾：每个文件各自新建（`shared_glossary` 都是 None，由 `_translate` 自己建）。"""
    from app.offline.job import JobOptions, SubtitleJob

    job, _cfg = _job_with_provider("llm")
    job.options = JobOptions(translate=True, keep_glossary=False)

    seen = _run_with_fake_transcriber(monkeypatch, job, ["a.mp3", "b.mp3"])

    assert seen == [None, None]


def test_run_glossary_is_not_persisted(monkeypatch):
    """批次结束后表就丢：不落盘、下一个批次是全新一张。"""
    from app.offline.job import JobOptions, SubtitleJob

    job, _cfg = _job_with_provider("llm")
    job.options = JobOptions(translate=True, keep_glossary=True)

    first = _run_with_fake_transcriber(monkeypatch, job, ["a.mp3"])[0]
    first.learn({"甲": "甲某"})

    second = _run_with_fake_transcriber(monkeypatch, job, ["b.mp3"])[0]
    assert second is not first
    assert second.entries == {}
