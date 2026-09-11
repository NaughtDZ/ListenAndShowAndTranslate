"""翻译缓存 / 术语表 / 调度中心的单元测试（不联网）。

用一个"假通道"来验证 Hub 的缓存、熔断、降级、上下文与成本护栏——
这些逻辑出错的表现是"字幕串了"或"钱白花了"，比崩溃更难发现。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.config import TranslateConfig
from app.translate.base import Segment, TranslateRequest, TranslateResult
from app.translate.cache import TranslationCache, cache_key
from app.translate.glossary import Glossary, load_glossary
from app.translate.hub import FAILURE_THRESHOLD, TranslatorHub

# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
@pytest.fixture
def cache(tmp_path: Path) -> TranslationCache:
    c = TranslationCache(db_path=tmp_path / "t.db")
    yield c
    c.close()


def test_cache_roundtrip(cache):
    cache.put("こんにちは", "你好", "ja", "zh", "llm", "m1", "tpl")
    got = cache.get("こんにちは", "ja", "zh", "llm", "m1", "tpl")
    assert got == "你好"


def test_cache_miss_on_different_model(cache):
    cache.put("a", "甲", "ja", "zh", "llm", "m1", "tpl")
    assert cache.get("a", "ja", "zh", "llm", "m2", "tpl") is None


def test_cache_miss_on_different_glossary(cache):
    """术语表改了必须让缓存失效，否则用户改完译名却看到旧译文。"""
    cache.put("a", "甲", extra="hash1")
    assert cache.get("a", extra="hash2") is None
    assert cache.get("a", extra="hash1") == "甲"


def test_cache_different_template_is_separate(cache):
    cache.put("a", "甲", template="t1")
    assert cache.get("a", template="t2") is None


def test_cache_get_many_and_hits_counter(cache):
    cache.put("a", "甲")
    cache.put("b", "乙")
    k1 = cache_key("a")
    k2 = cache_key("b")
    got = cache.get_many([k1, k2, "nonexistent"])
    assert got == {k1: "甲", k2: "乙"}
    assert cache.stats()["total_hits"] == 2


def test_cache_get_many_chunks_over_sqlite_limit(cache):
    """SQLite 的 IN 变量上限是 999，缓存查询必须分块（这个坑在别的项目踩过）。"""
    for i in range(1200):
        cache.put(f"text{i}", f"t{i}")
    keys = [cache_key(f"text{i}") for i in range(1200)]
    got = cache.get_many(keys)
    assert len(got) == 1200


def test_cache_skips_empty(cache):
    cache.put("", "x")
    cache.put("y", "")
    assert cache.stats()["entries"] == 0


def test_cache_clear_and_prune(cache):
    for i in range(5):
        cache.put(f"t{i}", f"v{i}")
    assert cache.clear() == 5
    assert cache.stats()["entries"] == 0


def test_cache_key_is_stable_and_distinct():
    assert cache_key("abc") == cache_key("abc")
    assert cache_key("abc") != cache_key("abd")
    assert cache_key(" abc ") == cache_key("abc"), "首尾空白不应影响键"


# --------------------------------------------------------------------------- #
# 术语表
# --------------------------------------------------------------------------- #
def test_glossary_add_and_len():
    g = Glossary()
    g.add("リンファン", "林凡")
    g.add("", "空")
    g.add("x", "")
    assert len(g) == 1


def test_glossary_subset_only_matched():
    """只注入命中的术语——整表注入浪费 token 且会干扰模型。"""
    g = Glossary()
    g.add("リンファン", "林凡")
    g.add("声堂", "青铜")
    g.add("関係ない", "无关")
    assert g.subset_for("リンファンは声堂を持った") == {"リンファン": "林凡", "声堂": "青铜"}
    assert g.subset_for("関係ない話") == {"関係ない": "无关"}


def test_glossary_subset_case_insensitive_by_default():
    g = Glossary()
    g.add("Sword", "剑")
    assert g.subset_for("the SWORD is here") == {"Sword": "剑"}


def test_glossary_case_sensitive_mode():
    g = Glossary(case_sensitive=True)
    g.add("Sword", "剑")
    assert g.subset_for("the sword") == {}
    assert g.subset_for("the Sword") == {"Sword": "剑"}


def test_glossary_hash_changes_with_entries():
    g1 = Glossary()
    g1.add("a", "甲")
    g2 = Glossary()
    g2.add("a", "乙")
    assert g1.hash() != g2.hash()


def test_glossary_tsv_roundtrip(tmp_path: Path):
    p = tmp_path / "g.tsv"
    p.write_text("# 注释行\nリンファン\t林凡\t主人公\n声堂\t青铜\n", encoding="utf-8")
    g = Glossary.from_file(p)
    assert g.entries == {"リンファン": "林凡", "声堂": "青铜"}
    assert g.notes["リンファン"] == "主人公"


def test_glossary_json_roundtrip(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text(
        '{"name":"我的表","entries":{"a":{"target":"甲","note":"备注"}}}', encoding="utf-8"
    )
    g = Glossary.from_file(p)
    assert g.name == "我的表"
    assert g.entries == {"a": "甲"}
    assert g.notes["a"] == "备注"


def test_glossary_missing_file_returns_empty(tmp_path: Path):
    assert len(load_glossary(tmp_path / "nope.tsv")) == 0
    assert len(load_glossary("")) == 0


def test_glossary_save_and_reload(tmp_path: Path):
    g = Glossary(name="测试")
    g.add("a", "甲", "备注")
    p = tmp_path / "out.tsv"
    g.to_file(p)
    g2 = Glossary.from_file(p)
    assert g2.entries == {"a": "甲"}
    assert g2.notes["a"] == "备注"


def test_glossary_auto_extract_finds_repeated_ngrams():
    """候选词提取只做提示，但必须真能提取出反复出现的人名（n-gram 频次法）。"""
    g = Glossary()
    texts = ["林凡握紧了钥匙。", "林凡说道。", "林凡走了。", "不过林凡还是回来了。"]
    cands = g.auto_extract_candidates(texts, min_count=3)
    assert "林凡" in cands, cands
    assert cands["林凡"] >= 3


def test_glossary_auto_extract_drops_longer_supersets():
    """「林凡说」比「林凡」长且高频时，不该同时留下碎片词。"""
    g = Glossary()
    texts = ["林凡说了话。"] * 5
    cands = g.auto_extract_candidates(texts, min_count=3)
    assert cands, "应至少提取出候选"
    # 不该出现被更长候选包含的碎片
    for w in list(cands):
        assert not any(w != other and w in other for other in cands), f"{w} 是被包含的碎片"


def test_glossary_auto_extract_respects_min_count():
    g = Glossary()
    assert g.auto_extract_candidates(["只出现一次的词"], min_count=3) == {}


# --------------------------------------------------------------------------- #
# Hub：用假通道验证调度逻辑
# --------------------------------------------------------------------------- #
class FakeTranslator:
    """可控的假通道。"""

    def __init__(self, name="fake", fail=False, partial=False, delay=0.0):
        self.name = name
        self.supports_batch = True
        self.supports_streaming = False
        self.fail = fail
        self.partial = partial
        self.delay = delay
        self.calls: list[TranslateRequest] = []
        self.closed = False

    def translate(self, req: TranslateRequest) -> TranslateResult:
        self.calls.append(req)
        if self.delay:
            time.sleep(self.delay)
        r = TranslateResult(provider=self.name, model="fake-model", prompt_tokens=10, completion_tokens=5)
        segs = req.segments if not self.partial else req.segments[:1]
        for s in segs:
            if self.fail:
                continue
            r.translations[s.id] = f"[{self.name}]{s.text}"
        if self.fail:
            for s in req.segments:
                r.failures[s.id] = "假通道故意失败"
        return r

    def close(self):
        self.closed = True


@pytest.fixture
def hub(tmp_path: Path):
    cfg = TranslateConfig()
    cfg.context_lines = 3
    h = TranslatorHub(cfg, glossary=Glossary(), cache=TranslationCache(db_path=tmp_path / "h.db"))
    yield h
    h.close()


def test_hub_translates_and_caches(hub):
    fake = FakeTranslator()
    hub.register(fake, priority=10)
    segs = [Segment(id=1, text="こんにちは"), Segment(id=2, text="さようなら")]

    r1 = hub.translate(segs, "ja", "zh")
    assert r1.ok_count == 2
    assert r1.provider == "fake"
    assert r1.from_cache == 0

    # 第二次应该全部命中缓存，不再调通道
    r2 = hub.translate(segs, "ja", "zh")
    assert r2.ok_count == 2
    assert r2.from_cache == 2
    assert len(fake.calls) == 1, "命中缓存后不该再请求通道"


def test_hub_partial_cache_hit(hub):
    fake = FakeTranslator()
    hub.register(fake)
    hub.translate([Segment(id=1, text="A")], "ja", "zh")
    r = hub.translate([Segment(id=1, text="A"), Segment(id=2, text="B")], "ja", "zh")
    assert r.from_cache == 1
    assert len(fake.calls) == 2
    assert {s.text for s in fake.calls[1].segments} == {"B"}


def test_hub_falls_back_to_second_provider(hub):
    bad = FakeTranslator(name="bad", fail=True)
    good = FakeTranslator(name="good")
    hub.register(bad, priority=1)
    hub.register(good, priority=2)

    r = hub.translate([Segment(id=1, text="x")], "ja", "zh")
    assert r.provider == "good"
    assert r.ok_count == 1
    assert hub.stats.fallbacks == 1


def test_hub_circuit_breaker_after_threshold(hub):
    bad = FakeTranslator(name="bad", fail=True)
    good = FakeTranslator(name="good")
    hub.register(bad, priority=1)
    hub.register(good, priority=2)

    for i in range(FAILURE_THRESHOLD):
        hub.translate([Segment(id=i + 1, text=f"t{i}")], "ja", "zh")

    status = {s["name"]: s for s in hub.status()}
    assert status["bad"]["available"] is False, "连续失败到阈值应被熔断"
    assert "假通道故意失败" in status["bad"]["last_error"]


def test_hub_reports_when_no_provider_available(hub):
    r = hub.translate([Segment(id=1, text="x")], "ja", "zh")
    assert r.fail_count == 1
    assert "没有可用的翻译通道" in r.failures[1]


def test_hub_disabled_translation(hub):
    hub.config.enabled = False
    hub.register(FakeTranslator())
    r = hub.translate([Segment(id=1, text="x")], "ja", "zh")
    assert "已关闭" in r.failures[1]


def test_hub_daily_char_limit(hub):
    hub.config.daily_char_limit = 5
    hub.register(FakeTranslator())
    r = hub.translate([Segment(id=1, text="这是一句很长的话超过五个字")], "ja", "zh")
    assert r.fail_count == 1
    assert "字符上限" in r.failures[1]


def test_hub_injects_only_matched_glossary(hub):
    hub.glossary.add("リンファン", "林凡")
    hub.glossary.add("無関係", "无关")
    fake = FakeTranslator()
    hub.register(fake)

    hub.translate([Segment(id=1, text="リンファンだ")], "ja", "zh")
    assert fake.calls[0].glossary == {"リンファン": "林凡"}


def test_hub_maintains_context(hub):
    fake = FakeTranslator()
    hub.register(fake)
    hub.translate([Segment(id=1, text="第一句")], "ja", "zh")
    hub.translate([Segment(id=2, text="第二句")], "ja", "zh")
    # 第二次请求应带上第一次的上下文
    assert fake.calls[1].context == [("第一句", "[fake]第一句")]
    assert len(hub.recent_context()) == 2


def test_hub_context_respects_maxlen(hub):
    hub.config.context_lines = 2
    hub = TranslatorHub(hub.config, glossary=hub.glossary, cache=None)
    for i in range(5):
        hub.add_context(f"s{i}", f"t{i}")
    assert len(hub.recent_context()) == 2
    hub.close()


def test_hub_tolerates_provider_exception(hub):
    class Boom:
        name = "boom"
        supports_batch = True
        supports_streaming = False

        def translate(self, req):
            raise RuntimeError("网络炸了")

        def close(self):
            pass

    hub.register(Boom(), priority=1)
    good = FakeTranslator(name="good")
    hub.register(good, priority=2)

    r = hub.translate([Segment(id=1, text="x")], "ja", "zh")
    assert r.provider == "good"
    assert "网络炸了" in {s["name"]: s for s in hub.status()}["boom"]["last_error"]


def test_hub_stats_accounting(hub):
    hub.register(FakeTranslator())
    hub.translate([Segment(id=1, text="abc")], "ja", "zh")
    assert hub.stats.requests == 1
    assert hub.stats.segments == 1
    assert hub.stats.prompt_tokens == 10
    assert hub.stats.characters == 3
