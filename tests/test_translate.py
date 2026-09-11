"""翻译层单元测试（不联网）。

覆盖三类纯逻辑：
1. 编号输出解析（批量翻译的核心，解析错了字幕就会错位）
2. 术语表校验（含"繁体术语表 vs 简体译文"这类假阳性）
3. 提示词渲染（变量替换、空段落处理、批量化）
"""

from __future__ import annotations

from app.translate.base import Segment, TranslateRequest, TranslateResult
from app.translate.openai_compat import (
    glossary_violations,
    is_local_url,
    parse_numbered,
)
from app.translate.prompts import (
    build_context_block,
    build_glossary_block,
    build_messages,
    build_source_block,
    get_template,
    language_name,
    list_templates,
)

# --------------------------------------------------------------------------- #
# 编号解析
# --------------------------------------------------------------------------- #
def test_parse_numbered_simple():
    got = parse_numbered("1. 第一条\n2. 第二条\n3. 第三条", 3)
    assert got == {1: "第一条", 2: "第二条", 3: "第三条"}


def test_parse_numbered_various_separators():
    for text in ("1. a\n2. b", "1、a\n2、b", "1)a\n2)b", "1．a\n2．b", "1：a\n2：b"):
        assert parse_numbered(text, 2) == {1: "a", 2: "b"}, text


def test_parse_numbered_joins_wrapped_lines():
    """模型把一条译文折成多行时，续行要接上而不是丢掉。"""
    text = "1. 第一行\n这是续行\n2. 第二条"
    got = parse_numbered(text, 2)
    assert got is not None
    assert "续行" in got[1]
    assert got[2] == "第二条"


def test_parse_numbered_rejects_count_mismatch():
    """条数不符必须返回 None，让调用方退回逐条——否则会错位。"""
    assert parse_numbered("1. a\n2. b", 3) is None
    assert parse_numbered("1. a", 2) is None


def test_parse_numbered_rejects_missing_index():
    """跳号（缺 2）也要拒绝。"""
    assert parse_numbered("1. a\n3. c", 2) is None


def test_parse_numbered_rejects_empty():
    assert parse_numbered("", 1) is None
    assert parse_numbered("   \n  ", 1) is None


def test_parse_numbered_single_line_without_number():
    """单条时模型常不带编号，parse_numbered 对这种情况返回 None（由调用方直接取正文）。"""
    assert parse_numbered("就是一句译文", 1) is None


# --------------------------------------------------------------------------- #
# 术语表校验
# --------------------------------------------------------------------------- #
def test_glossary_violation_detected():
    bad = glossary_violations("リンファンは鍵を持った", "他拿着钥匙", {"リンファン": "林凡"})
    assert bad == ["リンファン→林凡"]


def test_glossary_ok_when_term_used():
    assert glossary_violations("リンファンは鍵を持った", "林凡拿着钥匙", {"リンファン": "林凡"}) == []


def test_glossary_ignores_terms_absent_from_source():
    """原文里没有这个词就不该要求译文里有——否则全是假阳性。"""
    assert glossary_violations("関係ない文", "无关的句子", {"リンファン": "林凡"}) == []


def test_glossary_tolerates_traditional_vs_simplified():
    """实测踩过的坑：术语表写繁体「青銅」，模型输出简体「青铜」，
    严格比较会误判为漏译并触发无意义的重翻。"""
    assert glossary_violations("声堂の鍵", "青铜钥匙", {"声堂": "青銅"}) == []


def test_glossary_tolerates_punctuation_and_spacing():
    assert glossary_violations("声堂の鍵", "拿着 青铜、钥匙", {"声堂": "青铜"}) == []


def test_glossary_tolerates_case_for_latin():
    assert glossary_violations("the Sword of Dawn", "黎明之sword", {"Sword": "Sword"}) == []


def test_glossary_skips_empty_entries():
    assert glossary_violations("abc", "abc", {"": "x", "abc": ""}) == []


# --------------------------------------------------------------------------- #
# 提示词渲染
# --------------------------------------------------------------------------- #
def test_templates_exist_and_have_required_rules():
    tpls = list_templates()
    assert len(tpls) >= 3
    for t in tpls:
        # 忠实原文与不审查是产品红线，必须写进每一套模板
        assert "忠实原文" in t.system, t.id
        assert "内容审查" in t.system, t.id
        assert "{source_block}" in t.user, t.id


def test_get_template_falls_back_to_default():
    assert get_template("不存在的模板").id == "subtitle_direct"


def test_language_name():
    assert language_name("ja") == "日语"
    assert language_name("zh") == "简体中文"
    assert language_name("") == "原文语言"
    assert language_name("xx") == "xx"


def test_build_glossary_block():
    assert build_glossary_block({}) == ""
    got = build_glossary_block({"リンファン": "林凡"})
    assert "リンファン → 林凡" in got
    assert "术语表" in got


def test_build_context_block():
    assert build_context_block([]) == ""
    got = build_context_block([("原文A", "译文A")])
    assert "原文A" in got and "译文A" in got
    assert "不要翻译" in got


def test_build_source_block_single_and_batch():
    single = TranslateRequest(segments=[Segment(id=1, text="一句话")])
    text, numbered = build_source_block(single)
    assert text == "一句话" and numbered is False

    multi = TranslateRequest(segments=[Segment(id=1, text="A"), Segment(id=2, text="B")])
    text, numbered = build_source_block(multi)
    assert numbered is True
    assert text == "1. A\n2. B"


def test_build_messages_has_two_roles_and_variables_substituted():
    req = TranslateRequest(
        segments=[Segment(id=1, text="リンファン")],
        source_language="ja",
        target_language="zh",
        glossary={"リンファン": "林凡"},
        context=[("前文原", "前文译")],
    )
    msgs = build_messages(req)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "日语" in user and "简体中文" in user
    assert "林凡" in user          # 术语表注入
    assert "前文译" in user         # 前文注入
    assert "リンファン" in user     # 待翻译
    assert "{" not in user.replace("{", "", 0) or "}" not in user.split("待翻译")[0]


def test_custom_prompt_overrides_system():
    req = TranslateRequest(
        segments=[Segment(id=1, text="x")],
        custom_prompt="我的自定义系统提示",
    )
    msgs = build_messages(req)
    assert msgs[0]["content"] == "我的自定义系统提示"


def test_batch_message_asks_for_numbered_output():
    req = TranslateRequest(segments=[Segment(id=1, text="A"), Segment(id=2, text="B")])
    user = build_messages(req)[1]["content"]
    assert "编号" in user


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
def test_is_local_url():
    assert is_local_url("http://127.0.0.1:1234/v1")
    assert is_local_url("http://localhost:8080/v1")
    assert not is_local_url("https://api.openai.com/v1")
    assert not is_local_url("https://api.deepseek.com/v1")


def test_translate_result_merge():
    a = TranslateResult(translations={1: "一"}, prompt_tokens=10, retries=1)
    b = TranslateResult(translations={2: "二"}, failures={3: "err"}, prompt_tokens=20)
    a.merge(b)
    assert a.translations == {1: "一", 2: "二"}
    assert a.failures == {3: "err"}
    assert a.prompt_tokens == 30
    assert a.ok_count == 2 and a.fail_count == 1


def test_segment_strips_text():
    assert Segment(id=1, text="  hi  ").text == "hi"


def test_request_is_batch():
    assert not TranslateRequest(segments=[Segment(id=1, text="a")]).is_batch
    assert TranslateRequest(segments=[Segment(id=1, text="a"), Segment(id=2, text="b")]).is_batch
