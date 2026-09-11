"""字幕模型与换行逻辑的单元测试。

换行用的是"假 metrics"（每个字符固定宽度），这样断言是精确的、且不依赖系统字体——
用真 QFontMetrics 的话，不同机器字体不同、无字体环境下宽度为 0，测试会变得不可靠。
"""

from __future__ import annotations

from app.subtitle.model import SubtitleState, SubtitleLine, _srt_time
from app.ui.subtitle_overlay import wrap_text


class FakeMetrics:
    """每字符等宽；超出宽度的部分用 '…' 省略。"""

    def __init__(self, char_width: int = 10) -> None:
        self.char_width = char_width

    def horizontalAdvance(self, text: str) -> int:  # noqa: N802 - 对齐 Qt 命名
        return len(text) * self.char_width

    def elidedText(self, text: str, _mode, max_width: int) -> str:  # noqa: N802
        limit = max_width // self.char_width
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"


# --------------------------------------------------------------------------- #
# 换行
# --------------------------------------------------------------------------- #
def test_wrap_returns_single_line_when_fits():
    fm = FakeMetrics()
    assert wrap_text(fm, "短句", max_width=100, max_lines=2) == ["短句"]


def test_wrap_empty_text():
    assert wrap_text(FakeMetrics(), "", 100, 2) == []


def test_wrap_respects_max_lines():
    fm = FakeMetrics()
    lines = wrap_text(fm, "一" * 100, max_width=50, max_lines=2)
    assert len(lines) == 2
    assert lines[-1].endswith("…"), "放不下时最后一行应省略收尾"


def test_wrap_cjk_breaks_anywhere():
    fm = FakeMetrics(char_width=10)
    lines = wrap_text(fm, "一二三四五六", max_width=30, max_lines=3)
    assert lines == ["一二三", "四五六"]


def test_wrap_latin_prefers_spaces():
    """英文不该从单词中间劈开。"""
    fm = FakeMetrics(char_width=10)
    lines = wrap_text(fm, "hello world foo", max_width=110, max_lines=3)
    assert lines == ["hello world", "foo"]


def test_wrap_kinsoku_keeps_punctuation_off_line_start():
    """禁则：句号不能落到行首（实测预览里出现过"最后一行只有一个句号"）。"""
    fm = FakeMetrics(char_width=10)
    text = "一二三四五。六七八"
    lines = wrap_text(fm, text, max_width=50, max_lines=3)
    for ln in lines:
        assert not ln.startswith("。"), lines
    # 内容不能丢（禁则是"把前一个字一起挪下去"，不是删标点）
    assert "".join(lines) == text, lines


def test_wrap_kinsoku_does_not_orphan_trailing_period():
    """最典型的坏排版：最后一行只有一个句号。"""
    fm = FakeMetrics(char_width=10)
    lines = wrap_text(fm, "一二三四五六七八九。", max_width=50, max_lines=2)
    assert not any(ln == "。" for ln in lines), lines


def test_wrap_kinsoku_keeps_opening_quote_off_line_end():
    fm = FakeMetrics(char_width=10)
    lines = wrap_text(fm, "一二三四「五六七八九十", max_width=50, max_lines=3)
    for ln in lines:
        assert not ln.endswith("「"), lines


def test_wrap_does_not_duplicate_or_drop_characters():
    """曾出现过的 bug：算剩余内容时漏算丢掉的空格，导致省略号前多一个字符。"""
    fm = FakeMetrics(char_width=10)
    text = "The quick brown fox jumps over the lazy dog."
    lines = wrap_text(fm, text, max_width=170, max_lines=2)
    joined = "".join(ln.replace("…", "") for ln in lines)
    # 去重后应为原文本的前缀（空格可能在断行处被吃掉，所以按"无空格字符串"比）
    assert text.replace(" ", "").startswith(joined.replace(" ", "")), (lines, joined)


def test_wrap_handles_single_char_wider_than_line():
    """单字就超宽时不能死循环或返回空。"""
    fm = FakeMetrics(char_width=100)
    lines = wrap_text(fm, "一二三", max_width=50, max_lines=3)
    assert lines and all(lines)


# --------------------------------------------------------------------------- #
# 字幕状态
# --------------------------------------------------------------------------- #
def test_state_add_final_and_translation():
    st = SubtitleState()
    line = st.add_final("こんにちは", "ja")
    assert line.source == "こんにちは"
    assert not line.has_translation

    assert st.set_translation(line.id, "你好")
    assert st.get(line.id).translation == "你好"
    assert st.get(line.id).translate_latency_s is not None


def test_state_partial_cleared_by_final():
    st = SubtitleState()
    st.set_partial("途中", "ja")
    assert st.partial_source == "途中"
    st.add_final("完成した", "ja")
    assert st.partial_source == ""


def test_state_set_translation_unknown_id_returns_false():
    st = SubtitleState()
    assert st.set_translation(999, "x") is False


def test_state_visible_lines_order_and_limit():
    st = SubtitleState()
    for i in range(5):
        st.add_final(f"行{i}")
    assert [ln.source for ln in st.visible_lines(2)] == ["行3", "行4"]
    assert st.visible_lines(0) == []


def test_state_max_lines_is_bounded():
    st = SubtitleState(max_lines=3)
    for i in range(10):
        st.add_final(f"行{i}")
    assert len(st.lines) == 3


def test_state_clear():
    st = SubtitleState()
    st.add_final("a")
    st.set_partial("b")
    st.clear()
    assert len(st.lines) == 0 and st.partial_source == ""


def test_render_rows_bilingual_makes_pairs():
    st = SubtitleState()
    line = st.add_final("src", "ja")
    st.set_translation(line.id, "dst")
    rows = st.render_rows("bilingual", max_lines=3)
    assert rows == [("src", False), ("dst", True)]


def test_render_rows_target_falls_back_to_source():
    """还没翻译出来时显示原文，比显示空行好。"""
    st = SubtitleState()
    st.add_final("src", "ja")
    rows = st.render_rows("target", max_lines=3)
    assert rows == [("src", True)]


def test_render_rows_includes_partial_last():
    st = SubtitleState()
    st.add_final("a")
    st.set_partial("未定稿")
    rows = st.render_rows("source", max_lines=3)
    assert rows[-1] == ("未定稿", False)


def test_line_text_for_modes():
    line = SubtitleLine(id=1, source="src", translation="dst")
    assert line.text_for("source") == "src"
    assert line.text_for("target") == "dst"
    assert line.text_for("bilingual") == "dst"
    line2 = SubtitleLine(id=2, source="src")
    assert line2.text_for("target") == "src"


# --------------------------------------------------------------------------- #
# SRT 导出
# --------------------------------------------------------------------------- #
def test_srt_time_format():
    assert _srt_time(0) == "00:00:00,000"
    assert _srt_time(3661.5) == "01:01:01,500"


def test_to_srt_contains_source_and_translation():
    st = SubtitleState()
    line = st.add_final("こんにちは", "ja")
    st.set_translation(line.id, "你好")
    srt = st.to_srt()
    assert "1" in srt
    assert "こんにちは" in srt
    assert "你好" in srt
    assert "-->" in srt


def test_to_srt_empty_state():
    assert SubtitleState().to_srt() == ""


def test_to_srt_handles_equal_timestamps():
    """两行时间戳相同时不能让 end <= start（否则播放器会拒收）。"""
    st = SubtitleState()
    a = st.add_final("a")
    b = st.add_final("b")
    a.created_at = 100.0
    b.created_at = 100.0
    srt = st.to_srt()
    assert srt.count("-->") == 2
