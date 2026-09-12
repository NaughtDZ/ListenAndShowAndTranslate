"""悬浮字幕窗的尺寸 / 字号自适应测试（offscreen 平台，不弹窗）。

背景（用户 2026-09-12 反馈的两条）：

1. **上下不能拖**：拖高拖矮都弹回原大小。根因是自动高度按"最多可能几行"预留
   （3 条 × 每条 2 行 × 双语 = 12 行），而且手动高度只在 ``window_height == 0``
   时记录，存过一次以后就再也不更新 → 下一帧 `_relayout()` 又把旧值顶回去。
2. **顶上会空一行**：同一个预留造成的——内容没那么高时底部对齐，空行堆在顶上。

顺带落地用户要的"动态字号"：窗口装不下时把字号缩小到刚好放得下，
长句/多行过去以后**自动回到基准字号**。

这些用例都自己算期望值（用 overlay 自己的 `_line_height()` / `_wrap_rows()`），
不硬编码像素，免得换字体或换 DPI 就挂。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.subtitle.model import SubtitleState  # noqa: E402
from app.ui.subtitle_overlay import (  # noqa: E402
    MIN_WINDOW_HEIGHT,
    PADDING,
    STATUS_HEIGHT,
    SubtitleOverlay,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _overlay(**overlay_kwargs) -> SubtitleOverlay:
    cfg = AppConfig()
    cfg.overlay.always_on_top = False  # 测试里不需要每 200ms 重申置顶
    for key, value in overlay_kwargs.items():
        setattr(cfg.overlay, key, value)
    return SubtitleOverlay(cfg.overlay)


def _state(*texts: str, translate: bool = True) -> SubtitleState:
    st = SubtitleState()
    for i, text in enumerate(texts, start=1):
        st.add_final(text, "zh", line_id=i)
        if translate:
            st.set_translation(i, f"translation {i}")
    return st


def _drag_height(overlay: SubtitleOverlay, height: int, edge: str = "b") -> None:
    """模拟用户鼠标拖上下边缘：先真的改几何，再走释放处理。"""
    overlay.setGeometry(overlay.x(), overlay.y(), overlay.width(), height)
    overlay._apply_resize(edge)


def _rows(overlay: SubtitleOverlay) -> int:
    return len(overlay._build_draw_rows())


def _body_h(overlay: SubtitleOverlay) -> int:
    return overlay.height() - PADDING * 2 - STATUS_HEIGHT


# --------------------------------------------------------------------------- #
# 自动高度：按内容实际行数，不留空行
# --------------------------------------------------------------------------- #
def test_auto_height_exactly_fits_content(qapp):
    o = _overlay(display_mode="source", max_lines=3, lines_per_subtitle=1, window_height=0)
    try:
        o.update_state(_state("短句", translate=False))
        assert _rows(o) == 1
        assert o.height() == o._line_height() + PADDING * 2 + STATUS_HEIGHT
    finally:
        o.deleteLater()


def test_auto_height_shrinks_back_when_content_gets_shorter(qapp):
    """**回归**：以前高度按最大行数预留，内容变少时空行堆在顶上。"""
    o = _overlay(display_mode="source", max_lines=3, lines_per_subtitle=1, window_height=0)
    try:
        o.update_state(_state("第一条字幕", "第二条字幕", "第三条字幕", translate=False))
        assert _rows(o) == 3
        tall = o.height()
        assert tall == 3 * o._line_height() + PADDING * 2 + STATUS_HEIGHT

        o.update_state(_state("只剩一条", translate=False))
        assert _rows(o) == 1
        assert o.height() == o._line_height() + PADDING * 2 + STATUS_HEIGHT
        assert o.height() < tall
    finally:
        o.deleteLater()


# --------------------------------------------------------------------------- #
# 上下拖动：必须生效，而且不能被自动高度顶回去
# --------------------------------------------------------------------------- #
def test_manual_height_is_remembered(qapp):
    """**回归**：拖出来的高度必须被记住，并且下一批字幕不会把它弹回去。"""
    o = _overlay(display_mode="source", lines_per_subtitle=1, window_height=0)
    try:
        o.update_state(_state("第一条", "第二条", translate=False))
        target = o.height() + 60
        _drag_height(o, target, edge="t")

        assert o.config.window_height == target
        assert o.height() == target

        # 内容变了（下一个字幕到来）也不能弹回自动高度
        o.update_state(_state("第一条", "第二条", "第三条更长一点的字幕", translate=False))
        assert o.height() == target, "拖动出来的高度又被自动高度顶回去了"
    finally:
        o.deleteLater()


def test_manual_height_never_below_minimum(qapp):
    o = _overlay(window_height=0)
    try:
        _drag_height(o, 5)
        assert o.config.window_height == MIN_WINDOW_HEIGHT
        assert o.height() == MIN_WINDOW_HEIGHT
    finally:
        o.deleteLater()


def test_vertical_drag_does_not_change_font_size(qapp):
    """只拖上下时字号不许动。

    这里特意让 ``config.window_width`` 与"实际生效宽度"不一致（小屏会被夹窄），
    来钉住以前那个坑：拿 config 宽度当基准，第一次拖高就会误判"宽度变了"顺手改字号。
    """
    o = _overlay(window_width=4000, display_mode="source", lines_per_subtitle=1)
    try:
        o._relayout()
        base = o.config.font_size
        assert o.width() < o.config.window_width, "这个用例需要屏幕把宽度夹窄"
        _drag_height(o, 200, edge="t")
        assert o.config.font_size == base
        assert o._draw_font_size == base
    finally:
        o.deleteLater()


def test_width_drag_scales_font_and_updates_spin(qapp):
    """拉宽 → 字号按比例放大，并且同步回设置窗里的字号框（_font_spin）。"""
    o = _overlay(window_width=1200, display_mode="source", lines_per_subtitle=1)
    try:
        o._relayout()
        base = o.config.font_size
        spin = _FakeSpin()
        o._font_spin = spin
        o.setGeometry(o.x(), o.y(), int(o.width() * 1.5), o.height())
        o._apply_resize("r")
        assert o.config.font_size > base
        assert spin.value == o.config.font_size
        assert spin.blocked, "同步字号时要屏蔽信号，否则会触发回调回环"
    finally:
        o.deleteLater()


class _FakeSpin:
    """够用的假 QSpinBox：只记录 setValue / blockSignals。"""

    def __init__(self) -> None:
        self.value = None
        self.blocked = False

    def blockSignals(self, flag: bool) -> None:  # noqa: N802
        self.blocked = self.blocked or bool(flag)  # 记录"屏蔽过信号"这件事

    def setValue(self, value: int) -> None:  # noqa: N802
        self.value = int(value)


# --------------------------------------------------------------------------- #
# 动态字号：装不下就缩，装得下就回到基准
# --------------------------------------------------------------------------- #
def test_font_shrinks_so_that_all_rows_still_fit(qapp):
    o = _overlay(display_mode="bilingual")
    try:
        o.update_state(_state("第一条字幕内容", "第二条字幕内容", "第三条字幕内容"))
        base = o.config.font_size
        assert o._draw_font_size == base
        rows_at_base = _rows(o)

        # 拖矮：矮到基准字号放不下，但还没到缩字下限
        _drag_height(o, o.height() - 90, edge="b")
        assert o.config.window_height == o.height()
        assert o._draw_font_size < base, "窗口装不下却没缩字"
        assert o._draw_font_size >= o._min_font_size()
        # 缩完之后所有行都还得放得下（这正是缩字的目的）
        assert _rows(o) * o._line_height() <= _body_h(o)
        assert _rows(o) >= rows_at_base  # 缩字只会让换行更少或持平
    finally:
        o.deleteLater()


def test_font_returns_to_base_when_window_grows_again(qapp):
    """长内容过去 / 窗口拉大之后，字号必须回到设置值（用户明确要求）。"""
    o = _overlay(display_mode="bilingual")
    try:
        o.update_state(_state("第一条字幕内容", "第二条字幕内容", "第三条字幕内容"))
        base = o.config.font_size
        _drag_height(o, o.height() - 90, edge="b")
        assert o._draw_font_size < base

        _drag_height(o, o.height() + 400, edge="b")
        assert o._draw_font_size == base
    finally:
        o.deleteLater()


def test_auto_shrink_can_be_turned_off(qapp):
    o = _overlay(display_mode="bilingual", auto_shrink_font=False)
    try:
        o.update_state(_state("第一条字幕内容", "第二条字幕内容", "第三条字幕内容"))
        base = o.config.font_size
        _drag_height(o, 120, edge="b")
        assert o._draw_font_size == base  # 关了就不许动字号
    finally:
        o.deleteLater()


def test_shrink_respects_floor(qapp):
    """内容远超窗口时缩到下限就停，不会缩成看不见的字。"""
    o = _overlay(display_mode="bilingual", lines_per_subtitle=6, max_lines=3)
    try:
        o.update_state(_state(*[f"第{i}条很长的字幕内容" * 3 for i in range(1, 4)]))
        _drag_height(o, 90, edge="b")
        assert o._draw_font_size == o._min_font_size()
    finally:
        o.deleteLater()
