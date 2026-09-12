"""设置窗「不能比屏幕高」的测试（offscreen 平台）。

用户反馈：「设置页加了模型选择之后窗口会很长很长，标题栏被挤到屏幕顶部外面，
用户拖动不了了」。

根因是**最小高度**：实测 ``minimumSizeHint`` 曾经是 **692×1275**——因为「识别」
那一页是一根长竖列（语言 + 阈值 + 电平表 + 延迟档位 + 五个滑杆 + 模型选择 + 向导），
它的最小高度 1199px 被 QTabWidget 继承成了整窗的最小高度。Qt 会照这个最小尺寸摆窗口，
于是标题栏落在屏幕外头。

两条一起修：

1. 拆页 + 每页可滚动：新增「模型」分页，每个分页都套 QScrollArea；
2. 显示前把窗口夹进屏幕（``fit_window_to_screen``），尺寸和位置都夹。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea, QTabWidget  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.ui.lifecycle import fit_window_to_screen  # noqa: E402
from app.ui.settings import SettingsWindow  # noqa: E402

# 一台 768p 笔记本的可用高度大约就这么点；窗口最小高度必须比它小，
# 否则标题栏一定会被顶到屏幕外。
SMALL_SCREEN_H = 700


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(qapp, monkeypatch):
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    w = SettingsWindow(AppConfig())
    yield w
    w.hide()
    w.deleteLater()


def test_window_can_shrink_to_small_screen(win):
    """**核心回归**：窗口最小高度必须小到能放进 768p 屏幕。"""
    assert win.minimumSizeHint().height() <= SMALL_SCREEN_H, (
        f"最小高度 {win.minimumSizeHint().height()}px —— 标题栏又会被顶出屏幕"
    )


def test_tabs_count_and_model_tab_exists(win):
    tabs = win.findChild(QTabWidget)
    labels = [tabs.tabText(i) for i in range(tabs.count())]
    assert "模型" in labels, "模型选择应该单独一页，别全堆在「识别」里"
    assert labels[:2] == ["网络", "翻译"]
    # 模型下拉框在「模型」页里
    model_page = tabs.widget(labels.index("模型"))
    assert model_page is not None
    for combo in win.model_combos.values():
        assert model_page.isAncestorOf(combo), "模型下拉框应该在「模型」页"


def test_every_tab_is_scrollable_or_small(win):
    """每个分页要么自己带滚动区域，要么最小高度本来就很小。"""
    tabs = win.findChild(QTabWidget)
    for i in range(tabs.count()):
        page = tabs.widget(i)
        name = tabs.tabText(i)
        has_scroll = isinstance(page, QScrollArea) or page.findChild(QScrollArea) is not None
        assert has_scroll, f"「{name}」页没有滚动区域，窗口一缩内容就够不着了"
        assert page.minimumSizeHint().height() <= 400, f"「{name}」页最小高度太大"


def test_fit_window_to_screen_clamps_size_and_position(qapp):
    """把窗口夹进屏幕：尺寸不超过屏幕，标题栏留在屏幕里。"""
    w = SettingsWindow(AppConfig())
    try:
        screen = qapp.primaryScreen().availableGeometry()
        w.resize(screen.width() * 3, screen.height() * 3)
        w.move(-screen.width(), -screen.height())
        fit_window_to_screen(w)
        assert w.width() <= int(screen.width() * 0.95)
        assert w.height() <= int(screen.height() * 0.95)
        assert w.x() >= 0 and w.y() >= 0, "标题栏跑到屏幕外了"
        assert w.y() + 30 <= screen.height()
    finally:
        w.hide()
        w.deleteLater()


def test_first_show_fits_even_if_content_is_huge(qapp, monkeypatch):
    """内容把窗口撑爆时，第一次显示就该被夹回屏幕内。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    w = SettingsWindow(AppConfig())
    try:
        screen = qapp.primaryScreen().availableGeometry()
        w.resize(screen.width() * 4, screen.height() * 4)
        w.move(-50, -50)
        w.show()  # showEvent → fit_window_to_screen
        assert w.width() <= int(screen.width() * 0.95)
        assert w.height() <= int(screen.height() * 0.95)
        assert w.y() >= 0
    finally:
        w.hide()
        w.deleteLater()


def test_user_resize_is_not_fought_after_first_show(qapp, monkeypatch):
    """夹一次就够：之后用户自己拖大的尺寸不该被反复纠正。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    w = SettingsWindow(AppConfig())
    try:
        w.show()
        w.resize(600, 480)
        w.hide()
        w.show()
        assert w.width() == 600 and w.height() == 480
    finally:
        w.hide()
        w.deleteLater()
