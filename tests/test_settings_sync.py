"""设置窗与字幕窗的字段同步测试（offscreen 平台，不弹窗）。

守住两件容易悄悄坏掉的事：

1. **保存路径**：设置窗以前有个 ``self.prompt_edit.text()``（QTextEdit 没有
   ``text()``）导致**整个保存流程死掉**——后来才发现。这个用例点一次
   「保存设置」，把关键字段全验一遍。
2. **别处改过的值不能被覆盖**：字幕窗拖边框会改宽度/高度/字号，控制窗有透明度
   滑杆和显示模式按钮；设置窗每次打开都该先把这些值读回来，否则用户拖完窗口
   进来点一下保存，就把自己刚拖的尺寸覆盖回旧值。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.ui.settings import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(qapp, monkeypatch, tmp_path):
    """构造设置窗，但**绝不写真实配置文件**（红线：data/ 是用户数据）。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    cfg = AppConfig()
    w = SettingsWindow(cfg)
    yield w
    w.hide()
    w.deleteLater()


def test_show_syncs_fields_changed_elsewhere(win):
    """字幕窗拖出来的宽/高/字号、控制窗改的透明度与显示模式，都要被读回来。"""
    ov = win.config.overlay
    ov.window_width = 1536
    ov.window_height = 321
    ov.font_size = 41
    ov.window_opacity = 0.7
    ov.display_mode = "source"
    ov.scroll_mode = "replace"

    win.show()  # showEvent → _sync_live_fields()

    assert win.win_width.value() == 1536
    assert win.win_height.value() == 321
    assert win.font_size.value() == 41
    assert win.win_opacity.value() == pytest.approx(0.7)
    assert win.mode_combo.currentData() == "source"
    assert win.scroll_combo.currentData() == "replace"


def test_save_writes_overlay_fields(win):
    """保存路径必须是通的，而且新加的两个外观项真的落到配置里。"""
    win.win_height.setValue(240)
    win.shrink_font.setChecked(False)
    win.font_size.setValue(28)
    win.auto_font.setChecked(False)

    win._save()

    ov = win.config.overlay
    assert ov.window_height == 240
    assert ov.auto_shrink_font is False
    assert ov.font_size == 28
    assert ov.auto_font_scale is False
    assert "已保存" in win.status.text()  # 保存失败时会写 ❌


def test_save_keeps_autoshink_default_on(win):
    win._save()
    assert win.config.overlay.auto_shrink_font is True
