"""设置窗里的电平表 / 静音阈值测试（offscreen 平台，不弹窗）。

用户反馈的原话：「这里静音阈值这里也画个电平表和阈值线条呗，前面启动给个 ui，
这里又不给了」——所以这几条守住：设置窗里确实有电平表、阈值线跟着数值实时动、
采集在跑时是真实时读数、没采集时如实显示"未采集"。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.audio.levels import MIN_DB, LevelTracker, threshold_from_db  # noqa: E402
from app.config import AppConfig  # noqa: E402
from app.ui.meter import LevelMeterWidget  # noqa: E402
from app.ui.runner import capture_level  # noqa: E402
from app.ui.settings import MEASURED_SPEECH_DB, SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(qapp, monkeypatch):
    """设置窗，但**绝不写真实配置文件**（data/ 是用户数据）。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    settings = SettingsWindow(AppConfig())
    yield settings
    settings.hide()
    settings.deleteLater()


# --------------------------------------------------------------------------- #
# 电平表控件本身
# --------------------------------------------------------------------------- #
def test_meter_widget_draws_with_markers(qapp):
    """带参考线的电平表能画出来（不能崩，也不能不占位）。"""
    plain = LevelMeterWidget()
    marked = LevelMeterWidget()
    try:
        marked.set_markers([(-66.0, "真实语音 ≈ -66 dBFS")])
        assert marked.markers == [(-66.0, "真实语音 ≈ -66 dBFS")]
        assert marked.minimumHeight() > plain.minimumHeight(), "参考线要占一行，控件得变高"
        pm = marked.grab()  # 真画一遍
        assert pm.width() > 0 and pm.height() >= marked.minimumHeight()
    finally:
        plain.deleteLater()
        marked.deleteLater()


def _line_pixels_are(image, x_center: int, *, want: str, half: int = 3) -> bool:
    """在 x 附近扫一圈，看有没有/偏蓝/或/偏橙/的线像素。

    不能直接比 RGB 常量：虚线是 1.5px 宽 + 抗锯齿，落在绿条或黑卡片上会混色。
    """
    for x in range(x_center - half, x_center + half + 1):
        for y in range(12, 60):
            c = image.pixelColor(x, y)
            if want == "blue" and c.blue() > 140 and c.blue() > c.red() + 40:
                return True
            if want == "orange" and c.red() > 150 and c.blue() < 110 and c.green() < c.red() - 20:
                return True
    return False


def test_both_lines_are_actually_drawn(qapp):
    """橙色阈值线 + 蓝色参考线都必须真的画出来（用户要的就是这两条线）。"""
    m = LevelMeterWidget()
    try:
        m.resize(700, m.minimumHeight())
        m.set_markers([(-66.0, "实测参考")])
        m.set_threshold(threshold_from_db(-45.0))
        m.set_level(LevelTracker().update_values(0.02, 0.09, 0.1))
        image = m.grab().toImage()

        x0, x1 = 10, m.width() - 10
        assert _line_pixels_are(image, int(m._x_for_db(-45.0, x0, x1)), want="orange"), "阈值线没画出来"
        assert _line_pixels_are(image, int(m._x_for_db(-66.0, x0, x1)), want="blue"), "参考线没画出来"
    finally:
        m.deleteLater()


def test_default_threshold_is_visible_on_scale(qapp):
    """默认阈值 -80 必须落在刻度范围内，不能贴在左边缘（否则分不出 -80 和 -70）。"""
    m = LevelMeterWidget()
    try:
        m.resize(700, m.minimumHeight())
        m.set_threshold(threshold_from_db(-80.0))
        x = m._x_for_db(-80.0, 10, m.width() - 10)
        assert x > 15, f"-80 dBFS 被挤到左边缘了（x={x}）"
    finally:
        m.deleteLater()


def test_meter_threshold_line_follows_spinbox(win):
    """阈值数值一变，橙色虚线立刻跟着动（不用点保存）。"""
    win.silence_db.setValue(-50.0)
    assert win.threshold_meter.threshold_linear == pytest.approx(
        threshold_from_db(-50.0)
    )
    win.silence_db.setValue(-75.0)
    assert win.threshold_meter.threshold_linear == pytest.approx(
        threshold_from_db(-75.0)
    )


def test_meter_has_measured_reference_marker(win):
    """蓝色参考线要标出实测电平，用户才知道阈值取在哪合适。"""
    dbs = [db for db, _ in win.threshold_meter.markers]
    assert MEASURED_SPEECH_DB in dbs


def test_show_reads_threshold_changed_by_meter_window(win):
    """独立电平表窗口改的是同一个配置项：设置窗打开时要回读，不能显示旧值。"""
    win.config.audio.silence_rms_threshold_db = -45.0
    win.show()
    assert win.silence_db.value() == pytest.approx(-45.0)
    assert win.threshold_meter.threshold_linear == pytest.approx(
        threshold_from_db(-45.0)
    )


# --------------------------------------------------------------------------- #
# 实时读数
# --------------------------------------------------------------------------- #
def test_without_source_shows_not_collecting(win):
    win.show()
    assert win._level_source is None
    assert win._level_timer.isActive() is False
    win._poll_level()
    assert "未采集" in win.threshold_meter.status


def test_live_source_feeds_meter(qapp, monkeypatch):
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    reading = {"rms": 0.1, "peak": 0.2}
    w = SettingsWindow(AppConfig(), level_source=lambda: (reading["rms"], reading["peak"]))
    w.silence_db.setValue(-80.0)
    try:
        w.show()
        assert w._level_timer.isActive() is True  # 可见时才开始轮询
        w._poll_level()
        assert w.threshold_meter.level.db_rms > MIN_DB + 1
        assert w.threshold_meter.status == "有声"

        # 目标安静下来：读数掉到阈值以下 → 状态变"静音"
        reading["rms"] = 1e-5
        reading["peak"] = 1e-5
        for _ in range(30):  # 平滑是渐进的，多喂几帧
            w._poll_level()
        assert "静音" in w.threshold_meter.status

        w.hide()
        assert w._level_timer.isActive() is False  # 藏起来就别烧 CPU
    finally:
        w.hide()
        w.deleteLater()


def test_broken_source_does_not_break_settings(qapp, monkeypatch):
    """电平来源抛异常时，设置界面必须照常能用（不能连设置都打不开）。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)

    def boom():
        raise RuntimeError("采集线程炸了")

    w = SettingsWindow(AppConfig(), level_source=boom)
    try:
        w.show()
        w._poll_level()
        assert "未采集" in w.threshold_meter.status
        w.win_width.setValue(1000)
        w._save()  # 保存照常
        assert w.config.overlay.window_width == 1000
    finally:
        w.hide()
        w.deleteLater()


def test_clipping_is_reported(qapp, monkeypatch):
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    w = SettingsWindow(AppConfig(), level_source=lambda: (0.999, 1.0))
    try:
        w.show()
        w._poll_level()
        assert "削波" in w.threshold_meter.status
    finally:
        w.hide()
        w.deleteLater()


# --------------------------------------------------------------------------- #
# 字幕进程里的读数来源
# --------------------------------------------------------------------------- #
class _FakePipelineStats:
    last_rms = 0.0123
    last_peak = 0.456


class _FakePipeline:
    """电平读数现在统一走 ``pipeline.pipeline_stats()``。

    进程模式与浏览器标签页模式共用同一套 AudioPipeline，
    所以 runner 不再直接摸 ``pipeline.capture``。
    """

    def __init__(self, stats):
        self._stats = stats

    def pipeline_stats(self):
        return self._stats


def test_capture_level_reads_current_chunk_not_high_water():
    """必须读**最近一块**的峰值：``CaptureStats.peak`` 是整段高水位，
    拿它画 PEAK 条会一直顶在最右边。"""
    assert capture_level(_FakePipeline(_FakePipelineStats())) == (0.0123, 0.456)


def test_capture_level_is_none_when_not_capturing():
    assert capture_level(_FakePipeline(None)) is None


# --------------------------------------------------------------------------- #
# 接线：open_settings_window 要把 options 传下去
# --------------------------------------------------------------------------- #
def test_open_settings_window_passes_level_source(qapp, monkeypatch):
    from app.ui.lifecycle import open_settings_window

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)

    class Owner:
        pass

    owner = Owner()
    source = lambda: (0.5, 0.5)  # noqa: E731
    win = open_settings_window(owner, AppConfig(), options={"level_source": source})
    try:
        assert win._level_source is source
    finally:
        win.hide()
        win.deleteLater()
