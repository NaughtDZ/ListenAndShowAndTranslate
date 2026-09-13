"""运行中换音频来源（用户在问："为什么进了程序反而调不了监听目标？"）。

这里验三件事：

1. ``SubtitlePipeline.switch_source / restart`` 的调用顺序正确
   （停 → 重建引擎 → 按新音源开工），进程模式与标签页模式都验；
2. ``start()`` 在"目标现在没在发声"时**不再拒绝启动**（先开窗、等它出声自动接上）；
3. 控制窗里的「换音频来源…」真的接上了 pipeline，并且会清掉旧字幕。
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from app.audio.capture import TargetSpec
from app.config import AppConfig
from app.pipeline import SubtitlePipeline


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def pipe(qapp):
    p = SubtitlePipeline(AppConfig())
    yield p
    p.stop(timeout=0.2)


# --------------------------------------------------------------------------- #
# 切换音源：调用顺序
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, monkeypatch, *, start_ok: bool = True, tab_ok: bool = True) -> None:
        monkeypatch.setattr(SubtitlePipeline, "stop", lambda self, timeout=8.0: self.calls_append("stop"), raising=True)
        monkeypatch.setattr(SubtitlePipeline, "prepare", lambda self: (True, "ok"))
        monkeypatch.setattr(SubtitlePipeline, "start", lambda self, spec: self.calls_append("start") or start_ok)
        monkeypatch.setattr(SubtitlePipeline, "start_tab_audio", lambda self, server=None: self.calls_append("start_tab") or tab_ok)

    # 把记录器挂到实例上
    def bind(self, pipeline: SubtitlePipeline) -> None:
        pipeline.calls_append = self.calls.append  # type: ignore[attr-defined]


def test_switch_source_to_another_process(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)

    ok = pipe.switch_source(TargetSpec(pid=4321))

    assert ok is True
    assert rec.calls == ["stop", "start"], "必须先停、再按新音源开工"
    assert pipe._target_spec is not None and pipe._target_spec.pid == 4321
    assert pipe.tab_source is False


def test_switch_source_to_browser_tab(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)

    ok = pipe.switch_source(tab_mode=True)

    assert ok is True
    assert rec.calls == ["stop", "start_tab"]
    assert pipe.tab_source is True
    assert pipe._target_spec is None, "标签页模式不该留着旧的 PID 目标"


def test_switch_back_from_tab_to_process(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)

    pipe.switch_source(tab_mode=True)
    rec.calls.clear()
    pipe.switch_source(TargetSpec(pid=99))

    assert rec.calls == ["stop", "start"]
    assert pipe.tab_source is False


def test_switch_source_reports_failure(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch, start_ok=False)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)
    errors: list[str] = []
    pipe.errorOccurred.connect(errors.append)

    assert pipe.switch_source(TargetSpec(pid=1)) is False
    assert errors and "切换音源失败" in errors[0]


def test_reload_keeps_current_source(qapp, monkeypatch):
    """设置保存后的 reload 不能把音源弄丢（它现在与 switch_source 共用 restart）。"""
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)
    pipe._target_spec = TargetSpec(pid=777)

    assert pipe.reload() is True
    assert rec.calls == ["stop", "start"]
    assert pipe._target_spec is not None and pipe._target_spec.pid == 777


def test_reload_without_source_is_a_noop(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)

    assert pipe.reload() is False
    assert rec.calls == []


def test_restart_without_source_returns_false(qapp, monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    pipe = SubtitlePipeline(AppConfig())
    rec.bind(pipe)

    assert pipe.restart() is False
    assert rec.calls == ["stop"]


# --------------------------------------------------------------------------- #
# 启动时目标不在发声：不再拒绝启动
# --------------------------------------------------------------------------- #
def test_start_waits_when_target_is_silent(qapp, monkeypatch):
    """用户可能"先开字幕窗、再开播放器"：这时不该退出，而是等待并自动重连。"""
    monkeypatch.setattr("app.pipeline.resolve_target", lambda spec: None)
    started: list[str] = []
    monkeypatch.setattr(
        "app.pipeline.CaptureWorker.start", lambda self: started.append(self.spec.describe())
    )
    monkeypatch.setattr("app.pipeline.CaptureWorker.stop", lambda self, timeout=5.0: None)

    pipe = SubtitlePipeline(AppConfig())
    pipe.prepare = lambda: (True, "ok")  # type: ignore[assignment]
    statuses: list[str] = []
    errors: list[str] = []
    pipe.statusChanged.connect(statuses.append)
    pipe.errorOccurred.connect(errors.append)

    ok = pipe.start(TargetSpec(pid=1234))

    assert ok is True, "应该先跑起来，而不是直接失败"
    assert started, "采集线程仍要启动（它会自己重试解析目标）"
    assert any("等待" in s for s in statuses)
    assert errors == []
    pipe.stop(timeout=0.2)


def test_start_reports_resolved_target(qapp, monkeypatch):
    class _Target:
        name = "喜马拉雅.exe"
        pid = 505
        executable = "C:/x/喜马拉雅.exe"

    monkeypatch.setattr("app.pipeline.resolve_target", lambda spec: _Target())
    monkeypatch.setattr("app.pipeline.CaptureWorker.start", lambda self: None)
    monkeypatch.setattr("app.pipeline.CaptureWorker.stop", lambda self, timeout=5.0: None)

    pipe = SubtitlePipeline(AppConfig())
    pipe.prepare = lambda: (True, "ok")  # type: ignore[assignment]
    statuses: list[str] = []
    pipe.statusChanged.connect(statuses.append)

    assert pipe.start(TargetSpec(pid=505)) is True
    assert any("喜马拉雅" in s and "505" in s for s in statuses)
    pipe.stop(timeout=0.2)


# --------------------------------------------------------------------------- #
# 选择对话框
# --------------------------------------------------------------------------- #
class _Proc:
    pid = 111
    name = "喜马拉雅.exe"
    title = "正在播第 3 回"
    executable = "C:/x/喜马拉雅.exe"
    display_label = "喜马拉雅.exe (PID 111)"
    is_system_sounds = False


def _fake_procs(*_a, **_k):
    return [_Proc()]


def test_picker_lists_tab_and_processes(qapp, monkeypatch):
    from app.ui import source_picker

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", _fake_procs)
    dlg = source_picker.SourcePickerDialog()
    try:
        labels = [dlg.list.item(i).text() for i in range(dlg.list.count())]
        assert len(labels) == 2
        assert "浏览器标签页" in labels[0]
        assert "喜马拉雅.exe" in labels[1] and "PID 111" in labels[1]
    finally:
        dlg.deleteLater()


def test_picker_choice_process(qapp, monkeypatch):
    from app.ui import source_picker

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", _fake_procs)
    dlg = source_picker.SourcePickerDialog()
    try:
        dlg.list.setCurrentRow(1)  # 第一个程序
        dlg._accept_selected()
        assert dlg.choice == ("process", 111)
    finally:
        dlg.deleteLater()


def test_picker_choice_tab(qapp, monkeypatch):
    from app.ui import source_picker

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", _fake_procs)
    dlg = source_picker.SourcePickerDialog()
    try:
        dlg.list.setCurrentRow(0)
        dlg._accept_selected()
        assert dlg.choice == ("tab", None)
    finally:
        dlg.deleteLater()


def test_picker_selecting_current_source_acts_like_cancel(qapp, monkeypatch):
    """选的就是当前音源时不要白重启一次（等于什么都没换）。"""
    from app.ui import source_picker

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", _fake_procs)
    dlg = source_picker.SourcePickerDialog(current_pid=111)
    try:
        dlg.list.setCurrentRow(1)
        dlg._accept_selected()
        assert dlg.choice is None
        assert dlg.result() == QDialog.Rejected
    finally:
        dlg.deleteLater()


def test_picker_marks_current_and_empty_hint(qapp, monkeypatch):
    from app.ui import source_picker

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", lambda *a, **k: [])
    dlg = source_picker.SourcePickerDialog(tab_active=True)
    try:
        assert "（当前）" in dlg.list.item(0).text()
        assert "没有程序在输出音频" in dlg.empty.text()
    finally:
        dlg.deleteLater()


def test_picker_survives_enumeration_failure(qapp, monkeypatch):
    from app.ui import source_picker

    def boom(*_a, **_k):
        raise RuntimeError("WASAPI 挂了")

    monkeypatch.setattr(source_picker, "enumerate_audio_processes", boom)
    dlg = source_picker.SourcePickerDialog()
    try:
        assert "❌" in dlg.empty.text()
        assert dlg.list.count() == 1  # 至少还有"浏览器标签页"
    finally:
        dlg.deleteLater()


# --------------------------------------------------------------------------- #
# 控制窗按钮
# --------------------------------------------------------------------------- #
def _control_window(qapp, monkeypatch):
    from app.ui.runner import SubtitleControlWindow
    from app.ui.subtitle_overlay import SubtitleOverlay

    cfg = AppConfig()
    overlay = SubtitleOverlay(cfg.overlay)
    pipe = SubtitlePipeline(cfg)
    win = SubtitleControlWindow(overlay, pipe)
    return win, overlay, pipe


def test_control_window_has_source_button(qapp, monkeypatch):
    win, overlay, pipe = _control_window(qapp, monkeypatch)
    try:
        assert win.source_btn.text() == "换音频来源…"
        assert "换音源" in win.source_btn.toolTip()
    finally:
        win.deleteLater()
        overlay.deleteLater()


def test_control_window_switch_updates_pipeline_and_clears_subtitles(qapp, monkeypatch):
    from app.ui import source_picker

    class _FakeDialog:
        choice = ("process", 4321)

        def __init__(self, *a, **k) -> None:
            pass

        def exec(self) -> int:
            return QDialog.Accepted

    monkeypatch.setattr(source_picker, "SourcePickerDialog", _FakeDialog)
    win, overlay, pipe = _control_window(qapp, monkeypatch)
    try:
        calls: list[tuple] = []
        monkeypatch.setattr(
            SubtitlePipeline,
            "switch_source",
            lambda self, spec=None, tab_mode=False, timeout=15.0: calls.append((spec, tab_mode)) or True,
        )
        # 放一条旧字幕，切完必须清掉（否则看起来像新音源出的）
        pipe.state.add_final("旧音源的一句话", "zh", line_id=1)
        assert pipe.state.lines

        win._switch_source()

        assert calls and calls[0][1] is False
        assert calls[0][0] is not None and calls[0][0].pid == 4321
        assert not pipe.state.lines, "切音源后旧字幕要清掉"
        assert "4321" in win.target_label.text()
    finally:
        win.deleteLater()
        overlay.deleteLater()


def test_control_window_switch_to_tab_mode(qapp, monkeypatch):
    from app.ui import source_picker

    class _FakeDialog:
        choice = ("tab", None)

        def __init__(self, *a, **k) -> None:
            pass

        def exec(self) -> int:
            return QDialog.Accepted

    monkeypatch.setattr(source_picker, "SourcePickerDialog", _FakeDialog)
    win, overlay, pipe = _control_window(qapp, monkeypatch)
    try:
        calls: list[tuple] = []
        monkeypatch.setattr(
            SubtitlePipeline,
            "switch_source",
            lambda self, spec=None, tab_mode=False, timeout=15.0: calls.append((spec, tab_mode)) or True,
        )
        win._switch_source()
        assert calls == [(None, True)]
        assert "浏览器标签页" in win.target_label.text()
    finally:
        win.deleteLater()
        overlay.deleteLater()


def test_control_window_switch_cancelled_does_nothing(qapp, monkeypatch):
    from app.ui import source_picker

    class _CancelDialog:
        choice = None

        def __init__(self, *a, **k) -> None:
            pass

        def exec(self) -> int:
            return QDialog.Rejected

    monkeypatch.setattr(source_picker, "SourcePickerDialog", _CancelDialog)
    win, overlay, pipe = _control_window(qapp, monkeypatch)
    try:
        calls: list[tuple] = []
        monkeypatch.setattr(
            SubtitlePipeline,
            "switch_source",
            lambda self, spec=None, tab_mode=False, timeout=15.0: calls.append((spec, tab_mode)) or True,
        )
        win._switch_source()
        assert calls == []
    finally:
        win.deleteLater()
        overlay.deleteLater()
