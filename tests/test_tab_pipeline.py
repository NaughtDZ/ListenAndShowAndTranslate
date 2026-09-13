"""标签页音源在流水线/入口上的接线。

这里只验"接线"（信号映射、读数转发、命令行入口、配置默认值），
真实浏览器端到端在 `scripts/probe_tab_capture_e2e.py`（需要 Edge + 扩展）。
"""

from __future__ import annotations

import inspect

import pytest
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.pipeline import SubtitlePipeline


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = AppConfig()
    assert cfg.tab_audio.enabled is True
    assert cfg.tab_audio.port == 38991
    assert cfg.tab_audio.token == ""
    assert cfg.tab_audio.require_extension_origin is True


def test_source_mode_accepts_tab():
    cfg = AppConfig()
    cfg.audio.source_mode = "tab"
    assert cfg.audio.source_mode == "tab"


def test_config_roundtrip_keeps_tab_audio(tmp_path):
    cfg = AppConfig()
    cfg.tab_audio.port = 40001
    cfg.tab_audio.token = "秘密"
    path = tmp_path / "config.json"
    cfg.save(path)
    back = AppConfig.load(path)
    assert back.tab_audio.port == 40001
    assert back.tab_audio.token == "秘密"


def test_redacted_config_masks_token(tmp_path):
    """导出配置不能把配对码带出去（字段名含 token，走 _SENSITIVE_HINTS）。"""
    cfg = AppConfig()
    cfg.tab_audio.token = "abc123"
    assert cfg.redacted()["tab_audio"]["token"] == "<redacted>"


# --------------------------------------------------------------------------- #
# 状态信号
# --------------------------------------------------------------------------- #
class _Stats:
    """TabAudioStats 的最小替身。"""

    connected = True
    capturing = True
    browser = "Edge"
    extension_id = "abcdefg"
    likely_playing = True
    last_error = ""

    class tab:
        id = 12
        title = "某个视频"

    def describe(self):
        return "标签页: 某个视频"


def _pipeline() -> SubtitlePipeline:
    return SubtitlePipeline(AppConfig())


def test_tab_state_maps_to_signal(qapp):
    """状态回调在 WS 线程里跑，只允许 emit 信号（不能碰控件）。"""
    pipe = _pipeline()
    seen: list[dict] = []
    pipe.tabStateChanged.connect(seen.append)

    pipe._on_tab_state(_Stats())

    assert seen, "应当发出 tabStateChanged"
    payload = seen[-1]
    assert payload["connected"] is True
    assert payload["capturing"] is True
    assert payload["tab_title"] == "某个视频"
    assert payload["tab_id"] == 12
    assert payload["extension_id"] == "abcdefg"
    assert "某个视频" in payload["describe"]
    pipe.stop(timeout=0.1)


def test_tab_state_error_is_forwarded(qapp):
    class Bad(_Stats):
        connected = False
        capturing = False
        last_error = "配对码不正确"

        def describe(self):
            return "等待浏览器扩展连接…"

    pipe = _pipeline()
    errors: list[str] = []
    pipe.errorOccurred.connect(errors.append)
    pipe._on_tab_state(Bad())
    assert errors == ["配对码不正确"]
    pipe.stop(timeout=0.1)


def test_pipeline_stats_is_none_without_any_source(qapp):
    pipe = _pipeline()
    assert pipe.pipeline_stats() is None
    pipe.stop(timeout=0.1)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def test_cli_has_tab_flag():
    from main import build_parser

    args = build_parser().parse_args(["--tab"])
    assert args.tab is True
    assert build_parser().parse_args([]).tab is False


def test_run_subtitles_accepts_tab_mode():
    from app.ui.runner import run_subtitles

    sig = inspect.signature(run_subtitles)
    assert "tab_mode" in sig.parameters
    assert sig.parameters["tab_mode"].default is False


# --------------------------------------------------------------------------- #
# 设置界面：端口 / 配对码
# --------------------------------------------------------------------------- #
def test_settings_roundtrips_tab_audio_fields(qapp, monkeypatch):
    """端口与配对码必须在设置界面里能改、能存、能被读回来。

    不能只让用户去手改配置文件——端口冲突是最常见的求助场景。
    """
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    win = SettingsWindow(AppConfig())
    try:
        win.tab_port.setValue(40002)
        win.tab_token.setText("配对码123")
        win.tab_enabled.setChecked(False)
        win._save()

        cfg = win.config
        assert cfg.tab_audio.port == 40002
        assert cfg.tab_audio.token == "配对码123"
        assert cfg.tab_audio.enabled is False
    finally:
        win.hide()
        win.deleteLater()


def test_settings_show_syncs_tab_audio_fields(qapp, monkeypatch):
    """配置文件被手改过（或向导改过）时，打开设置窗要读回来。"""
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)
    cfg = AppConfig()
    cfg.tab_audio.port = 41111
    cfg.tab_audio.token = "abc"
    win = SettingsWindow(cfg)
    try:
        win.show()  # showEvent → _sync_live_fields()
        assert win.tab_port.value() == 41111
        assert win.tab_token.text() == "abc"
    finally:
        win.hide()
        win.deleteLater()


def test_start_tab_audio_binds_callbacks_and_stops(qapp, monkeypatch):
    """start_tab_audio 必须把 on_chunk/on_state 接上，并在 stop 时清理干净。"""
    from app.audio.tab_audio import TabAudioServer

    started: list[TabAudioServer] = []

    def fake_start(self):
        started.append(self)

    monkeypatch.setattr(TabAudioServer, "start", fake_start)
    monkeypatch.setattr(SubtitlePipeline, "prepare", lambda self: (True, ""))

    pipe = _pipeline()
    assert pipe.start_tab_audio() is True
    server = pipe.tab_server
    assert server is not None and started == [server]
    assert server._on_chunk == pipe._on_audio
    assert server._on_state == pipe._on_tab_state

    pipe.stop(timeout=0.5)
    assert pipe.tab_server is None


def test_start_tab_audio_reports_port_conflict(qapp, monkeypatch):
    """端口被占用要变成一条明确的错误，而不是抛异常把程序带崩。"""
    from app.audio.tab_audio import TabAudioServer

    def boom(self):
        raise OSError("端口被占用")

    monkeypatch.setattr(TabAudioServer, "start", boom)
    monkeypatch.setattr(SubtitlePipeline, "prepare", lambda self: (True, ""))

    pipe = _pipeline()
    errors: list[str] = []
    pipe.errorOccurred.connect(errors.append)
    assert pipe.start_tab_audio() is False
    assert errors and "被占用" in errors[0]
    pipe.stop(timeout=0.5)
