"""「重新运行首次运行向导」入口 + 重跑安全性的测试（offscreen 平台）。

用户反馈：跑了几次之后想重新跑向导来**改模型下载**，但**没有入口**——
以前只有 ``first_run_done`` 为假时才自动跑一次。

补入口很容易，难的是**重跑别把用户的设置清回出厂值**：
向导第 ③ 页的代理输入框默认是空的、第 ④ 页翻译通道默认选 LLM，
如果重跑时不用当前配置预填，用户"只想再下个日语包"就会顺手丢掉：
代理、翻译通道（甚至百度/有道这些向导里没有的通道）、延迟档位。
下面四条用例专门守这个。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.ui import wizard as wizard_mod  # noqa: E402
from app.ui.settings import SettingsWindow  # noqa: E402
from app.ui.wizard import FirstRunWizard  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def no_save(monkeypatch):
    """绝不让测试往真实 data/config.json 里写（红线：那是用户数据）。"""
    monkeypatch.setattr(AppConfig, "save", lambda self: None, raising=True)


class _FakeHardware:
    recommended_tier = "mid"
    recommended_packs = ["core", "zh"]
    notes = ["（测试用假硬件信息）"]

    def summary_lines(self) -> list[str]:
        return ["✅ 假硬件：测试环境"]


@pytest.fixture()
def fake_hardware(monkeypatch):
    monkeypatch.setattr(wizard_mod, "detect_hardware", lambda: _FakeHardware())


# --------------------------------------------------------------------------- #
# 入口：启动窗口
# --------------------------------------------------------------------------- #
def test_launcher_has_wizard_entry(qapp, monkeypatch):
    from app.ui.launcher import LauncherWindow

    seen: list[AppConfig] = []

    def fake_run_wizard(config=None, **_kw):
        seen.append(config)
        return 1  # Accepted

    monkeypatch.setattr(wizard_mod, "run_wizard", fake_run_wizard)
    win = LauncherWindow(AppConfig())
    win._timer.stop()
    try:
        assert win.wizard_btn.isEnabled()
        win.wizard_btn.click()
        assert seen and seen[0] is win.config, "向导必须吃同一份 config 对象"
        assert "向导已完成" in win.hint.text()
    finally:
        win._timer.stop()
        win.deleteLater()


# --------------------------------------------------------------------------- #
# 入口：设置窗（并且完成要立刻生效）
# --------------------------------------------------------------------------- #
def _settings(qapp, no_save, *, wizard_result=1, mutate=None):
    """返回 (settings_mod, 假的 run_wizard)。"""

    def fake_run_wizard(config=None, **_kw):
        if mutate is not None:
            mutate(config)
        return wizard_result

    import app.ui.settings as settings_mod  # noqa: PLC0415

    return settings_mod, fake_run_wizard


def test_settings_wizard_entry_applies_immediately(qapp, monkeypatch, no_save):
    def mutate(cfg: AppConfig) -> None:
        # 模拟用户在向导里把档位从"平衡"改成"最准"
        cfg.asr.apply_latency_preset("accurate")

    settings_mod, fake = _settings(qapp, no_save, mutate=mutate)
    monkeypatch.setattr(wizard_mod, "run_wizard", fake)
    win = SettingsWindow(AppConfig())
    applied: list[bool] = []
    win.saved.connect(lambda: applied.append(True))
    try:
        assert win.wizard_btn.isEnabled()
        win.wizard_btn.click()
        assert applied == [True], "向导完成后必须发 saved（否则引擎不会重建）"
        assert "向导已完成" in win.status.text()
        # 档位按钮与五条旋钮都要跟着向导改后的配置走
        assert win.config.asr.latency_preset == "accurate"
        assert win.preset_buttons["accurate"].isChecked()
        for knob in settings_mod.LATENCY_KNOBS:
            section = win.config.asr if knob.section == "asr" else win.config.asr.vad
            assert win.knob_sliders[knob.field].value() == getattr(section, knob.field)
            assert win.knob_labels[knob.field].text() == f"{getattr(section, knob.field)} ms"
    finally:
        win.hide()
        win.deleteLater()


def test_settings_wizard_cancel_changes_nothing(qapp, monkeypatch, no_save):
    _, fake = _settings(qapp, no_save, wizard_result=0)
    monkeypatch.setattr(wizard_mod, "run_wizard", fake)
    win = SettingsWindow(AppConfig())
    applied: list[bool] = []
    win.saved.connect(lambda: applied.append(True))
    try:
        win.wizard_btn.click()
        assert applied == [], "取消向导不许触发引擎重建"
        assert "取消" in win.status.text()
    finally:
        win.hide()
        win.deleteLater()


# --------------------------------------------------------------------------- #
# 重跑安全性：以"现状"为默认值
# --------------------------------------------------------------------------- #
def test_rerun_is_auto_detected_from_config(qapp):
    """调用方不用记传 rerun：``first_run_done`` 为真就按"重新配置"跑。"""
    done = AppConfig()
    done.first_run_done = True
    wiz = FirstRunWizard(done)
    try:
        assert wiz.rerun is True
        assert "重新配置" in wiz.windowTitle()
    finally:
        wiz.deleteLater()

    fresh = AppConfig()
    fresh.first_run_done = False
    wiz2 = FirstRunWizard(fresh)
    try:
        assert wiz2.rerun is False
        assert "首次运行" in wiz2.windowTitle()
    finally:
        wiz2.deleteLater()


def test_run_wizard_passes_explicit_rerun_through(qapp, monkeypatch):
    seen: list[object] = []

    class _Fake:
        def __init__(self, config=None, *, rerun=None) -> None:  # noqa: D107
            seen.append(rerun)

        def show(self) -> None: ...

        def exec(self) -> int:
            return 1

    monkeypatch.setattr(wizard_mod, "FirstRunWizard", _Fake)
    # 这个替身没有真窗口，跳过"夹进屏幕"那一步（那条逻辑另有专门用例守着）
    monkeypatch.setattr("app.ui.lifecycle.fit_window_to_screen", lambda window: None)
    cfg = AppConfig()
    assert wizard_mod.run_wizard(cfg) == 1
    assert wizard_mod.run_wizard(cfg, rerun=False) == 1
    assert seen == [None, False]  # 不传 = 让向导自己按配置判断


def test_rerun_preselects_installed_packs(qapp, fake_hardware, monkeypatch):
    """重跑时勾"已经装好的包"，而不是把用户重置成硬件推荐那一套。"""
    monkeypatch.setattr(wizard_mod, "installed_packs", lambda: ["core", "zh"])
    cfg = AppConfig()
    cfg.first_run_done = True
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        wiz.page2.initializePage()
        assert wiz.page2.selected_packs() == ["core", "zh"]
    finally:
        wiz.deleteLater()


def test_rerun_uses_current_tier(qapp, fake_hardware, monkeypatch):
    monkeypatch.setattr(wizard_mod, "installed_packs", lambda: ["core"])
    cfg = AppConfig()
    cfg.first_run_done = True
    cfg.asr.latency_preset = "realtime"
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        wiz.page2.initializePage()
        assert wiz.page2.tier_buttons["low"].isChecked()
    finally:
        wiz.deleteLater()


def test_first_run_still_uses_hardware_recommendation(qapp, fake_hardware, monkeypatch):
    monkeypatch.setattr(wizard_mod, "installed_packs", lambda: ["core", "zh"])
    cfg = AppConfig()
    cfg.first_run_done = False
    wiz = FirstRunWizard(cfg, rerun=False)
    try:
        wiz.page2.initializePage()
        assert wiz.page2.tier_buttons["mid"].isChecked()  # 假硬件推荐 mid
        assert "ja-ko-yue" in wiz.page2.selected_packs()  # 按档位的推荐组合，不是「已装的」
    finally:
        wiz.deleteLater()


def test_rerun_size_label_says_what_will_really_download(
    qapp, fake_hardware, monkeypatch
):
    """用户是来"改模型下载"的：得让他看出哪些**不会**再下。"""
    monkeypatch.setattr(wizard_mod, "installed_packs", lambda: ["core", "zh"])
    cfg = AppConfig()
    cfg.first_run_done = True
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        wiz.page2.initializePage()
        text = wiz.page2.size_label.text()
        assert "已经装好" in text and "2" in text
    finally:
        wiz.deleteLater()


def test_first_run_size_label_has_no_installed_claim(qapp, fake_hardware, monkeypatch):
    monkeypatch.setattr(wizard_mod, "installed_packs", lambda: ["core", "zh"])
    cfg = AppConfig()
    cfg.first_run_done = False
    wiz = FirstRunWizard(cfg, rerun=False)
    try:
        wiz.page2.initializePage()
        assert "已经装好" not in wiz.page2.size_label.text()
    finally:
        wiz.deleteLater()


def test_rerun_prefills_proxy(qapp, monkeypatch):
    cfg = AppConfig()
    cfg.first_run_done = True
    cfg.proxy = "http://127.0.0.1:2333"
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        monkeypatch.setattr(AppConfig, "save", lambda self: None)
        wiz.page3.initializePage()
        assert wiz.page3.proxy_edit.text() == "http://127.0.0.1:2333"
        wiz._apply()
        assert cfg.proxy == "http://127.0.0.1:2333", "重跑不许把代理清空"
    finally:
        wiz.deleteLater()


def test_first_run_proxy_stays_empty(qapp, monkeypatch):
    """首次运行时代理必须留空（不预填任何人的地址）。"""
    cfg = AppConfig()
    cfg.first_run_done = False
    wiz = FirstRunWizard(cfg, rerun=False)
    try:
        wiz.page3.initializePage()
        assert wiz.page3.proxy_edit.text() == ""
    finally:
        wiz.deleteLater()


def test_rerun_keeps_translate_channel_not_offered_by_wizard(qapp, monkeypatch, no_save):
    """向导里没有百度/有道这些通道；重跑完成时**不能**把它们偷偷换成 LLM。"""
    cfg = AppConfig()
    cfg.first_run_done = True
    cfg.translate.provider = "baidu"
    cfg.translate.llm.base_url = "http://127.0.0.1:1234/v1"
    cfg.translate.llm.model = "qwen3-8b"
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        wiz.page4.initializePage()
        assert wiz.page4.provider.currentData() == "baidu"
        wiz._apply()
        assert cfg.translate.provider == "baidu"
        assert cfg.translate.enabled is True
    finally:
        wiz.deleteLater()


def test_rerun_prefills_translate_fields(qapp, monkeypatch, no_save):
    cfg = AppConfig()
    cfg.first_run_done = True
    cfg.translate.provider = "llm"
    cfg.translate.llm.base_url = "http://127.0.0.1:5555/v1"
    cfg.translate.llm.model = "sakura-galtransl"
    cfg.translate.llm.prompt_style = "plain"
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        wiz.page4.initializePage()
        assert wiz.page4.provider.currentData() == "llm"
        assert wiz.page4.base_url.text() == "http://127.0.0.1:5555/v1"
        assert wiz.page4.model.text() == "sakura-galtransl"
        assert wiz.page4.style.currentData() == "plain"
        wiz._apply()
        assert cfg.translate.llm.model == "sakura-galtransl"
        assert cfg.translate.llm.prompt_style == "plain"
    finally:
        wiz.deleteLater()


def test_wizard_title_shows_rerun(qapp):
    cfg = AppConfig()
    cfg.first_run_done = True
    wiz = FirstRunWizard(cfg, rerun=True)
    try:
        assert "重新配置" in wiz.windowTitle()
    finally:
        wiz.deleteLater()
    cfg2 = AppConfig()
    cfg2.first_run_done = False
    wiz2 = FirstRunWizard(cfg2, rerun=False)
    try:
        assert "首次运行" in wiz2.windowTitle()
    finally:
        wiz2.deleteLater()
