"""前端启动器（``听显译.exe`` 的源码）测试。

它不跑业务，只做"找目录 → 检查 venv → 用 pythonw 拉起主程序 → 自己退出"。
这几条守住的是：**双击不弹控制台**（GUI 子系统 + pythonw）、找不到环境时有人话提示、
以及参数能透传（``听显译.exe --settings``）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import frontend


def _fake_root(tmp_path: Path, *, pythonw: bool = True, python: bool = False) -> Path:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    if pythonw:
        (scripts / "pythonw.exe").write_bytes(b"")
    if python:
        (scripts / "python.exe").write_bytes(b"")
    (tmp_path / "main.py").write_text("# 假的入口\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# 找目录 / 找解释器
# --------------------------------------------------------------------------- #
def test_app_root_is_project_root():
    root = frontend.app_root()
    assert (root / "main.py").is_file()


def test_app_root_walks_up_from_exe_location(monkeypatch, tmp_path):
    """exe 被放进 dist/ 之类子目录时，也要能往上找到 main.py。"""
    nested = tmp_path / "dist" / "exe"
    nested.mkdir(parents=True)
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    fake_exe = nested / "听显译.exe"
    fake_exe.write_bytes(b"")

    monkeypatch.setattr(frontend.sys, "frozen", True, raising=False)
    monkeypatch.setattr(frontend.sys, "executable", str(fake_exe))

    assert frontend.app_root() == tmp_path


def test_find_pythonw_prefers_pythonw(tmp_path):
    root = _fake_root(tmp_path, pythonw=True, python=True)
    exe = frontend.find_pythonw(root)
    assert exe is not None and exe.name == "pythonw.exe", "pythonw 才是没有控制台的那个"


def test_find_pythonw_falls_back_to_python(tmp_path):
    root = _fake_root(tmp_path, pythonw=False, python=True)
    exe = frontend.find_pythonw(root)
    assert exe is not None and exe.name == "python.exe"


def test_find_pythonw_missing(tmp_path):
    assert frontend.find_pythonw(tmp_path) is None


# --------------------------------------------------------------------------- #
# 拉起主程序
# --------------------------------------------------------------------------- #
def test_launch_uses_pythonw_detached_and_forwards_args(monkeypatch, tmp_path):
    root = _fake_root(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return object()

    monkeypatch.setattr(frontend.subprocess, "Popen", fake_popen)
    code = frontend.launch(root, root / ".venv" / "Scripts" / "pythonw.exe", ["--settings"])

    assert code == 0
    cmd, kwargs = calls[0]
    assert cmd[1].endswith("main.py") and cmd[2] == "--settings"
    assert "pythonw.exe" in cmd[0]
    assert kwargs["cwd"] == str(root)
    assert kwargs["creationflags"] & frontend.subprocess.DETACHED_PROCESS
    """必须 DETACHED：父进程（exe）立刻退出，主程序要活下来。"""


def test_launch_reports_failure_with_dialog(monkeypatch, tmp_path):
    root = _fake_root(tmp_path)
    messages: list[str] = []

    def boom(*_a, **_k):
        raise OSError("no such file")

    monkeypatch.setattr(frontend.subprocess, "Popen", boom)
    monkeypatch.setattr(frontend, "_message", lambda text, **kw: messages.append(text) or 0)

    code = frontend.launch(root, root / ".venv" / "Scripts" / "pythonw.exe", [])
    assert code == 3
    assert messages and "启动失败" in messages[0]


# --------------------------------------------------------------------------- #
# 没装环境时的引导
# --------------------------------------------------------------------------- #
def test_main_warns_when_venv_missing(monkeypatch, tmp_path):
    messages: list[str] = []
    monkeypatch.setattr(frontend, "app_root", lambda: tmp_path)
    monkeypatch.setattr(frontend, "_message", lambda text, **kw: messages.append(text) or 7)  # 7 = 否

    assert frontend.main([]) == 2
    assert messages and "首次安装" in messages[0]


def test_main_runs_installer_when_user_says_yes(monkeypatch, tmp_path):
    (tmp_path / frontend.INSTALL_BAT).write_text("@echo off\n", encoding="utf-8")
    launched: list[list[str]] = []

    monkeypatch.setattr(frontend, "app_root", lambda: tmp_path)
    monkeypatch.setattr(frontend, "_message", lambda text, **kw: 6)  # 6 = 是
    monkeypatch.setattr(
        frontend.subprocess, "Popen", lambda cmd, **kw: launched.append(list(cmd))
    )

    assert frontend.main([]) == 0
    assert launched and launched[0][0].endswith("cmd.exe")
    assert launched[0][2].endswith(frontend.INSTALL_BAT)


def test_main_passes_argv_to_launch(monkeypatch, tmp_path):
    root = _fake_root(tmp_path)
    seen: list[list[str]] = []
    monkeypatch.setattr(frontend, "app_root", lambda: root)
    monkeypatch.setattr(frontend, "launch", lambda r, p, argv: seen.append(argv) or 0)

    assert frontend.main(["--list-audio"]) == 0
    assert seen == [["--list-audio"]]
