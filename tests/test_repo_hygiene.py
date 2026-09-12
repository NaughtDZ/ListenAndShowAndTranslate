"""仓库卫生：**源码必须真的进了 Git**。

为什么专门写这个：2026-09-12 复查公开仓库时发现 `.gitignore` 里一句
``models/``（本意是忽略运行时下载的模型目录）把 **``app/models/`` 整个源码包**
也忽略掉了 —— ``registry.py`` / ``downloader.py`` / ``hardware.py``
**从来没进过仓库**，而 ``tests/test_models.py`` 却引用了它们：
别人 clone 下来直接 import 失败，本机却一切正常（文件在磁盘上）。

这类问题在本地永远看不见，所以这里用 git 自己来查：
只要 ``.py`` 被 ignore，或者必需文件没被 track，就报错。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# 这些目录下的文件都是"要跟着仓库走的"（.py 源码 + 文档）
CHECKED_DIRS = ("app", "scripts", "tests", "docs")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )


@pytest.fixture(scope="module", autouse=True)
def _need_git():
    if not (ROOT / ".git").exists():
        pytest.skip("不是 git 工作区（比如下载的 ZIP），跳过仓库卫生检查")
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("git 不可用")


def _python_files() -> list[Path]:
    out: list[Path] = []
    for d in CHECKED_DIRS:
        base = ROOT / d
        if base.is_dir():
            out.extend(
                p for p in base.rglob("*.py") if "__pycache__" not in p.parts
            )
    return sorted(out)


def test_no_python_source_is_gitignored():
    """任何 .py 都不许被 .gitignore 吃掉（曾经 app/models/ 整包被吃）。"""
    ignored: list[str] = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if _git("check-ignore", "-q", rel).returncode == 0:
            ignored.append(rel)
    assert not ignored, (
        "这些源码被 .gitignore 忽略了（.gitignore 里的目录模式要加 `/` 锚定）：\n  "
        + "\n  ".join(ignored)
    )


def test_all_python_source_is_tracked():
    """源码不只是"没被忽略"，还必须**真的被 git 跟踪**（否则 clone 不到）。"""
    untracked: list[str] = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if _git("ls-files", "--error-unmatch", rel).returncode != 0:
            untracked.append(rel)
    assert not untracked, "这些源码没进 Git（新文件记得 git add）：\n  " + "\n  ".join(untracked)


def test_key_files_are_tracked():
    """显式点名几个"少了就彻底跑不起来"的文件，失败信息更好读。"""
    must_have = (
        "main.py",
        "frontend.py",
        "app/models/registry.py",
        "app/models/downloader.py",
        "app/models/hardware.py",
        "app/ui/lifecycle.py",
        "requirements.txt",
        "启动.bat",
    )
    missing = [rel for rel in must_have if _git("ls-files", "--error-unmatch", rel).returncode != 0]
    assert not missing, "仓库里缺少必需文件：\n  " + "\n  ".join(missing)


def test_no_user_data_is_tracked():
    """红线：data/ 与 .venv/ 永远不许进仓库。"""
    tracked = _git("ls-files").stdout.splitlines()
    bad = [
        f for f in tracked
        if f.startswith(("data/", ".venv/")) or f.endswith((".onnx", ".bin", ".wav", ".mp3"))
    ]
    assert not bad, "用户数据/模型/音频进了仓库：\n  " + "\n  ".join(bad)
