"""ListenAndShowAndTranslate 入口。

P0 阶段提供环境自检：
    python main.py --selftest      检查解释器/依赖/目录/硬件
    python main.py --list-audio    列出当前正在发声的进程（等价音量合成器）
"""

from __future__ import annotations

import argparse
import platform
import sys
from typing import Callable

from app import __version__, paths
from app.config import AppConfig
from app.utils.log import get_logger

log = get_logger("main")

REQUIRED_PY = (3, 12)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) == REQUIRED_PY
    in_venv = sys.prefix != sys.base_prefix
    detail = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if not in_venv:
        return False, detail + "  ✗ 未在虚拟环境中运行"
    if not ok:
        want = ".".join(map(str, REQUIRED_PY))
        return False, detail + f"  ✗ 需要 {want}.x（见 计划书.md 第 1.1 节）"
    return True, detail


def _check_import(module: str, label: str | None = None) -> tuple[bool, str]:
    label = label or module
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            from importlib.metadata import version as _v  # noqa: PLC0415

            try:
                ver = _v(label.replace("_", "-"))
            except Exception:
                ver = "?"
        return True, f"{label} {ver}"
    except Exception as exc:  # noqa: BLE001 - 自检要吞掉所有异常
        return False, f"{label} 导入失败: {exc}"


def _check_dirs() -> tuple[bool, str]:
    try:
        paths.ensure_dirs()
        return True, str(paths.DATA_DIR)
    except OSError as exc:
        return False, f"无法创建数据目录: {exc}"


def _check_proxy(cfg: AppConfig) -> tuple[bool, str]:
    if not cfg.proxy:
        return True, "未配置（直连）"
    import socket
    from urllib.parse import urlparse

    u = urlparse(cfg.proxy)
    host, port = u.hostname or "", u.port or 0
    if not host or not port:
        return False, f"代理地址无法解析: {cfg.proxy}"
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect((host, port))
        return True, f"{cfg.proxy} 可达"
    except OSError as exc:
        return False, f"{cfg.proxy} 不可达: {exc}"
    finally:
        s.close()


def _check_gpu() -> tuple[bool, str]:
    """用 nvidia-smi 探测 N 卡；失败不算错误（A 卡/核显正常）。"""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return True, "未检测到 nvidia-smi（A 卡/核显属正常，将走 Vulkan 或 CPU）"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        line = (out.stdout or "").strip().splitlines()
        return True, line[0] if line else "未知"
    except Exception as exc:  # noqa: BLE001
        return True, f"探测失败（不影响运行）: {exc}"


def selftest() -> int:
    print(f"\n=== ListenAndShowAndTranslate v{__version__} 环境自检 ===\n")
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("Python / venv", _check_python),
        ("OS", lambda: (True, f"{platform.system()} {platform.release()} {platform.version()}")),
        ("PySide6", lambda: _check_import("PySide6", "PySide6")),
        ("proc-tap（音频采集）", lambda: _check_import("proctap", "proc-tap")),
        ("sherpa-onnx（ASR）", lambda: _check_import("sherpa_onnx", "sherpa-onnx")),
        ("numpy", lambda: _check_import("numpy", "numpy")),
        ("soxr（重采样）", lambda: _check_import("soxr", "soxr")),
        ("soundfile", lambda: _check_import("soundfile", "soundfile")),
        ("httpx", lambda: _check_import("httpx", "httpx")),
        ("pydantic", lambda: _check_import("pydantic", "pydantic")),
        ("platformdirs", lambda: _check_import("platformdirs", "platformdirs")),
        ("数据目录", _check_dirs),
        ("配置文件", lambda: _config_check()),
        ("网络代理", lambda: _check_proxy(AppConfig.load())),
        ("GPU", _check_gpu),
    ]

    failed = 0
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"检查本身抛异常: {exc!r}"
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name:<22} {detail}")
        if not ok:
            failed += 1

    print()
    if failed:
        print(f"自检未通过：{failed} 项失败。请对照 计划书.md 第 1.1 节排查。\n")
        return 1
    print("自检全部通过。\n")
    return 0


def _config_check() -> tuple[bool, str]:
    cfg = AppConfig.load()
    return True, f"{paths.CONFIG_FILE} (engine={cfg.asr.engine}, translate={cfg.translate.provider})"


# --------------------------------------------------------------------------- #
# 列出正在发声的进程
# --------------------------------------------------------------------------- #
def list_audio_processes() -> int:
    """枚举当前正在输出音频的进程——UI 里那个"选择音频来源"下拉框的数据源。"""
    from app.audio.process_list import enumerate_audio_processes

    procs = enumerate_audio_processes()
    if not procs:
        print("当前没有任何进程在输出音频。")
        return 0

    print(f"\n{'PID':>8}  {'进程名':<32} {'窗口标题'}")
    print("-" * 90)
    for p in procs:
        print(f"{p.pid:>8}  {p.name:<32} {p.title}")
    print(f"\n共 {len(procs)} 个进程在发声。\n")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lst",
        description="只监听指定程序音频的实时字幕/翻译悬浮窗",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--selftest", action="store_true", help="环境自检并退出")
    p.add_argument("--list-audio", action="store_true", help="列出当前正在发声的进程")
    p.add_argument("--config", metavar="PATH", help="使用指定配置文件")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config:
        from pathlib import Path

        paths.CONFIG_FILE = Path(args.config).resolve()

    paths.ensure_dirs()
    log.info("启动 v%s | Python %s | %s", __version__, sys.version.split()[0], paths.DATA_DIR)

    if args.selftest:
        return selftest()
    if args.list_audio:
        return list_audio_processes()

    # 无参数：P0 阶段先做自检；P3 起换成启动 GUI
    print(f"ListenAndShowAndTranslate v{__version__}")
    print("GUI 尚未实现（计划书 P3 阶段）。当前可用：")
    print("  python main.py --selftest    环境自检")
    print("  python main.py --list-audio  列出正在发声的进程")
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
