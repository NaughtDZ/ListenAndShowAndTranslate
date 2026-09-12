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
from app.utils.log import get_logger, install_excepthook

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
# --------------------------------------------------------------------------- #
# 采集目标音频
# --------------------------------------------------------------------------- #
def _parse_target(text: str):
    """把命令行参数解析成 TargetSpec：纯数字当 PID，否则当进程名。"""
    from app.audio.capture import TargetSpec

    text = text.strip()
    try:
        return TargetSpec(pid=int(text))
    except ValueError:
        return TargetSpec(process_name=text)


def _rms_bar(rms: float, width: int = 40) -> str:
    """把 RMS 画成一根电平条，直观看出"目标到底有没有在出声"。"""
    import math

    if rms <= 0:
        return "·" * width
    db = 20 * math.log10(max(rms, 1e-6))
    level = int(max(0.0, min(1.0, (db + 60) / 60)) * width)
    return "#" * level + "·" * (width - level)


def capture_target(args) -> int:
    """采集指定进程的音频，打印实时电平；可选落盘 WAV。"""
    import time as _time

    from app.audio.capture import CaptureWorker, record_to_wav, resolve_target

    spec = _parse_target(args.capture)
    target = resolve_target(spec)
    if target is None:
        print(f"未找到活跃音频会话: {spec.describe()}")
        print("提示：先运行 `python main.py --list-audio` 查看当前正在发声的进程与 PID。")
        print("      注意：只有真正在输出声音的进程才有活跃音频会话。")
        return 2

    print(f"\n目标: {target.display_label}")
    print(f"可执行文件: {target.executable}")

    if args.record:
        print(f"录制 {args.seconds:.1f} 秒到 {args.record} ...")
        stats = record_to_wav(spec, args.record, args.seconds)
        print(
            f"\n完成：{stats.chunks} 块 / 输入 {stats.bytes_in / 1024:.0f} KiB / "
            f"输出 {stats.samples_out} 样本 @16k（{stats.samples_out / 16000:.2f} 秒）"
        )
        print(f"峰值幅度: {stats.peak:.4f}")
        if stats.peak <= 0.001:
            print("⚠️ 峰值接近 0：目标进程当时没有真正在放音（这是正常的静音流，不是采集失败）")
        return 0

    print("开始采集，实时电平:\n")
    worker = CaptureWorker(spec, follow=not args.no_follow)
    worker.start()

    t0 = _time.time()
    try:
        while _time.time() - t0 < args.seconds:
            _time.sleep(0.2)
            st = worker.snapshot()
            status = "采集中" if st.running else "未连接"
            silent = f"静音 {st.silent_seconds:.1f}s" if st.silent_seconds > 0.5 else "有声"
            print(
                f"\r  [{_rms_bar(st.last_rms)}] rms={st.last_rms:.4f} "
                f"peak={st.peak:.3f} {status} {silent} "
                f"块={st.chunks} pid={st.target_pid}   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()

    st = worker.snapshot()
    print("\n")
    print(f"共 {st.chunks} 块，输出 {st.samples_out} 样本 @16k")
    print(f"峰值 {st.peak:.4f}，末次 RMS {st.last_rms:.4f}，重连 {st.reconnects} 次")
    if st.last_error:
        print(f"最后错误: {st.last_error}")
    if st.peak <= 0.001:
        print("⚠️ 全程静音：目标进程当时没有真正在放音。")
    return 0


# --------------------------------------------------------------------------- #
# 模型管理
# --------------------------------------------------------------------------- #
def models_command(args) -> int:
    from app.models.downloader import ModelDownloader
    from app.models.registry import (
        MODELS,
        PACKS,
        default_pack_ids,
        human_size,
        models_for_packs,
        total_bytes_for_packs,
    )

    action = args.models
    cfg = AppConfig.load()
    downloader = ModelDownloader(proxy=cfg.proxy)

    if action == "list":
        print("\n=== 语言包（按包下载，也可单独装模型）===\n")
        for pack in PACKS.values():
            mark = "★推荐 " if pack.recommended else "      "
            print(f"  {mark}{pack.id:<14} {pack.display_name:<22} "
                  f"{human_size(pack.total_bytes):>10}   {pack.description}")
        selected = default_pack_ids()
        print(f"\n  默认勾选合计: {human_size(total_bytes_for_packs(selected))}")
        print(f"  全部语言包合计: {human_size(total_bytes_for_packs(list(PACKS)))}\n")

        print("=== 模型明细 ===\n")
        for m in MODELS.values():
            print(f"  {m.id:<22} {m.display_name:<34} {human_size(m.total_bytes):>10}")
            print(f"      引擎={m.engine:<15} 语言={'/'.join(m.languages)}")
            if m.note:
                print(f"      {m.note}")
        print()
        return 0

    if action == "status":
        print("\n=== 模型安装状态 ===\n")
        total_installed = 0
        for m in MODELS.values():
            st = downloader.status(m.id)
            got = downloader.installed_bytes(m.id)
            total_installed += got
            icon = {"installed": "✓", "partial": "~", "missing": "·"}[st]
            pct = (got / m.total_bytes * 100) if m.total_bytes else 0
            print(f"  [{icon}] {m.id:<22} {st:<10} "
                  f"{human_size(got):>10} / {human_size(m.total_bytes):>10} ({pct:5.1f}%)")
        print(f"\n  合计占用: {human_size(total_installed)}")
        print(f"  模型目录: {downloader.models_dir}\n")
        return 0

    if action in ("install", "uninstall"):
        raw = (args.packs or "default").strip()
        if raw == "all":
            pack_ids = list(PACKS)
        elif raw == "default":
            pack_ids = default_pack_ids()
        else:
            pack_ids = [x.strip() for x in raw.split(",") if x.strip()]

        unknown = [p for p in pack_ids if p not in PACKS]
        if unknown:
            print(f"未知语言包: {unknown}")
            print(f"可用: {', '.join(PACKS)} 或 all / default")
            return 2

        model_ids = models_for_packs(pack_ids)
        total = total_bytes_for_packs(pack_ids)

        if action == "uninstall":
            print(f"\n将删除 {len(model_ids)} 个模型（{human_size(total)}）")
            for mid in model_ids:
                ok = downloader.uninstall(mid)
                print(f"  {'已删除' if ok else '本就不存在'} {mid}")
            return 0

        print(f"\n即将下载 {len(model_ids)} 个模型，共 {human_size(total)}")
        print(f"语言包: {', '.join(pack_ids)}")
        print(f"目标目录: {downloader.models_dir}")
        if cfg.proxy:
            print(f"代理: {cfg.proxy}")
        print("（支持断点续传，中断后重跑本命令可接着下）\n")

        last_line = {"len": 0}

        def on_progress(p) -> None:
            line = "  " + p.describe()
            pad = max(0, last_line["len"] - len(line))
            print("\r" + line + " " * pad, end="", flush=True)
            last_line["len"] = len(line)

        def on_log(msg: str) -> None:
            print("\r" + " " * (last_line["len"] + 2) + "\r" + msg)

        results = downloader.install_many(model_ids, on_progress=on_progress, on_log=on_log)

        ok_count = sum(1 for v in results.values() if v)
        print(f"\n完成：{ok_count}/{len(results)} 个模型就绪")
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"失败: {', '.join(failed)}")
            return 1
        return 0

    print(f"未知的 --models 动作: {action}")
    return 2


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
    p.add_argument("--capture", metavar="PID|进程名", help="采集目标音频并显示实时电平")
    p.add_argument("--meter", metavar="PID", help="打开电平表悬浮窗（RMS/PEAK 条 + 可调阈值）")
    p.add_argument("--run", metavar="PID", help="启动字幕程序：透明悬浮窗 + 实时翻译")
    p.add_argument("--process", metavar="NAME", help="配合 --run：按进程名（如 喜马拉雅.exe）")
    p.add_argument("--settings", action="store_true", help="打开设置界面（网络/翻译/识别/外观）")
    p.add_argument("--wizard", action="store_true", help="重新运行首次运行向导（硬件/档位/模型/翻译）")
    p.add_argument(
        "--models",
        choices=["list", "status", "install", "uninstall"],
        help="模型管理：list 看清单、status 看状态、install/uninstall 装或删",
    )
    p.add_argument(
        "--packs",
        metavar="PACKS",
        default="default",
        help="语言包，逗号分隔（zh,zh-en,en,ja-ko-yue,multilingual,lid,core）或 all / default",
    )
    p.add_argument(
        "--threshold-db",
        type=float,
        default=None,
        help="静音阈值 dBFS（默认 -80；越小越灵敏）",
    )
    p.add_argument("--seconds", type=float, default=10.0, help="采集时长（默认 10 秒）")
    p.add_argument("--record", metavar="WAV", help="把采集结果写成 16k 单声道 WAV")
    p.add_argument("--no-follow", action="store_true", help="目标消失后不自动重连")
    p.add_argument("--config", metavar="PATH", help="使用指定配置文件")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config:
        from pathlib import Path

        paths.CONFIG_FILE = Path(args.config).resolve()

    paths.ensure_dirs()
    # pythonw.exe 起的子进程没有控制台，未捕获异常必须落到日志里（否则凭空消失）
    install_excepthook()
    log.info("启动 v%s | Python %s | %s", __version__, sys.version.split()[0], paths.DATA_DIR)

    if args.selftest:
        return selftest()
    if args.list_audio:
        return list_audio_processes()
    if args.models:
        return models_command(args)
    if args.settings:
        from PySide6.QtWidgets import QApplication

        from app.ui.settings import SettingsWindow

        app = QApplication.instance() or QApplication([])
        win = SettingsWindow(AppConfig.load())
        win.show()
        return int(app.exec())
    if args.wizard:
        from app.ui.wizard import run_wizard

        return run_wizard(AppConfig.load())
    if args.run or args.process:
        from app.ui.runner import run_subtitles

        return run_subtitles(
            pid=int(args.run) if args.run else None,
            process_name=args.process or "",
        )
    if args.meter:
        from app.ui.meter import run_meter

        return run_meter(int(args.meter), args.threshold_db)
    if args.capture:
        return capture_target(args)

    # 无参数 = 正常启动（双击 启动.bat 走的就是这条路）
    #   没做过首次设置 → 先走向导；否则直接开主窗口选音频来源
    from app.ui.launcher import run_launcher

    return run_launcher(AppConfig.load())


if __name__ == "__main__":
    raise SystemExit(main())
