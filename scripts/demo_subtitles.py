"""整机演示：播放测试语音 → 实时字幕 + 翻译（不需要你手动操作任何东西）。

链路完全走真实路径：
    独立播放进程（winsound 循环播 WAV）
      → WASAPI 进程回环只抓它
        → VAD/端点检测分句
          → ASR（默认 auto 自动判语种，会选中对应引擎）
            → LLM 翻译（LM Studio）
              → 透明悬浮字幕窗

用法：
    .venv\\Scripts\\python.exe scripts\\demo_subtitles.py
    .venv\\Scripts\\python.exe scripts\\demo_subtitles.py --wav data\\test_speech\\zh_00.wav
    .venv\\Scripts\\python\\... --translate-model qwen3.8-2b-uncensored --language ja
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from app.config import AppConfig  # noqa: E402

_PLAYER = r"""
import os, sys, time, winsound
print("PID", os.getpid(), flush=True)
winsound.PlaySound(sys.argv[1], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
print("PLAYING", flush=True)
time.sleep(3600)
"""


def spawn_player(wav: Path) -> tuple[subprocess.Popen, int]:
    """播放进程自己上报真实 PID（uv 的 python.exe 是转发器，父子 PID 不同）。"""
    proc = subprocess.Popen(
        [sys.executable, "-c", _PLAYER, str(wav)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", bufsize=1,
    )
    real_pid = -1
    for _ in range(200):
        line = (proc.stdout.readline() or "").strip()
        if not line:
            break
        if line.startswith("PID "):
            real_pid = int(line.split()[1])
        if line == "PLAYING":
            break
    return proc, real_pid


def main() -> int:
    ap = argparse.ArgumentParser(description="字幕整机演示")
    ap.add_argument("--wav", default="data/test_speech/ja_03.wav",
                    help="要播放的测试语音（默认日文，最能体现翻译效果）")
    ap.add_argument("--language", default="auto", help="auto 或 zh/en/ja/ko/yue")
    ap.add_argument("--translate-model", default="qwen3.8-2b-uncensored")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--no-translate", action="store_true", help="只显示原文，不翻译")
    ap.add_argument("--display", default="bilingual", choices=["source", "target", "bilingual"])
    args = ap.parse_args()

    wav = Path(args.wav)
    if not wav.is_absolute():
        wav = PROJECT / wav
    if not wav.exists():
        print(f"测试语音不存在: {wav}")
        print("请先运行: python scripts/gen_test_speech.py")
        return 2

    proc, real_pid = spawn_player(wav)
    if real_pid <= 0:
        proc.terminate()
        print("播放进程启动失败")
        return 2
    print(f"播放进程: 启动器 pid={proc.pid} → 真实 pid={real_pid}")
    print(f"播放内容: {wav.name}")
    time.sleep(0.8)

    cfg = AppConfig.load()
    cfg.asr.language = args.language
    cfg.overlay.display_mode = args.display
    cfg.overlay.max_lines = 3

    if args.no_translate:
        cfg.translate.enabled = False
    else:
        cfg.translate.enabled = True
        cfg.translate.provider = "llm"
        cfg.translate.llm.enabled = True
        cfg.translate.llm.base_url = args.base_url
        cfg.translate.llm.model = args.translate_model
        cfg.translate.context_lines = 20
        # 演示用的术语表：正好覆盖测试语音里的专名与 ASR 误识别
        cfg.translate.glossary = {
            "リンファン": "林凡",
            "林凡": "林凡",
            "声堂": "青铜",
            "青銅": "青铜",
            "太虚": "太虚",
        }

    print(f"翻译: {'关闭' if args.no_translate else args.translate_model}  语言: {args.language}")
    print("字幕窗与控制窗即将出现；关掉控制窗即退出。\n")

    try:
        from app.ui.runner import run_subtitles

        return run_subtitles(pid=real_pid, config=cfg)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
