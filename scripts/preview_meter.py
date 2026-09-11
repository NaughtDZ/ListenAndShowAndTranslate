"""把电平表渲染成 PNG，用于人工/自动检查绘制效果（不需要真实音频）。

注意：**默认用原生平台插件**，这样能用系统字体正确渲染中文。
若用 QT_QPA_PLATFORM=offscreen，Qt 找不到字体目录，中文会全变成方块（tofu），
那是渲染环境问题、不是绘制代码问题。

用法：
    .venv\\Scripts\\python.exe scripts\\preview_meter.py
    → data/selftest_audio/meter_preview.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.audio.levels import LevelState, threshold_from_db  # noqa: E402
from app.ui.meter import LevelMeterWidget, _pick_cjk_font  # noqa: E402

W, H = 460, 108

CASES: list[tuple[str, LevelState, str, str]] = [
    (
        "① 完全静音（目标没在放音）",
        LevelState(rms=0.0, peak=0.0, peak_hold=0.0, db_rms=-90, db_peak=-90, db_peak_hold=-90,
                   is_silent=True),
        "未连接（目标没在放音？）",
        "",
    ),
    (
        "② 小音量语音（用户把小说调轻了）",
        LevelState(rms=0.00316, peak=0.0126, peak_hold=0.0158,
                   db_rms=-50.0, db_peak=-38.0, db_peak_hold=-36.0, is_silent=False),
        "有声",
        "",
    ),
    (
        "③ 正常语音",
        LevelState(rms=0.063, peak=0.2, peak_hold=0.21,
                   db_rms=-24.0, db_peak=-14.0, db_peak_hold=-13.6, is_silent=False),
        "有声",
        "",
    ),
    (
        "④ 削波 + 会话音量过低警告",
        LevelState(rms=0.56, peak=0.999, peak_hold=0.999,
                   db_rms=-5.0, db_peak=-0.01, db_peak_hold=-0.01, is_silent=False,
                   clipping=True),
        "有声",
        "会话音量仅 8%，建议调大或勾选自动增益",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染电平表预览图")
    ap.add_argument("--offscreen", action="store_true", help="用离屏渲染（中文会变方块）")
    args = ap.parse_args()

    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

    app = QApplication.instance() or QApplication([])
    print("Qt 平台插件:", app.platformName())

    gap = 30
    total_h = len(CASES) * (H + gap)
    canvas = QPixmap(W + 40, total_h + 20)
    canvas.fill(QColor(28, 30, 36))

    p = QPainter(canvas)
    p.setRenderHint(QPainter.Antialiasing)

    label_font = _pick_cjk_font()
    label_font.setPointSize(10)
    label_font.setBold(True)

    y = 20
    for title, state, status, warning in CASES:
        p.setPen(QColor(240, 240, 240))
        p.setFont(label_font)
        p.drawText(20, y - 6, title)

        widget = LevelMeterWidget()
        widget.resize(W, H)
        widget.set_threshold(threshold_from_db(-80.0))
        widget.set_level(state)
        widget.set_texts("msedge.exe  (PID 23784)   已运行 42s", status, warning)

        pm = widget.grab()
        p.drawPixmap(20, y, pm)
        y += H + gap

    p.end()

    out = PROJECT / "data" / "selftest_audio" / "meter_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = canvas.save(str(out))
    print(f"保存{'成功' if ok else '失败'}: {out}  ({canvas.width()}x{canvas.height()})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
