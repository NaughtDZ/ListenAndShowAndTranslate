"""把字幕悬浮窗渲染成 PNG 自检（不需要音频、不需要开窗）。

用法：
    .venv\\Scripts\\python.exe scripts\\preview_subtitle.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import OverlayConfig  # noqa: E402
from app.subtitle.model import SubtitleState  # noqa: E402
from app.ui.subtitle_overlay import SubtitleOverlay  # noqa: E402

LINES = [
    ("列車?リンファンは手の中の声堂の鍵を強く握りしめ。", "列车？林凡握着手中青铜钥匙的手紧了紧。"),
    ("今度こそ。", "这一次，不能再让它溜走。"),
    ("誰も失望させたりはしないと小さくつぶやいた。", "他在心里低语，绝不再让任何人失望。"),
]


def make_state(mode_lines: int, partial: str = "") -> SubtitleState:
    st = SubtitleState()
    now = time.time()
    for i, (src, dst) in enumerate(LINES[:mode_lines]):
        line = st.add_final(src, "ja")
        line.created_at = now + i
        st.set_translation(line.id, dst)
    if partial:
        st.set_partial(partial, "ja")
    return st


def render(overlay: SubtitleOverlay, state: SubtitleState, status: str) -> QPixmap:
    overlay.update_state(state)
    overlay.set_status(status)
    return overlay.grab()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    print("Qt 平台插件:", app.platformName())

    cases = [
        ("① 双语 · 累积 3 行（带未定稿中间结果）",
         OverlayConfig(display_mode="bilingual", max_lines=3, scroll_mode="accumulate"),
         make_state(3, "窓の外には、見知らぬ"), "ja · sensevoice · 有声"),
        ("② 原文 · 累积", OverlayConfig(display_mode="source", max_lines=3), make_state(3), "ja · sensevoice"),
        ("③ 译文 · 累积", OverlayConfig(display_mode="target", max_lines=3), make_state(3), "ja · sensevoice"),
        ("④ 双语 · 单行替换", OverlayConfig(display_mode="bilingual", max_lines=1, scroll_mode="replace"),
         make_state(1), "ja · sensevoice"),
        ("⑤ 无描边（对比可读性）",
         OverlayConfig(display_mode="bilingual", max_lines=2, outline_width=0), make_state(2), "ja"),
    ]

    width = 760
    canvas = QPixmap(width, 40)
    shots = []
    for title, cfg, state, status in cases:
        cfg.window_width = width - 20
        ov = SubtitleOverlay(cfg)
        ov.resize(cfg.window_width, ov.height())
        pm = render(ov, state, status)
        shots.append((title, pm))
        ov.deleteLater()

    total_h = sum(pm.height() + 34 for _, pm in shots) + 20
    canvas = QPixmap(width, total_h)
    canvas.fill(QColor(66, 78, 92))   # 模拟"游戏画面"的中间灰

    p = QPainter(canvas)
    from app.ui.meter import _pick_cjk_font

    f = _pick_cjk_font()
    f.setPointSize(10)
    f.setBold(True)
    p.setFont(f)
    y = 22
    for title, pm in shots:
        p.setPen(QColor(245, 245, 245))
        p.drawText(12, y - 5, title)
        p.drawPixmap(12, y, pm)
        y += pm.height() + 34
    p.end()

    out = PROJECT / "data" / "selftest_audio" / "subtitle_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = canvas.save(str(out))
    print(f"保存{'成功' if ok else '失败'}: {out} ({canvas.width()}x{canvas.height()})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
