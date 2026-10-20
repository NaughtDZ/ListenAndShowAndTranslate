"""把 ``Cue`` 列表写成字幕文件：SRT / VTT / LRC。

三种格式的时间戳口径不一样，别混：

===========  =========================  ==========================
格式          时间写法                    有无结束时间
===========  =========================  ==========================
SRT          ``00:00:01,234``（逗号）    **有**
VTT          ``00:00:01.234``（点）      **有**
LRC          ``[01:23.45]``（百分秒）    **没有**，只有行开始
===========  =========================  ==========================

所以 LRC 之前必须先切行（见 ``app/offline/cue.py``），否则一行会显示一整句。

编码统一 **UTF-8 with BOM**：Windows 上的播放器/剪辑软件对 BOM 的兼容性最好，
而浏览器 ``<track>`` 也容忍 BOM。换行统一 LF（与仓库其它文本一致）。
"""

from __future__ import annotations

from pathlib import Path

from app.offline.cue import Cue

FORMATS: tuple[str, ...] = ("srt", "vtt", "lrc")
MODE_LABELS: dict[str, str] = {
    "source": "只有原文（转写结果）",
    "bilingual": "双语（原文 + 译文）",
    "target": "只有译文",
}

MIN_CUE_S = 0.2
"""比这更短的字幕某些播放器会一闪而过、甚至不显示；导出时补齐。"""


# --------------------------------------------------------------------------- #
# 时间戳
# --------------------------------------------------------------------------- #
def fmt_srt(seconds: float) -> str:
    total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def fmt_vtt(seconds: float) -> str:
    return fmt_srt(seconds).replace(",", ".")


def fmt_lrc(seconds: float) -> str:
    """LRC 只有"开始时间"，精度到百分之一秒。"""
    total = max(0, int(round(seconds * 100)))
    minutes, rem = divmod(total, 6000)
    secs, centis = divmod(rem, 100)
    return f"[{minutes:02d}:{secs:02d}.{centis:02d}]"


# --------------------------------------------------------------------------- #
# 规整
# --------------------------------------------------------------------------- #
def _prepare(cues: list[Cue], mode: str) -> list[Cue]:
    """丢掉空文本、按时间排序、把过短的行补到可显示。"""
    cleaned: list[Cue] = []
    for cue in sorted(cues, key=lambda c: c.start_s):
        if not cue.text_for(mode).strip():
            continue
        end = max(cue.end_s, cue.start_s + MIN_CUE_S)
        cleaned.append(Cue(cue.start_s, end, cue.text, cue.translation))
    return cleaned


# --------------------------------------------------------------------------- #
# 三种格式
# --------------------------------------------------------------------------- #
def to_srt(cues: list[Cue], mode: str = "source") -> str:
    blocks: list[str] = []
    for idx, cue in enumerate(_prepare(cues, mode), start=1):
        blocks.append(str(idx))
        blocks.append(f"{fmt_srt(cue.start_s)} --> {fmt_srt(cue.end_s)}")
        blocks.append(cue.text_for(mode))
        blocks.append("")
    return "\n".join(blocks)


def to_vtt(cues: list[Cue], mode: str = "source") -> str:
    blocks: list[str] = ["WEBVTT", ""]
    for cue in _prepare(cues, mode):
        blocks.append(f"{fmt_vtt(cue.start_s)} --> {fmt_vtt(cue.end_s)}")
        blocks.append(cue.text_for(mode))
        blocks.append("")
    return "\n".join(blocks)


def to_lrc(cues: list[Cue], mode: str = "source") -> str:
    """LRC：双语写成"同时间戳的两行"（原文一行、译文一行），这是播放器通行的做法。"""
    lines: list[str] = []
    for cue in _prepare(cues, mode):
        stamp = fmt_lrc(cue.start_s)
        if mode == "bilingual" and cue.translation and cue.translation != cue.text:
            lines.append(f"{stamp}{cue.text}")
            lines.append(f"{stamp}{cue.translation}")
        else:
            lines.append(f"{stamp}{cue.text_for(mode)}")
    return "\n".join(lines) + ("\n" if lines else "")


_RENDERERS = {"srt": to_srt, "vtt": to_vtt, "lrc": to_lrc}


def render(cues: list[Cue], fmt: str, mode: str = "source") -> str:
    fmt = fmt.lower().lstrip(".")
    renderer = _RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"不支持的字幕格式: {fmt}（可选 {list(_RENDERERS)}）")
    return renderer(cues, mode)


def write_subtitles(
    path: str | Path,
    cues: list[Cue],
    fmt: str,
    mode: str = "source",
    bom: bool = True,
) -> Path:
    """写出一个字幕文件，返回实际路径。"""
    path = Path(path)
    content = render(cues, fmt, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig" if bom else "utf-8", newline="\n")
    return path
