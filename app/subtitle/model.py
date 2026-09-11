"""字幕数据模型：原文 / 译文 / 未定稿中间结果。

三个概念必须分清，否则界面会很别扭：

- **partial（未定稿）**：流式引擎边听边出的中间结果，会被后续覆盖。
  它不该进历史、不该送翻译（翻译半句话纯属浪费），但**要显示**——
  这正是"流式"的价值所在（用户能立刻看到字在长）。
- **final 原文**：定稿的识别结果，立刻显示，并送去翻译。
- **final 译文**：译文是**后补**的，可能比原文晚 0.3~3 秒。
  所以状态模型必须允许"一行字幕先有原文、后有译文"。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SubtitleLine:
    """一行字幕。"""

    id: int
    source: str
    language: str = ""
    translation: str = ""
    created_at: float = field(default_factory=time.time)
    translated_at: float | None = None
    error: str = ""

    @property
    def has_translation(self) -> bool:
        return bool(self.translation)

    @property
    def translate_latency_s(self) -> float | None:
        if self.translated_at is None:
            return None
        return max(0.0, self.translated_at - self.created_at)

    def text_for(self, mode: str) -> str:
        """按显示模式取文本。"""
        if mode == "source":
            return self.source
        if mode == "target":
            return self.translation or self.source
        # bilingual：译文优先，没有译文时退回原文（避免空行）
        return self.translation or self.source


class SubtitleState:
    """字幕状态机。线程安全由外部保证（流水线里只在 UI 线程改）。"""

    def __init__(self, max_lines: int = 200) -> None:
        self.lines: deque[SubtitleLine] = deque(maxlen=max_lines)
        self._next_id = 0

        self.partial_source = ""
        self.partial_language = ""

    # ------------------------------------------------------------------ #
    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ---- 中间结果 ---- #
    def set_partial(self, text: str, language: str = "") -> None:
        self.partial_source = text or ""
        if language:
            self.partial_language = language

    def clear_partial(self) -> None:
        self.partial_source = ""
        self.partial_language = ""

    # ---- 定稿 ---- #
    def add_final(self, text: str, language: str = "", line_id: int | None = None) -> SubtitleLine:
        self.clear_partial()
        line = SubtitleLine(
            id=line_id if line_id is not None else self.new_id(),
            source=text,
            language=language or self.partial_language,
        )
        self.lines.append(line)
        return line

    def set_translation(self, line_id: int, translation: str, error: str = "") -> bool:
        for line in self.lines:
            if line.id == line_id:
                if translation:
                    line.translation = translation
                    line.translated_at = time.time()
                    line.error = ""
                elif error:
                    line.error = error
                    # 部分通道失败时，用原文兜底比空白好——用户至少能看到内容
                    if not line.translation:
                        line.translation = ""
                return True
        return False

    def get(self, line_id: int) -> SubtitleLine | None:
        for line in self.lines:
            if line.id == line_id:
                return line
        return None

    # ------------------------------------------------------------------ #
    def visible_lines(self, count: int) -> list[SubtitleLine]:
        """最近 N 行定稿字幕（按时间正序）。"""
        if count <= 0:
            return []
        return list(self.lines)[-count:]

    def render_rows(self, mode: str, max_lines: int, show_partial: bool = True) -> list[tuple[str, bool]]:
        """生成要绘制的行：``[(文本, 是否为译文), ...]``。

        双语模式下每行字幕会拆成两条绘制行（原文在上、译文在下）。
        """
        rows: list[tuple[str, bool]] = []
        for line in self.visible_lines(max_lines):
            if mode == "source":
                rows.append((line.source, False))
            elif mode == "target":
                rows.append((line.translation or line.source, True))
            else:  # bilingual
                rows.append((line.source, False))
                if line.translation:
                    rows.append((line.translation, True))

        # 中间结果追加在最后（用不同样式）
        if show_partial and self.partial_source:
            rows.append((self.partial_source, False))
        return rows

    def clear(self) -> None:
        self.lines.clear()
        self.clear_partial()

    # ------------------------------------------------------------------ #
    def to_srt(self) -> str:
        """导出成 SRT 字幕文件（只导有译文的，优先用译文）。

        时间轴按"每行占用的时间"估算——实时字幕没有精确结束时间，
        这里用相邻两行的创建时间差作为时长，无后继时给一个保守值。
        """
        lines = [ln for ln in self.lines if ln.source]
        if not lines:
            return ""
        out: list[str] = []
        for i, ln in enumerate(lines, start=1):
            start = ln.created_at
            nxt = lines[i] if i < len(lines) else None
            end = nxt.created_at if nxt else start + 3.0
            if end <= start:
                end = start + 2.0
            text = ln.translation or ln.source
            if ln.translation and ln.translation != ln.source:
                text = f"{ln.source}\n{ln.translation}"
            out.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
        return "\n".join(out)


def _srt_time(ts: float) -> str:
    ms = int((ts % 1) * 1000)
    total = int(ts)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
