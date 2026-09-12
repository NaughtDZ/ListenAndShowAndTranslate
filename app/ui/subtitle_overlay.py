"""字幕悬浮窗。

直接复用电平表验证过的那套原生窗口能力（app/utils/win32.py）：
无边框 / 透明 / 置顶 / 点击穿透 / 不抢焦点 / 不进任务栏 / 定时重申置顶。

绘制上最要紧的是**描边**：字幕要压在亮暗不定的游戏画面上，
纯色文字在某些场景完全看不清。这里用 QPainterPath 描边 + 填充，
效果等价于常见播放器字幕。

双语模式下每行字幕占两条绘制行（原文在上、译文在下），
未定稿的中间结果用**半透明**画在最后一行——用户能立刻看到字在长，
但一眼能看出"这还不是最终结果"。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from app.config import OverlayConfig
from app.subtitle.model import SubtitleState
from app.ui.meter import _pick_cjk_font
from app.utils import win32
from app.utils.log import get_logger

log = get_logger(__name__)

STATUS_HEIGHT = 16
PADDING = 10


# CJK 禁则处理：这些标点不能出现在行首 / 行尾，否则会出现"孤零零一个句号"这种难看排版
_NO_LINE_START = "。，、！？；：）〕」』】〉》”’…·%℃"
_NO_LINE_END = "（〔「『【〈《“‘"


def wrap_text(
    fm: QFontMetrics,
    text: str,
    max_width: int,
    max_lines: int = 2,
) -> list[str]:
    """把一段文字折成不超过 ``max_lines`` 行。

    **CJK 可以任意位置断行，拉丁文优先在空格处断**——这是中英混排字幕的常识，
    否则英文单词会被从中间劈开（"subti / tle"）。

    另外做了**禁则处理**：句号/逗号等不能落到行首，左引号/左括号不能留在行尾
    （实测预览里出现过"最后一行只有一个句号"的排版事故）。

    实在放不下时，最后一行用省略号收尾，而不是丢掉整段。
    """
    if not text:
        return []
    if max_width <= 10 or fm.horizontalAdvance(text) <= max_width:
        return [text]

    lines: list[str] = []
    cur = ""
    dropped = 0   # 断行时丢弃的空格数：算"已消费多少字符"必须把它算上，
                  # 否则省略号那一步会从错误的位置取剩余文本（实测出现过多一个字母）
    for ch in text:
        trial = cur + ch
        if fm.horizontalAdvance(trial) <= max_width:
            cur = trial
            continue

        # 放不下了，决定从哪儿断
        if ch != " " and " " in cur:
            # 拉丁文：回退到最近的空格
            idx = cur.rfind(" ")
            head, tail = cur[:idx], cur[idx + 1:]
            new_line, cur = head, ((tail + ch) if tail else ch)
            dropped += 1
        elif ch in _NO_LINE_START and len(cur) > 1:
            # 禁则：标点不能起行 → 把前一个字一起挪下去
            new_line, cur = cur[:-1], cur[-1] + ch
        elif cur and cur[-1] in _NO_LINE_END:
            # 禁则：左引号/左括号不能结尾 → 一起挪下去
            new_line, cur = cur[:-1], cur[-1] + ch
        else:
            if ch == " ":
                dropped += 1
            new_line, cur = cur, ("" if ch == " " else ch)

        if new_line:
            lines.append(new_line)
        if len(lines) >= max_lines:
            break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    # 还有剩余内容 → 最后一行省略收尾
    consumed = sum(len(x) for x in lines) + dropped
    if consumed < len(text) and lines:
        remainder = text[consumed:]
        # 断行时把空格丢了，拼回来时要补一个，否则会连成 "Englishsentence" 这种
        if dropped > 0 and remainder and not remainder.startswith(" ") and not lines[-1].endswith(" "):
            remainder = " " + remainder
        lines[-1] = fm.elidedText(lines[-1] + remainder, Qt.ElideRight, max_width)
    return [ln for ln in lines if ln]


class SubtitleOverlay(QWidget):
    """浮在游戏上的字幕条。"""

    def __init__(self, config: OverlayConfig | None = None) -> None:
        super().__init__(None)
        self.config = config or OverlayConfig()
        self.state = SubtitleState()
        self.status_text = ""
        self.show_status = True

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("听·显·译 字幕")

        self.setFont(_pick_cjk_font())
        self._drag_from: QPoint | None = None
        self._resize_edge: str = ""
        self._resize_origin: QPoint | None = None
        self._resize_geo = None
        self._click_through = self.config.click_through

        # 对抗游戏抢 Z 序：定时重申置顶（计划书第 2.4 节）
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._reassert_topmost)
        self._topmost_timer.start(max(200, self.config.topmost_reassert_ms))

        self._relayout()

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def update_state(self, state: SubtitleState) -> None:
        self.state = state
        self._relayout()
        self.update()

    def set_status(self, text: str) -> None:
        if text != self.status_text:
            self.status_text = text
            self.update()

    def set_mode(self, mode: str) -> None:
        self.config.display_mode = mode  # type: ignore[assignment]
        self._relayout()
        self.update()

    def apply_config(self, config: OverlayConfig) -> None:
        self.config = config
        self._click_through = config.click_through
        self._apply_opacity()
        self.apply_native_flags()
        self._relayout()
        self.update()

    def _apply_opacity(self) -> None:
        """整体不透明度。

        与 ``background_opacity`` 的区别：那个只淡化底衬，这个**连文字一起淡化**，
        用来让字幕"融进"游戏画面（也有人反过来要它更醒目，所以做成可调）。
        """
        try:
            self.setWindowOpacity(max(0.2, min(1.0, float(self.config.window_opacity))))
        except Exception as exc:  # noqa: BLE001
            log.debug("设置窗口不透明度失败: %s", exc)

    # ------------------------------------------------------------------ #
    # 原生窗口
    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_opacity()
        QTimer.singleShot(0, self.apply_native_flags)

    def apply_native_flags(self) -> None:
        hwnd = win32.hwnd_of(self)
        if not hwnd:
            return
        win32.set_no_activate(hwnd, True)
        win32.set_taskbar_visible(hwnd, self.config.show_in_taskbar)
        win32.set_click_through(hwnd, self._click_through)
        win32.reassert_topmost(hwnd)

    def _reassert_topmost(self) -> None:
        if not self.config.always_on_top:
            return
        hwnd = win32.hwnd_of(self)
        if hwnd:
            win32.reassert_topmost(hwnd)

    def set_click_through(self, enabled: bool) -> None:
        self._click_through = enabled
        hwnd = win32.hwnd_of(self)
        if hwnd:
            win32.set_click_through(hwnd, enabled)
        log.info("字幕窗点击穿透: %s", "开" if enabled else "关")

    # ------------------------------------------------------------------ #
    # 布局
    # ------------------------------------------------------------------ #
    def _line_height(self) -> int:
        fm = QFontMetrics(self._font(self.config.font_size, self.config.bold))
        return int(fm.height() * self.config.line_spacing)

    def _max_subtitles(self) -> int:
        return 1 if self.config.scroll_mode == "replace" else max(1, self.config.max_lines)

    def _max_rows(self) -> int:
        """最多几条绘制行（每条字幕可换 lines_per_subtitle 行，双语再乘 2）。"""
        per_sub = self.config.lines_per_subtitle
        if self.config.display_mode == "bilingual":
            per_sub *= 2
        return self._max_subtitles() * per_sub

    def _relayout(self) -> None:
        rows = self._max_rows()
        auto_h = max(
            40,
            rows * self._line_height() + PADDING * 2
            + (STATUS_HEIGHT if self.show_status else 0),
        )
        # 用户手动拖过高就沿用他的高度，否则按内容自动算
        height = self.config.window_height if self.config.window_height > 0 else auto_h
        width = self.config.window_width
        screen = self.screen()
        if screen is not None:
            geo = screen.availableGeometry()
            width = min(width, geo.width() - 40) if width else geo.width() - 40
        self.resize(width, max(auto_h, height))
        self._reposition()

    def _build_draw_rows(self) -> list[tuple[str, bool, bool]]:
        """按需换行，返回 ``[(文本, 是否译文, 是否未定稿)]``。

        换行而不是截断——截断会丢内容（听小说时丢半句不可接受）。
        """
        cfg = self.config
        width = max(40, self.width() - PADDING * 2)
        fm = QFontMetrics(self._font(cfg.font_size, cfg.bold))
        per_sub = cfg.lines_per_subtitle

        rows: list[tuple[str, bool, bool]] = []
        for line in self.state.visible_lines(self._max_subtitles()):
            if cfg.display_mode in ("source", "bilingual"):
                for t in wrap_text(fm, line.source, width, per_sub):
                    rows.append((t, False, False))
            if cfg.display_mode in ("target", "bilingual"):
                text = line.translation or (line.source if cfg.display_mode == "target" else "")
                for t in wrap_text(fm, text, width, per_sub):
                    rows.append((t, True, False))

        if self.state.partial_source:
            for t in wrap_text(fm, self.state.partial_source, width, per_sub):
                rows.append((t, False, True))
        return rows

    def _reposition(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        w, h = self.width(), self.height()
        margin = self.config.margin_px
        pos = self.config.position

        if self.config.lock_position and pos != "custom":
            x = {
                "bottom-center": geo.left() + (geo.width() - w) // 2,
                "top-center": geo.left() + (geo.width() - w) // 2,
                "bottom-left": geo.left() + margin,
                "top-left": geo.left() + margin,
                "bottom-right": geo.right() - w - margin,
                "top-right": geo.right() - w - margin,
            }.get(pos, geo.left() + (geo.width() - w) // 2)
            y = (
                geo.top() + margin if pos.startswith("top")
                else geo.bottom() - h - margin
            )
            self.move(x, y)
        elif pos == "custom":
            self.move(self.config.custom_x, self.config.custom_y)

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def _font(self, size: int, bold: bool) -> QFont:
        f = QFont(self.font())
        f.setPointSize(size)
        f.setBold(bold)
        return f

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        w, h = self.width(), self.height()
        cfg = self.config

        # 背景（可调透明度，默认很淡，避免遮挡游戏）
        if cfg.background_opacity > 0.01:
            bg = QColor(cfg.background_color)
            bg.setAlphaF(max(0.0, min(1.0, cfg.background_opacity)))
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(0, 0, w - 1, h - 1, 8, 8)

        rows = self._build_draw_rows()
        row_h = self._line_height()
        status_h = STATUS_HEIGHT if self.show_status else 0
        body_h = h - PADDING * 2 - status_h

        # 底部对齐：新行从下往上堆，像真正的字幕
        visible_rows = rows[-self._max_rows():] if self._max_rows() else rows
        total = len(visible_rows) * row_h
        y = PADDING + max(0, body_h - total)

        for text, is_translation, is_partial in visible_rows:
            if not text:
                continue
            color = QColor(cfg.target_color if is_translation else cfg.source_color)
            if is_partial:
                color.setAlphaF(0.6)   # 未定稿：半透明，一眼可辨
            fm = QFontMetrics(self._font(cfg.font_size, cfg.bold))
            self._draw_outlined_text(
                p, text, PADDING, y, w - PADDING * 2, row_h,
                self._font(cfg.font_size, cfg.bold), color, cfg, fm,
            )
            y += row_h

        if self.show_status and self.status_text:
            p.setFont(self._font(8, False))
            sc = QColor(255, 255, 255, 140)
            self._draw_outlined_text(
                p, self.status_text, PADDING, h - status_h - PADDING // 2,
                w - PADDING * 2, status_h,
                self._font(8, False), sc, cfg, QFontMetrics(self._font(8, False)),
                outline_width=1,
            )

    def _draw_outlined_text(
        self,
        p: QPainter,
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        font: QFont,
        color: QColor,
        cfg: OverlayConfig,
        fm: QFontMetrics,
        outline_width: int | None = None,
    ) -> None:
        """画一行带描边的文字（描边宽度不足时自动降级为无描边）。"""
        display = text
        if fm.horizontalAdvance(display) > width:
            display = fm.elidedText(display, Qt.ElideRight, width)

        baseline = y + (height + fm.ascent() - fm.descent()) // 2

        if cfg.outline_width > 0 or (outline_width or 0) > 0:
            path = QPainterPath()
            path.addText(float(x), float(baseline), font, display)
            pen = QPen(QColor(cfg.outline_color))
            pen.setWidthF(float(outline_width if outline_width is not None else cfg.outline_width))
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.fillPath(path, QBrush(color))
        else:
            p.setPen(color)
            p.setFont(font)
            p.drawText(x, baseline, display)

    # ------------------------------------------------------------------ #
    # 拖动 / 缩放（未开穿透时）
    # ------------------------------------------------------------------ #
    def _hit_edge(self, pos) -> str:
        """判断鼠标压在哪个边/角上，返回形如 ``"br"`` 的字符串。

        无边框窗口没有系统的缩放边框，只能自己判定——所以这里手工算，
        否则用户根本没法调字幕窗大小（用户明确提过这个需求）。
        """
        m = 8  # 边缘判定宽度（px）
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        out = ""
        if x <= m:
            out += "l"
        elif x >= w - m:
            out += "r"
        if y <= m:
            out += "t"
        elif y >= h - m:
            out += "b"
        return out

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        edge = self._hit_edge(event.position().toPoint())
        if edge and self.config.resizable:
            self._resize_edge = edge
            self._resize_origin = event.globalPosition().toPoint()
            self._resize_geo = self.geometry()
            return
        if not self.config.lock_position:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        # 缩放
        if self._resize_edge:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = self._resize_geo
            left, top, right, bottom = geo.left(), geo.top(), geo.right(), geo.bottom()
            if "l" in self._resize_edge:
                left = min(left + delta.x(), right - 120)
            if "r" in self._resize_edge:
                right = max(right + delta.x(), left + 120)
            if "t" in self._resize_edge:
                top = min(top + delta.y(), bottom - 30)
            if "b" in self._resize_edge:
                bottom = max(bottom + delta.y(), top + 30)
            self.setGeometry(left, top, right - left, bottom - top)
            return

        # 拖动
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)
            self.config.position = "custom"  # type: ignore[assignment]
            self.config.custom_x = self.x()
            self.config.custom_y = self.y()
            return

        # 光标反馈：让用户知道哪里能拖
        if self.config.resizable and not self._click_through:
            edge = self._hit_edge(event.position().toPoint())
            cursors = {
                "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            }
            self.setCursor(cursors.get(edge, Qt.ArrowCursor))

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._resize_edge:
            # 记下手动尺寸，下次启动沿用（横向拖动改宽度，纵向拖动改高度）
            self.config.window_width = self.width()
            self.config.window_height = self.height()
            log.info("字幕窗尺寸已记录：%dx%d", self.width(), self.height())
        self._resize_edge = None
        self._drag_from = None
