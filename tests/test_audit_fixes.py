"""代码审计（2026-10）修掉的那批缺陷的回归测试。

每个用例对应一个**真实存在过**的问题，注释里写清"以前怎么坏的、
为什么平时不容易发现"。目的很直接：谁把这些改回去，测试就红。

覆盖的三类：

1. **逻辑 bug** —— 400 重试漏清字段、最后一句不翻译、auto 模式无限重试
2. **线程安全** —— 跨线程分配行号、跨线程读 Qt 控件
3. **性能** —— 逐字度量是 O(n²)、逐档缩字号、定时器无条件重绘
"""

from __future__ import annotations

import os
import threading

import pytest
import numpy as np

from app.config import AppConfig
from app.subtitle.model import SubtitleState
from app.translate.base import TranslateRequest, TranslateResult
from app.translate.openai_compat import _THINK_OFF_KEYS, thinking_extras


# --------------------------------------------------------------------------- #
# 1. 关思考的字段清单必须覆盖所有服务端
# --------------------------------------------------------------------------- #
def test_think_off_keys_cover_every_host():
    """``_THINK_OFF_KEYS`` 漏掉某个 key ⇒ 400 重试还是 400。

    这是 DeepSeek 接入时踩出来的：重试代码里手写了 pop 列表，
    新加的 ``thinking`` 没跟着加进去，于是"去掉参数重试一次"变成
    "带着同样的非法参数再发一次"。现在改成单一真源，这条测试守着它。
    """
    hosts = (
        "https://api.openai.com/v1",
        "https://api.deepseek.com",
        "https://openrouter.ai/api/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "http://127.0.0.1:1234/v1",
    )
    for base in hosts:
        extras = thinking_extras(base)
        assert extras, base
        for key in extras:
            assert key in _THINK_OFF_KEYS, f"{base} 用了 {key}，但清单里没有它"


def test_400_retry_strips_thinking_field(monkeypatch):
    """端点拒收 ``thinking`` 时，第二次请求不能再带上它。

    用一个假的 httpx client 记下每次 payload：第一次回 400，
    第二次必须 200，且 payload 里不能还有 ``thinking``。
    """

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self._body = {
                "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._body

    class _Client:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def post(self, url, json=None, **_kw):  # noqa: A002 - httpx 的签名就叫 json
            self.payloads.append(dict(json or {}))
            return _Resp(400 if len(self.payloads) == 1 else 200)

        def close(self) -> None:
            pass

    from app.translate.base import Segment
    from app.translate.openai_compat import OpenAICompatTranslator

    # 构造 httpx.Client 是惰性的（不联网），所以这里不需要任何 mock，
    # 之后直接把 _client 换成会记账的假对象。
    t = OpenAICompatTranslator(base_url="https://api.deepseek.com", model="deepseek-flash")
    client = _Client()
    t._client = client

    req = TranslateRequest(
        segments=[Segment(id=1, text="こんにちは", language="ja")],
        source_language="ja",
        target_language="zh",
    )
    content, _usage, _trunc = t._call(req, 128)

    assert content == "译文"
    assert len(client.payloads) == 2, "应该只重试一次"
    assert "thinking" in client.payloads[0], "第一次应该带上关思考参数"
    assert "thinking" not in client.payloads[1], "重试时必须清掉 thinking"


# --------------------------------------------------------------------------- #
# 2. 行号分配必须线程安全
# --------------------------------------------------------------------------- #
def test_new_id_is_thread_safe():
    """``new_id`` 在采集线程被调用，以前是裸的 ``+= 1``。

    裸自增在 CPython 里靠 GIL 侥幸不重复，但那是实现细节，不是保证。
    """
    state = SubtitleState()
    got: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [state.new_id() for _ in range(200)]
        with lock:
            got.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(got) == len(set(got)) == 8 * 200, "行号出现重复"
    assert max(got) == 8 * 200


# --------------------------------------------------------------------------- #
# 3. stop() 必须把最后一句翻译掉，并关掉引擎
# --------------------------------------------------------------------------- #
def _pipeline():
    from app.pipeline import SubtitlePipeline

    return SubtitlePipeline(AppConfig())


def test_translate_loop_drains_queue_after_stop():
    """``_stop`` 置上之后，队列里剩下的字幕仍然要被翻译。

    以前 ``stop()`` 是「先 join 翻译线程，再 router.finish()」：
    最后半句被冲出来丢进队列，而翻译线程已经退出了——用户表现就是
    "最后一句话只有原文、没有译文"，而且只在**停止时**出现，很难复现。
    """

    class _Hub:
        def __init__(self) -> None:
            from app.translate.hub import HubStats

            # 必须用 HubStats：pipeline._emit_stats() 会调 .as_dict()
            self.stats = HubStats()

        def close(self) -> None:
            pass

        def translate(self, segments, _src):
            res = TranslateResult(provider="fake")
            for seg in segments:
                res.translations[seg.id] = f"[{seg.text}]"
            return res

        def cache_stats(self) -> dict:
            return {}

    pipe = _pipeline()
    pipe.hub = _Hub()
    pipe._queue.put((7, "最后一句", "ja"))
    pipe._stop.set()          # 模拟 stop() 已经把停止标志置上

    seen: list[tuple[int, str]] = []
    pipe.translationReceived.connect(lambda i, t, _e: seen.append((i, t)))
    pipe._translate_loop()    # 同步跑完：应当处理完队列才退出

    assert seen == [(7, "[最后一句]")], seen


def test_stop_closes_router_and_hub():
    """``stop()`` 必须显式关闭识别引擎。

    不关的话 ``reload()`` 重建 router 时，旧的 onnx session 要等 GC 才释放
    （模型几百 MB），切几次设置就能把内存顶上去。
    """

    class _Router:
        def __init__(self) -> None:
            self.finished = False
            self.closed = False

        def finish(self):
            self.finished = True
            return []

        def close(self) -> None:
            self.closed = True

    class _Hub:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pipe = _pipeline()
    router, hub = _Router(), _Hub()
    pipe.router, pipe.hub = router, hub

    pipe.stop(timeout=1.0)

    assert router.finished and router.closed
    assert hub.closed
    assert pipe.router is None and pipe.hub is None, "置 None 才不会挡住下一次 prepare()"


# --------------------------------------------------------------------------- #
# 4. auto 模式：加载不了的语言不许无限重试
# --------------------------------------------------------------------------- #
def test_router_does_not_retry_failed_language():
    """引擎构建失败后，**每一帧音频**都重跑整条降级链是灾难。

    模型缺失时 ``_engines`` 里永远没有条目，于是 10ms 一块的音频会触发
    每帧 6 次模型加载尝试 + 6 条 WARNING —— 实测表现为"日志每秒刷几千行、
    CPU 莫名跑满"。修法是记住失败的语言。
    """
    from app.asr.router import LanguageRouter

    cfg = AppConfig()
    cfg.asr.language = "zh"          # 跳过 LID，直接走"取引擎"这条路
    router = LanguageRouter(cfg.asr)

    calls = {"n": 0}

    def fake_build(language: str):
        calls["n"] += 1
        return None

    router._create_engine_with_fallback = fake_build  # type: ignore[method-assign]

    chunk = np.zeros(1600, dtype=np.float32)
    assert router.feed(chunk) == []
    assert router.feed(chunk) == []
    assert router.feed(chunk) == []
    assert calls["n"] == 1, f"失败后重试了 {calls['n']} 次，应该只试 1 次"


# --------------------------------------------------------------------------- #
# 5. 性能：换行不能是 O(n²)
# --------------------------------------------------------------------------- #
class _FixedMetrics:
    """确定性的"假字体"：CJK 每字 10px、ASCII 每字 5px，并记录被叫了几次。

    为什么不直接用真 ``QFontMetrics``：``QT_QPA_PLATFORM=offscreen`` 下
    Qt 找不到字体目录，``horizontalAdvance()`` **一律返回 0**——
    于是任何文本都"一行放得下"，换行分支根本走不到，测了等于没测。
    这里用固定宽度既跨平台确定，又保证真的能触发断行。
    """

    def __init__(self, ascii_w: float = 5.0, cjk_w: float = 10.0) -> None:
        self.ascii_w = ascii_w
        self.cjk_w = cjk_w
        self.calls = 0
        self.chars = 0
        """累计被度量的**字符数**——这才是 O(n²) 的真实体现。

        只看调用次数会误判：旧实现每行仍是"每字调用一次"（次数 O(n)），
        但每次都把**整行**重新量一遍（每次长度 O(n)）→ 总量 O(n²)。
        """

    def _char_w(self, ch: str) -> float:
        return self.cjk_w if ord(ch) > 0x2E80 else self.ascii_w

    def horizontalAdvance(self, text, *_args, **_kwargs):  # noqa: N802 - 模仿 Qt 命名
        self.calls += 1
        self.chars += len(text)
        return sum(self._char_w(c) for c in text)

    def elidedText(self, text, _mode, width):  # noqa: N802
        if not text or self.horizontalAdvance(text) <= width:
            return text
        ell = "…"
        budget = width - self.horizontalAdvance(ell)
        out, w = "", 0.0
        for ch in text:
            cw = self._char_w(ch)
            if w + cw > budget:
                break
            out += ch
            w += cw
        return out + ell


def _wrap_text_reference(fm, text, max_width, max_lines=2):
    """**旧实现**（每次把整行重量一遍）。留在这里只当参照物。"""
    from app.ui.subtitle_overlay import _NO_LINE_END, _NO_LINE_START

    if not text:
        return []
    if max_width <= 10 or fm.horizontalAdvance(text) <= max_width:
        return [text]

    lines: list[str] = []
    cur = ""
    dropped = 0
    for ch in text:
        trial = cur + ch
        if fm.horizontalAdvance(trial) <= max_width:
            cur = trial
            continue
        if ch != " " and " " in cur:
            idx = cur.rfind(" ")
            head, tail = cur[:idx], cur[idx + 1:]
            new_line, cur = head, ((tail + ch) if tail else ch)
            dropped += 1
        elif ch in _NO_LINE_START and len(cur) > 1:
            new_line, cur = cur[:-1], cur[-1] + ch
        elif cur and cur[-1] in _NO_LINE_END:
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

    consumed = sum(len(x) for x in lines) + dropped
    if consumed < len(text) and lines:
        remainder = text[consumed:]
        if dropped > 0 and remainder and not remainder.startswith(" ") and not lines[-1].endswith(" "):
            remainder = " " + remainder
        from PySide6.QtCore import Qt

        lines[-1] = fm.elidedText(lines[-1] + remainder, Qt.ElideRight, max_width)
    return [ln for ln in lines if ln]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


SAMPLES = (
    "夜の列車に乗って、彼は静かに本を読んでいた。" * 3,
    "The quick brown fox jumps over the lazy dog. " * 4,
    "中英混排 mixed 的一句话，看看会不会把 Englishword 劈开。" * 2,
    "短句。",
    "",
)


def test_wrap_text_matches_old_implementation():
    """优化后的换行结果必须和旧实现**逐字一致**（只是算法不同）。"""
    from app.ui.subtitle_overlay import wrap_text

    fm = _FixedMetrics()
    for text in SAMPLES:
        for width in (40, 120, 300, 900):
            for max_lines in (1, 2, 3):
                got = wrap_text(fm, text, width, max_lines)
                want = _wrap_text_reference(fm, text, width, max_lines)
                assert got == want, (
                    f"换行结果变了：width={width} lines={max_lines} text={text[:20]!r}"
                    f"\n got={got}\nwant={want}"
                )


def test_wrap_text_is_linear_in_measure_calls():
    """度量**字符总量**必须从 O(n²) 降到 O(n)。

    旧写法每加一个字就量一遍整行：一行 90 字就要量 1+2+…+90 ≈ 4000 个字符。
    而它每 200ms（partial 刷新）就跑一次，还被逐档缩字号反复触发。
    """
    from app.ui.subtitle_overlay import wrap_text

    text = "字幕测试文本，一二三四五六七八九十。" * 10   # 180 字
    width = 900      # 每行约 90 字：让"整行重量"的代价充分暴露
    lines = 10       # 不因 max_lines 提前 break

    fast = _FixedMetrics()
    wrap_text(fast, text, width, lines)

    slow = _FixedMetrics()
    _wrap_text_reference(slow, text, width, lines)

    assert fast.chars < len(text) * 5, f"新实现度量了 {fast.chars} 个字符（应该 O(n)）"
    assert slow.chars > fast.chars * 8, (
        f"优化幅度不够：旧 {slow.chars} vs 新 {fast.chars} 个字符——"
        "检查是不是又改回'每加一个字把整行重量一遍'了"
    )


def test_wrap_text_works_with_real_font(qapp):
    """真字体下也不许抛异常（offscreen 的度量恒为 0，只做烟雾测试）。"""
    from PySide6.QtGui import QFont, QFontMetrics

    from app.ui.subtitle_overlay import wrap_text

    fm = QFontMetrics(QFont("Microsoft YaHei UI", 34))
    assert wrap_text(fm, "测试 text 混合", 200, 2)


# --------------------------------------------------------------------------- #
# 6. 性能：缩字号用二分，不许逐档试
# --------------------------------------------------------------------------- #
def test_fit_font_size_matches_linear_scan(qapp):
    """二分找到的字号必须和"从大到小逐档试"的结果一致。"""
    from app.config import OverlayConfig
    from app.subtitle.model import SubtitleState
    from app.ui.subtitle_overlay import PADDING, SubtitleOverlay

    cfg = OverlayConfig()
    cfg.window_width = 900
    cfg.font_size = 48
    cfg.always_on_top = False
    ov = SubtitleOverlay(cfg)
    st = SubtitleState()
    st.add_final("很长很长的一句话，用来把窗口撑满，逼它缩字号。" * 4, "zh", line_id=1)
    ov.state = st

    width, body_h = 880, 120
    got = ov._fit_font_size(width, body_h)

    # 参照：从大到小找第一个放得下的
    wrap_w = max(40, width - PADDING * 2)
    expected = ov._min_font_size()
    for size in range(cfg.font_size - 1, ov._min_font_size() - 1, -1):
        if ov._rows_height(wrap_w, size) <= body_h:
            expected = size
            break

    assert got == expected, f"二分选了 {got}pt，逐档扫是 {expected}pt"


# --------------------------------------------------------------------------- #
# 7. 设置页：后台线程不许碰控件
# --------------------------------------------------------------------------- #
@pytest.fixture()
def settings_win(qapp, monkeypatch):
    """打开设置窗，但**掐掉 save**（绝不允许碰真实 data/config.json）。"""
    from app.config import AppConfig as _Cfg
    from app.models.downloader import ModelDownloader
    from app.ui.settings import SettingsWindow

    monkeypatch.setattr(_Cfg, "save", lambda self: None, raising=True)
    monkeypatch.setattr(ModelDownloader, "status", lambda self, mid: "installed", raising=True)

    cfg = _Cfg()
    cfg.translate.provider = "llm"
    win = SettingsWindow(cfg)
    yield win
    win.hide()
    win.deleteLater()


def test_test_thread_returns_models_instead_of_touching_ui():
    """后台测试线程只能把结果**带回来**，由 UI 线程去填下拉框。

    以前 ``_test_all`` 直接在 QThread 里调 ``self._fill_models(...)``——
    跨线程改 QComboBox 属于未定义行为（偶发崩溃，且只在点了"测试全部通道"
    之后才出现，极难定位）。
    """
    from app.ui.settings import _TestThread

    t = _TestThread(lambda: ("报告正文", ["m1", "m2"]))
    got: list[tuple[str, list]] = []
    t.done.connect(lambda text, models: got.append((text, list(models))))
    t.run()   # 直接跑，不起线程

    assert got == [("报告正文", ["m1", "m2"])]


def test_test_thread_wraps_plain_string_result():
    from app.ui.settings import _TestThread

    t = _TestThread(lambda: "只有文本")
    got: list[tuple[str, list]] = []
    t.done.connect(lambda text, models: got.append((text, list(models))))
    t.run()
    assert got == [("只有文本", [])]


def test_on_test_done_fills_model_box_on_ui_thread(settings_win):
    win = settings_win
    win._on_test_done("报告", ["deepseek-flash", "deepseek-v4-pro"])

    models = [win.llm_model.itemText(i) for i in range(win.llm_model.count())]
    assert "deepseek-flash" in models
    assert "deepseek-v4-pro" in models


def test_provider_none_disables_every_llm_input_but_keeps_visible(settings_win):
    """通道选「不翻译」时，LLM 那一块**整块可看不可改**。

    漏掉快捷预设下拉（就是这次的 bug）会让用户以为"点这里就能生效"。
    """
    win = settings_win
    win.provider_combo.setCurrentIndex(win.provider_combo.findData("none"))

    for widget in (
        win.llm_base, win.llm_key, win.llm_model,
        win.llm_style, win.llm_no_think, win.llm_preset,
    ):
        assert not widget.isEnabled(), f"{widget} 本该是灰的"
    # 仍然看得到：藏起来用户就不知道 LLM 要在这里配。
    # 用 isHidden()（只看"自身是否被 setVisible(False)"）而不是 isVisibleTo()：
    # 这个分页此刻不是当前标签页，祖先链本来就不显示。
    assert not win.llm_box.isHidden(), "LLM 设置块不该被藏起来"
    assert not win.llm_switch_btn.isHidden(), "一键切换按钮不该被藏起来"


def test_proxy_without_scheme_is_understood(settings_win):
    """用户常只填 ``127.0.0.1:2333``（不带 http://）。

    以前 urlparse 会把整串当成 path、hostname 为空，然后连 "" 报错，
    界面显示"不可达"——用户以为代理坏了，其实是地址没被认出来。
    """
    win = settings_win
    msg = win._test_proxy("127.0.0.1:1")      # 端口 1 必然拒绝连接
    assert "看不懂" not in msg, msg
    assert "不可达" in msg, msg
