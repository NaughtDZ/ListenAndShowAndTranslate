"""离线字幕工具：切行 / 三种格式 / 端到端。

守两件容易回归的事：

1. **时间戳必须是真实语音边界**（来自 ``SpeechSegment.start_sample``），
   不是 ``ASREvent`` 那种"上次出结果 → 本次喂完"（含静音，字幕会提前冒出来）；
2. **喂 VAD 的块必须够小**：实测 silero 在 320ms 起就会把两句话并成一句，
   10 秒块更是只吐出末尾一小截（见 ``VAD_BLOCK_SECONDS`` 的实测表）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.offline.cue import (
    Cue,
    display_width,
    slice_by_weights,
    split_cue,
    split_text,
)
from app.offline.export import (
    fmt_lrc,
    fmt_srt,
    fmt_vtt,
    to_lrc,
    to_srt,
    to_vtt,
    write_subtitles,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WAV = ROOT / "data" / "test_speech" / "ja_03.wav"


# --------------------------------------------------------------------------- #
# 切行
# --------------------------------------------------------------------------- #
def test_display_width_counts_cjk_double():
    assert display_width("こんにちは") == 5.0
    assert display_width("hello") == 2.5
    assert display_width("リン ファン") > display_width("ab")


def test_split_text_short_sentence_stays_whole():
    assert split_text("第印象、夜の列車。", 24) == ["第印象、夜の列車。"]


def test_split_text_breaks_at_sentence_end_first():
    text = "第一句。第二句。"
    assert split_text(text, 6) == ["第一句。", "第二句。"]


def test_split_text_keeps_closing_bracket_with_sentence():
    text = "「そんなはずはない。」と彼女は言った。"
    chunks = split_text(text, 14)
    assert chunks[0].endswith("。」")


def test_split_cue_preserves_total_span_and_order():
    cue = Cue(1.0, 5.0, "これはとても長い文です。" * 4)
    parts = split_cue(cue, 12)
    assert len(parts) > 1
    assert parts[0].start_s == pytest.approx(1.0)
    assert parts[-1].end_s == pytest.approx(5.0)
    for prev, nxt in zip(parts, parts[1:]):
        assert nxt.start_s == pytest.approx(prev.end_s), "切出来的行必须首尾相接"
    assert all(part.duration_s > 0 for part in parts)


def test_split_cue_distributes_translation_by_width():
    cue = Cue(0.0, 4.0, "前半句很长很长。后半句。", "first half is long. second one.")
    parts = split_cue(cue, 10)
    assert len(parts) > 1
    joined = "".join(p.translation for p in parts).replace(" ", "")
    assert joined == cue.translation.replace(" ", ""), "译文不能被丢字"


def test_slice_by_weights_covers_all_text():
    pieces = slice_by_weights("abcdefghij", [1, 1, 2])
    assert "".join(pieces) == "abcdefghij"


# --------------------------------------------------------------------------- #
# 时间戳格式
# --------------------------------------------------------------------------- #
def test_time_formats():
    assert fmt_srt(1.234) == "00:00:01,234"
    assert fmt_srt(3725.5) == "01:02:05,500"
    assert fmt_vtt(1.234) == "00:00:01.234"
    assert fmt_lrc(83.45) == "[01:23.45]"
    assert fmt_lrc(0.0) == "[00:00.00]"


def test_srt_structure():
    cues = [Cue(0.07, 2.19, "第一句"), Cue(2.89, 6.04, "第二句")]
    text = to_srt(cues)
    assert text.startswith("1\n00:00:00,070 --> 00:00:02,190\n第一句")
    assert "\n2\n00:00:02,890 --> 00:00:06,040\n第二句" in text


def test_vtt_has_header_and_dot_separator():
    text = to_vtt([Cue(0.0, 1.0, "hi")])
    assert text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in text
    assert "," not in text.split("-->")[0]


def test_lrc_has_no_end_time():
    """LRC 是歌词格式：只有开始时间，不能写出区间。"""
    line = to_lrc([Cue(1.0, 5.0, "一句")]).strip()
    assert line == "[00:01.00]一句"
    assert "-->" not in line


def test_lrc_bilingual_emits_two_lines_with_same_stamp():
    line = to_lrc([Cue(1.0, 3.0, "原文", "译文")], mode="bilingual")
    assert line.count("[00:01.00]") == 2
    assert "原文" in line and "译文" in line


def test_target_mode_falls_back_to_source():
    """没有译文时不能输出空白——回退原文。"""
    text = to_srt([Cue(0.0, 1.0, "原文")], mode="target")
    assert "原文" in text


def test_export_skips_empty_and_orders_by_time():
    cues = [Cue(5.0, 6.0, "后"), Cue(1.0, 2.0, " "), Cue(0.0, 1.0, "前")]
    text = to_srt(cues)
    assert "后" in text and "前" in text
    assert text.index("前") < text.index("后")
    assert text.count("-->") == 2


def test_write_subtitles_uses_bom_and_lf(tmp_path):
    path = write_subtitles(tmp_path / "a.srt", [Cue(0.0, 1.0, "x")], "srt")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "带 BOM，Windows 播放器才认"
    assert b"\r\n" not in raw


# --------------------------------------------------------------------------- #
# 参数守护
# --------------------------------------------------------------------------- #
def test_vad_block_stays_small():
    """回归护栏：块一大，silero 的断句就会静默失效（实测 320ms 起并句）。"""
    from app.offline.transcribe import VAD_BLOCK_SECONDS

    assert VAD_BLOCK_SECONDS <= 0.25


def test_job_defaults_are_sane():
    from app.offline.job import JobOptions

    options = JobOptions()
    assert options.formats == ("srt",)
    assert options.mode == "source"
    assert options.transcribe.max_segment_ms <= 10000, (
        "成句上限不能调得比常见连续语音还长，否则 VAD 一直不出结果"
    )


# --------------------------------------------------------------------------- #
# 翻译接线（用假 hub，不联网）
# --------------------------------------------------------------------------- #
def test_translate_fills_cues_and_counts_failures():
    from app.offline.job import JobOptions, SubtitleJob
    from app.translate.base import TranslateResult

    class _HalfBrokenHub:
        """只翻译第一条，第二条算失败——模拟通道抽风。"""

        def translate(self, segments, source_language="", target_language=""):
            out = TranslateResult()
            for i, segment in enumerate(segments):
                if i == 0:
                    out.translations[segment.id] = "译:" + segment.text
                else:
                    out.failures[segment.id] = "通道超时"
            return out

    job = SubtitleJob(JobOptions(translate=True))
    cues = [Cue(0.0, 1.0, "甲"), Cue(1.0, 2.0, "乙")]
    ok, failed = job._translate(cues, _HalfBrokenHub(), 0.0, 1.0)

    assert ok == 1 and failed == 1
    assert cues[0].translation == "译:甲"
    assert cues[1].translation == ""      # 失败就留空，导出时回退原文
    assert "甲" in to_srt(cues, mode="target")


# --------------------------------------------------------------------------- #
# 端到端（需要已装模型 + 样例音频）
# --------------------------------------------------------------------------- #
def _installed(model_id: str) -> bool:
    try:
        from app.models.downloader import ModelDownloader

        return ModelDownloader().status(model_id) == "installed"
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="样例音频不在")
@pytest.mark.skipif(not _installed("sensevoice-int8"), reason="SenseVoice 未安装")
def test_end_to_end_writes_three_formats(tmp_path):
    from app.offline.job import JobOptions, SubtitleJob
    from app.offline.transcribe import TranscribeOptions

    options = JobOptions(
        transcribe=TranscribeOptions(model_id="sensevoice-int8", language="ja"),
        formats=("srt", "vtt", "lrc"),
        max_line_width=24,
        out_dir=str(tmp_path),
    )
    result = SubtitleJob(options).run([SAMPLE_WAV])

    item = result.files[0]
    assert item.error == "", item.error
    assert item.cues >= 1
    assert {p.suffix for p in item.outputs} == {".srt", ".vtt", ".lrc"}

    srt = (tmp_path / "ja_03.srt").read_text(encoding="utf-8-sig")
    # 第一句应该在 0 秒附近开始（不是"上一段结束"式的错位）
    assert "00:00:00," in srt.splitlines()[1]
    assert "第" in srt, "识别文本应包含日语原文"


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="样例音频不在")
def test_decode_chunking():
    from app.offline import decode

    info = decode.probe(SAMPLE_WAV)
    assert info.duration_s == pytest.approx(11.4, abs=0.3)

    chunks = list(decode.iter_pcm16k(SAMPLE_WAV, chunk_seconds=5.0))
    assert len(chunks) >= 3
    total = sum(c.size for c in chunks)
    assert total == pytest.approx(info.duration_s * 16000, rel=0.02)
    assert all(c.dtype.name == "float32" for c in chunks)
