"""离线说话人分离：区间映射、导出前缀、缺模型降级。

真模型推理不在这里测（需要 33MB 模型 + 真实多人音频），
这里守的是"**逻辑**"：谁和谁重叠、缺模型怎么办、前缀怎么加。
"""

from __future__ import annotations

import pytest

from app.offline.cue import Cue
from app.offline.diarize import SpeakerSpan, assign_speakers, find_models, models_available
from app.offline.export import speaker_prefix, to_lrc, to_srt, to_vtt


# --------------------------------------------------------------------------- #
# 区间映射
# --------------------------------------------------------------------------- #
def _cue(start: float, end: float, text: str = "x") -> Cue:
    return Cue(start, end, text)


def test_assign_by_max_overlap():
    """按**重叠时长最大**的区间定角色（不是"中心点落在哪"）。"""
    cues = [_cue(0.0, 2.0), _cue(2.0, 4.0)]
    spans = [
        SpeakerSpan(0.0, 2.1, speaker=0),
        SpeakerSpan(2.1, 4.0, speaker=1),
    ]
    assign_speakers(cues, spans)

    assert cues[0].speaker == 0
    assert cues[1].speaker == 1


def test_short_overlap_is_unknown():
    """重叠不到该句 30% 就标 -1——宁可不标，也不要标错人。"""
    cue = _cue(0.0, 2.0)
    spans = [SpeakerSpan(0.0, 0.2, speaker=5)]     # 只覆盖 10%
    assign_speakers([cue], spans)
    assert cue.speaker == -1


def test_no_spans_leaves_unknown():
    cue = _cue(0.0, 1.0)
    assign_speakers([cue], [])
    assert cue.speaker == -1


def test_cue_outside_all_spans_is_unknown():
    cue = _cue(10.0, 12.0)
    spans = [SpeakerSpan(0.0, 5.0, speaker=0)]
    assign_speakers([cue], spans)
    assert cue.speaker == -1


def test_speaker_span_duration():
    assert SpeakerSpan(1.0, 3.5, 0).duration_s == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# 导出前缀
# --------------------------------------------------------------------------- #
def test_prefix_starts_at_one():
    """内部 0 起，给人看从 1 起（S0 很别扭）。"""
    assert speaker_prefix(Cue(0, 1, "x", speaker=0)) == "[S1] "
    assert speaker_prefix(Cue(0, 1, "x", speaker=2)) == "[S3] "


def test_no_prefix_when_unknown():
    """没做分离时输出必须和以前**一模一样**（不留任何痕迹）。"""
    assert speaker_prefix(Cue(0, 1, "x", speaker=-1)) == ""


def test_srt_has_speaker_prefix():
    cues = [Cue(0.0, 1.0, "こんにちは", speaker=0), Cue(1.0, 2.0, "どうも", speaker=1)]
    srt = to_srt(cues)
    assert "[S1] こんにちは" in srt
    assert "[S2] どうも" in srt


def test_srt_unchanged_without_diarization():
    """回归：不标角色时，SRT 内容不能多出任何前缀。"""
    srt = to_srt([Cue(0.0, 1.0, "こんにちは")])
    assert "こんにちは" in srt
    assert "[S" not in srt


def test_vtt_has_speaker_prefix():
    vtt = to_vtt([Cue(0.0, 1.0, "hi", speaker=0)])
    assert "[S1] hi" in vtt


def test_lrc_bilingual_prefixes_both_lines():
    """LRC 双语是同时间戳两行，两行都要带角色，否则看不出该行属于谁。"""
    cue = Cue(1.0, 2.0, "原文", translation="译文", speaker=0)
    lrc = to_lrc([cue], mode="bilingual")
    lines = [ln for ln in lrc.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all("[S1] " in ln for ln in lines), lines


def test_split_keeps_speaker():
    """切成多行后每行都要保留角色（否则后半句就没角色了）。"""
    from app.offline.cue import split_cue

    cue = Cue(0.0, 4.0, "很长的一句话" * 10, speaker=2)
    parts = split_cue(cue, max_width=12)
    assert len(parts) > 1
    assert all(p.speaker == 2 for p in parts)


# --------------------------------------------------------------------------- #
# 缺模型时降级
# --------------------------------------------------------------------------- #
def test_find_models_returns_none_when_missing(tmp_path):
    assert find_models(tmp_path) is None
    assert models_available(tmp_path) is False


def test_find_models_finds_complete_pair(tmp_path):
    seg = tmp_path / "pyannote-segmentation-3-0" / "model.onnx"
    emb = tmp_path / "campplus-sv" / "campplus.onnx"
    seg.parent.mkdir(parents=True)
    emb.parent.mkdir(parents=True)
    seg.write_bytes(b"x")
    emb.write_bytes(b"x")

    found = find_models(tmp_path)
    assert found is not None
    assert found[0] == seg and found[1] == emb


def test_find_models_needs_both(tmp_path):
    """只有分割模型、没有声纹模型 → 也算缺（两个都要）。"""
    seg = tmp_path / "pyannote-segmentation-3-0" / "model.onnx"
    seg.parent.mkdir(parents=True)
    seg.write_bytes(b"x")
    assert find_models(tmp_path) is None


def test_diarize_is_off_by_default():
    """默认关：它要多跑一遍推理、还要把整段音频留在内存里。"""
    from app.offline.transcribe import TranscribeOptions

    assert TranscribeOptions().diarize is False


def test_diarizer_without_models_raises_clear_error(tmp_path):
    """缺模型时抛的错要说清缺什么，而不是"No module"这种让人猜的话。"""
    from app.offline.diarize import SpeakerDiarizer

    d = SpeakerDiarizer(models_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        d.load()
    assert "pyannote-segmentation-3-0" in str(exc.value)
    assert "campplus-sv" in str(exc.value)
