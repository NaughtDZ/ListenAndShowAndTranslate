"""混合 VAD 可行性实验（独立脚本，不改主流程、不接主流程）。

## 要验证什么

Whisper-VAD 慢的根因是"每 1 秒重算整个 30 秒窗口"。而大量音频段本来就是
**正常音量的语音**，Silero 判得又快又准（RTF ≈ 0.01）。

所以想的是：**Silero 先扫一遍，只把"它说没语音、但电平不低"的可疑区间
交给 Whisper-VAD**。ASMR 素材里正常音量的部分不少，理论上能砍掉大部分推理。

## 三种做法对比

1. 纯 Whisper-VAD（现状）
2. 混合（Silero 粗筛 + 可疑区间才跑 Whisper）
3. 纯 Silero（下限对照，看漏多少）

看三件事：耗时、检出段数、检出语音总时长。

## 结论怎么用

- 混合的段和纯 Whisper 基本一致（段数差 ≤1、时长差 <5%）且明显更快 → 值得接
- 段明显变少 → Silero 漏掉的区间没被补回来，粗筛逻辑有问题
- 没变快 → 可疑区间太多，混了个寂寞

## 留后路

这个脚本不 import 任何主流程改动。要接进 ``transcribe.py`` 之前，
先确认这里的数字站得住。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.asr.vad import SAMPLE_RATE, SpeechSegment, VadSegmenter
from app.asr.whisper_vad import WhisperVadSegmenter

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 1600  # 100ms，和 transcribe.VAD_BLOCK_SECONDS 一致


# --------------------------------------------------------------------------- #
def run_silero(audio: np.ndarray, models_dir: Path) -> list[SpeechSegment]:
    seg = VadSegmenter(min_silence_ms=800, min_speech_ms=250, max_speech_ms=8000,
                       num_threads=1, models_dir=models_dir)
    seg.load()
    for i in range(0, audio.size, BLOCK):
        seg.accept(audio[i : i + BLOCK])
    out = seg.flush()
    seg.reset()
    return out


def run_whisper(audio: np.ndarray, models_dir: Path, threads: int = 8) -> list[SpeechSegment]:
    seg = WhisperVadSegmenter(min_silence_ms=800, min_speech_ms=250, max_speech_ms=8000,
                              hop_ms=1000, num_threads=threads, models_dir=models_dir)
    seg.load()
    for i in range(0, audio.size, BLOCK):
        seg.accept(audio[i : i + BLOCK])
    out = seg.flush()
    seg.reset()
    return out


# --------------------------------------------------------------------------- #
def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(max(r, 1e-9))


def spans_of(segs: list[SpeechSegment]) -> list[tuple[int, int]]:
    return [(s.start_sample, s.start_sample + s.samples.size) for s in segs]


def _gaps_of(covered: np.ndarray) -> list[tuple[int, int]]:
    """布尔掩码里 False 的连续区间。"""
    out: list[tuple[int, int]] = []
    n = covered.size
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def merge_spans(spans: list[tuple[int, int]], gap_s: float = 0.25) -> list[tuple[int, int]]:
    """按时间合并重叠/相邻的区间。"""
    if not spans:
        return []
    ordered = sorted((int(a), int(b)) for a, b in spans if b > a)
    out = [list(ordered[0])]
    gap = int(gap_s * SAMPLE_RATE)
    for a, b in ordered[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def make_segments(audio: np.ndarray, spans: list[tuple[int, int]]) -> list[SpeechSegment]:    return [
        SpeechSegment(samples=audio[a:b].copy(), start_sample=a)
        for a, b in spans
        if b > a
    ]


def suspicious_spans(
    audio: np.ndarray,
    speech: list[SpeechSegment],
    floor_db: float = -50.0,
    pad_s: float = 0.3,
    min_gap_s: float = 0.3,
) -> list[tuple[int, int]]:
    """找出"Silero 说没语音、但电平不低"的区间（样本下标，左闭右开）。"""
    covered = np.zeros(audio.size, dtype=bool)
    for s in speech:
        a = max(0, int(s.start_sample))
        b = min(audio.size, a + int(s.samples.size))
        if b > a:
            covered[a:b] = True

    gap = ~covered
    spans: list[tuple[int, int]] = []
    pad = int(pad_s * SAMPLE_RATE)
    min_gap = int(min_gap_s * SAMPLE_RATE)
    i = 0
    while i < audio.size:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j < audio.size and gap[j]:
            j += 1
        if j - i >= min_gap and rms_db(audio[i:j]) > floor_db:
            spans.append((max(0, i - pad), min(audio.size, j + pad)))
        i = j
    return spans


def report(name: str, segs: list[SpeechSegment], dt: float, dur: float) -> dict:
    speech = sum(s.samples.size for s in segs) / SAMPLE_RATE
    print(f"  {name:26s} {dt:6.2f}s  RTF {dt / dur:5.2f}  段数 {len(segs):3d}  语音 {speech:6.1f}s")
    return {"name": name, "dt": dt, "n": len(segs), "speech": speech}


# --------------------------------------------------------------------------- #
def main() -> None:
    import soundfile as sf
    import soxr

    models_dir = ROOT / "data" / "models"
    src = ROOT / "data/parakeet_ja_test/test_ja_2.wav"
    a, sr = sf.read(str(src), dtype="float32", always_2d=True)
    m = a.mean(axis=1)
    if sr != SAMPLE_RATE:
        m = soxr.resample(m, sr, SAMPLE_RATE, quality="HQ")
    repeats = 6
    audio = np.tile(np.ascontiguousarray(m, dtype=np.float32), repeats)
    dur = audio.size / SAMPLE_RATE
    print(f"测试音频 {dur:.1f}s（{src.name} ×{repeats}）\n")

    res: dict[str, dict] = {}

    t0 = time.perf_counter()
    sil = run_silero(audio, models_dir)
    res["silero"] = report("① 纯 Silero", sil, time.perf_counter() - t0, dur)

    t0 = time.perf_counter()
    wh = run_whisper(audio, models_dir)
    res["whisper"] = report("② 纯 Whisper-VAD", wh, time.perf_counter() - t0, dur)

    # ---- 混合 ----------------------------------------------------------- #
    t0 = time.perf_counter()
    spans = suspicious_spans(audio, sil)
    sus_s = sum(b - a for a, b in spans) / SAMPLE_RATE
    extra: list[SpeechSegment] = []
    for a0, b0 in spans:
        for s in run_whisper(audio[a0:b0], models_dir):
            extra.append(SpeechSegment(samples=s.samples, start_sample=s.start_sample + a0))
    hybrid_spans = merge_spans(spans_of(sil) + spans_of(extra))
    hybrid = make_segments(audio, hybrid_spans)
    res["hybrid"] = report("③ 混合（RMS 筛可疑）", hybrid, time.perf_counter() - t0, dur)
    print(f"     可疑区间 {len(spans)} 个，共 {sus_s:.1f}s（占 {sus_s / dur:.0%}）")

    # ---- 变体：不筛电平，所有 Silero 静音区间都跑 Whisper --------------- #
    # 逻辑上更自洽：Silero 的正检直接采用，只把它"没听到"的补一遍。
    # 代价是省下的只有 Silero 检出语音的那部分比例。
    t0 = time.perf_counter()
    sil_spans = spans_of(sil)
    covered = np.zeros(audio.size, dtype=bool)
    for a0, b0 in sil_spans:
        covered[a0:b0] = True
    gaps = merge_spans(
        [
            (max(0, a - int(0.3 * SAMPLE_RATE)), min(audio.size, b + int(0.3 * SAMPLE_RATE)))
            for a, b in _gaps_of(covered)
            if b - a >= int(0.6 * SAMPLE_RATE)   # 太短的空隙不值得跑
        ]
    )
    extra2: list[SpeechSegment] = []
    for a0, b0 in gaps:
        for s in run_whisper(audio[a0:b0], models_dir):
            extra2.append(SpeechSegment(samples=s.samples, start_sample=s.start_sample + a0))
    hybrid2_spans = merge_spans(sil_spans + spans_of(extra2))
    hybrid2 = make_segments(audio, hybrid2_spans)
    res["hybrid2"] = report("④ 混合（静音区间全跑）", hybrid2, time.perf_counter() - t0, dur)
    gap_s = sum(b - a for a, b in gaps) / SAMPLE_RATE
    print(f"     补齐区间 {len(gaps)} 个，共 {gap_s:.1f}s（占 {gap_s / dur:.0%}）")

    # ---- 对比 ----------------------------------------------------------- #
    print("\n=== 与纯 Whisper-VAD 对比 ===")
    base = res["whisper"]
    for key in ("silero", "hybrid", "hybrid2"):
        r = res[key]
        print(
            f"  {r['name']:26s} 段数 {r['n'] - base['n']:+d}  "
            f"语音 {r['speech'] - base['speech']:+6.1f}s  "
            f"耗时 {r['dt'] / base['dt']:.0%}"
        )


if __name__ == "__main__":
    main()
