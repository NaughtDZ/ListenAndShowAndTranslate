"""把 WAV 喂给 ASR 引擎并打印结果（含延迟与字准率）。

用法：
    .venv\\Scripts\\python.exe scripts\\transcribe_wav.py data\\test_speech\\zh_00.wav --model zipformer-zh-int8
    .venv\\Scripts\\python.exe scripts\\transcribe_wav.py --all          # 跑 manifest 里全部
    .venv\\Scripts\\python.exe scripts\\transcribe_wav.py --all --realtime  # 按真实速度喂，测实时延迟
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soxr
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from app.asr.base import ASREventType  # noqa: E402
from app.asr.sherpa_offline import SherpaOfflineEngine  # noqa: E402
from app.asr.sherpa_stream import SherpaStreamingEngine  # noqa: E402

TARGET_SR = 16000

# 模型 id → 语言（仅用于给结果打标签与选对引擎）
MODEL_LANGUAGE = {
    "zipformer-zh-int8": "zh",
    "zipformer-zh-en-int8": "zh-en",
    "zipformer-en-int8": "en",
    "sensevoice-int8": "ja",
    "whisper-turbo-int8": "auto",
}


def make_engine(model_id: str, min_silence_ms: int, num_threads: int, language: str):
    """按模型 id 自动选择流式或分块引擎。"""
    if "zipformer" in model_id:
        return SherpaStreamingEngine(
            model_id=model_id, language=language,
            num_threads=num_threads, min_silence_ms=min_silence_ms,
        )
    return SherpaOfflineEngine(
        model_id=model_id, language=language,
        num_threads=num_threads, min_silence_ms=min_silence_ms,
    )


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（按字符），用于算字准率。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(text: str) -> str:
    """去掉标点与空白，只比字（ASR 通常不出标点，直接比会冤枉它）。"""
    drop = set("，。、！？；：“”‘’（）《》〈〉「」『』【】…—　 \t\n·,.!?;:\"'()[]<>-_/\\|")
    return "".join(ch for ch in text if ch not in drop).lower()


def transcribe(
    wav: Path,
    model_id: str,
    expected: str = "",
    realtime: bool = False,
    chunk_ms: int = 100,
    min_silence_ms: int = 350,
    num_threads: int = 2,
    language: str = "",
    verbose: bool = True,
) -> dict:
    audio = load_wav(wav)
    duration = audio.size / TARGET_SR
    lang = language or MODEL_LANGUAGE.get(model_id, "auto")

    engine = make_engine(model_id, min_silence_ms, num_threads, lang)
    engine.start()

    finals: list[str] = []
    partials = 0
    result_latency: list[float] = []
    silence_s = min_silence_ms / 1000.0

    wall_t0 = time.monotonic()
    chunk = int(TARGET_SR * chunk_ms / 1000)

    def handle(events, now_audio: float) -> None:
        """统一处理事件与延迟统计。"""
        nonlocal partials
        for ev in events:
            if ev.type is ASREventType.PARTIAL:
                partials += 1
            elif ev.type is ASREventType.FINAL:
                finals.append(ev.text)
                if realtime:
                    # 「语音结束 → 字幕出现」的真实延迟：
                    # 事件是在尾部静音攒够 min_silence 之后才触发的，
                    # 所以要把这段等待时间从 audio_end 里扣掉，否则会算成 0 甚至负数。
                    speech_end_at = max(0.0, ev.audio_end_s - silence_s)
                    lat = (time.monotonic() - wall_t0 - speech_end_at) * 1000
                else:
                    lat = ev.latency_ms
                result_latency.append(lat)
                if verbose:
                    print(f"    [音频 {now_audio:6.2f}s / 实际 {time.monotonic() - wall_t0:6.2f}s] "
                          f"定稿(延迟 {lat:5.0f}ms): {ev.text}")

    if realtime:
        for start in range(0, audio.size, chunk):
            piece = audio[start:start + chunk]
            now_audio = (start + piece.size) / TARGET_SR
            handle(engine.feed(piece), now_audio)
            target = wall_t0 + now_audio
            sleep = target - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    else:
        for start in range(0, audio.size, chunk):
            piece = audio[start:start + chunk]
            now_audio = (start + piece.size) / TARGET_SR
            handle(engine.feed(piece), now_audio)

    handle(engine.finish(), duration)

    wall = time.monotonic() - wall_t0
    engine.close()

    text = "".join(finals)
    stats = {
        "wav": wav.name,
        "model": model_id,
        "duration_s": duration,
        "wall_s": wall,
        "rtf": engine.stats.process_seconds / duration if duration else 0,
        "finals": len(finals),
        "partials": partials,
        "text": text,
        "expected": expected,
        "latency_p50": _pct(result_latency, 50),
        "latency_p90": _pct(result_latency, 90),
    }

    if expected:
        a, b = normalize(text), normalize(expected)
        dist = edit_distance(a, b)
        stats["cer"] = dist / max(1, len(b))
        stats["accuracy"] = max(0.0, 1 - stats["cer"])

    return stats


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))]


def main() -> int:
    ap = argparse.ArgumentParser(description="WAV → ASR 识别验证")
    ap.add_argument("wav", nargs="?", help="WAV 文件")
    ap.add_argument("--all", action="store_true", help="跑 manifest 里全部")
    ap.add_argument("--model", default="zipformer-zh-int8")
    ap.add_argument("--realtime", action="store_true", help="按真实速度喂，测实际延迟")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--min-silence-ms", type=int, default=350)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--language", default="", help="覆盖语言（zh/en/ja/ko/yue/auto）")
    args = ap.parse_args()

    speech_dir = PROJECT / "data" / "test_speech"
    manifest = speech_dir / "manifest.tsv"
    expected_map: dict[str, str] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, _lang, text = line.split("\t", 2)
                expected_map[name] = text

    if args.all:
        wavs = sorted(speech_dir.glob("*.wav"))
    elif args.wav:
        wavs = [Path(args.wav)]
    else:
        ap.error("请给出 WAV 路径或 --all")

    print("=" * 100)
    print(f"识别验证  model={args.model}  realtime={args.realtime}  断句={args.min_silence_ms}ms")
    print("=" * 100)

    summary = []
    for wav in wavs:
        exp = expected_map.get(wav.name, "")
        print(f"\n--- {wav.name} ---")
        if exp:
            print(f"  期望: {exp}")
        st = transcribe(
            wav, args.model, expected=exp,
            realtime=args.realtime, chunk_ms=args.chunk_ms,
            min_silence_ms=args.min_silence_ms, num_threads=args.threads,
            language=args.language,
        )
        print(f"  实际: {st['text']}")
        print(f"  音频 {st['duration_s']:.2f}s | 处理 {st['wall_s']:.2f}s | RTF {st['rtf']:.3f} | "
              f"定稿 {st['finals']} 段 / 中间结果 {st['partials']} 次")
        if "accuracy" in st:
            print(f"  ★ 字准率 {st['accuracy'] * 100:.1f}%  (CER {st['cer'] * 100:.1f}%)")
        if st["latency_p50"]:
            print(f"  延迟 P50 {st['latency_p50']:.0f}ms  P90 {st['latency_p90']:.0f}ms")
        summary.append(st)

    if len(summary) > 1:
        print("\n" + "=" * 100)
        print(f"{'文件':<14}{'RTF':>8}{'字准率':>10}{'段数':>6}")
        print("-" * 100)
        for st in summary:
            acc = f"{st['accuracy'] * 100:.1f}%" if "accuracy" in st else "-"
            print(f"{st['wav']:<14}{st['rtf']:>8.3f}{acc:>10}{st['finals']:>6}")
        ok = [st for st in summary if st.get("accuracy", 0) > 0.8]
        print(f"\n字准率 > 80% 的: {len(ok)}/{len(summary)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
