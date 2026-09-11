"""验收：语言路由器 + 语种识别（language=auto）。

这是"多语言"需求的端到端验收：程序不知道音频是什么语言，
要自己判断出来，再选对引擎，最后还要把文本清理到可上屏。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_router.py
    .venv\\Scripts\\python.exe scripts\\verify_router.py --language zh   # 强制指定语言做对照
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
from app.asr.router import LanguageRouter  # noqa: E402
from app.config import ASRConfig  # noqa: E402

TARGET_SR = 16000


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def normalize(t: str) -> str:
    drop = set("，。、！？；：“”‘’（）《》〈〉「」『』【】…—　 \t\n·,.!?;:\"'()[]<>-_/\\|")
    return "".join(ch for ch in t if ch not in drop).lower()


def cer(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)


def main() -> int:
    ap = argparse.ArgumentParser(description="语言路由器验收")
    ap.add_argument("--language", default="auto", help="auto 或指定语言（做对照）")
    ap.add_argument("--chunk-ms", type=int, default=100)
    args = ap.parse_args()

    speech = PROJECT / "data" / "test_speech"
    manifest = {}
    for line in (speech / "manifest.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            n, lang, text = line.split("\t", 2)
            manifest[n] = (lang, text)

    cfg = ASRConfig()
    cfg.language = args.language

    print("=" * 96)
    print(f"语言路由器验收   language={args.language}")
    print("=" * 96)

    results = []
    for wav in sorted(speech.glob("*.wav")):
        true_lang, expected = manifest.get(wav.name, ("", ""))
        router = LanguageRouter(cfg, on_log=lambda m: print(f"    · {m}"))

        audio = load_wav(wav)
        chunk = int(TARGET_SR * args.chunk_ms / 1000)
        finals: list[str] = []
        t0 = time.monotonic()
        for start in range(0, audio.size, chunk):
            for ev in router.feed(audio[start:start + chunk]):
                if ev.type is ASREventType.FINAL:
                    finals.append(ev.text)
        for ev in router.finish():
            if ev.type is ASREventType.FINAL:
                finals.append(ev.text)
        elapsed = time.monotonic() - t0
        detected = router.language
        stats = router.stats_summary()
        router.close()

        text = "".join(finals)
        acc = 1 - cer(text, expected) if expected else 0.0
        results.append({
            "wav": wav.name, "true": true_lang, "detected": detected,
            "acc": acc, "text": text, "stats": stats,
            "wall": elapsed, "dur": audio.size / TARGET_SR,
        })

        print(f"\n--- {wav.name} ---")
        print(f"  真实语言={true_lang}  识别语言={detected}  "
              f"{'✓ 正确' if detected == true_lang or args.language != 'auto' else '✗ 判错'}")
        print(f"  期望: {expected}")
        print(f"  实际: {text}")
        print(f"  ★ 字准率 {acc * 100:.1f}%   处理 {elapsed:.2f}s / 音频 {audio.size / TARGET_SR:.2f}s")
        for lang, s in stats.get("engines", {}).items():
            print(f"  引擎[{lang}] {s['model']} ({'流式' if s['streaming'] else '分块'}) "
                  f"RTF={s['rtf']} 段数={s['segments']}")

    print("\n" + "=" * 96)
    print(f"{'文件':<12}{'真实':>6}{'识别':>6}{'字准率':>10}{'引擎':>12}")
    print("-" * 96)
    for r in results:
        eng = next(iter(r["stats"].get("engines", {})), "-")
        print(f"{r['wav']:<12}{r['true']:>6}{r['detected']:>6}{r['acc'] * 100:>9.1f}%{eng:>12}")

    ok = all(r["acc"] > 0.8 for r in results)
    if args.language == "auto":
        lang_ok = all(r["detected"] == r["true"] for r in results)
        print(f"\n语种判断: {sum(1 for r in results if r['detected'] == r['true'])}/{len(results)} 正确")
    else:
        lang_ok = True
    print(f"字准率 > 80%: {sum(1 for r in results if r['acc'] > 0.8)}/{len(results)}")
    print("\n" + ("✅ 通过" if ok and lang_ok else "❌ 未通过"))
    return 0 if ok and lang_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
