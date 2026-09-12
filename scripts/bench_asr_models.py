"""ASR 模型横向对拍：同一批音频、同一套引擎代码，比**字准率 / 实时率 / 延迟**。

用法::

    .venv\\Scripts\\python.exe scripts\\bench_asr_models.py            # 全部语言
    .venv\\Scripts\\python.exe scripts\\bench_asr_models.py --lang ja  # 只跑日语
    .venv\\Scripts\\python.exe scripts\\bench_asr_models.py --threads 8 --provider cuda

为什么单独一个脚本：换模型这种决定必须**用实测说话**（计划书第 9 节红线），
而"同一批音频 + 同一套引擎代码 + 同一台机器"才叫可比。
它复用 ``transcribe_wav.py`` 的引擎构造与字准率算法，保证和正式程序走同一条路。

音频来自 ``scripts/gen_test_speech.py``（Windows SAPI 合成，因此**知道正确文本**）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from transcribe_wav import transcribe  # noqa: E402  （复用引擎构造与字准率口径）

from app.models.registry import MODELS  # noqa: E402

SPEECH_DIR = PROJECT / "data" / "test_speech"

# 每个语言要对比哪些模型（顺序 = 报告顺序，第一个是**当前默认路由**）
CANDIDATES: dict[str, list[str]] = {
    "zh": [
        "zipformer-zh-int8",
        "fire-red-asr2-ctc-zh_en-int8",
        "dolphin-base-ctc-int8",
        "omnilingual-300m-ctc-int8",
        "whisper-turbo-int8",
    ],
    "en": [
        "zipformer-en-int8",
        "fire-red-asr2-ctc-zh_en-int8",
        "omnilingual-300m-ctc-int8",
        "whisper-turbo-int8",
    ],
    "ja": [
        "parakeet-ja-int8",
        "dolphin-base-ctc-int8",
        "omnilingual-300m-ctc-int8",
        "whisper-turbo-int8",
    ],
    # 韩语/粤语：本机 SAPI 没有对应音色，造不出带标注的音频 →
    # 只能用模型自带 test_wavs 做"能加载出字"的冒烟检查（见 --smoke-only）。
    "ko": ["dolphin-base-ctc-int8", "omnilingual-300m-ctc-int8", "whisper-turbo-int8"],
    "yue": ["dolphin-base-ctc-int8", "omnilingual-300m-ctc-int8", "whisper-turbo-int8"],
}


def load_manifest() -> list[tuple[Path, str, str]]:
    manifest = SPEECH_DIR / "manifest.tsv"
    if not manifest.is_file():
        print(f"❌ 没有测试音频清单：{manifest}\n   先跑 scripts/gen_test_speech.py")
        raise SystemExit(2)
    rows: list[tuple[Path, str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append((SPEECH_DIR / parts[0], parts[1], parts[2]))
    return rows


def installed(model_id: str) -> bool:
    from app.models.downloader import ModelDownloader

    try:
        return ModelDownloader().status(model_id) == "installed"
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="", help="只测某个语言（zh/en/ja/ko/yue），留空=全部")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-silence-ms", type=int, default=350)
    ap.add_argument("--provider", default="cpu", help="cpu / cuda / directml / vulkan")
    args = ap.parse_args()

    rows = load_manifest()
    langs = [args.lang] if args.lang else list(CANDIDATES)
    results: list[tuple[str, str, float, float, float, int]] = []

    for lang in langs:
        clips = [(p, exp) for p, lg, exp in rows if lg == lang]
        print(f"\n================ {lang}（{len(clips)} 句） ================")
        if not clips:
            print("  （本机没有这个语言的测试音频，跳过）")
            continue
        for model_id in CANDIDATES.get(lang, []):
            if model_id not in MODELS:
                print(f"  - {model_id}: 注册表里没有，跳过")
                continue
            if not installed(model_id):
                print(f"  - {model_id}: 未下载，跳过")
                continue

            audio_s = 0.0
            compute_s = 0.0
            accs: list[float] = []
            latencies: list[float] = []
            texts: list[str] = []
            for wav, expected in clips:
                st = transcribe(
                    wav, model_id, expected,
                    realtime=True, num_threads=args.threads,
                    min_silence_ms=args.min_silence_ms,
                    language=lang, verbose=False,
                )
                dur = st.get("duration_s", 0.0)
                audio_s += dur
                # engine.stats.process_seconds / duration：**纯计算**耗时（不含喂音频的 sleep），
                # 所以能真实反映"能不能实时跑"
                compute_s += st.get("rtf", 0.0) * dur
                if "accuracy" in st:
                    accs.append(st["accuracy"])
                if st.get("latency_p50"):
                    latencies.append(st["latency_p50"])
                texts.append(st.get("text", ""))

            acc = sum(accs) / len(accs) * 100 if accs else 0.0
            rtf = (compute_s / audio_s) if audio_s else 0.0
            lat = sum(latencies) / len(latencies) if latencies else 0.0
            results.append((lang, model_id, acc, rtf, lat, len(accs)))
            print(f"  ✓ {model_id:32} 字准率 {acc:5.1f}%  计算RTF {rtf:5.2f}  "
                  f"延迟 {lat:6.0f}ms  ({len(accs)} 句)")
            if args.lang:
                for wav, text in zip([c[0] for c in clips], texts):
                    print(f"      {wav.name}: {text}")

    print("\n================ 汇总 ================")
    print(f"{'语言':<6}{'模型':<34}{'字准率':>8}{'RTF':>8}{'延迟ms':>9}")
    for lang, model_id, acc, rtf, lat, n in results:
        print(f"{lang:<6}{model_id:<34}{acc:>7.1f}%{rtf:>8.2f}{lat:>9.0f}")

    missing = [
        f"{lang}:{m}" for lang, ids in CANDIDATES.items() if lang in langs
        for m in ids if m in MODELS and not installed(m)
    ]
    if missing:
        print("\n未下载（未参与对比）：" + "、".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
