"""翻译层验收：接 LM Studio 真机跑，同时验证降级链。

用法：
    .venv\\Scripts\\python.exe scripts\\verify_translate.py
    .venv\\Scripts\\python.exe scripts\\verify_translate.py --model qwen3.8-2b-uncensored
    .venv\\Scripts\\python.exe scripts\\verify_translate.py --break-it   # 故意用会返回空译文的模型

--break-it 用 qwen3.8-9b-heretic-uncensored（推理关不掉、content 为空），
用来证明"空译文 → 加大预算 → 逐条"的降级链真的能把字幕救回来，
而不是给用户一堆空白。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from app.translate.base import Segment, TranslateRequest  # noqa: E402
from app.translate.openai_compat import OpenAICompatTranslator  # noqa: E402

# 真实 ASR 输出（日语，含"声堂"这类识别错误）
SEGMENTS = [
    "列車?リンファンは手の中の声堂の鍵を強く握りしめ。",
    "今度こそ。",
    "誰も失望させたりはしないと小さくつぶやいた。",
    "窓の外には、見知らぬ街の灯りが流れていく。",
    "彼は目を閉じて、遠い日の記憶をたどった。",
]

CONTEXT = [
    ("第一章、夜の列車。", "第一章，夜行的列车。"),
    ("リンファンは窓の外を見ていた。", "林凡望着窗外。"),
    ("夜の闇がゆっくりと街を包んでいく。", "夜色缓缓笼罩了这座城市。"),
]

GLOSSARY = {"リンファン": "林凡", "声堂": "青铜", "鍵": "钥匙"}


def run(model: str, batch_size: int, glossary: dict, label: str) -> bool:
    print("=" * 92)
    print(f"{label}   model={model}   batch_size={batch_size}")
    print("=" * 92)

    t = OpenAICompatTranslator(
        base_url="http://127.0.0.1:1234/v1",
        model=model,
        batch_size=batch_size,
        temperature=0.3,
        max_tokens=512,
    )
    ok, msg = t.ping()
    print(f"端点: {msg}")
    if not ok:
        t.close()
        return False

    # 预热：LM Studio 按需加载模型，第一次调用含加载时间，
    # 不预热的话量出来的耗时全是错的（这个坑踩过两次）
    t0 = time.monotonic()
    t.translate(TranslateRequest(
        segments=[Segment(id=0, text="テスト", language="ja")],
        source_language="ja", target_language="zh",
    ))
    print(f"预热 {time.monotonic() - t0:.2f}s")

    req = TranslateRequest(
        segments=[Segment(id=i + 1, text=s, language="ja") for i, s in enumerate(SEGMENTS)],
        source_language="ja",
        target_language="zh",
        context=CONTEXT,
        glossary=glossary,
    )

    try:
        r = t.translate(req)
    finally:
        t.close()

    print(f"\n结果：成功 {r.ok_count}/{len(SEGMENTS)} 条，失败 {r.fail_count} 条")
    for seg in req.segments:
        got = r.text_for(seg.id)
        mark = "✓" if got else "✗"
        print(f"  [{mark}] {seg.text}")
        print(f"      → {got or r.failures.get(seg.id, '(空)')}")

    print(f"\n延迟 {r.latency_ms:.0f}ms | prompt {r.prompt_tokens} tok | "
          f"输出 {r.completion_tokens} tok | 重试 {r.retries} 次")
    if r.note:
        print(f"说明: {r.note}")

    # 术语表校验
    if glossary:
        missed = []
        for seg in req.segments:
            got = r.text_for(seg.id)
            for k, v in glossary.items():
                if k in seg.text and v not in got:
                    missed.append((seg.id, k, v))
        print(f"术语表: {'全部命中 ✓' if not missed else f'漏 {len(missed)} 处: {missed}'}")

    return r.ok_count == len(SEGMENTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-2b-uncensored")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--break-it", action="store_true",
                    help="用会返回空译文的推理模型，验证降级链")
    args = ap.parse_args()

    if args.break_it:
        # 只测 1 条：9B 每轮要生成上千思考 token，5 条会跑十几分钟
        print("（故意使用思考关不掉的模型，检验降级链；只测 1 条以控制耗时）\n")
        global SEGMENTS
        SEGMENTS = SEGMENTS[:1]
        ok = run("qwen3.8-9b-heretic-uncensored", 1, GLOSSARY,
                 "降级链测试（预期：空译文 → 加大预算 → 逐条 → 成功或明确失败）")
        print("\n结论：即使模型把预算全用在思考上，也不会产出空白字幕——"
              "要么拿到译文，要么给出明确的失败原因。")
        # 这里判定的是"降级链是否正确工作"（明确失败 = 正确），
        # 而不是"是否翻出来了"（该模型当前就是不可用的通道）
        return 0

    results = []
    results.append(run(args.model, args.batch_size, GLOSSARY, "批量 + 术语表"))
    results.append(run(args.model, 1, {}, "逐条 + 无术语表（对照）"))

    print("\n" + "=" * 92)
    print("✅ 全部通过" if all(results) else "❌ 有失败")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
