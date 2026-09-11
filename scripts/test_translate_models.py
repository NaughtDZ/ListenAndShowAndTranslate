"""翻译模型实测：对比 LM Studio 里的模型在'日译中字幕'任务上的质量、速度与 token 消耗。

用**真实的 ASR 输出**做输入（含识别错误），比干净文本更能反映实际效果。

用法：
    .venv\\Scripts\\python.exe scripts\\test_translate_models.py
    .venv\\Scripts\\python.exe scripts\\test_translate_models.py --models qwen3.8-2b-uncensored
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:1234/v1"

# 真实 ASR 输出（SenseVoice 日文结果，含"声堂"（应为青銅）这类识别错误）
SOURCE = (
    "列車?リンファンは手の中の声堂の鍵を強く握りしめ。"
    "今度こそ。誰も失望させたりはしないと小さくつぶやいた。"
)

GLOSSARY = {
    "リンファン": "林凡",
    "声堂": "青铜",
    "鍵": "钥匙",
}

# 前文（原文+已有译文），用于测"上下文是否被正确利用"
CONTEXT = [
    ("第一章、夜の列車。", "第一章，夜行的列车。"),
    ("リンファンは窓の外を見ていた。", "林凡望着窗外。"),
]

TARGET = "简体中文"
SRC_LANG = "日语"

SYSTEM = f"""你是专业的字幕翻译。把{SRC_LANG}文本翻译成{TARGET}。

要求：
1. 忠实原文：不增删、不概括、不弱化、不做内容审查
2. 与「前文」保持人称、称呼与语气一致
3. 术语表里的词必须用指定译法
4. 只输出译文本身，不要解释、不要加引号、不要重复原文"""


def build_user(glossary: dict, context: list[tuple[str, str]], source: str) -> str:
    parts = []
    if glossary:
        g = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        parts.append(f"【术语表】\n{g}")
    if context:
        c = "\n".join(f"  原文：{s}\n  译文：{t}" for s, t in context)
        parts.append(f"【前文（仅供理解上下文，不要翻译）】\n{c}")
    parts.append(f"【待翻译】\n{source}")
    return "\n\n".join(parts)


def list_models() -> list[str]:
    with httpx.Client(trust_env=False, timeout=15) as c:
        r = c.get(f"{BASE}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


def translate(
    model: str,
    system: str,
    user: str,
    timeout: float = 600.0,
    disable_thinking: bool = True,
    max_tokens: int = 1024,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        # Qwen3 系推理模型：不关思考的话，512 预算会全被推理 token 吃掉，
        # content 返回空字符串（实测 qwen3.8-9b / 27b 都是这样）
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.monotonic()
    with httpx.Client(trust_env=False, timeout=timeout) as c:
        r = c.post(f"{BASE}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    dt = time.monotonic() - t0
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    return {
        "text": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "seconds": dt,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="逗号分隔；默认自动挑几个")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--no-warmup", action="store_true", help="跳过预热（不推荐）")
    args = ap.parse_args()

    available = list_models()
    print("=== LM Studio 里的模型 ===")
    for m in available:
        print(f"  {m}")

    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        prefer = [
            "qwen3.8-2b-uncensored",
            "qwen3.8-9b-heretic-uncensored",
            "sakura-galtransl-7b-v3.7",
            "qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp-nvfp4",
        ]
        wanted = [m for m in prefer if m in available]
        if not wanted:
            wanted = [m for m in available if "embed" not in m and "rerank" not in m][:3]

    user = build_user(GLOSSARY, CONTEXT, SOURCE)
    print("\n=== 输入 ===")
    print(f"  原文: {SOURCE}")

    results = []
    for model in wanted:
        if model not in available:
            print(f"\n--- {model} ---\n  ✗ 模型不在列表里，跳过")
            continue
        print(f"\n--- {model} ---")

        if not args.no_warmup:
            # LM Studio 是按需加载：第一次调用包含加载时间，必须预热后才能量速度
            print("  预热（含模型加载）…", flush=True)
            try:
                w = translate(model, SYSTEM, user, timeout=args.timeout)
                print(f"    预热耗时 {w['seconds']:.2f}s")
            except Exception as exc:  # noqa: BLE001
                print(f"   预热失败: {exc}")
                continue

        t0 = time.monotonic()
        try:
            r = translate(model, SYSTEM, user, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 失败: {exc}")
            continue
        extra = time.monotonic() - t0 - r["seconds"]

        results.append((model, r))
        if not r["text"]:
            print(f"  ✗ 译文为空（completion={r['completion_tokens']} tok）"
                  f"{'，思考内容长度=' + str(len(r['reasoning'])) if r['reasoning'] else ''}")
        else:
            print(f"  译文: {r['text']}")
        # 术语表遵守检查
        checks = []
        if "林凡" in r["text"]:
            checks.append("术语✓林凡")
        if "青铜" in r["text"]:
            checks.append("术语✓青铜")
        if checks:
            print(f"  {', '.join(checks)}")
        print(f"  实测推理 {r['seconds']:.2f}s | prompt {r['prompt_tokens']} tok | "
              f"输出 {r['completion_tokens']} tok | 共 {r['total_tokens']} tok")

    if results:
        print("\n" + "=" * 92)
        print(f"{'模型':<50}{'推理耗时':>10}{'prompt':>8}{'输出':>7}{'合计':>8}")
        print("-" * 92)
        for model, r in results:
            print(f"{model:<50}{r['seconds']:>9.2f}s{r['prompt_tokens']:>8}"
                  f"{r['completion_tokens']:>7}{r['total_tokens']:>8}")

        ok = [(m, r) for m, r in results if r["text"]]
        if ok:
            fastest = min(ok, key=lambda x: x[1]["seconds"])
            print(f"\n最快且成功: {fastest[0]}（{fastest[1]['seconds']:.2f}s）")
        mx = max(r["total_tokens"] for _, r in results)
        print(f"单次请求最大 token 用量: {mx}（16000 上下文下占 {mx / 16000 * 100:.1f}%）")
        print(f"按 16k 窗口算，最多可容纳约 {16000 // max(1, mx)} 个这样的并发请求")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
