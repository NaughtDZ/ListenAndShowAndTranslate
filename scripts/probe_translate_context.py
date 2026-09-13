"""实测「带入前文」到底往请求里放了什么。

方法：起一个**假的 OpenAI 兼容服务端**，把每一次 `/v1/chat/completions` 的
请求体原样录下来，然后逐条核对：

  1. 第 1 条请求里前文应为空；
  2. 第 k 条请求的前文 = 前 k-1 条 (原文, 译文)，**顺序正确**；
  3. 前文**不能包含正在翻译的这一条**（自我包含会让模型把上一条译文复读一遍）；
  4. 前文条数严格受「设置 → 翻译 → 带入前文」限制（例如设 3，就从不超过 3 对）；
  5. 设为 0 时永远不带前文；
  6. 编号/分行解析失败走"逐条重试"时，前文也要在（不能只在正常路径上有）。

用法::

    .venv\\Scripts\\python.exe scripts\\probe_translate_context.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import LLMConfig, TranslateConfig  # noqa: E402
from app.translate.base import Segment  # noqa: E402
from app.translate.hub import TranslatorHub  # noqa: E402
from app.translate.openai_compat import OpenAICompatTranslator  # noqa: E402

CONTEXT_HEADER = "【前文"
CONTEXT_END = "【待翻译"


class MockEndpoint:
    """假端点：记下每个请求，回一个像模像样的编号式译文。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                mock.requests.append(body)

                user = next(
                    (m["content"] for m in body.get("messages", []) if m.get("role") == "user"),
                    "",
                )
                count = int(
                    (re.search(r"【待翻译（共 (\d+) 条）】", user) or [None, "1"])[1]
                )
                if count <= 1:
                    text = "译文一号"
                else:
                    text = "\n".join(f"{i}. 译文{i}号" for i in range(1, count + 1))

                payload = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": body.get("model", "mock"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - /v1/models 探活用
                data = json.dumps({"data": [{"id": "mock-translator"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self.httpd.shutdown()

    # ------------------------------------------------------------------ #
    def user_contents(self) -> list[str]:
        out = []
        for body in self.requests:
            out.append(
                next(
                    (m["content"] for m in body.get("messages", []) if m.get("role") == "user"),
                    "",
                )
            )
        return out

    def context_pairs(self, user: str) -> list[tuple[str, str]]:
        """从用户消息里把【前文】块解析成 (原文, 译文) 列表。"""
        start = user.find(CONTEXT_HEADER)
        end = user.find(CONTEXT_END)
        if start < 0 or end < 0 or end <= start:
            return []
        block = user[start:end]
        pairs: list[tuple[str, str]] = []
        src = dst = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("原文："):
                if src is not None and dst is not None:
                    pairs.append((src, dst))
                src, dst = line[len("原文："):], None
            elif line.startswith("译文："):
                dst = line[len("译文："):]
        if src is not None and dst is not None:
            pairs.append((src, dst))
        return pairs

    def pending(self, user: str) -> list[str]:
        """【待翻译】块里的行。"""
        pos = user.find(CONTEXT_END)
        if pos < 0:
            return []
        tail = user[pos:]
        return [ln.strip() for ln in tail.splitlines()[1:] if ln.strip()]


def build_hub(mock: MockEndpoint, context_lines: int) -> TranslatorHub:
    cfg = TranslateConfig(
        enabled=True,
        provider="llm",
        context_lines=context_lines,
        llm=LLMConfig(
            enabled=True,
            base_url=mock.base_url,
            api_key="",
            model="mock-translator",
            prompt_style="chat",
            disable_thinking=True,
        ),
    )
    hub = TranslatorHub(cfg)
    hub.register(
        OpenAICompatTranslator(
            base_url=mock.base_url,
            model="mock-translator",
            prompt_style="chat",
            disable_thinking=True,
        ),
        priority=10,
    )
    return hub


def run_case(context_lines: int, lines: list[str]) -> tuple[int, int]:
    """逐条喂进去，返回 (通过项, 总项)。"""
    mock = MockEndpoint()
    hub = build_hub(mock, context_lines)
    try:
        for i, text in enumerate(lines, start=1):
            res = hub.translate([Segment(id=i, text=text, language="ja")], "ja")
            assert res.translations, f"第 {i} 条没翻出来：{res.failures}"

        users = mock.user_contents()
        print(f"\n{'=' * 74}")
        print(f"带入前文 = {context_lines}，共翻译 {len(lines)} 条，实际发出 {len(users)} 个请求")
        print("=" * 74)
        for idx, user in enumerate(users, start=1):
            pairs = mock.context_pairs(user)
            pend = mock.pending(user)
            print(f"\n-- 请求 #{idx}：" + "  ".join(f"待翻译={p}" for p in pend))
            if pairs:
                for src, dst in pairs:
                    print(f"     前文  原文：{src}  →  译文：{dst}")
            else:
                print("     前文  （空）")

        checks: list[tuple[bool, str]] = []
        # 1) 前文条数上限
        for idx, user in enumerate(users, start=1):
            pairs = mock.context_pairs(user)
            checks.append(
                (len(pairs) <= context_lines,
                 f"请求 #{idx} 前文条数 {len(pairs)} ≤ {context_lines}")
            )
        # 2) 前文必须严格是"之前那些行"、且顺序一致
        done: list[str] = []
        for idx, user in enumerate(users, start=1):
            pairs = mock.context_pairs(user)
            expect = [p for p in done][-context_lines:] if context_lines else []
            got_src = [s for s, _ in pairs]
            checks.append((got_src == expect,
                           f"请求 #{idx} 前文原文序列正确（期望 {expect} / 实际 {got_src}）"))
            done.extend(mock.pending(user))
        # 3) 不自包含
        for idx, user in enumerate(users, start=1):
            pairs = mock.context_pairs(user)
            pend = mock.pending(user)
            checks.append((not (set(pend) & {s for s, _ in pairs}),
                           f"请求 #{idx} 前文不含正在翻译的行"))
        # 4) 只有第一条允许没有前文（顺序翻译时）
        if context_lines:
            checks.append((not mock.context_pairs(users[0]) if users else False,
                           "第 1 条请求不带前文（还没有历史）"))

        bad = [msg for ok, msg in checks if not ok]
        for ok, msg in checks:
            print(f"   {'✅' if ok else '❌'} {msg}")
        print(f"   小计：{len(checks) - len(bad)}/{len(checks)} 通过")
        return len(checks) - len(bad), len(checks)
    finally:
        mock.stop()
        hub.close()


def main() -> int:
    ja = ["おおよし、かわい。", "可愛くするためには。", "もっと近くに来て。",
          "その手を離さないで。", "ずっと一緒にいよう。", "明日も会える？"]
    total_ok = total = 0
    for ctx in (0, 1, 3, 20):
        ok, n = run_case(ctx, ja)
        total_ok += ok
        total += n

    # 带术语表 + 故意让"编号解析"失败一次，验证重试路径也带前文
    print(f"\n{'=' * 74}\n汇总：{total_ok}/{total} 项通过\n{'=' * 74}")
    return 0 if total_ok == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
