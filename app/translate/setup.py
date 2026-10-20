"""按配置构建 ``TranslatorHub``。

**实时管线与离线字幕工具共用这一份**：通道选择、凭据、代理、思考开关、
网页版内部接口的优先级…… 两边各写一套，必然漂移（改了 LLM 参数忘了改另一边）。

返回值里的"说明列表"是给 UI/日志看的（``LLM(qwen): 可用`` 这种），
不是错误——通道没通不代表整个功能不可用（hub 内部还会降级）。
"""

from __future__ import annotations

from app.config import AppConfig
from app.translate.glossary import Glossary
from app.translate.hub import TranslatorHub
from app.utils.log import get_logger

log = get_logger(__name__)


def build_translator_hub(
    config: AppConfig,
    glossary: Glossary | None = None,
) -> tuple[TranslatorHub, list[str]]:
    """按 ``config.translate`` 构建 hub 并注册可用通道。

    返回 ``(hub, notes)``；**不抛异常**——构造不出通道只体现在 notes 里，
    真正要用的时候 hub 会给出"没有可用的翻译通道"的具体原因。
    """
    cfg = config
    hub = TranslatorHub(cfg.translate, glossary=glossary or Glossary())
    notes: list[str] = []
    provider = cfg.translate.provider

    if provider == "llm" or provider.startswith("llm"):
        from app.translate.openai_compat import OpenAICompatTranslator

        llm = cfg.translate.llm
        if not llm.enabled:
            notes.append("LLM 通道未启用")
        else:
            try:
                translator = OpenAICompatTranslator(
                    base_url=llm.base_url, api_key=llm.api_key, model=llm.model,
                    temperature=llm.temperature, max_tokens=llm.max_tokens,
                    timeout_s=llm.timeout_s, proxy=cfg.proxy,
                    verify_glossary=True, name="llm",
                    prompt_style=getattr(llm, "prompt_style", "") or "",
                    # 思考默认关（字幕翻译不需要思考，且会把预算烧光导致空译文）
                    disable_thinking=getattr(llm, "disable_thinking", True),
                )
            except Exception as exc:  # noqa: BLE001 - 配错凭据不该让整个流程起不来
                notes.append(f"LLM({llm.model}): 构造失败 {exc}")
            else:
                hub.register(translator, priority=10)
                try:
                    _ok, msg = translator.ping()
                except Exception as exc:  # noqa: BLE001
                    msg = f"探测失败 {exc}"
                notes.append(f"LLM({llm.model}): {msg}")

    elif provider != "none":
        from app.translate.traditional.providers import ALL_PROVIDERS, build_provider

        if provider not in ALL_PROVIDERS:
            notes.append(f"未知通道: {provider}")
        else:
            # web_* 是免 key 的网页版内部接口（非官方，可能失效）
            creds = cfg.translate.providers.get(provider, {})
            try:
                translator = build_provider(
                    provider, creds, proxy=cfg.proxy, qps_limit=cfg.translate.qps_limit
                )
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{provider}: 构造失败 {exc}")
            else:
                if translator is None:
                    notes.append(f"无法构造通道: {provider}")
                else:
                    hub.register(translator, priority=20 if provider.startswith("web_") else 10)
                    try:
                        _ok, msg = translator.ping()
                    except Exception as exc:  # noqa: BLE001
                        msg = f"探测失败 {exc}"
                    tag = "（非官方接口，可能失效）" if provider.startswith("web_") else ""
                    notes.append(f"{provider}{tag}: {msg}")

    log.info("翻译通道：%s", "；".join(notes) if notes else "未配置")
    return hub, notes
