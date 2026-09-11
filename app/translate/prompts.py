"""提示词模板：内置多套，全部可被用户覆盖。

模板变量（用 ``str.format`` 风格）：
    {source_language}  源语言（中文名，如「日语」）
    {target_language}  目标语言（如「简体中文」）
    {glossary_block}   术语表段落（可能为空）
    {context_block}    前文段落（可能为空）
    {source_block}     待翻译文本段落
    {count}            本次待翻译条数

**所有内置模板都要求"忠实原文、不删减、不做内容审查"**——
这是产品红线：翻译层只做语言转换，不对内容做取舍或软化。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.translate.base import TranslateRequest

LANGUAGE_NAMES = {
    "zh": "简体中文", "zh-en": "简体中文", "en": "英语", "ja": "日语",
    "ko": "韩语", "yue": "粤语", "fr": "法语", "de": "德语", "es": "西班牙语",
    "ru": "俄语", "auto": "原文语言",
}

# 所有模板共用的硬要求。单独抽出来是因为这几条是红线，不该被模板作者漏掉。
_COMMON_RULES = """硬性要求：
1. **忠实原文**：不增删、不概括、不弱化、不做内容审查或道德说教；原文有的就翻，没有的不加
2. **保持人名与称谓一致**：与「前文」中的译法保持一致
3. **术语表里的词必须使用指定译法**
4. **只输出译文本身**：不要解释、不要加引号、不要复述原文、不要加任何前后缀
5. 标点使用全角中文标点"""


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    name: str
    description: str
    system: str
    user: str


_TEMPLATES: dict[str, PromptTemplate] = {
    "subtitle_direct": PromptTemplate(
        id="subtitle_direct",
        name="字幕直译（默认）",
        description="逐句忠实翻译，语序尽量贴近原文，适合听小说/看剧情",
        system="你是一名专业的字幕翻译。" + _COMMON_RULES,
        user="""请把下面的{source_language}文本翻译成{target_language}。

{glossary_block}{context_block}【待翻译（共 {count} 条）】
{source_block}""",
    ),
    "subtitle_natural": PromptTemplate(
        id="subtitle_natural",
        name="字幕口语化",
        description="在忠实的前提下改用中文自然口语，读起来更像中文台词",
        system=(
            "你是一名专业的影视字幕翻译，擅长把外语台词写成自然的中文口语。"
            + _COMMON_RULES
            + "\n6. 在忠实原文的前提下，可以调整语序、补出中文习惯的主语，让句子像中文台词"
        ),
        user="""请把下面的{source_language}文本翻译成{target_language}。

{glossary_block}{context_block}【待翻译（共 {count} 条）】
{source_block}""",
    ),
    "novel_literary": PromptTemplate(
        id="novel_literary",
        name="小说文学风",
        description="偏书面语与文学表达，适合有声书/小说朗读",
        system=(
            "你是一名文学翻译，负责把小说内容译成中文。"
            + _COMMON_RULES
            + "\n6. 用词偏书面与文学化，避免过于现代的网络口语；"
            "叙述部分保持流畅的书面语，对话部分保留人物口吻"
        ),
        user="""请把下面的{source_language}文本翻译成{target_language}。

{glossary_block}{context_block}【待翻译（共 {count} 条）】
{source_block}""",
    ),
}

DEFAULT_TEMPLATE = "subtitle_direct"


def list_templates() -> list[PromptTemplate]:
    return list(_TEMPLATES.values())


def get_template(template_id: str) -> PromptTemplate:
    return _TEMPLATES.get(template_id) or _TEMPLATES[DEFAULT_TEMPLATE]


def register_template(tpl: PromptTemplate) -> None:
    """注册/覆盖一套模板（用户自定义用）。"""
    _TEMPLATES[tpl.id] = tpl


def language_name(code: str) -> str:
    if not code:
        return "原文语言"
    return LANGUAGE_NAMES.get(code, code)


# --------------------------------------------------------------------------- #
# 段落构造
# --------------------------------------------------------------------------- #
def build_glossary_block(glossary: dict[str, str]) -> str:
    if not glossary:
        return ""
    lines = "\n".join(f"  {src} → {dst}" for src, dst in glossary.items())
    return f"【术语表（必须严格遵守）】\n{lines}\n\n"


def build_context_block(context: list[tuple[str, str]]) -> str:
    if not context:
        return ""
    lines = "\n".join(f"  原文：{src}\n  译文：{dst}" for src, dst in context)
    return f"【前文（仅供理解上下文与保持译名一致，不要翻译它）】\n{lines}\n\n"


def build_source_block(
    request: TranslateRequest, numbered: bool | None = None
) -> tuple[str, bool]:
    """构造待翻译文本段。

    Returns:
        (文本, 是否为编号批量格式)

    多条时必须编号——否则模型无法保证输出与输入一一对应。
    """
    batch = request.is_batch if numbered is None else numbered
    if not batch:
        return request.segments[0].text, False
    lines = "\n".join(f"{i + 1}. {seg.text}" for i, seg in enumerate(request.segments))
    return lines, True


def build_messages(request: TranslateRequest) -> list[dict[str, str]]:
    """把请求渲染成 chat messages。"""
    tpl = get_template(request.template)
    system = (request.custom_prompt or "").strip() or tpl.system

    source_block, numbered = build_source_block(request)
    user = tpl.user.format(
        source_language=language_name(request.source_language),
        target_language=language_name(request.target_language),
        glossary_block=build_glossary_block(request.glossary),
        context_block=build_context_block(request.context),
        source_block=source_block,
        count=len(request.segments),
    )
    if numbered:
        user += "\n\n请按相同编号逐条输出译文，每行一条，格式为「编号. 译文」。不要合并或省略任何一条。"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
