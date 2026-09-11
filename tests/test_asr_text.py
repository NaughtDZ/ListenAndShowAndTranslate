"""ASR 文本后处理、语种识别、路由决策的单元测试。

这些都不需要加载真模型，纯逻辑，任何机器可跑。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.asr.lid import LanguageStabilizer, normalize_language
from app.asr.postprocess import (
    append_ending,
    clean_text,
    is_cjk,
    needs_ending_punctuation,
    remove_cjk_spaces,
    restore_sentence_case,
)
from app.config import ASRConfig

# --------------------------------------------------------------------------- #
# 后处理：CJK 空格
# --------------------------------------------------------------------------- #
def test_remove_spaces_between_cjk():
    """实测：SenseVoice 日文会把 ``誰も`` 输出成 ``誰 も``。"""
    assert remove_cjk_spaces("誰 も 失望 させ たり") == "誰も失望させたり"


def test_keeps_space_between_cjk_and_latin():
    assert remove_cjk_spaces("使用 Windows 系统") == "使用 Windows 系统"


def test_remove_cjk_spaces_leaves_english_alone():
    assert remove_cjk_spaces("hello world foo") == "hello world foo"


def test_is_cjk():
    assert is_cjk("中")
    assert is_cjk("あ")
    assert is_cjk("ア")
    assert not is_cjk("A")
    assert not is_cjk("1")


# --------------------------------------------------------------------------- #
# 后处理：英文大小写
# --------------------------------------------------------------------------- #
def test_restore_sentence_case_from_all_caps():
    """实测：流式英文模型输出是全大写。"""
    assert restore_sentence_case("CHAPTER ONE. THE NIGHT TRAIN") == "Chapter one. The night train"


def test_restore_sentence_case_handles_i():
    got = restore_sentence_case("I THINK I KNOW")
    assert "I think I know" == got


def test_restore_sentence_case_does_not_touch_mixed_case():
    text = "Chapter One the night train"
    assert restore_sentence_case(text) == text


def test_restore_sentence_case_ignores_short_allcaps():
    """全大写但很短（很可能是缩写）不该动。"""
    assert restore_sentence_case("OK") == "OK"


# --------------------------------------------------------------------------- #
# 后处理：clean_text
# --------------------------------------------------------------------------- #
def test_clean_text_japanese():
    got = clean_text("誰 も 失望 させ たり はし ないと 小さく つぶや いた。", "ja")
    assert got == "誰も失望させたりはしないと小さくつぶやいた。"


def test_clean_text_english_all_caps():
    got = clean_text("CHAPTER ONE THE NIGHT TRAIN", "en")
    assert got == "Chapter one the night train"


def test_clean_text_chinese_keeps_punctuation():
    got = clean_text("林凡握紧了手里的青铜钥匙。", "zh")
    assert got == "林凡握紧了手里的青铜钥匙。"


def test_clean_text_collapses_spaces():
    assert clean_text("a    b", "en") == "a b"


def test_clean_text_empty():
    assert clean_text("", "zh") == ""
    assert clean_text("   ", "en") == ""


# --------------------------------------------------------------------------- #
# 后处理：标点
# --------------------------------------------------------------------------- #
def test_needs_ending_punctuation():
    assert needs_ending_punctuation("没有标点的话") is True
    assert needs_ending_punctuation("有句号。") is False
    assert needs_ending_punctuation("With period.") is False


def test_append_ending():
    assert append_ending("没有标点的话", "zh") == "没有标点的话。"
    assert append_ending("no period", "en") == "no period."
    assert append_ending("已有。", "zh") == "已有。"
    assert append_ending("", "zh") == ""


# --------------------------------------------------------------------------- #
# 语种标签归一
# --------------------------------------------------------------------------- #
def test_normalize_language():
    assert normalize_language("Chinese") == "zh"
    assert normalize_language("ENGLISH") == "en"
    assert normalize_language("japanese") == "ja"
    assert normalize_language("ko") == "ko"
    assert normalize_language("<|yue|>") == "yue"
    assert normalize_language("") == "auto"


# --------------------------------------------------------------------------- #
# 语种稳定器（没有它，LID 偶尔误判会让字幕来回跳）
# --------------------------------------------------------------------------- #
def test_stabilizer_requires_agreement():
    st = LanguageStabilizer(window=3, min_agree=2)
    assert st.push("zh") == ""       # 第一次还不够
    assert st.push("zh") == "zh"     # 第二次一致 → 确定


def test_stabilizer_ignores_single_outlier():
    st = LanguageStabilizer(window=3, min_agree=2)
    st.push("zh")
    st.push("zh")
    assert st.current == "zh"
    st.push("ja")                    # 偶发误判
    assert st.current == "zh"        # 不该被带跑


def test_stabilizer_switches_when_genuinely_changed():
    """多语言小说集中途换语言时应该能切过去。"""
    st = LanguageStabilizer(window=3, min_agree=2)
    st.push("zh")
    st.push("zh")
    assert st.current == "zh"
    st.push("ja")
    st.push("ja")
    assert st.current == "ja"


def test_stabilizer_ignores_uninstalled_language():
    st = LanguageStabilizer(window=2, min_agree=2)
    st.push("zh", allowed={"zh", "ja"})
    st.push("fr", allowed={"zh", "ja"})   # 没装法语模型
    assert st.current != "fr"


def test_stabilizer_ignores_auto():
    st = LanguageStabilizer()
    st.force("zh")
    assert st.push("auto") == "zh"


def test_stabilizer_reset():
    st = LanguageStabilizer()
    st.push("zh")
    st.push("zh")
    st.reset()
    assert st.current == ""
    assert st.push("ja") == ""


# --------------------------------------------------------------------------- #
# 路由决策（不加载模型，只测选路逻辑）
# --------------------------------------------------------------------------- #
def test_router_route_for_known_and_unknown_language():
    cfg = ASRConfig()
    assert cfg.route_for("ja").engine == "sherpa_offline"
    assert cfg.route_for("zh").engine == "sherpa_stream"
    # 未配置的语言走 * 兜底
    assert cfg.route_for("sw").engine == "whispercpp"


def test_router_fallback_chain_includes_whisper_as_last_resort():
    from app.asr.router import LanguageRouter

    router = LanguageRouter(ASRConfig(), models_dir=None, num_threads=1)
    for lang in ("zh", "ja", "sw"):
        chain = router._fallback_chain(lang)
        assert chain, lang
        assert "whisper-turbo-int8" in chain, f"{lang} 的降级链里必须有 Whisper 兜底"
        assert len(chain) == len(set(chain)), "降级链不应有重复"


def test_router_fallback_chain_starts_with_primary():
    from app.asr.router import LanguageRouter

    router = LanguageRouter(ASRConfig(), models_dir=None, num_threads=1)
    assert router._fallback_chain("ja")[0] == "sensevoice-int8"
    assert router._fallback_chain("zh")[0] == "zipformer-zh-int8"


def test_all_routing_models_exist_in_registry():
    """路由表里的 model 必须是注册表键名，写错会导致引擎找不到模型。

    （这个测试就是为抓一次真实 bug 而写的：默认路由曾写成
    "sense-voice-zh-en-ja-ko-yue-int8" 这种显示名，注册表里根本没有。）
    """
    from app.models.registry import MODELS

    cfg = ASRConfig()
    for lang, route in cfg.routing.items():
        if not route.model:
            continue
        assert route.model in MODELS, (
            f"路由 {lang} 引用了不存在的模型键名 {route.model!r}；"
            f"可用：{sorted(MODELS)}"
        )


def test_routing_engine_matches_registry_engine():
    """路由声明的引擎类型必须和注册表里该模型的引擎一致，否则会加载失败。"""
    from app.models.registry import MODELS

    cfg = ASRConfig()
    for lang, route in cfg.routing.items():
        if not route.model:
            continue
        spec = MODELS[route.model]
        assert spec.engine == route.engine, (
            f"路由 {lang}: 声明 {route.engine} 但注册表里 {route.model} 是 {spec.engine}"
        )


def test_router_starts_decided_when_language_explicit():
    from app.asr.router import LanguageRouter

    r1 = LanguageRouter(ASRConfig(language="ja"), models_dir=None, num_threads=1)
    assert r1.is_decided and r1.language == "ja"

    cfg = ASRConfig()
    cfg.language = "auto"
    r2 = LanguageRouter(cfg, models_dir=None, num_threads=1)
    assert not r2.is_decided and r2.language == ""


def test_router_buffers_until_enough_audio_for_lid():
    """auto 模式下音频不够时不该急着建引擎。"""
    from app.asr.router import LanguageRouter

    cfg = ASRConfig()
    cfg.language = "auto"
    router = LanguageRouter(cfg, models_dir=None, num_threads=1, lid_min_audio_s=2.0)

    # 只喂 0.5 秒
    events = router.feed(np.zeros(8000, dtype=np.float32))
    assert events == []
    assert not router.is_decided
    assert router._pending_samples == 8000
