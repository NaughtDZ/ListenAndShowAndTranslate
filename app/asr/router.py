"""语言路由器：把「音频 + 语言设置」变成「用哪个引擎识别」。

这是多语言需求的核心。三种情况：

1. **用户指定了语言**（如 ``ja``）→ 直接用配置里该语言的路由，跳过语种识别，最快最准
2. **``language="auto"``** → 先缓冲一小段音频跑 LID，判定后再建引擎，
   并把已缓冲的音频补喂给它（不能白丢）
3. **目标语言模型缺失/加载失败** → 按降级链依次尝试，并向上层报告实际用了哪个

另外统一在出口做文本后处理（英文大小写、日文多余空格等，见 postprocess.py），
这样上层拿到的永远是"可上屏"的文本。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from app.asr.base import ASREngine, ASREvent, ASREventType, BaseEngine
from app.asr.lid import LanguageIdentifier, LanguageStabilizer
from app.asr.locate import ModelNotFound
from app.asr.postprocess import clean_text
from app.asr.sherpa_offline import SherpaOfflineEngine
from app.asr.sherpa_stream import SherpaStreamingEngine
from app.config import ASRConfig, LanguageRoute
from app.models.registry import MODELS, models_for_language, supports_language
from app.utils.log import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000
# auto 模式下，至少要攒这么多音频才敢做语种判断（LID 对短音频不稳）
DEFAULT_LID_MIN_AUDIO_S = 2.0

LogCb = Callable[[str], None]


class LanguageRouter:
    """按语言选择并持有识别引擎。"""

    def __init__(
        self,
        config: ASRConfig,
        models_dir: Path | None = None,
        num_threads: int | None = None,
        provider: str | None = None,
        on_log: LogCb | None = None,
        lid_min_audio_s: float = DEFAULT_LID_MIN_AUDIO_S,
    ) -> None:
        self.config = config
        self.models_dir = models_dir
        self.num_threads = num_threads if num_threads is not None else self._auto_threads()
        self.provider = provider or (config.provider if config.provider != "auto" else "cpu")
        self.on_log = on_log
        self.lid_min_audio_s = lid_min_audio_s

        self._engines: dict[str, ASREngine] = {}
        self._routes: dict[str, LanguageRoute] = {}
        self._lid: LanguageIdentifier | None = None
        self._stabilizer = LanguageStabilizer()
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._decided = config.language != "auto"
        self._language = "" if config.language == "auto" else config.language
        self._closed = False

    # ------------------------------------------------------------------ #
    def _auto_threads(self) -> int:
        """默认用物理核数的一半，给游戏留资源。"""
        import os

        try:
            return max(1, (os.cpu_count() or 4) // 4)
        except Exception:  # noqa: BLE001
            return 2

    @property
    def language(self) -> str:
        """当前实际使用的语言；``auto`` 且尚未判定时返回空串。"""
        return self._language

    @property
    def is_decided(self) -> bool:
        return self._decided

    @property
    def engine_name(self) -> str:
        eng = self._engines.get(self._language)
        return getattr(eng, "name", "") if eng else ""

    def _log(self, msg: str) -> None:
        log.info(msg)
        if self.on_log:
            try:
                self.on_log(msg)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # 引擎构建与降级
    # ------------------------------------------------------------------ #
    def _active_route(self, language: str) -> LanguageRoute:
        """按语言取路由，并做一道"模型真的能用吗"的兜底。

        用户可以在设置里选识别模型，也可以直接手改 ``data/config.json``；
        所以这里必须自己判断：模型不存在、不是识别模型（比如把语种识别模型
        或 VAD 填进来了）、或者**不支持这门语言**时，一律退回默认路由并说明原因，
        而不是硬拿它去加载（用户最怕的就是"乱用模型"却看不出为什么不对）。

        结果按语言缓存：同一次会话里同一门语言只提示一次，不刷屏
        （配置改了会重建 LanguageRouter，见 ``pipeline.reload()``）。
        """
        cached = self._routes.get(language)
        if cached is not None:
            return cached

        route = self.config.route_for(language)
        if route.model and not supports_language(route.model, language):
            default = self.config.default_route_for(language)
            self._log(
                f"⚠️ {language} 配置的模型 {route.model} 不能用（不存在 / 不是识别模型 / "
                f"不支持该语言），已改回默认：{default.model}"
            )
            route = default
        self._routes[language] = route
        return route

    def _build_engine(self, language: str) -> ASREngine:
        route = self._active_route(language)
        vad = self.config.vad

        if route.engine == "sherpa_stream":
            return SherpaStreamingEngine(
                model_id=route.model or "zipformer-zh-int8",
                language=language,
                num_threads=self.num_threads,
                provider=self.provider,
                models_dir=self.models_dir,
                min_silence_ms=vad.min_silence_ms,
                max_segment_ms=vad.max_segment_ms,
            )
        # sherpa_offline 与 whispercpp 都走分块离线识别器
        return SherpaOfflineEngine(
            model_id=route.model or "dolphin-base-ctc-int8",
            language=language,
            num_threads=self.num_threads,
            provider=self.provider,
            models_dir=self.models_dir,
            min_silence_ms=vad.min_silence_ms,
            min_speech_ms=vad.min_speech_ms,
            max_segment_ms=vad.max_segment_ms,
        )

    def _fallback_chain(self, language: str) -> list[str]:
        """降级链：主模型 → 注册表里其它**支持这门语言**的模型 → Whisper turbo 兜底。

        以前这里写死"sensevoice 或 whisper 二选一"，换个模型就得改代码；
        现在直接从注册表推（``models_for_language``），加新模型不用动这里，
        而且每个语言的次选都是"真的支持它"的模型——不会拿中文模型去兜日语。
        """
        chain: list[str] = []
        primary = self._active_route(language).model
        if primary:
            chain.append(primary)
        chain.extend(m.id for m in models_for_language(language))
        # 最后一道：用户要求保留的 Whisper turbo（99 语言）
        chain.append("whisper-turbo-int8")
        return [m for m in dict.fromkeys(chain) if m and m in MODELS]

    def _create_engine_with_fallback(self, language: str) -> ASREngine | None:
        errors: list[str] = []
        route = self._active_route(language)  # 只取一次，避免重复提示
        for model_id in self._fallback_chain(language):
            engine_kind = route.engine
            if model_id != route.model:
                # 降级时按模型所属引擎类型重建
                from app.models.registry import MODELS

                spec = MODELS.get(model_id)
                if spec is None:
                    continue
                engine_kind = spec.engine

            try:
                eng = self._build_engine_for(model_id, engine_kind, language)
                eng.start()
                if model_id != route.model:
                    self._log(f"⚠️ {language} 主模型不可用，已降级到 {model_id}")
                else:
                    self._log(f"✓ 引擎就绪：{language} → {model_id}（{'流式' if eng.is_streaming else '分块'}）")
                return eng
            except (ModelNotFound, FileNotFoundError) as exc:
                errors.append(f"{model_id}: 缺文件")
                log.warning("模型 %s 不可用: %s", model_id, exc)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_id}: {exc}")
                log.warning("加载模型 %s 失败: %s", model_id, exc)

        self._log(f"✗ 无法为 {language} 加载任何模型：{'；'.join(errors)}")
        return None

    def _build_engine_for(self, model_id: str, engine_kind: str, language: str) -> ASREngine:
        vad = self.config.vad
        if engine_kind == "sherpa_stream":
            return SherpaStreamingEngine(
                model_id=model_id, language=language,
                num_threads=self.num_threads, provider=self.provider,
                models_dir=self.models_dir,
                min_silence_ms=vad.min_silence_ms, max_segment_ms=vad.max_segment_ms,
            )
        return SherpaOfflineEngine(
            model_id=model_id, language=language,
            num_threads=self.num_threads, provider=self.provider,
            models_dir=self.models_dir,
            min_silence_ms=vad.min_silence_ms, min_speech_ms=vad.min_speech_ms,
            max_segment_ms=vad.max_segment_ms,
        )

    def _ensure_engine(self, language: str) -> ASREngine | None:
        eng = self._engines.get(language)
        if eng is not None:
            return eng
        eng = self._create_engine_with_fallback(language)
        if eng is not None:
            self._engines[language] = eng
        return eng

    # ------------------------------------------------------------------ #
    # 语种识别
    # ------------------------------------------------------------------ #
    def _ensure_lid(self) -> LanguageIdentifier | None:
        if self._lid is not None:
            return self._lid
        lid = LanguageIdentifier(models_dir=self.models_dir)
        try:
            lid.load()
            self._lid = lid
            return lid
        except ModelNotFound as exc:
            self._log(
                "⚠️ 语种识别模型未安装，无法自动判断语言。"
                "请在设置里指定语言，或运行: python main.py --models install --packs lid"
            )
            log.warning("LID 模型缺失: %s", exc)
            return None

    def _decide_language(self) -> str:
        """用缓冲的音频判定语种；失败时用兜底路由的语言。"""
        allowed = {lang for lang in self.config.routing if lang != "*"}
        allowed.discard("zh-en")

        lid = self._ensure_lid()
        if lid is None:
            return "zh"  # 没装 LID 时的现实兜底：中文最常见

        audio = np.concatenate(self._pending) if self._pending else np.zeros(0, dtype=np.float32)
        detected = lid.identify(audio)
        stable = self._stabilizer.push(detected, allowed=allowed)
        if stable:
            self._log(f"识别到语种：{detected} → 使用 {stable}")
            return stable
        # 只判到一次，先用它；稳定器后续会纠正
        return detected if detected in allowed else "zh"

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def feed(self, samples: np.ndarray) -> list[ASREvent]:
        """喂入 16k 单声道音频。"""
        if self._closed or samples.size == 0:
            return []

        # --- auto：先攒够音频再决定语言 --- #
        if not self._decided:
            self._pending.append(samples)
            self._pending_samples += samples.size
            if self._pending_samples < int(self.lid_min_audio_s * SAMPLE_RATE):
                return []

            language = self._decide_language()
            self._language = language
            eng = self._ensure_engine(language)
            if eng is None:
                self._pending.clear()
                self._pending_samples = 0
                return [ASREvent(
                    type=ASREventType.ERROR, text="", segment_id=0,
                    error=f"无法为语言 {language} 加载模型",
                )]
            # 把攒下的音频补给引擎，别浪费
            buffered = np.concatenate(self._pending)
            self._pending.clear()
            self._pending_samples = 0
            events = eng.feed(buffered)
            self._decided = True
            return self._postprocess(events, language)

        eng = self._engines.get(self._language)
        if eng is None:
            eng = self._ensure_engine(self._language)
            if eng is None:
                return []
        return self._postprocess(eng.feed(samples), self._language)

    def finish(self) -> list[ASREvent]:
        """音频结束，冲出残留结果。"""
        if self._closed:
            return []
        # auto 且一直没攒够音频：用兜底语言强行识别
        if not self._decided and self._pending:
            language = self._decide_language()
            self._language = language
            eng = self._ensure_engine(language)
            self._pending.clear()
            self._pending_samples = 0
            self._decided = True
            if eng is None:
                return []
            return self._postprocess(eng.finish(), language)

        eng = self._engines.get(self._language)
        if eng is None:
            return []
        return self._postprocess(eng.finish(), self._language)

    def _postprocess(self, events: list[ASREvent], language: str) -> list[ASREvent]:
        """统一在出口清理文本，让上层拿到的永远是可上屏的文本。"""
        for ev in events:
            if ev.text:
                ev.text = clean_text(ev.text, language)
        return [ev for ev in events if ev.text or ev.type is ASREventType.ERROR]

    def close(self) -> None:
        for eng in self._engines.values():
            try:
                eng.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("关闭引擎失败: %s", exc)
        self._engines.clear()
        if self._lid is not None:
            self._lid.close()
            self._lid = None
        self._closed = True

    # ------------------------------------------------------------------ #
    def stats_summary(self) -> dict:
        """汇总各引擎的统计（供基准脚本/UI 显示）。"""
        out = {
            "language": self._language,
            "decided": self._decided,
            "engines": {},
        }
        for lang, eng in self._engines.items():
            st = getattr(eng, "stats", None)
            if st is None:
                continue
            out["engines"][lang] = {
                "engine": getattr(eng, "name", ""),
                "model": getattr(eng, "model_id", ""),
                "streaming": getattr(eng, "is_streaming", False),
                "segments": st.segments,
                "partials": st.partials,
                "audio_s": round(st.audio_seconds, 2),
                "rtf": round(st.rtf, 4),
                "latency_p50_ms": round(st.percentile(50), 1),
                "latency_p90_ms": round(st.percentile(90), 1),
            }
        return out
