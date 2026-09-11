"""日志：滚动文件 + 控制台，并对 API Key 之类的敏感串打码。

红线要求（计划书第 9 节）：日志中不得出现明文凭据。
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys

from app import paths

_LOGGER_NAME = "lst"
_configured = False

# (模式, 替换文本)。替换文本中的 \1 \2 对应模式里的捕获组。
# 覆盖：OpenAI 风格 sk-xxx、GitHub ghp_xxx、Bearer 头、以及 key=value 形式的密钥。
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "<redacted>"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "<redacted>"),
    (re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._\-]{8,})"), r"\1<redacted>"),
    (
        re.compile(
            r"(?i)\b(api[-_]?key|app[-_]?id|app[-_]?key|app[-_]?secret|secret|token|password)"
            r"(\s*[=:]\s*)"
            # 值可能是 "双引号"、'单引号'，或裸串
            r"""(?:"[^"]{4,}"|'[^']{4,}'|[^\s,;]{4,})"""
        ),
        r"\1\2<redacted>",
    ),
)


def scrub(text: str) -> str:
    """把敏感串替换成掩码，供日志使用。"""
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - 极端情况下的格式化失败
            return True
        scrubbed = scrub(msg)
        if scrubbed != msg:
            record.msg = scrubbed
            record.args = ()
        return True


def get_logger(name: str | None = None) -> logging.Logger:
    """取得配置好的 logger。首次调用时完成初始化。"""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)

    if not _configured:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        scrub_filter = _ScrubFilter()

        try:
            paths.ensure_dirs()
            fh = logging.handlers.RotatingFileHandler(
                paths.LOGS_DIR / "lst.log",
                maxBytes=4 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(fmt)
            fh.addFilter(scrub_filter)
            logger.addHandler(fh)
        except OSError:
            # 日志目录不可写时不能因此崩溃
            pass

        # 控制台在打包的窗口模式下可能不存在
        if sys.stderr is not None:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            sh.addFilter(scrub_filter)
            logger.addHandler(sh)

        _configured = True

    return logger if name is None else logger.getChild(name)
