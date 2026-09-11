"""翻译缓存（SQLite）。

为什么必须有：听小说时**重复内容极多**——章标题、口头禅、人物自称、
"他说道"这类句子会反复出现。实测一次字幕请求 300~400 tokens，
没有缓存的话同样的句子要反复付费/反复推理。

缓存键必须包含所有会影响译文的因素，否则会出现"换了模型却拿到旧译文"这种脏问题：
    原文 + 源语言 + 目标语言 + 通道 + 模型 + 模板 + 自定义提示词 + 命中的术语子集
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from app.paths import DB_FILE
from app.utils.log import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_cache (
    key         TEXT PRIMARY KEY,
    source_text TEXT NOT NULL,
    target_text TEXT NOT NULL,
    source_lang TEXT DEFAULT '',
    target_lang TEXT DEFAULT '',
    provider    TEXT DEFAULT '',
    model       TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    hits        INTEGER DEFAULT 0,
    last_hit_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tc_created ON translation_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_tc_provider ON translation_cache(provider);
"""


def cache_key(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "zh",
    provider: str = "",
    model: str = "",
    template: str = "",
    extra: str = "",
) -> str:
    """生成缓存键（sha1，足够短且碰撞概率可忽略）。"""
    payload = "\x1f".join([
        (text or "").strip(),
        source_lang or "auto",
        target_lang or "zh",
        provider or "",
        model or "",
        template or "",
        extra or "",
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class TranslationCache:
    """翻译结果缓存。线程安全（字幕线程与 UI 线程都会碰它）。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or DB_FILE)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def get(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "zh",
        provider: str = "",
        model: str = "",
        template: str = "",
        extra: str = "",
    ) -> str | None:
        key = cache_key(text, source_lang, target_lang, provider, model, template, extra)
        return self.get_by_key(key)

    def get_by_key(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT target_text FROM translation_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE translation_cache SET hits = hits + 1, last_hit_at = ? WHERE key = ?",
                (time.time(), key),
            )
            self._conn.commit()
            return row[0]

    def get_many(self, keys: list[str]) -> dict[str, str]:
        """批量取（一次 SQL，避免 N 次查询）。"""
        if not keys:
            return {}
        out: dict[str, str] = {}
        # SQLite 变量数上限 999，分块避免超限（这个坑在别的项目里踩过）
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT key, target_text FROM translation_cache WHERE key IN ({marks})",
                    chunk,
                ).fetchall()
                for k, v in rows:
                    out[k] = v
                if rows:
                    now = time.time()
                    self._conn.executemany(
                        "UPDATE translation_cache SET hits = hits + 1, last_hit_at = ? WHERE key = ?",
                        [(now, k) for k, _ in rows],
                    )
            if out:
                self._conn.commit()
        return out

    # ------------------------------------------------------------------ #
    def put(
        self,
        text: str,
        translation: str,
        source_lang: str = "auto",
        target_lang: str = "zh",
        provider: str = "",
        model: str = "",
        template: str = "",
        extra: str = "",
    ) -> None:
        if not text or not translation:
            return
        key = cache_key(text, source_lang, target_lang, provider, model, template, extra)
        self.put_many([(key, text, translation, source_lang, target_lang, provider, model)])

    def put_many(self, rows: list[tuple]) -> None:
        """rows: (key, source_text, target_text, src_lang, dst_lang, provider, model)"""
        if not rows:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                """INSERT INTO translation_cache
                   (key, source_text, target_text, source_lang, target_lang,
                    provider, model, created_at, hits)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(key) DO UPDATE SET
                     target_text = excluded.target_text,
                     provider = excluded.provider,
                     model = excluded.model""",
                [(*r, now) for r in rows],
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM translation_cache").fetchone()[0]
            hits = self._conn.execute("SELECT COALESCE(SUM(hits), 0) FROM translation_cache").fetchone()[0]
            saved = self._conn.execute(
                "SELECT COUNT(*) FROM translation_cache WHERE hits > 0"
            ).fetchone()[0]
        return {"entries": total, "total_hits": hits, "entries_with_hits": saved}

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM translation_cache").fetchone()[0]
            self._conn.execute("DELETE FROM translation_cache")
            self._conn.commit()
        log.info("已清空翻译缓存（%d 条）", n)
        return n

    def prune(self, keep_days: int = 90, max_entries: int = 200_000) -> int:
        """清理旧条目，控制数据库体积。"""
        removed = 0
        with self._lock:
            cutoff = time.time() - keep_days * 86400
            cur = self._conn.execute(
                "DELETE FROM translation_cache WHERE created_at < ? AND hits = 0", (cutoff,)
            )
            removed += cur.rowcount or 0
            total = self._conn.execute("SELECT COUNT(*) FROM translation_cache").fetchone()[0]
            if total > max_entries:
                cur = self._conn.execute(
                    """DELETE FROM translation_cache WHERE key IN (
                         SELECT key FROM translation_cache
                         ORDER BY hits ASC, created_at ASC LIMIT ?)""",
                    (total - max_entries,),
                )
                removed += cur.rowcount or 0
            self._conn.commit()
        if removed:
            log.info("清理翻译缓存 %d 条", removed)
        return removed

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("关闭缓存连接失败: %s", exc)
