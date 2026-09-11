"""术语表：小说翻译的刚需。

人名、地名、功法名如果不固定，同一个角色会一会叫"林凡"一会叫"林繁"，
读者直接懵。术语表做两件事：

1. **注入提示词**（强制译法）
2. **事后校验**（翻完检查有没有漏用，见 openai_compat.glossary_violations）

关键优化：**只注入本条字幕真正命中的术语**。
整张表几百条全塞进每次请求里既浪费 token 又会干扰模型
（无关术语会让它把不相关的词也硬套上去）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.utils.log import get_logger

log = get_logger(__name__)


@dataclass
class GlossaryEntry:
    source: str
    target: str
    note: str = ""
    case_sensitive: bool = False


@dataclass
class Glossary:
    """术语表。``entries`` 是 原文 → 译法 的映射。"""

    entries: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    case_sensitive: bool = False
    name: str = "默认"

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def add(self, source: str, target: str, note: str = "") -> None:
        src, dst = (source or "").strip(), (target or "").strip()
        if not src or not dst:
            return
        self.entries[src] = dst
        if note:
            self.notes[src] = note

    def update(self, mapping: dict[str, str]) -> None:
        for k, v in (mapping or {}).items():
            self.add(k, v)

    def remove(self, source: str) -> None:
        self.entries.pop(source, None)
        self.notes.pop(source, None)

    # ------------------------------------------------------------------ #
    def subset_for(self, text: str) -> dict[str, str]:
        """只返回本条文本里真正出现的术语。

        这是省 token 与提升命中率的关键：整表注入会让模型把无关的词也硬套。
        """
        if not text or not self.entries:
            return {}
        if self.case_sensitive:
            return {k: v for k, v in self.entries.items() if k in text}
        low = text.lower()
        return {k: v for k, v in self.entries.items() if k.lower() in low}

    def matched(self, text: str) -> list[str]:
        return list(self.subset_for(text))

    # ------------------------------------------------------------------ #
    def hash(self) -> str:
        """术语表指纹，用于缓存键——术语改了必须让缓存失效。"""
        items = sorted(self.entries.items())
        blob = json.dumps(items, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def subset_hash(self, subset: dict[str, str]) -> str:
        blob = json.dumps(sorted(subset.items()), ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: Path) -> "Glossary":
        """从文件加载。支持 JSON 与 TSV 两种格式。

        JSON:  ``{"リンファン": "林凡", ...}``
               或 ``{"name": "...", "entries": {"...": "..."}}``
        TSV:   每行 ``原文<TAB>译法[<TAB>备注]``，``#`` 开头为注释

        用 TSV 是因为它好用 Excel 编辑——用户维护几百条术语时这点很重要。
        """
        path = Path(path)
        if not path.exists():
            log.warning("术语表不存在: %s", path)
            return cls()
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
            return cls._from_json(text, path)
        return cls._from_tsv(text, path)

    @classmethod
    def _from_json(cls, text: str, path: Path) -> "Glossary":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error("术语表 JSON 解析失败 %s: %s", path, exc)
            return cls()
        g = cls(name=path.stem)
        if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], dict):
            g.name = data.get("name", path.stem)
            g.case_sensitive = bool(data.get("case_sensitive", False))
            for k, v in data["entries"].items():
                if isinstance(v, dict):
                    g.add(k, v.get("target", ""), v.get("note", ""))
                else:
                    g.add(k, str(v))
        elif isinstance(data, dict):
            for k, v in data.items():
                g.add(k, str(v))
        return g

    @classmethod
    def _from_tsv(cls, text: str, path: Path) -> "Glossary":
        g = cls(name=path.stem)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= 2:
                g.add(parts[0], parts[1], parts[2] if len(parts) > 2 else "")
        return g

    def to_file(self, path: Path, as_json: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_json or path.suffix.lower() == ".json":
            data = {
                "name": self.name,
                "case_sensitive": self.case_sensitive,
                "entries": {
                    k: ({"target": v, "note": self.notes[k]} if k in self.notes else v)
                    for k, v in self.entries.items()
                },
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            lines = ["# 原文\t译法\t备注（用 Tab 分隔，# 开头是注释）"]
            for k, v in self.entries.items():
                lines.append(f"{k}\t{v}\t{self.notes.get(k, '')}")
            path.write_text("\n".join(lines), encoding="utf-8")
        log.info("术语表已保存: %s（%d 条）", path, len(self))

    # ------------------------------------------------------------------ #
    def auto_extract_candidates(
        self,
        texts: list[str],
        min_len: int = 2,
        max_len: int = 4,
        min_count: int = 3,
        top: int = 200,
    ) -> dict[str, int]:
        """从已识别文本里挑"可能是专名"的候选词（供 UI 提示用户确认）。

        **只做候选提示，绝不自动写入**——猜错的译名比没有更糟。

        做法是中文新词发现的经典手段：统计 2~4 字的 n-gram 频次，
        再丢掉"被更长的高频候选包含"的短词（否则「林凡」会被「林凡说」这种盖住，
        或者反过来只留下「林」这种无意义的碎片）。

        它**只是帮你少打几个字**，最终要人来判断。
        """
        import re
        from collections import Counter

        counter: Counter[str] = Counter()
        for t in texts:
            if not t:
                continue
            # 按非文字字符切开（中文/日文假名/字母数字都算文字）
            for seg in re.split(r"[^\w\u3040-\u30ff\u4e00-\u9fff]+", t):
                seg = seg.strip()
                if len(seg) < min_len:
                    continue
                for n in range(min_len, max_len + 1):
                    for i in range(len(seg) - n + 1):
                        counter[seg[i:i + n]] += 1

        items = [(w, c) for w, c in counter.items() if c >= min_count]
        # 频次高的优先；同频次时长的优先
        items.sort(key=lambda x: (-x[1], -len(x[0])))

        result: dict[str, int] = {}
        for word, count in items:
            # 若已被某个更长的候选包含，说明它多半是那个词的碎片，跳过
            if any(word != kept and word in kept for kept in result):
                continue
            result[word] = count
            if len(result) >= top:
                break
        return result


def load_glossary(path: str | Path) -> Glossary:
    """便捷函数：路径为空返回空术语表。"""
    if not path:
        return Glossary()
    return Glossary.from_file(Path(path))
