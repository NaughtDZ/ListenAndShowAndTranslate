"""离线字幕工具的内核：文件 → 识别 → 字幕文件。

与实时管线（``app/asr`` + ``app/pipeline.py``）的区别：

- **没有"未来"压力**：不用边听边出，每句都用最准的模型整句识别
- **输入是文件**：可能是 mp3、也可能是带音轨的 mp4/mkv
- **输出是字幕文件**：srt / vtt / lrc，要带**精确的起止时间**

复用：``app/asr/vad.py``（分段与精确起点）、``app/models/registry.py``（模型清单）、
``app/translate``（翻译、缓存、术语表）。
"""

from __future__ import annotations

__all__ = ["Cue"]
