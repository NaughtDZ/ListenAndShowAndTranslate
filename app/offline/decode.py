"""音视频文件 → 16 kHz 单声道 float32 音频流。

**为什么优先 ffmpeg**：``soundfile``（libsndfile 1.2.2，本机实测）只认
wav / flac / ogg / **mp3** / aiff / caf / w64；**m4a、aac、wma、mp4、mkv 全都不行**。
而"拖一个 mp4 进来出字幕"是这工具最常见的用法，所以优先用 ffmpeg 抽第一条音轨；
机器上没有 ffmpeg 时退回 soundfile（纯 Python、零外部依赖，覆盖上面那几种格式）。

两条路都是**流式**的：几小时的音频不会一次性读进内存，
喂给 VAD/ASR 的永远是 10 秒左右的块。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 16000
CHUNK_SECONDS = 10.0
"""每次产出的音频块长度。离线场景不需要更小，块大反而省开销。"""

SOUNDFILE_EXTS: tuple[str, ...] = (
    ".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff", ".caf", ".w64",
)
"""没有 ffmpeg 时还能处理的扩展名。"""


class DecodeError(RuntimeError):
    """解码失败（没有音轨、格式不支持、ffmpeg 报错等）。"""


@dataclass
class MediaInfo:
    path: Path
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    backend: str = ""
    """``ffmpeg`` 或 ``soundfile``。"""

    @property
    def duration_label(self) -> str:
        if not self.duration_s:
            return "未知时长"
        total = int(round(self.duration_s))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


# --------------------------------------------------------------------------- #
# 后端探测
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _no_window() -> int:
    """Windows 下别给每个文件弹一个黑框。"""
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


# --------------------------------------------------------------------------- #
# 基本信息（时长用于进度条）
# --------------------------------------------------------------------------- #
def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    info = MediaInfo(path=path)

    ffprobe = find_ffprobe()
    if ffprobe:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, creationflags=_no_window()
            )
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout)
                streams = data.get("streams") or []
                if streams:
                    info.sample_rate = int(streams[0].get("sample_rate") or 0)
                    info.channels = int(streams[0].get("channels") or 0)
                dur = (data.get("format") or {}).get("duration")
                if dur:
                    info.duration_s = float(dur)
                info.backend = "ffmpeg"
                return info
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # 落到 soundfile 兜底

    try:
        meta = sf.info(str(path))
        info.duration_s = float(meta.frames) / meta.samplerate if meta.samplerate else 0.0
        info.sample_rate = int(meta.samplerate)
        info.channels = int(meta.channels)
        info.backend = "soundfile"
    except (RuntimeError, OSError):
        # 探测不出来不算失败：ffmpeg 可能仍然能解（只是拿不到时长）
        info.backend = "ffmpeg" if find_ffmpeg() else ""

    return info


# --------------------------------------------------------------------------- #
# 解码（流式）
# --------------------------------------------------------------------------- #
def iter_pcm16k(
    path: str | Path,
    chunk_seconds: float = CHUNK_SECONDS,
) -> Iterator[np.ndarray]:
    """逐块产出 16 kHz 单声道 float32（取值范围 -1..1）。

    调用方 ``close()`` 掉生成器即可中止 ffmpeg 子进程。
    """
    path = str(path)
    chunk_samples = max(1600, int(TARGET_SR * max(0.1, chunk_seconds)))

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        yield from _iter_ffmpeg(ffmpeg, path, chunk_samples)
        return
    yield from _iter_soundfile(path, chunk_samples)


def _iter_ffmpeg(ffmpeg: str, path: str, chunk_samples: int) -> Iterator[np.ndarray]:
    cmd = [
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-i", path,
        "-map", "0:a:0",           # 只取第一条音轨（视频文件里音乐/对白通常在 a:0）
        "-vn",
        "-ac", "1", "-ar", str(TARGET_SR),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=chunk_samples * 2,
        creationflags=_no_window(),
    )
    assert proc.stdout is not None
    chunk_bytes = chunk_samples * 2

    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                break
            if len(raw) % 2:          # 半个采样点没有任何意义，丢掉
                raw = raw[:-1]
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size:
                yield pcm

        returncode = proc.wait()
        if returncode != 0:
            detail = ""
            if proc.stderr is not None:
                detail = proc.stderr.read().decode("utf-8", "replace").strip()[-400:]
            raise DecodeError(
                "ffmpeg 解码失败" + (f"（退出码 {returncode}）" if returncode else "")
                + (f"：{detail}" if detail else "：这个文件里可能没有音轨")
            )
    finally:
        for stream in (proc.stdout, proc.stderr):
            try:
                stream and stream.close()
            except OSError:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            try:
                proc.kill()
            except OSError:
                pass


def _iter_soundfile(path: str, chunk_samples: int) -> Iterator[np.ndarray]:
    try:
        handle = sf.SoundFile(path)
    except (RuntimeError, OSError) as exc:
        hint = ""
        if not find_ffmpeg():
            hint = (
                "（本机没装 ffmpeg，所以只支持 wav/flac/ogg/mp3 等；"
                "装 ffmpeg 后即可处理 m4a/mp4/mkv）"
            )
        raise DecodeError(f"打不开这个音频：{exc}{hint}") from exc

    with handle:
        src_rate = handle.samplerate
        channels = max(1, handle.channels)
        resampler = None
        if src_rate != TARGET_SR:
            resampler = soxr.ResampleStream(
                src_rate, TARGET_SR, 1, dtype="float32", quality="HQ"
            )
        read_frames = max(1, int(chunk_samples * src_rate / TARGET_SR))

        while True:
            block = handle.read(read_frames, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            mono = block.mean(axis=1) if channels > 1 else block[:, 0]
            out = resampler.resample_chunk(mono, last=False) if resampler else mono
            if out.size:
                yield np.ascontiguousarray(out, dtype=np.float32)

        if resampler is not None:
            tail = resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            if tail.size:
                yield np.ascontiguousarray(tail, dtype=np.float32)


def decode_all(path: str | Path, max_seconds: float = 0.0) -> np.ndarray:
    """整段读进内存——只给测试与小文件用；长音频请用 ``iter_pcm16k``。"""
    limit = int(TARGET_SR * max_seconds) if max_seconds > 0 else 0
    parts: list[np.ndarray] = []
    total = 0
    for chunk in iter_pcm16k(path):
        parts.append(chunk)
        total += chunk.size
        if limit and total >= limit:
            break
    if not parts:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(parts)
    return audio[:limit] if limit else audio
