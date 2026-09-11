"""进程音频采集：把 proc-tap 包装成"带目标解析、静音检测、断线重连"的工作线程。

⚠️ 本模块严格遵守 P1 实测得出的硬约束（见 docs/P1-实测记录.md 第 3 节）：

1. **绝不假定"我拿到的 PID"就是音频会话的 PID。**
   uv 的 ``.venv\\Scripts\\python.exe`` 是转发器；浏览器/Electron 应用有几十个子进程。
   因此目标一律用 :func:`resolve_target` 从**活跃音频会话列表**里解析。

2. **"有没有声音"只能靠能量判断。**
   目标未播声时 proc-tap 照样每次返回数据（速率恒定 384 KB/s），只是幅度为 0。
   ``read()`` 超时也不代表进程死了。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from app.audio.pipeline import AudioPipeline, PipelineStats, RingBuffer
from app.audio.process_list import AudioProcess, enumerate_audio_processes
from app.utils.log import get_logger

log = get_logger(__name__)

# 一块 10 ms，静音 5 秒就提示"目标好像没在放音"
DEFAULT_SILENT_ALERT_SECONDS = 5.0


@dataclass(frozen=True)
class TargetSpec:
    """用户选择的音频来源。

    优先级：``pid`` > ``executable`` > ``process_name``。
    注意 ``pid`` 必须是**音频会话 PID**（即用户从下拉框里选的那个），
    不是"某个碰巧同名的进程"。
    """

    pid: int | None = None
    process_name: str = ""
    executable: str = ""

    def describe(self) -> str:
        if self.pid is not None:
            return f"pid={self.pid}"
        if self.executable:
            return f"exe={self.executable}"
        return f"name={self.process_name}"


def resolve_target(spec: TargetSpec) -> AudioProcess | None:
    """把 TargetSpec 解析成当前**真的持有活跃音频会话**的进程。

    返回 None 表示目标当前没有在输出音频（或进程不存在）。

    同名多进程（浏览器/Electron）时的策略：
    优先带窗口标题的，其次会话数多的；并用 WARNING 记录候选，方便排错。
    """
    active = enumerate_audio_processes(include_inactive=False, with_window_title=False)
    candidates = [p for p in active if not p.is_system_sounds]
    if not candidates:
        return None

    if spec.pid is not None:
        for p in candidates:
            if p.pid == spec.pid:
                return p
        # PID 不在活跃会话里：可能没在放音，或串口/子进程错位
        log.debug("pid=%s 不在活跃音频会话中（可能未播放）", spec.pid)
        return None

    if spec.executable:
        want = spec.executable.lower()
        matched = [p for p in candidates if p.executable and p.executable.lower() == want]
        if matched:
            return _pick_best(matched, spec)

    if spec.process_name:
        want = spec.process_name.lower()
        matched = [
            p for p in candidates
            if p.name.lower() == want or p.name.lower().removesuffix(".exe") == want.removesuffix(".exe")
        ]
        if matched:
            return _pick_best(matched, spec)

    return None


def _pick_best(matched: list[AudioProcess], spec: TargetSpec) -> AudioProcess:
    if len(matched) > 1:
        log.warning(
            "目标 %s 匹配到 %d 个活跃音频会话（%s），已自动选择第一个。"
            "多进程应用建议直接在 UI 里按 PID 指定。",
            spec.describe(), len(matched), ", ".join(str(p.pid) for p in matched),
        )
        with_title = [p for p in matched if p.title]
        if with_title:
            return with_title[0]
    return matched[0]


@dataclass
class CaptureStats:
    """采集状态快照，供 UI 显示。"""

    running: bool = False
    target_pid: int | None = None
    target_name: str = ""
    resolved_at: float = 0.0

    chunks: int = 0
    bytes_in: int = 0
    samples_out: int = 0

    last_rms: float = 0.0
    peak: float = 0.0

    silent_seconds: float = 0.0
    """当前连续静音时长；用来提示"目标程序没在放音"。"""

    total_silent_seconds: float = 0.0
    reconnects: int = 0
    last_error: str = ""

    @property
    def likely_playing(self) -> bool:
        return self.running and self.silent_seconds < DEFAULT_SILENT_ALERT_SECONDS


AudioCallback = Callable[[object], None]


class CaptureWorker:
    """在后台线程里做：解析目标 → 采集 → 降混/重采样 → 回调。

    用法::

        worker = CaptureWorker(TargetSpec(process_name="喜马拉雅.exe"))
        worker.start()
        ...
        worker.stop()
    """

    def __init__(
        self,
        spec: TargetSpec,
        pipeline: AudioPipeline | None = None,
        on_chunk: AudioCallback | None = None,
        follow: bool = True,
        reconnect_interval_s: float = 2.0,
        buffer_ms: int = 200,
    ) -> None:
        self.spec = spec
        self.pipeline = pipeline or AudioPipeline()
        self.on_chunk = on_chunk
        self.follow = follow
        self.reconnect_interval_s = reconnect_interval_s

        self.stats = CaptureStats()
        self.ring = RingBuffer(
            int(self.pipeline.out_rate * buffer_ms / 1000) if buffer_ms > 0 else 0
        )

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_loud_at = 0.0

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="lst-capture", daemon=True
        )
        self._thread.start()
        log.info("采集线程已启动，目标 %s", self.spec.describe())

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None
        with self._lock:
            self.stats.running = False
        log.info("采集线程已停止")

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while not self._stop.is_set():
            target = resolve_target(self.spec)
            if target is None:
                with self._lock:
                    self.stats.running = False
                    self.stats.target_pid = None
                    self.stats.last_error = "目标当前没有输出音频"
                if not self.follow:
                    break
                log.info("未找到活跃音频会话（%s），%.1fs 后重试", self.spec.describe(), self.reconnect_interval_s)
                self._stop.wait(self.reconnect_interval_s)
                continue

            self._capture_one(target)

            if not self.follow:
                break
            if not self._stop.is_set():
                with self._lock:
                    self.stats.reconnects += 1
                log.info("目标会话结束，%.1fs 后尝试重连", self.reconnect_interval_s)
                self._stop.wait(self.reconnect_interval_s)

        with self._lock:
            self.stats.running = False

    def _capture_one(self, target: AudioProcess) -> None:
        """采集单个目标直到它消失或被要求停止。异常全部吞掉并重试。"""
        try:
            from proctap import ProcessAudioCapture
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.stats.last_error = f"导入 proc-tap 失败: {exc}"
            log.error("导入 proc-tap 失败: %s", exc)
            return

        cap = None
        try:
            cap = ProcessAudioCapture(target.pid)
            cap.start()
        except Exception as exc:  # noqa: BLE001 - 目标是别人的进程，什么错都可能
            with self._lock:
                self.stats.last_error = f"启动采集失败: {exc}"
            log.error("启动采集失败 pid=%s: %s", target.pid, exc)
            if cap is not None:
                try:
                    cap.close()
                except Exception:  # noqa: BLE001
                    pass
            return

        fmt = {}
        try:
            fmt = cap.get_format()
        except Exception:  # noqa: BLE001
            pass

        with self._lock:
            self.stats.running = True
            self.stats.target_pid = target.pid
            self.stats.target_name = target.name
            self.stats.resolved_at = time.time()
            self.stats.last_error = ""
            self._last_loud_at = time.time()

        log.info("开始采集 %s (pid=%s) 格式=%s", target.name, target.pid, fmt)

        try:
            while not self._stop.is_set():
                data = cap.read(timeout=0.5)
                if not data:
                    # 实测：目标静音时也会持续返回数据，所以走到这里通常是目标消失了
                    if not _pid_alive(target.pid):
                        log.info("目标进程 pid=%s 已退出", target.pid)
                        break
                    continue

                out = self.pipeline.process(data)
                self._publish(data, out)

                # 目标进程已消失（或不再是活跃会话）→ 结束本轮，外层重连
                if not _pid_alive(target.pid):
                    log.info("目标进程 pid=%s 已退出，准备重连", target.pid)
                    break
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.stats.last_error = str(exc)
            log.error("采集过程中出错 pid=%s: %s", target.pid, exc)
        finally:
            try:
                tail = self.pipeline.flush()
                if tail.size and self.on_chunk is not None:
                    self.on_chunk(tail)
            except Exception:  # noqa: BLE001
                pass
            try:
                cap.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                cap.close()
            except Exception:  # noqa: BLE001
                pass
            with self._lock:
                self.stats.running = False

    def _publish(self, raw: bytes, out) -> None:
        """更新统计、写环形缓冲、分发回调。"""
        now = time.time()
        rms = self.pipeline.stats.last_rms

        with self._lock:
            self.stats.chunks += 1
            self.stats.bytes_in += len(raw)
            self.stats.samples_out += int(out.size)
            self.stats.last_rms = rms
            self.stats.peak = self.pipeline.stats.peak

            if rms >= self.pipeline.silence_threshold:
                self._last_loud_at = now
                self.stats.silent_seconds = 0.0
            else:
                silent = now - self._last_loud_at
                self.stats.silent_seconds = silent
                self.stats.total_silent_seconds = self.pipeline.stats.silent_seconds_total

        if out.size:
            self.ring.push(out)
            if self.on_chunk is not None:
                try:
                    self.on_chunk(out)
                except Exception as exc:  # noqa: BLE001 - 回调的锅不能让采集线程崩
                    log.error("音频回调抛异常: %s", exc)

    def snapshot(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(**vars(self.stats))

    def pipeline_stats(self) -> PipelineStats:
        return self.pipeline.stats


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        return True  # 判断不了就别中断采集


def record_to_wav(
    spec: TargetSpec,
    wav_path,
    seconds: float,
    out_rate: int = 16000,
    startup_timeout: float = 15.0,
) -> CaptureStats:
    """把目标音频录成 16k 单声道 WAV —— 手动验证用。

    注意：录的是 **N 秒音频**，不是"墙钟 N 秒"。
    实测采集启动本身要 ~0.6s（COM 激活 + 音频会话枚举），重采样器还要 40ms 缓冲；
    若按墙钟计时，请求 3 秒只会得到约 2.4 秒音频。
    """
    import wave

    import numpy as np

    chunks: list[np.ndarray] = []
    collected = [0]
    target_samples = int(seconds * out_rate)

    def _collect(chunk: np.ndarray) -> None:
        chunks.append(chunk)
        collected[0] += int(chunk.size)

    pipeline = AudioPipeline(out_rate=out_rate)
    worker = CaptureWorker(spec, pipeline=pipeline, on_chunk=_collect, follow=False)

    worker.start()
    deadline = time.time() + seconds + startup_timeout
    try:
        while collected[0] < target_samples and time.time() < deadline:
            if not worker.is_running and collected[0] == 0:
                # 启动失败/目标消失，没必要空等
                break
            time.sleep(0.05)
    finally:
        worker.stop()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if audio.size > target_samples:
        audio = audio[:target_samples]

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(out_rate)
        w.writeframes(pcm16.tobytes())

    return worker.snapshot()
