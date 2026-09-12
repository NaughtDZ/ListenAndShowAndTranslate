"""模型下载器：代理、HF 镜像回退、断点续传、原子落盘、大小校验。

设计要点：
- **断点续传**：先下到 ``*.part``，用 Range 头接着下；服务器不支持 Range 就重头来
- **原子落盘**：只有大小校验通过才 ``replace`` 成正式文件名，
  避免"下了一半但看起来已存在"的假成功
- **镜像回退**：huggingface.co 失败自动改用 hf-mirror.com
- **不猜**：每个文件的期望字节数来自 registry（HF API 实测值）
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from app.models.registry import MODELS, ModelFile, ModelSpec
from app.paths import MODELS_DIR
from app.utils.log import get_logger

log = get_logger(__name__)

CHUNK = 1024 * 256  # 256 KiB
PROGRESS_INTERVAL_S = 0.25

ModelStatus = Literal["installed", "partial", "missing"]


@dataclass
class DownloadProgress:
    """一次下载任务的实时进度。"""

    model_id: str
    filename: str
    file_index: int          # 从 1 开始
    file_count: int
    file_downloaded: int
    file_size: int
    total_downloaded: int
    total_size: int
    speed_bps: float = 0.0
    eta_s: float = 0.0
    mirror: bool = False

    @property
    def fraction(self) -> float:
        return self.total_downloaded / self.total_size if self.total_size else 0.0

    def describe(self) -> str:
        pct = self.fraction * 100
        return (
            f"[{pct:5.1f}%] {self.model_id} "
            f"{self.filename} ({self.file_index}/{self.file_count}) "
            f"{self.total_downloaded / 1e6:.1f}/{self.total_size / 1e6:.1f} MB "
            f"{self.speed_bps / 1e6:.1f} MB/s ETA {self.eta_s:.0f}s"
            + ("  [mirror]" if self.mirror else "")
        )


ProgressCb = Callable[[DownloadProgress], None]
LogCb = Callable[[str], None]


class DownloadCancelled(Exception):
    """用户取消下载。"""


class ModelDownloader:
    """把模型下到 ``data/models/<model_id>/``。"""

    def __init__(
        self,
        models_dir: Path | None = None,
        proxy: str = "",
        timeout_s: float = 60.0,
        prefer_mirror: bool = False,
    ) -> None:
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.proxy = proxy or ""
        self.timeout_s = timeout_s
        self.prefer_mirror = prefer_mirror
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    # ------------------------------------------------------------------ #
    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def status(self, model_id: str) -> ModelStatus:
        """检查模型是否已完整下载。"""
        spec = MODELS.get(model_id)
        if spec is None:
            return "missing"
        d = self.model_dir(model_id)
        if not d.exists():
            return "missing"
        present = 0
        for f in spec.files:
            p = d / f.name
            if p.exists() and p.stat().st_size == f.size:
                present += 1
        if present == len(spec.files):
            return "installed"
        return "partial" if present else "missing"

    def installed_bytes(self, model_id: str) -> int:
        spec = MODELS.get(model_id)
        if spec is None:
            return 0
        d = self.model_dir(model_id)
        total = 0
        for f in spec.files:
            p = d / f.name
            if p.exists():
                total += min(p.stat().st_size, f.size)
        return total

    # ------------------------------------------------------------------ #
    def install(
        self,
        model_id: str,
        on_progress: ProgressCb | None = None,
        on_log: LogCb | None = None,
    ) -> bool:
        """下载并校验一个模型。返回是否成功（已装好也算成功）。"""
        spec = MODELS.get(model_id)
        if spec is None:
            raise KeyError(f"未知模型: {model_id}")

        if self.status(model_id) == "installed":
            if on_log:
                on_log(f"已安装，跳过: {spec.display_name}")
            return True

        target = self.model_dir(model_id)
        target.mkdir(parents=True, exist_ok=True)

        total_size = spec.total_bytes
        done_before = self.installed_bytes(model_id)
        total_done = done_before
        started = time.time()

        mirrors = [self.prefer_mirror, not self.prefer_mirror]

        for idx, f in enumerate(spec.files, start=1):
            if self._cancel.is_set():
                raise DownloadCancelled(model_id)

            if (target / f.name).exists() and (target / f.name).stat().st_size == f.size:
                total_done += f.size
                continue

            ok = False
            last_err: Exception | None = None
            for use_mirror in mirrors:
                if self._cancel.is_set():
                    raise DownloadCancelled(model_id)
                try:
                    self._download_one(
                        spec, f, target, idx, len(spec.files),
                        total_size, total_done, started, use_mirror, on_progress,
                    )
                    ok = True
                    break
                except DownloadCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - 换镜像重试
                    last_err = exc
                    log.warning("下载 %s 失败（mirror=%s）: %s", f.name, use_mirror, exc)
                    if on_log:
                        on_log(f"  ! {f.name} 下载失败（{'镜像' if use_mirror else '官方源'}）: {exc}")

            if not ok:
                if on_log:
                    on_log(f"✗ {spec.display_name} 下载失败: {last_err}")
                return False

            total_done += f.size

        # 全部校验
        final = self.status(model_id)
        if on_log:
            on_log(f"{'✓' if final == 'installed' else '✗'} {spec.display_name} "
                   f"({spec.total_mb:.0f} MB) → {final}")
        return final == "installed"

    def install_many(
        self,
        model_ids: Iterable[str],
        on_progress: ProgressCb | None = None,
        on_log: LogCb | None = None,
    ) -> dict[str, bool]:
        return {mid: self.install(mid, on_progress, on_log) for mid in model_ids}

    # ------------------------------------------------------------------ #
    def _download_one(
        self,
        spec: ModelSpec,
        f: ModelFile,
        target: Path,
        file_index: int,
        file_count: int,
        total_size: int,
        total_done_before: int,
        started: float,
        use_mirror: bool,
        on_progress: ProgressCb | None,
    ) -> None:
        dest = target / f.name
        part = dest.with_suffix(dest.suffix + ".part")
        url = spec.url_for(f.name, mirror=use_mirror)

        already = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={already}-"} if already else {}

        kwargs: dict = {"follow_redirects": True, "timeout": self.timeout_s}
        if self.proxy:
            kwargs["proxy"] = self.proxy

        with httpx.Client(**kwargs) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416:
                    # Range 越界：part 其实已经完整，直接进入校验
                    already = part.stat().st_size if part.exists() else 0
                else:
                    resp.raise_for_status()
                    if already and resp.status_code != 206:
                        # 服务器不支持续传，重头下
                        already = 0
                        part.unlink(missing_ok=True)

                    mode = "ab" if already and resp.status_code == 206 else "wb"
                    last_report = 0.0
                    downloaded = already
                    with open(part, mode) as fh:
                        for chunk in resp.iter_bytes(CHUNK):
                            if self._cancel.is_set():
                                raise DownloadCancelled(spec.id)
                            fh.write(chunk)
                            downloaded += len(chunk)

                            now = time.monotonic()
                            if on_progress and now - last_report >= PROGRESS_INTERVAL_S:
                                last_report = now
                                elapsed = max(1e-6, time.time() - started)
                                done = total_done_before + downloaded
                                speed = done / elapsed
                                remain = max(0, total_size - done)
                                on_progress(DownloadProgress(
                                    model_id=spec.id,
                                    filename=f.name,
                                    file_index=file_index,
                                    file_count=file_count,
                                    file_downloaded=downloaded,
                                    file_size=f.size,
                                    total_downloaded=done,
                                    total_size=total_size,
                                    speed_bps=speed,
                                    eta_s=remain / speed if speed > 0 else 0.0,
                                    mirror=use_mirror,
                                ))

        got = part.stat().st_size if part.exists() else 0
        if got != f.size:
            raise IOError(f"大小校验失败: {f.name} 期望 {f.size} 实得 {got}")

        part.replace(dest)  # 原子生效：只有校验通过才会出现正式文件名

    # ------------------------------------------------------------------ #
    def uninstall(self, model_id: str) -> bool:
        """删除模型目录。"""
        d = self.model_dir(model_id)
        if not d.exists():
            return False
        for p in sorted(d.rglob("*"), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError as exc:
                log.warning("删除 %s 失败: %s", p, exc)
        try:
            d.rmdir()
        except OSError:
            pass
        return not d.exists()

    # ------------------------------------------------------------------ #
    def repair(self, model_id: str) -> list[str]:
        """找出大小不符的文件并删除，返回被删列表（下次 install 会重下）。"""
        spec = MODELS.get(model_id)
        if spec is None:
            return []
        d = self.model_dir(model_id)
        broken: list[str] = []
        for f in spec.files:
            p = d / f.name
            if p.exists() and p.stat().st_size != f.size:
                p.unlink(missing_ok=True)
                broken.append(f.name)
        return broken
