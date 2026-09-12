"""硬件探测与档位推荐。

首次运行向导用它回答一个问题：**这台机器该装哪几个模型、用哪档识别。**

只做"能确定的事"：
- CPU 核数、内存：psutil 直接读，准确
- NVIDIA 显卡：nvidia-smi，拿得到型号与显存
- AMD / Intel 核显：**拿不到可靠的显存信息**，所以不猜——
  按"有 GPU 但未知型号"处理，推荐档位偏保守，并如实说明

档位不是硬性限制，用户可以在向导里改。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from app.models.registry import PACKS, total_bytes_for_packs
from app.utils.log import get_logger

log = get_logger(__name__)

TierName = str  # "low" | "mid" | "high"


@dataclass
class GpuInfo:
    name: str
    vram_mb: int = 0
    vendor: str = "unknown"   # nvidia / amd / intel / unknown

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024 if self.vram_mb else 0.0


@dataclass
class HardwareInfo:
    cpu_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    ram_gb: float = 0.0
    gpus: list[GpuInfo] = field(default_factory=list)
    recommended_tier: TierName = "mid"
    recommended_packs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def best_gpu(self) -> GpuInfo | None:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_mb)

    def summary_lines(self) -> list[str]:
        lines = [
            f"CPU：{self.cpu_name or '未知'}（{self.physical_cores} 物理核 / "
            f"{self.logical_cores} 线程）",
            f"内存：{self.ram_gb:.1f} GB",
        ]
        if self.gpus:
            for g in self.gpus:
                vram = f"{g.vram_gb:.0f} GB 显存" if g.vram_mb else "显存未知"
                lines.append(f"显卡：{g.name}（{vram}）")
        else:
            lines.append("显卡：未检测到独立显卡")
        return lines


# --------------------------------------------------------------------------- #
def detect_hardware() -> HardwareInfo:
    """探测本机硬件并给出档位与语言包建议。"""
    info = HardwareInfo()
    _detect_cpu_ram(info)
    _detect_gpus(info)
    _recommend(info)
    return info


def _detect_cpu_ram(info: HardwareInfo) -> None:
    try:
        import platform

        info.logical_cores = 0
        try:
            import psutil

            info.physical_cores = psutil.cpu_count(logical=False) or 0
            info.logical_cores = psutil.cpu_count(logical=True) or 0
            info.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception as exc:  # noqa: BLE001
            log.debug("psutil 读取 CPU/内存失败: %s", exc)
        info.cpu_name = platform.processor() or ""
        if not info.cpu_name:
            info.cpu_name = _cpu_name_from_windows()
    except Exception as exc:  # noqa: BLE001
        log.debug("探测 CPU/内存失败: %s", exc)


def _cpu_name_from_windows() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=10,
        )
        return (out.stdout or "").strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


def _detect_gpus(info: HardwareInfo) -> None:
    # NVIDIA：nvidia-smi 最可靠
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            for line in (out.stdout or "").strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if parts and parts[0]:
                    vram = int(float(parts[1])) if len(parts) > 1 and parts[1] else 0
                    info.gpus.append(GpuInfo(name=parts[0], vram_mb=vram, vendor="nvidia"))
        except Exception as exc:  # noqa: BLE001
            log.debug("nvidia-smi 探测失败: %s", exc)

    # 其它厂商：只拿型号，不猜显存
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=15,
        )
        for raw in (out.stdout or "").strip().splitlines():
            name = raw.strip()
            if not name or any(g.name.lower() == name.lower() for g in info.gpus):
                continue
            low = name.lower()
            vendor = ("amd" if any(k in low for k in ("radeon", "amd", "ati"))
                      else "intel" if "intel" in low else "unknown")
            # 虚拟显示器（远程桌面/串流）不是真显卡，别拿来推荐档位
            if any(k in low for k in ("virtual", "gameviewer", "parsec", "sunshine", "idd")):
                info.notes.append(f"忽略虚拟显示适配器：{name}")
                continue
            info.gpus.append(GpuInfo(name=name, vram_mb=0, vendor=vendor))
    except Exception as exc:  # noqa: BLE001
        log.debug("Win32_VideoController 探测失败: %s", exc)


def _recommend(info: HardwareInfo) -> None:
    """按"够不够跑得动"给档位，宁可保守。"""
    gpu = info.best_gpu
    cores = info.physical_cores or info.logical_cores
    ram = info.ram_gb
    tier = "low"

    if gpu is not None and gpu.vendor == "nvidia" and gpu.vram_gb >= 6:
        tier = "high"
    elif ram >= 32 and cores >= 8:
        tier = "high"
    elif ram >= 16 and cores >= 6:
        tier = "mid"

    if gpu is not None and gpu.vendor in ("amd", "intel") and not gpu.vram_mb:
        info.notes.append(
            f"检测到 {gpu.vendor.upper()} 显卡但拿不到显存信息："
            "本次按 CPU 档推荐。你仍可在设置里手动选高档（走 CPU/Vulkan 也能跑）"
        )

    packs = {
        "high": ["core", "zh", "zh-en", "en", "ja-ko-yue", "multilingual", "lid"],
        "mid": ["core", "zh", "zh-en", "ja-ko-yue", "lid"],
        "low": ["core", "zh", "ja-ko-yue"],
    }[tier]

    info.recommended_tier = tier
    info.recommended_packs = packs

    tier_label = {"low": "轻量档（CPU 为主）", "mid": "平衡档", "high": "性能档（可用 GPU 与大模型）"}
    info.notes.append(
        f"推荐档位：{tier_label[tier]}，语言包合计 {total_bytes_for_packs(packs) / 1e6:.0f} MB"
    )
    if tier == "low":
        info.notes.append("内存或核心数偏少，建议只装中文包并用流式引擎（延迟低、占用小）")


def tier_label(tier: str) -> str:
    return {"low": "轻量档", "mid": "平衡档", "high": "性能档"}.get(tier, tier)


def all_pack_ids() -> list[str]:
    return list(PACKS)
