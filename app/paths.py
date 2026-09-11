"""统一的路径管理。

所有运行时可写数据都落在 data/ 下（.gitignore 红线，绝不进 Git）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目根目录（app/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent

# 运行时数据根目录：默认 <项目根>/data，可用环境变量覆盖（便于绿色版/移动盘）
DATA_DIR = Path(os.environ.get("LST_DATA_DIR", ROOT / "data")).resolve()

CONFIG_FILE = DATA_DIR / "config.json"
DB_FILE = DATA_DIR / "listen.db"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ICONS_DIR = ASSETS_DIR / "icons"
PROMPTS_DIR = ASSETS_DIR / "prompts"
DOCS_DIR = ROOT / "docs"

_ALL_DIRS = (DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR, CACHE_DIR)


def ensure_dirs() -> None:
    """确保所有运行时目录存在。幂等，可反复调用。"""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    """是否为打包后的可执行文件运行（PyInstaller/Nuitka）。"""
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts: str) -> Path:
    """读取随程序分发的只读资源（打包后会指向解包目录）。"""
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return base.joinpath(*parts)


def default_data_dir() -> Path:
    """默认数据目录（供首次运行向导展示用）。"""
    return DATA_DIR
