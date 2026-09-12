"""从模型注册表解析出引擎需要的文件路径。

把"注册表里的文件名"翻译成"引擎要的 encoder/decoder/joiner/tokens"，
让引擎代码不必知道具体文件名（不同 zipformer 版本命名差异很大：
有 ``encoder.int8.onnx``、也有 ``encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx``）。
"""

from __future__ import annotations

from pathlib import Path

from app.models.registry import MODELS, ModelSpec
from app.paths import MODELS_DIR

# 引擎角色 → 文件名里应出现的关键字（按优先级）
_ROLE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("encoder", ("encoder",)),
    ("decoder", ("decoder",)),
    ("joiner", ("joiner",)),
    ("tokens", ("tokens",)),
)


class ModelNotFound(RuntimeError):
    """模型未安装或文件缺失。"""


def model_dir(model_id: str, models_dir: Path | None = None) -> Path:
    return Path(models_dir or MODELS_DIR) / model_id


def model_paths(model_id: str, models_dir: Path | None = None) -> dict[str, Path]:
    """返回该模型已存在的文件，键为引擎角色名。

    例：``{"encoder": Path(.../encoder.int8.onnx), "decoder": ..., "joiner": ..., "tokens": ...}``
    """
    spec = MODELS.get(model_id)
    if spec is None:
        raise ModelNotFound(f"注册表里没有模型 {model_id}")

    base = model_dir(model_id, models_dir)
    out: dict[str, Path] = {}

    for role, hints in _ROLE_HINTS:
        for f in spec.files:
            lower = f.name.lower()
            if any(h in lower for h in hints):
                # 同一角色有多个候选时，**优先 int8**（体积小、CPU 快）
                cand = base / f.name
                prev = out.get(role)
                if prev is None:
                    out[role] = cand
                elif "int8" in cand.name and "int8" not in prev.name:
                    out[role] = cand
                break

    # SenseVoice / Whisper / VAD 这类结构：只有一个主 onnx，没有 encoder/decoder 之分。
    if "encoder" not in out:
        onnx = [f for f in spec.files if f.name.endswith(".onnx")]
        chosen = None
        for f in onnx:
            if "model" in f.name.lower():
                chosen = f
                break
        if chosen is None and len(onnx) == 1:
            # 例如 silero_vad.onnx —— 文件名里没有 "model" 关键字
            chosen = onnx[0]
        if chosen is not None:
            out["model"] = base / chosen.name

    return out


def require(model_id: str, models_dir: Path | None = None, roles: tuple[str, ...] = ()) -> dict[str, Path]:
    """取路径并校验文件真的存在；缺失时抛出带**可执行修复命令**的错误。

    错误信息里带上命令是有意的——用户最需要的是"那我该怎么办"。
    """
    spec = MODELS.get(model_id)
    if spec is None:
        raise ModelNotFound(f"未知模型: {model_id}")

    paths = model_paths(model_id, models_dir)
    missing = [r for r in (roles or tuple(paths)) if r not in paths or not paths[r].exists()]
    if missing:
        raise ModelNotFound(
            f"模型 {model_id}（{spec.display_name}）缺少文件: {', '.join(missing)}\n"
            f"请运行: python main.py --models install --packs "
            f"{_pack_for(model_id) or 'all'}"
        )
    return paths


def _pack_for(model_id: str) -> str:
    """反查该模型属于哪个语言包，用于给出准确的修复命令。"""
    from app.models.registry import pack_for_model

    return pack_for_model(model_id)


def describe(spec: ModelSpec) -> str:
    return f"{spec.id}（{spec.display_name}，{spec.total_mb:.0f} MB）"
