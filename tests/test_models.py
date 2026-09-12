"""模型注册表与下载器测试（不联网）。

下载逻辑用 monkeypatch 把注册表换成小文件，这样能在测试里真实创建/删除文件，
验证"缺文件 / 半成品 / 已完整 / 大小不符"这几种状态的判定。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.models.downloader as dl
from app.models.downloader import ModelDownloader
from app.models.registry import (
    HF_HOST,
    HF_MIRROR,
    MODELS,
    PACKS,
    ModelFile,
    ModelSpec,
    default_pack_ids,
    find_by_engine,
    human_size,
    models_for_packs,
    total_bytes_for_packs,
)

# --------------------------------------------------------------------------- #
# 注册表完整性
# --------------------------------------------------------------------------- #
def test_registry_ids_match_keys():
    for key, spec in MODELS.items():
        assert spec.id == key, f"{key} 的 id 字段是 {spec.id}"


def test_every_model_has_files_with_positive_size():
    for spec in MODELS.values():
        assert spec.files, f"{spec.id} 没有文件"
        for f in spec.files:
            assert f.size > 0, f"{spec.id}/{f.name} 大小为 0"


def test_asr_models_have_onnx_and_tokens():
    for spec in MODELS.values():
        if spec.kind in ("asr_stream", "asr_offline", "lid"):
            assert spec.onnx_files, f"{spec.id} 缺 onnx"
            assert spec.tokens_file is not None, f"{spec.id} 缺 tokens"


def test_vad_uses_direct_url_not_hf():
    """VAD 来自 GitHub release，不是 HF 仓库。"""
    vad = MODELS["silero-vad"]
    assert vad.repo == ""
    assert vad.files[0].direct_url.startswith("https://github.com/")


def test_whisper_turbo_uses_int8_selfcontained_files():
    """非 int8 的 turbo encoder 只有 0.74MB，权重在外置 .weights 里，不能用。"""
    spec = MODELS["whisper-turbo-int8"]
    names = [f.name for f in spec.files]
    assert any("encoder.int8" in n for n in names)
    assert all(".weights" not in n for n in names)
    # 自包含 int8 encoder 应该是几百 MB，不是几百 KB
    enc = next(f for f in spec.files if "encoder" in f.name)
    assert enc.size > 100_000_000


def test_hf_url_and_mirror():
    spec = MODELS["zipformer-zh-int8"]
    official = spec.url_for("encoder.int8.onnx")
    mirror = spec.url_for("encoder.int8.onnx", mirror=True)
    assert official.startswith(HF_HOST)
    assert mirror.startswith(HF_MIRROR)
    assert official.endswith("/resolve/main/encoder.int8.onnx")


def test_direct_url_overrides_hf():
    vad = MODELS["silero-vad"]
    assert "github.com" in vad.url_for("silero_vad.onnx")
    # 未知文件仍回退到 HF 拼接（不会崩）
    assert vad.url_for("nope.onnx").startswith(HF_HOST)


def test_file_spec_lookup():
    spec = MODELS["dolphin-base-ctc-int8"]
    assert spec.file_spec("model.int8.onnx") is not None
    assert spec.file_spec("不存在.onnx") is None


# --------------------------------------------------------------------------- #
# 语言包
# --------------------------------------------------------------------------- #
def test_packs_reference_existing_models():
    for pack in PACKS.values():
        for mid in pack.model_ids:
            assert mid in MODELS, f"语言包 {pack.id} 引用了不存在的模型 {mid}"


def test_default_packs_include_core_and_zh():
    defaults = default_pack_ids()
    assert "core" in defaults
    assert "zh" in defaults
    # 默认要带上日语/韩粤这些新模型，但不该勾 1GB 的 Whisper 兜底包
    assert "ja-ko-yue" in defaults and "multilingual" in defaults
    # 可选候选（实测没赢过默认）不该默认勾上
    assert "ja-parakeet" not in defaults and "zh-accurate" not in defaults


def test_models_for_packs_dedupes_and_keeps_order():
    ids = models_for_packs(["zh", "zh", "core"])
    assert ids == ["zipformer-zh-int8", "silero-vad"]


def test_total_bytes_for_packs():
    total = total_bytes_for_packs(["core", "zh"])
    expected = MODELS["silero-vad"].total_bytes + MODELS["zipformer-zh-int8"].total_bytes
    assert total == expected
    assert 100_000_000 < total < 300_000_000


def test_all_packs_total_is_reported_to_user():
    """用户勾了全部语言包的总量（会用 MB/GB 展示给他看）。

    2026-09 换模型后是 ~3.3GB（新模型更准也更占地方；Whisper turbo 那 1GB
    现在只在"全部勾上"时才包含）。
    """
    total = total_bytes_for_packs(list(PACKS))
    assert 3_400_000_000 < total < 4_200_000_000, total / 1e6


def test_find_by_engine():
    streams = find_by_engine("sherpa_stream")
    assert all(m.engine == "sherpa_stream" for m in streams)
    ja = find_by_engine("sherpa_offline", "ja")
    assert any("sensevoice" in m.id for m in ja)  # 实测日语最优


def test_human_size():
    assert human_size(500) == "500 B"
    assert human_size(2_500_000) == "2.5 MB"
    assert human_size(1_815_900_000).endswith("GB")


# --------------------------------------------------------------------------- #
# 下载器状态判定（用假注册表，避免真的造 GB 级文件）
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_registry(monkeypatch) -> dict[str, ModelSpec]:
    spec = ModelSpec(
        id="fake-model",
        display_name="假模型",
        kind="vad",
        engine="vad",
        languages=("*",),
        repo="owner/repo",
        files=(
            ModelFile("a.onnx", 10),
            ModelFile("b.txt", 5),
        ),
    )
    reg = {"fake-model": spec}
    monkeypatch.setattr(dl, "MODELS", reg)
    return reg


def test_status_missing_when_no_dir(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    assert d.status("fake-model") == "missing"


def test_status_partial_when_some_files(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 10)
    assert d.status("fake-model") == "partial"


def test_status_installed_when_all_sizes_match(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 10)
    (mdir / "b.txt").write_bytes(b"y" * 5)
    assert d.status("fake-model") == "installed"


def test_status_partial_when_size_wrong(tmp_path: Path, fake_registry):
    """大小不符不能算装好——否则会拿一个坏模型去跑推理。"""
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 3)  # 应为 10
    (mdir / "b.txt").write_bytes(b"y" * 5)
    assert d.status("fake-model") == "partial"


def test_install_skips_when_already_installed(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 10)
    (mdir / "b.txt").write_bytes(b"y" * 5)

    logs: list[str] = []
    assert d.install("fake-model", on_log=logs.append) is True
    assert any("跳过" in m for m in logs)


def test_repair_removes_wrong_size_files(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 3)   # 坏
    (mdir / "b.txt").write_bytes(b"y" * 5)    # 好

    broken = d.repair("fake-model")
    assert broken == ["a.onnx"]
    assert not (mdir / "a.onnx").exists()
    assert (mdir / "b.txt").exists()


def test_uninstall_removes_dir(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 10)

    assert d.uninstall("fake-model") is True
    assert not mdir.exists()
    assert d.uninstall("fake-model") is False  # 幂等


def test_installed_bytes_counts_partial(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    mdir = tmp_path / "fake-model"
    mdir.mkdir()
    (mdir / "a.onnx").write_bytes(b"x" * 4)
    assert d.installed_bytes("fake-model") == 4


def test_unknown_model_raises(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    with pytest.raises(KeyError):
        d.install("no-such-model")


def test_cancel_flag(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    d.cancel()
    assert d._cancel.is_set()
    d.reset_cancel()
    assert not d._cancel.is_set()


def test_status_of_unknown_model_is_missing(tmp_path: Path, fake_registry):
    d = ModelDownloader(models_dir=tmp_path)
    assert d.status("nope") == "missing"


def test_progress_fraction_and_describe():
    from app.models.downloader import DownloadProgress

    p = DownloadProgress(
        model_id="m", filename="f.onnx", file_index=1, file_count=2,
        file_downloaded=100, file_size=200,
        total_downloaded=500, total_size=1000, speed_bps=1_000_000, eta_s=0.5,
    )
    assert p.fraction == 0.5
    assert "50.0%" in p.describe()
    assert "m" in p.describe()
