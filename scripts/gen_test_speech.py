"""用 Windows 自带语音合成（SAPI）生成多语言测试语音。

为什么不用现成样本音频：**自己合成才知道正确文本**，
这样才能真正算字准率，而不是"听起来差不多"。

Windows 自带音色（本机实测）：
    中文 zh-CN：Huihui / Kangkang / Yaoyao
    英文 en-US：Zira / David / Mark
    日文 ja-JP：Haruka / Ayumi / Ichiro / Sayaka

输出：16 kHz / 16-bit / 单声道 WAV（ASR 模型的原生输入格式）

用法：
    .venv\\Scripts\\python.exe scripts\\gen_test_speech.py
    .venv\\Scripts\\python.exe scripts\\gen_test_speech.py --list-voices
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# ⚠️ Windows PowerShell 5.1 读取 .ps1 时按 ANSI/本地代码页解码（不是 UTF-8），
# 所以脚本文件**必须带 UTF-8 BOM**，否则中文/日文会变成乱码（实测踩过）。
# PowerShell 7 (pwsh) 默认按 UTF-8 读，优先用它。
def _ps_exe() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell"


PS_ENCODING = "utf-8-sig"  # 带 BOM，兼容 5.1 与 7

# (语言, 首选音色候选, 文本)
# 文本故意包含：数字、专有名词、中英混说、口语停顿，覆盖真实小说场景
CASES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "zh",
        ("Microsoft Huihui Desktop", "Microsoft Huihui", "Microsoft Yaoyao"),
        "第一章　夜行的列车。林凡握紧了手里的青铜钥匙，"
        "低声说道：“这一次，我不会再让任何人失望。”",
    ),
    (
        "zh",
        ("Microsoft Kangkang Desktop", "Microsoft Kangkang", "Microsoft Huihui"),
        "他花了三千二百块钱买下那本古籍，"
        "封面上的字迹已经模糊，但依稀能认出“太虚”两个字。",
    ),
    (
        "en",
        ("Microsoft Zira Desktop", "Microsoft Zira", "Microsoft David"),
        "Chapter one. The night train pulled out of the station, "
        "and Eleanor realised she had left her notebook behind.",
    ),
    (
        "ja",
        ("Microsoft Haruka Desktop", "Microsoft Haruka", "Microsoft Ayumi"),
        "第一章、夜の列車。リンファンは手の中の青銅の鍵を強く握りしめ、"
        "「今度こそ、誰も失望させたりはしない」と小さくつぶやいた。",
    ),
]


def list_voices() -> int:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture }"
    )
    out = subprocess.run(
        [_ps_exe(), "-NoProfile", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(out.stdout)
    return 0


def synthesize(text: str, voice_candidates: tuple[str, ...], out_path: Path) -> bool:
    """用 PowerShell 调 SAPI 合成到 16k/16bit/单声道 WAV。"""
    voices_ps = ", ".join(f"'{v}'" for v in voice_candidates)
    path_ps = str(out_path).replace("'", "''")
    text_ps = text.replace("'", "''")

    script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$installed = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }}
$wanted = @({voices_ps})
$picked = $null
foreach ($w in $wanted) {{
    if ($installed -contains $w) {{ $picked = $w; break }}
}}
if ($null -eq $picked) {{
    Write-Output "NOVOICE"
    exit 3
}}
$s.SelectVoice($picked)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{path_ps}', $fmt)
$s.Speak('{text_ps}')
$s.SetOutputToNull()
$s.Dispose()
Write-Output "OK $picked"
"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".ps1", delete=False, encoding=PS_ENCODING
    ) as fh:
        fh.write(script)
        ps1 = fh.name

    try:
        out = subprocess.run(
            [_ps_exe(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        line = (out.stdout or "").strip()
        if line.startswith("OK"):
            print(f"  ✓ {out_path.name}  音色={line[3:].strip()}")
            return True
        print(f"  ✗ {out_path.name}  合成失败: {line or out.stderr.strip()[:200]}")
        return False
    finally:
        Path(ps1).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成多语言测试语音")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    if args.list_voices:
        return list_voices()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT / "data" / "test_speech"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {out_dir}\n")
    ok = 0
    manifest: list[str] = []
    for idx, (lang, voices, text) in enumerate(CASES):
        path = out_dir / f"{lang}_{idx:02d}.wav"
        if synthesize(text, voices, path):
            ok += 1
            manifest.append(f"{path.name}\t{lang}\t{text}")

    (out_dir / "manifest.tsv").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\n完成 {ok}/{len(CASES)} 个；清单: {out_dir / 'manifest.tsv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
