# P2 ASR 实测记录（多语言）

> 记录日期：2026-02-21
> 复现：`scripts/gen_test_speech.py` + `scripts/transcribe_wav.py`

---

## 1. 测试方法（为什么可信）

**自己合成语音 → 因此知道正确文本 → 能算真实字准率**，而不是"听起来差不多"。

```
scripts/gen_test_speech.py   # Windows SAPI 合成 16k/16bit/单声道，中英日各若干句
   ↓ data/test_speech/*.wav + manifest.tsv（含正确文本）
scripts/transcribe_wav.py    # 送 ASR，与正确文本算 CER（去掉标点空白后按字符编辑距离）
```

生成语音时用的音色：中文 Huihui / Kangkang，英文 Zira，日文 Haruka。

> ⚠️ 踩坑：`gen_test_speech.py` 生成的 `.ps1` **必须带 UTF-8 BOM**，
> 否则 Windows PowerShell 5.1 会按 GBK 解码，中文日文全变乱码。
> 同时优先用 PowerShell 7（`pwsh`）。

---

## 2. 核心结论表

| 模型 | 日语 | 中文 | 英文 | RTF（越低越快） | 流式 | 标点 |
|---|---|---|---|---|---|---|
| `zipformer-zh-int8`（流式） | — | **97.3%** | — | 0.047 | ✅ | ❌ |
| `zipformer-en-int8`（流式） | — | — | **96.6%** | 0.028 | ✅ | ❌ |
| **`sensevoice-int8`（2024-07-17）** | **90.6%** | 97.3%* | 95.5%* | **0.015** | ❌ | ✅ |
| `sensevoice`（2025-09-09 int8） | **17.0%** ❌ | 91.9% | 94.4% | 0.017 | ❌ | ✅ |
| `whisper-turbo-int8` | 75.5% | — | — | 0.443 | ❌ | ✅ |

\* 单句直测（无 VAD 切分）的分数；经 VAD 分句后整体为 91.9% / 93.3%。
RTF 0.015 = 比实时快约 66 倍；0.443 = 比实时快约 2.3 倍。
测试机：Ryzen 9 9950X，CPU 2 线程，provider=cpu。

---

## 3. ⚠️ 最重要发现：**新版本反而更差，而且差得离谱**

**SenseVoice 2025-09-09 int8 版本的日语完全不可用。**

同一段日语音频：

| 版本 | 字准率 | 输出 |
|---|---|---|
| 2024-07-17 | **90.6%** | 第 印象、夜 の 列車、凛ン ファン は 手 の 中 の 聖堂 の 鍵 を 強く 握りしめ… |
| 2025-09-09 | **17.0%** | 大印象夜列车手中声堂嗅握今度失望小 |

根因线索：把 2025 版的 `result.lang` 打出来，**中/英/日三种输入全部返回 `<|yue|>`（粤语）**，
无论 `language` 传 `ja`、`zh`、`en` 还是 `auto` 都一样
（传 `<|ja|>` 时 sherpa-onnx 直接回：`Unknown language: <|ja|>. Use 0 instead.`）。

即：**该次导出把语言提示固定成了粤语**。中文英文因为字符集重叠碰巧还能出正确结果，
日语被按粤语解码就成了垃圾。

**行动**：注册表已把 `sensevoice-int8` 指向 **2024-07-17** 版本，并在 `note` 里写明不要换回 2025 版。

**教训**：模型版本号更大 ≠ 更好。**必须用带正确文本的测试集实测**，
凭"新版本应该更好"去选模型，日语功能会直接不可用，而且从中文测试里完全看不出来。

---

## 4. 标点与大小写：影响字幕可读性与翻译质量

| 模型 | 标点 | 英文大小写 | 说明 |
|---|---|---|---|
| SenseVoice 2024 | ✅ 自带 `，。、` | ✅ `Chapter 1.` | 可直接上屏 |
| 流式 zipformer（zh/en） | ❌ | ❌ **全大写无空格** | `CHAPTER ONETHE NIGHT TRAIN…`、`NOTE BOOK` |
| Whisper | ✅ | ✅ | |

**两个直接后果**：

1. **流式引擎的英文字幕必须做后处理**，否则用户看到的是
   `CHAPTER ONETHE NIGHT TRAIN PULLED OUT OF THE STATION` 这种全大写且词间粘连的文本。
2. **中文流式引擎没有标点** → 字幕是一整片文字墙，
   **而且会显著拖累翻译**（翻译模型依赖句子边界）。

因此存在一个真实的取舍：

| 中文用哪个 | 准确率 | 延迟 | 标点 |
|---|---|---|---|
| 流式 zipformer | 97.3% | 低（有中间结果，边听边出） | ❌ |
| SenseVoice 2024 | 97.3% | 高（整句说完才出） | ✅ |

考虑到用户明确表示"不是看视频、有延迟无所谓"，
**SenseVoice 在中文上可能是更好的默认选择**（标点对翻译质量的价值很大）。

进一步的方案（未实现，记在计划里）：**双引擎融合**——
流式引擎出即时草稿字幕，SenseVoice 出带标点的定稿替换。
两者 RTF 合计仅 0.06，在本机算力下完全跑得动。

---

## 5. 按本次实测修正后的语言路由

| 语言 | 引擎 | 模型 | 实测 | 理由 |
|---|---|---|---|---|
| 中文 | `sherpa_stream` | zipformer-zh-int8 | 97.3% | 真流式，最低延迟 |
| 中英混说 | `sherpa_stream` | zipformer-zh-en-int8 | — | 夹英文词更稳 |
| 英文 | `sherpa_stream` | zipformer-en-int8 | 96.6% | 真流式 |
| **日语** | `sherpa_offline` | **sensevoice-int8（2024）** | **90.6%** | 官方无日语流式模型；SenseVoice 比 Whisper 快 30 倍且更准 |
| 韩语 / 粤语 | `sherpa_offline` | sensevoice-int8 | — | 同一模型 |
| 小语种 | `whispercpp` | whisper-turbo-int8 | — | 99 语言兜底 |

> 日语用 Whisper 只有 75.5%，**已被 SenseVoice 取代**。
> 这条修正直接来自实测，计划书原来的"SenseVoice 或 Whisper 都行"是不够准确的。

---

## 6. 已知问题（待办）

| # | 问题 | 影响 | 计划 |
|---|---|---|---|
| 1 | 流式引擎英文输出全大写、词间粘连 | 字幕可读性差 | 加文本后处理（句子首字母大写 + 空格规整） |
| 2 | 流式引擎无标点 | 中文可读性与翻译质量 | 接标点模型，或中文默认改用 SenseVoice |
| 3 | 专有名词同音字错（林凡→林繁、リンファン→リンァン） | 人名/术语错 | 接 sherpa-onnx hotwords（热词偏置）+ 翻译层术语表 |
| 4 | 数字 ITN：三千二百 → 3200 | 与"期望文本"比对时显得不准，实际是**正确行为** | 评测脚本需做数字归一 |
| 5 | 日文分词间被插入空格（`誰 も 失望 させ`） | 显示略有瑕疵 | 日文后处理去掉多余空格 |
| 6 | 分块引擎无中间结果 | 日语字幕整句才出现 | 已知取舍，UI 上需说明（见 `docs/延迟调节.md` 第 3.5 节） |

---

## 7. 语言路由器验收（language=auto，端到端）

程序**事先不知道音频是什么语言**，要自己判断再选引擎。`scripts/verify_router.py`：

| 文件 | 真实语言 | 程序判定 | 字准率 | 自动选中的引擎 |
|---|---|---|---|---|
| en_02.wav | en | **en** ✓ | 96.6% | `zipformer-en-int8`（流式） |
| ja_03.wav | ja | **ja** ✓ | 86.8% | `sensevoice-int8`（分块） |
| zh_00.wav | zh | **zh** ✓ | 94.6% | `zipformer-zh-int8`（流式） |
| zh_01.wav | zh | **zh** ✓ | **100.0%** | `zipformer-zh-int8`（流式） |

- **语种判断 4/4 正确**，字准率 4/4 超过 80%
- 引擎选择与语言路由表一致（中英走流式、日语走分块）
- 后处理生效：日文 `誰 も 失望 させ` → `誰も失望させ`；英文 `CHAPTER` → `Chapter`

**已知的残留问题（见第 6 节）**：流式英文丢词间空格修不掉
（`Chapter oneThe night train`、`note book`），
这是模型 BPE 拼接导致的，需要有词典做分词才能还原。

## 8. 复现命令

```powershell
# 1) 生成多语言测试语音（需要 Windows 中文/日文语音包）
.venv\Scripts\python.exe scripts\gen_test_speech.py

# 2) 逐语言验证
.venv\Scripts\python.exe scripts\transcribe_wav.py data\test_speech\zh_00.wav --model zipformer-zh-int8
.venv\Scripts\python.exe scripts\transcribe_wav.py data\test_speech\en_02.wav --model zipformer-en-int8
.venv\Scripts\python.exe scripts\transcribe_wav.py data\test_speech\ja_03.wav --model sensevoice-int8 --language ja

# 3) 一次跑全部（SenseVoice 覆盖中英日）
.venv\Scripts\python.exe scripts\transcribe_wav.py --all --model sensevoice-int8 --language auto

# 4) 按真实速度喂入，测实际字幕延迟
.venv\Scripts\python.exe scripts\transcribe_wav.py data\test_speech\zh_00.wav --model zipformer-zh-int8 --realtime
```
# --------------------------------------------------------------------------- #
# 2026-09 模型换代实测：新模型 vs 旧模型，同一批音频对拍
# --------------------------------------------------------------------------- #

> 起因：用户找到 [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)，
> 问"要不要换成榜上更好更快的模型"。**先看榜单覆盖范围**（读它的 `constants.py`）：
> 它评的是 English + de/fr/it/es/pt/hi/nl —— **没有中文、日语、韩语、粤语**，
> 所以它回答不了我们的问题；榜上第一梯队还是 NVIDIA NeMo（Parakeet/Canary），
> 好在 sherpa-onnx 能跑 NeMo 的 onnx 导出（`from_nemo_ctc`），于是逐个核实并实测。

## 1. 候选（都是 2025~2026 的新模型，sherpa-onnx 现成可跑）

| 模型 | 体积 | 语言 | 说明 |
|---|---|---|---|
| `parakeet-ja-int8` | 626 MB | 日语专用 | NVIDIA NeMo Parakeet TDT-CTC 0.6B，官方卡 CER 6.4~13.2% |
| `dolphin-base-ctc-int8` | 99 MB | 40 种亚洲语言 + 22 种中国方言 | DataoceanAI Dolphin |
| `omnilingual-300m-ctc-int8` | 348 MB | 1600 种语言 | Meta Omnilingual ASR |
| `fire-red-asr2-ctc-zh_en-int8` | 740 MB | 中 / 英 | FireRedASR2 CTC 版（2026-02） |

全部通过 `--models install --packs ja-parakeet,dolphin,omnilingual,zh-accurate` 下载（走代理，40 MB/s）。

## 2. 怎么测的（可复现）

```powershell
# 同一批音频：Windows SAPI 合成的 11 句（中 4 / 英 2 / 日 5），因此知道正确文本
.venv\Scripts\python.exe scripts\gen_test_speech.py

# 横向对拍（同一套引擎代码、同一台机器；字准率 = 1-CER，延迟 = 语音结束→字幕）
.venv\Scripts\python.exe scripts\bench_asr_models.py --lang ja
.venv\Scripts\python.exe scripts\bench_asr_models.py --lang zh
.venv\Scripts\python.exe scripts\bench_asr_models.py --lang en
```

真实录音用 NVIDIA 仓库自带的 `test_ja_1/2.wav` + `transcripts.txt`
（存在 `data/parakeet_ja_test/`，已核对参考文本；句子是口语化会议录音，和 TTS 完全两回事）。

## 3. 结果

**A. 合成语音（11 句，干净、标准发音）**

| 语言 | 模型 | 字准率 | 计算 RTF | 延迟 |
|---|---|---|---|---|
| ja | **SenseVoice（旧默认）** | **94.8%** | 0.01 | ~400ms |
| ja | Parakeet-ja（新） | 88.1% | 0.03 | 439ms |
| ja | Dolphin（新） | 83.9% | 0.01 | 375ms |
| ja | Whisper turbo | 74.0% | 0.30 | 1138ms |
| ja | Omnilingual（新） | 71.7% | 0.04 | 457ms |
| zh | **zipformer-zh（流式，默认）** | **97.6%** | 0.06 | 361ms |
| zh | FireRedASR2-CTC（新） | 96.2% | 0.13 | 991ms |
| zh | Dolphin（新） | 93.8% | 0.01 | 397ms |
| zh | Omnilingual（新） | 86.6% | 0.05 | 596ms |
| zh | Whisper turbo | 62.5% | 0.27 | 1600ms |
| en | **zipformer-en（流式，默认）** | **98.3%** | 0.04 | 359ms |
| en | Whisper turbo | 97.8% | 0.28 | 1061ms |
| en | Omnilingual（新） | 93.3% | 0.05 | 444ms |
| en | FireRedASR2-CTC（新） | 90.5% | 0.12 | 632ms |

**B. 真实日语录音（2 句，口语化会议风格）**

| 模型 | test_ja_1 | test_ja_2 | 平均 |
|---|---|---|---|
| **SenseVoice** | 83.7% | **92.1%** | **87.9%** |
| Parakeet-ja | **85.7%** | 73.7% | 79.7% |
| Dolphin | 36.7% | 89.5% | 63.1% |

Parakeet-ja 在 `test_ja_2` 上**整句开头被吃掉**（"これはテスト文です" 没出），
字幕场景里这种"丢半句"比错几个字更难受。

## 4. 结论（和预期相反，所以写清楚）

1. **日语没有换的必要**：SenseVoice（2024-07-17 版）在两套测试上都赢
   Parakeet-ja（TTS 94.8% vs 88.1%；真实录音 87.9% vs 79.7%）。
   官方卡上那组 CER 是 **NeMo TDT 解码器**的成绩，而我们能跑的只有 sherpa-onnx 的
   **CTC 导出**，两者不是一回事——这大概就是落差的来源。
2. **中文/英文也不该换**：流式 zipformer 分别 97.6% / 98.3%，比新来的分块模型更准**且延迟低 2~3 倍**
   （361ms vs 991ms）。FireRedASR2 只在"想试方言/嘈杂音"时值得切。
3. **Whisper turbo 保留**（用户要求）：英文 97.8%、延迟 1061ms，作为 99 语言兜底仍然合格。
4. 新模型**没有删**，全部保留为**可在设置 → 模型里一键切换的候选**：
   它们不是没价值，只是"在**我们的音频**上没赢"；想换随时换，换完引擎立刻重建。
5. 韩语/粤语**未实测**：本机 SAPI 只有 zh-CN / en-US / ja-JP 音色，造不出带标注的
   韩语/粤语音频；Dolphin 的亚洲语言覆盖仍然值得留着给用户自选。
   Omnilingual（1600 语言）同理——没有小语种音频就不敢说它比 Whisper 好。

> 这条正好印证计划书第 9 节的规矩：**模型效果的结论一律以本机实测为准，不凭榜单或模型卡下判断。**

