# 耳语/轻声 VAD（Whisper-VAD ASMR）

> 记录日期：2026-09-26
> 复现：见文末第 6 节
> 代码：`app/asr/whisper_vad.py`、`tests/test_vad_whisper.py`

---

## 1. 为什么需要第二个 VAD

默认的 **Silero VAD 是按正常音量训练的**。轻声、耳语、ASMR 这类低电平人声经常被判成
"静音"，于是音频**压根不会送进识别引擎**——用户的感受是"明明在说话，字幕却什么都没有"。

这种问题**换 ASR 模型没有意义**：问题出在"哪一段算是人声"，不在"人声听成了什么"。
所以要换的是 VAD。

---

## 2. 用的是什么

[`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx)

| 项 | 值 |
|---|---|
| 结构 | whisper-base encoder（冻结）+ 2 层 transformer 解码器，**判别式 VAD** |
| 训练数据 | 约 500 小时**日语 ASMR** 录音（标签就写着 `whispered-speech` / `asmr`） |
| 输入 / 输出 | `[1, 80, 3000]`（30s log-mel）→ `[1, 1500]` 帧概率（20ms 一帧） |
| 体积 / 许可 | 119 MB / MIT |
| 依赖 | **onnxruntime**（Silero 走 sherpa-onnx 内置实现，不需要它） |

它是判别式模型，**不会像 Whisper 那样吐幻觉文本**；同时因为继承的是 whisper 编码器，
多语言也仍然可用（只是日语 ASMR 上最拿手）。

---

## 3. 为什么要自己写推理（而不是加一行注册表）

sherpa-onnx 的 `VoiceActivityDetector` **只认 silero / ten-vad 两种结构**，
这个 whisper-base 的 VAD 进不去，所以 `app/asr/whisper_vad.py` 自己实现了三件事：

1. **whisper log-mel 特征**：不引入 transformers / librosa，纯 numpy 复现
   （`n_fft=400`、`hop=160`、**periodic** Hann、`center=True` 的 reflect padding、
   STFT 去最后一帧、`log10` 下限 `1e-10`、动态范围压到 8、`(x+4)/4`）。
   少任何一条都会和模型训练时的输入对不上。
2. **滑动窗口**：官方示例是"攒满 30 秒跑一次"，对字幕等于 30 秒延迟，**不可用**。
   这里改成**每 `hop_ms` 拿最近 30 秒跑一次**，并且只采信窗口靠前的部分
   （最右侧 `MARGIN_MS=400ms` 不采信：模型看不到右边时那里的判定不稳）。
3. **双阈值滞回状态机**：`threshold` 起判、`threshold-0.15` 止判，
   再叠加最短语音 / 静音断句 / 最长憋句，行为与 Silero 版一致。

对外接口（`load / accept / pop_ready / flush / reset`）与 `VadSegmenter` 完全一致，
所以 `SherpaOfflineEngine` 不需要知道用的是哪个 VAD。

---

## 4. 实测

测试机：Ryzen 9 9950X，`provider=cpu`，**单线程**；音频为
`scripts/gen_test_speech.py` 合成的 16k 单声道日语（有正确文本可比）。

### 4.1 开销（这是它真正的代价）

| 指标 | 实测 |
|---|---|
| 模型加载 | 0.23 s |
| 单次前向（30 s 窗口，1 线程） | **0.534 s**（RTF 0.018，比实时快约 55 倍） |
| `hop_ms=1000` 时的持续占用 | **约 50% 单核**（Silero 几乎为 0） |
| 静音窗口的误报 | 概率 mean 0.027 / max 0.135（不会乱报） |

> 每次前向都要算满 30 秒窗口，所以 **hop 越小 CPU 越高**；
> 这与 Silero 那种"几乎免费"的 VAD 完全不是一个量级，别常态开着。

### 4.2 分段质量（同一套 SenseVoice 识别器，只换 VAD）

| 音频 | Silero CER | Whisper-VAD CER | 段数（Silero → Whisper） |
|---|---|---|---|
| ja_03 | 9.4% | **7.5%** | 5 → 2 |
| ja_04 | 7.9% | **0.0%** | 3 → 1 |
| ja_05 | **15.8%** | 18.4% | 2 → 1 |
| ja_06 | 3.4% | 3.4% | 3 → 3 |
| ja_07 | 5.7% | **0.0%** | 2 → 2 |
| **平均** | **8.4%** | **5.9%** | — |

**结论：在正常音量的语音上它也不掉点，反而略好**——因为它切出来的段更完整
（ja_03 从 5 段变 2 段），识别与翻译的上下文更完整。也就是说：

- 它的**质量**值得开；
- 它的**代价**（50% 单核）才是要不要开的关键；
- 而它真正的用武之地（轻声/耳语被 Silero 漏掉）本机没有带标注的耳语音频，
  **还没法用数字证明**——这一点必须说清楚，不能拿"日语正常语音没变差"冒充。

---

## 5. 怎么用

1. **下载**：向导第 ② 页勾「轻声/耳语 VAD：Whisper-VAD ASMR（可选）」，
   或命令行
   ```
   python main.py --models install --packs vad-asmr
   ```
   它不在任何"推荐档位"里——是可选包，默认不会勾。
2. **切换**：设置 → 模型 → **语音活动检测（VAD）** → 选 `Whisper-VAD ASMR`。
   未下载时会直接标出来并指出该勾哪个包。
3. **配置文件**：`asr.vad.model = "whisper-vad-asmr"`（默认 `"silero-vad"`）。
4. **前置条件**：需要 `onnxruntime`。没装时引擎会抛出**带修复命令**的提示
   （而不是一个 ImportError 堆栈），把 VAD 换回 Silero 即可继续用。

---

## 6. 调参（设置里就能调）

**设置 → 模型 → 语音活动检测（VAD） → 「轻声 VAD 步进」**（配置键 `asr.vad.hop_ms`，
范围 500~5000ms，默认 1000，选 Silero 时该控件置灰）。

它的含义：**Whisper-VAD 每隔多久重新判断一次「有没有人声」**。因为这个模型每次都要
算满 30 秒窗口，所以这个值就是「字幕快一点」和「CPU 少一点」的直接交换。
界面上会**按本机实测**把占用百分比算给用户看，超过 95% 时直接警告"跟不上实时"。

同一段 11.4 秒日语音频上的实测（CPU 单线程）：

| 步进 | 单核占用 | 末段结束 | 备注 |
|---|---|---|---|
| 500 ms | **110.8%** | 11.12 s | ⚠️ 已经跟不上实时（界面会警告） |
| **1000 ms（默认）** | **58.3%** | 11.12 s | 短音频含结尾那次推理；长音频趋近 53% |
| 2000 ms | 29.8% | 11.12 s | 省一半 CPU，断句更迟钝 |
| 3000 ms | 20.9% | 11.10 s | 空闲感明显，轻声内容才值得 |

`MARGIN_MS = 400` 是窗口右侧不采信的长度（模型看不到右边时判定不稳），
它是延迟的固定下限，不随步进变化——所以上表里末段都停在 11.12s。

> ⚠️ 踩坑：**`flush()` 必须强制再做一次推理**。正常路径只在攒够一个 hop 时才判断，
> 于是结尾不足一个 hop 的音频永远拿不到概率，最后那截语音会被整段丢掉——
> 实测 hop=2000 时第二段少了 1.4 秒。修完之后四个档位的末段结束都对齐到 11.12s。

- 复现分段对拍：
  ```powershell
  # 模型先下好（走代理快很多）
  python main.py --models install --packs vad-asmr

  # 单次前向耗时 + 流式分段 + 与 silero 的 CER 对拍
  # 脚本片段见本次会话；核心是同一段音频分别用
  # VadSegmenter 与 WhisperVadSegmenter 切段，再送同一个 SenseVoice 识别器
  .venv\Scripts\python.exe -m pytest -q tests/test_vad_whisper.py
  ```

> 踩坑：下载这个模型时**直连 hf-mirror 只有 0.3 MB/s**（119 MB 要几分钟到十几分钟），
> 挂上本机代理后 41 MB 只用了 4.3 秒。程序内下载会读 `config.proxy`，
> 记得在设置里填 `http://127.0.0.1:2333`。
