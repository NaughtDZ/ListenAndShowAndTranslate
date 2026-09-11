# ListenAndShowAndTranslate · 听·显·译

> **只监听你指定的那一个程序的音频**，实时生成字幕（原文 / 译文 / 双语），
> 以透明、置顶、点击穿透的悬浮窗盖在游戏画面上。
> 典型场景：一边用播放器听小说，一边打游戏——**只有小说声进识别，游戏声完全不理**。

---

## 这是什么

Windows 的声卡输出把"小说软件"和"游戏"的音频混在一起。本程序用 **WASAPI 进程级回环**
（`ActivateAudioInterfaceAsync` + PROCESS_LOOPBACK）从系统层面**只抓目标进程的音频**，
再经过 VAD → 语音识别 → 翻译 → 悬浮字幕窗，全程不需要安装虚拟声卡、不需要管理员权限。

## 功能清单

| 模块 | 状态 | 说明 |
|---|---|---|
| 选择音频来源程序 | ✅ **已可用** | 直接调用 Windows 音频会话 API（音量合成器同款），列出真正在发声的进程 |
| 进程级音频采集 | 🚧 P1 | proc-tap（WASAPI 进程回环） |
| 实时语音识别 | 🚧 P2 | sherpa-onnx 流式 zipformer（低延迟，中文优先） |
| 悬浮字幕窗 | 🚧 P3 | 透明 / 置顶 / 点击穿透 / 描边 / 多种滚动模式 |
| 翻译（API + LLM + 本地） | 🚧 P4 | 百度/有道/微软/谷歌/DeepL + OpenAI 兼容/Ollama/LM Studio |
| 字幕历史与 srt/txt 导出 | 🚧 P5 | |
| 首次运行向导 | 🚧 P5 | 硬件探测 → 推荐档位 → 一键下模型 |

进度按 `计划书.md` 的 P0–P7 推进。

---

## 快速开始

### 1. 环境要求

- Windows 10 20H1+ / Windows 11（进程回环的硬性要求）
- Python **3.12**（由 uv 管理，**不要用系统的 3.14**，见 `计划书.md` 第 1.1 节）
- [uv](https://docs.astral.sh/uv/)

### 2. 安装

```powershell
cd L:\AI_AudioAndChat\ListenAndShowAndTranslate

# 创建隔离虚拟环境（红线：一切依赖只装进 .venv）
uv venv --python 3.12.12 .venv

# 安装依赖（国内/受限网络可先设置代理）
$env:HTTP_PROXY='http://127.0.0.1:2333'; $env:HTTPS_PROXY='http://127.0.0.1:2333'
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

> ⚠️ **禁止全局 `pip install`**。所有依赖必须装在项目内的 `.venv/`。

### 3. 使用

```powershell
# 环境自检（解释器/依赖/目录/代理/GPU 一次全查）
.venv\Scripts\python.exe main.py --selftest

# 列出当前正在发声的进程（= UI 里"选择音频来源"的数据源）
.venv\Scripts\python.exe main.py --list-audio

# 或直接双击
start.bat --list-audio
```

`--list-audio` 输出示例（真实运行结果）：

```
     PID  进程名                              窗口标题
------------------------------------------------------------------------------------------
   43052  GenshinImpact.exe                原神
   23784  msedge.exe

共 2 个进程在发声。
```

---

## 技术选型一览

| 关注点 | 选择 | 理由 |
|---|---|---|
| 音频隔离 | **WASAPI 进程回环**（`proc-tap`，MIT） | 系统原生能力，零驱动、零权限、按 PID 精确隔离 |
| 发声进程枚举 | **Windows 音频会话 API**（`pycaw`） | 音量合成器同款接口，真实反映"谁在响" |
| 语音识别 | **sherpa-onnx 流式 zipformer**（首选） | 真流式、低延迟、体积小、中文友好 |
| GPU 加速 | **Vulkan + ONNX Runtime(DirectML/CUDA) + CPU** 三通道 | 单一后端无法覆盖 N/A/I 三家显卡 |
| 翻译 | 统一适配器 + 内置提示词模板 | 传统 API 与 LLM/本地模型同一接口，可自由切换 |
| 界面 | **PySide6**（Qt6） | 透明/点击穿透/置顶只有原生窗口才能可靠实现 |

---

## 已知边界（诚实说明）

- **真·独占全屏游戏无法被普通窗口覆盖**。请把游戏设为「**无边框窗口全屏**」（通常性能损失 < 2%）；
  开着 Windows「全屏优化」的"独占全屏"大多也能覆盖。
  注入式 Overlay（RTSS 那种）有反作弊封号风险，**本项目默认不做**。
- 枚举发声进程目前只覆盖**默认播放设备**上的会话（P1 完善多设备）。
- 目标程序如果是 DRM 保护音频路径，可能采不到声音（P1 实测确认）。

---

## 工程红线

1. 一切 Python 依赖只装进项目内 `.venv/`（**禁止全局 pip**）。
2. `data/`（配置、API Key、SQLite、模型、日志、导出）与 `.venv/` **永不进 Git**。
3. 导出配置时自动剔除 API Key / Secret / Token；日志中的密钥自动打码。
4. 翻译忠实原文，不做内容审查或软化。

详见 `.gitignore` 与 `计划书.md` 第 9 节。

---

## 项目文档

| 文件 | 内容 |
|---|---|
| `计划书.md` | 立项计划书：环境实测、选型、架构、里程碑与验收标准、风险登记册 |
| `docs/P0-实测记录.md` | P0 阶段的一手实测结论（含推翻假设的发现） |

## 许可证

MIT，见 `LICENSE`。
第三方组件许可见后续 `docs/第三方许可.md`。
