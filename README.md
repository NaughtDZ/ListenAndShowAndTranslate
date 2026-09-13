# ListenAndShowAndTranslate · 听·显·译
<img width="1047" height="1680" alt="image" src="https://github.com/user-attachments/assets/26cb7906-e715-4ec5-b7ea-3e4c0e111599" />

> **只监听你指定的那一个程序的音频**，实时生成字幕（原文 / 译文 / 双语），
> 以透明、置顶、点击穿透的悬浮窗盖在游戏画面上。
> 典型场景：一边用播放器听小说，一边打游戏——**只有小说声进识别，游戏声完全不理**。

Windows 的声卡输出把"小说软件"和"游戏"的音频混在一起。本程序用 **WASAPI 进程级回环**
（`ActivateAudioInterfaceAsync` + PROCESS_LOOPBACK）从系统层面**只抓目标进程的音频**，
再经过 VAD → 语音识别 → 翻译 → 悬浮字幕窗，全程不需要虚拟声卡、不需要管理员权限、
不做任何游戏注入（不碰反作弊）。

---

## 功能清单（当前实际状态）

| 模块 | 状态 | 说明 |
|---|---|---|
| 选择音频来源程序 | ✅ 已可用 | Windows 音频会话 API（音量合成器同款），只列**真正在发声**的进程 |
| **浏览器标签页音频** | ✅ 已可用 | 只听**指定的那一个标签页**，别的标签页照常有声；由一个很小的浏览器扩展经本机回环（127.0.0.1）送进来，识别链路复用同一套管线。原理/实测/安装见 `docs/浏览器标签页.md` |
| 进程级音频采集 | ✅ 已可用 | `proc-tap`（WASAPI 进程回环）+ 降混/重采样；隔离性 99.90% 实测见 `docs/P1-实测记录.md` |
| 实时语音识别 | ✅ 已可用 | sherpa-onnx：流式 zipformer（中/英）+ SenseVoice（中日韩粤，实测日语最优）+ Whisper turbo 兜底；**可按语言手动指定模型**（只列能用的）；2026-09 新增 Parakeet-ja / Dolphin / Omnilingual / FireRedASR2 四个候选，实测对拍见 `docs/P2-ASR实测.md` 末节 |
| 悬浮字幕窗 | ✅ 已可用 | 透明 / 置顶 / 点击穿透 / 描边 / 字体自适应 / 可拖拽缩放；见 `docs/字幕窗尺寸与字号.md` |
| 翻译（API + LLM + 本地） | ✅ 已可用 | 5 家官方 API + 谷歌/必应网页版 + OpenAI 兼容（LM Studio/Ollama）+ 术语表 + 缓存；**思考默认关闭**（按服务端类型选参数 + 400 降级 + `/no_think` 兜底），见 `docs/P4-翻译实测.md` |
| 静音阈值与电平表 | ✅ 已可用 | 启动窗口独立电平表窗口 + 设置里内嵌实时电平表；见 `docs/静音阈值与电平表.md` |
| 首次运行向导 | ✅ 已可用 | 硬件探测 → 推荐档位 → 一键下模型 → 代理/翻译通道配置；**随时可重跑换模型**，见 `docs/向导与重新配置.md` |
| 系统托盘 / 窗口生命周期 | ✅ 已可用 | 控制窗可收进托盘，退出行为有明确约定；见 `docs/窗口与退出行为.md` |
| 前端 exe（无控制台黑窗） | ✅ 已可用 | `听显译.exe`，见下面「启动方式」 |
| 字幕历史与 srt 导出 | 🚧 部分 | 数据模型与 `to_srt()` 已有，UI 入口待接线 |
| 全局热键 / 安装包 | ⏳ 未开始 | 计划书 P6/P7 |
<img width="1070" height="1687" alt="image" src="https://github.com/user-attachments/assets/044a48c2-6efe-4659-90bc-ff69b8d98249" />
<img width="1048" height="1682" alt="image" src="https://github.com/user-attachments/assets/f8808d8d-c1b4-474e-9d72-eec48775f29d" />
<img width="898" height="288" alt="image" src="https://github.com/user-attachments/assets/6904b2ca-4a52-4979-b416-c188297f07cd" />

进度按 `计划书.md` 的 P0–P7 推进（第 11 节有逐项产出与实测数据）。

---

## 两种听法

| 想听什么 | 怎么用 | 需要什么 |
|---|---|---|
| **某个程序**（小说软件 / 播放器 / 游戏语音） | 主窗口选中它 → 开始字幕 | 什么都不用装 |
| **浏览器里的某一个标签页** | 主窗口选「浏览器标签页」→ 开始字幕 → 在浏览器里切到目标标签页 → 点扩展图标（或 Ctrl+Shift+U） | 侧载一个很小的扩展（`browser_extension/`，主窗口有安装说明） |

为什么要装扩展：浏览器把整个实例的音频混成一个流，**操作系统层根本分不出标签页**
（实测两个标签页同时出声，音频会话仍只有 1 个），所以标签页边界只能由浏览器内部提供。
扩展把目标标签页的音频经 `127.0.0.1` 送进来，**别的标签页照常有声**。
细节与全部实测数据：`docs/浏览器标签页.md`。

---

## 启动方式（三个入口，随便挑一个）

| 入口 | 有没有控制台窗口 | 说明 |
|---|---|---|
| **`听显译.exe`**（推荐） | **没有** | 前端启动器：检查环境 → 用 `pythonw.exe` 拉起主程序 → 自己立刻退出。也能带参数：`听显译.exe --settings`、`听显译.exe --list-audio`、`听显译.exe --wizard` |
| `启动.bat` | 有（会一直留一个黑框） | 命令行入口，方便看日志/传参数；首次装环境也走它 |
| `首次安装.bat` | 有 | 只负责创建 `.venv` 并装依赖（`启动.bat` 发现没环境时也会引导过来） |

> ⚠️ `听显译.exe` **不是独立发行版**：它是个几十 KB 逻辑的前端壳（打包后约 8 MB），
> 真正跑字幕的仍然是项目里的 `.venv` + `main.py`，模型也还在 `data/models/`。
> 所以先按下面「安装」把环境装好，再双击 exe。
> exe 未做代码签名，Windows SmartScreen 可能提示"未知发布者"，选「仍要运行」即可；
> 也可以随时用 `构建exe.bat` 自己重新生成（源码就是 `frontend.py`）。

### 重新打包 exe

```powershell
构建exe.bat
# 等价于：.venv\Scripts\python.exe scripts\build_exe.py
```

打包用 PyInstaller（开发依赖，装在 `.venv` 里）。产物覆盖根目录的 `听显译.exe`；
中间文件在 `data/build/`（已被 `.gitignore` 忽略）。

---

## 安装（Windows / Python 3.12）

```powershell
git clone https://github.com/NaughtDZ/ListenAndShowAndTranslate.git
cd ListenAndShowAndTranslate

# 一键：建 .venv（uv 托管的 Python 3.12.12）并安装依赖
.\首次安装.bat
```

手动等价命令（**禁止全局 pip**，红线见下）：

```powershell
uv venv --python 3.12.12 .venv

# 国内/受限网络：uv 走环境变量代理（注意 uv pip 没有 --proxy 参数）
$env:HTTP_PROXY='http://127.0.0.1:2333'; $env:HTTPS_PROXY='http://127.0.0.1:2333'
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

- 必须是 **Python 3.12**：`proc-tap` 与 `sherpa-onnx` 没有 3.13+ 的轮子
  （系统只装了 3.14 也没关系，uv 会自己拉一个 3.12）。
- 首次启动会自动下载识别/翻译模型到 `data/models/`（向导里可以选档位与体积）。

---

## 使用

双击 `听显译.exe` → 在列表里选**正在播放**的小说软件 → 「开始字幕」。
（浏览器/Electron 类应用有多个同名子进程，选错了会一直静音，换一个同名项试试。）

字幕窗：**拖动任意位置**移动，**拖边缘/角**缩放；控制窗里可切原文/译文/双语、
点击穿透、暂停，并可「最小化到托盘」（关闭控制窗才会退出程序，见
`docs/窗口与退出行为.md`）。

### 想换识别模型 / 补下载语言包？

三个入口，随便挑一个（细节见 `docs/向导与重新配置.md`）：

| 入口 | 什么时候用 |
|---|---|
| 启动窗口 →「**向导…**」按钮 | 还没开始出字幕时，最直观 |
| 设置 → **模型** →「**重新运行「首次运行向导」…**」 | 字幕正在跑时也能用；完成后自动重建识别/翻译引擎，**不用重启** |
| `听显译.exe --wizard` | 脚本化 / 排障 |

重跑向导时**默认值就是你的现状**：语言包默认勾"已经装好"的那些（提示会写清
"其中 N 个已经装好，会自动跳过"）、档位取你当前的延迟档位、代理与翻译通道按
现状预填。所以"只想再补一个日语包"不会顺手把代理和翻译通道清掉；
向导里表示不了的官方 API 通道（百度/有道/Azure/谷歌/DeepL）会显示成
「保持当前：…」原样保留。

**具体用哪个模型**则在 设置 → **模型** →「**识别模型（按语言）**」里选：
每个语言一行，候选只有**注册表里真的能识别这门语言**的那些模型
（语种识别模型、VAD、中文专用模型都不会出现在日语那一行），
所以不存在"下了个模型却用错地方"这回事。详见 `docs/识别模型选择.md`。

### 命令行用法

```powershell
# 环境自检（解释器/依赖/目录/代理/GPU 一次全查）
.venv\Scripts\python.exe main.py --selftest

# 列出当前正在发声的进程（= UI 里"选择音频来源"的数据源）
.venv\Scripts\python.exe main.py --list-audio

# 采集目标程序的音频，实时电平表
.venv\Scripts\python.exe main.py --capture 43052 --seconds 10
.venv\Scripts\python.exe main.py --capture 喜马拉雅.exe --seconds 10

# 打开电平表悬浮窗（RMS/PEAK 条 + 峰值保持 + 可调静音阈值）
.venv\Scripts\python.exe main.py --meter 43052
.venv\Scripts\python.exe main.py --meter 43052 --threshold-db -70

# 直接开设置窗 / 重跑配置向导 / 录一段 16k 单声道 WAV
.venv\Scripts\python.exe main.py --settings
.venv\Scripts\python.exe main.py --wizard
.venv\Scripts\python.exe main.py --capture 43052 --record test.wav --seconds 3

# 模型管理（不走界面）：列出 / 安装指定语言包 / 校验
.venv\Scripts\python.exe main.py --models list
.venv\Scripts\python.exe main.py --models install --packs zh,ja-ko-yue
```

`--list-audio` 输出示例（真实运行结果）：

```
     PID  进程名                              窗口标题
------------------------------------------------------------------------------------------
   43052  GenshinImpact.exe                原神
   23784  msedge.exe

共 2 个进程在发声。
```

`--capture` 实时电平示例（真实运行结果）：

```
目标: python.exe  (PID 45008)
可执行文件: ...\python.exe
开始采集，实时电平:

  [################################········] rms=0.2831 peak=0.400 采集中 有声 块=341 pid=45008
共 342 块，输出 54316 样本 @16k
峰值 0.4000，末次 RMS 0.2810，重连 0 次
```

### 验收与调试脚本

```powershell
# 隔离性：验证不会混入其他程序的声音
.venv\Scripts\python.exe scripts\verify_process_isolation.py

# 保真性：验证采集→降混→16k 整条链路不破坏声音
.venv\Scripts\python.exe scripts\verify_capture_e2e.py

# 会话音量/静音对采集的影响（自动控制音量，只调小不调大）
.venv\Scripts\python.exe scripts\verify_session_volume_effect.py --pid <音频会话PID>

# 没有真实音频时，用测试音演示电平表 / 渲染成 PNG
.venv\Scripts\python.exe scripts\demo_meter.py
.venv\Scripts\python.exe scripts\preview_meter.py
.venv\Scripts\python.exe scripts\preview_subtitle.py

# 生成测试语音 + ASR 模型横向对拍（换模型前必跑，同一批音频/同一套引擎）
.venv\Scripts\python.exe scripts\gen_test_speech.py
.venv\Scripts\python.exe scripts\bench_asr_models.py --lang ja

# 单元测试（350+ 项，含 Qt offscreen UI 用例）
.venv\Scripts\python.exe -m pytest -q
```

---

## 技术选型一览

| 关注点 | 选择 | 理由 |
|---|---|---|
| 音频隔离 | **WASAPI 进程回环**（`proc-tap`，MIT） | 系统原生能力，零驱动、零权限、按 PID 精确隔离 |
| 发声进程枚举 | **Windows 音频会话 API**（`pycaw`） | 音量合成器同款接口，真实反映"谁在响" |
| 语音识别 | **sherpa-onnx**：流式 zipformer / SenseVoice / Whisper | 真流式、低延迟、体积小、中文友好，全本地推理 |
| GPU 加速 | **Vulkan + ONNX Runtime(DirectML/CUDA) + CPU** 三通道 | 单一后端无法覆盖 N/A/I 三家显卡 |
| 翻译 | 统一适配器 + 内置提示词模板 | 传统 API 与 LLM/本地模型同一接口，可自由切换 |
| 界面 | **PySide6**（Qt6） | 透明/点击穿透/置顶只有原生窗口才能可靠实现 |
| 前端入口 | **PyInstaller 单文件启动器** | 双击即开，不留控制台黑窗 |

---

## 已知边界（诚实说明）

- **真·独占全屏游戏无法被普通窗口覆盖**。请把游戏设为「**无边框窗口全屏**」（通常性能损失 < 2%）；
  开着 Windows「全屏优化」的"独占全屏"大多也能覆盖。
  注入式 Overlay（RTSS 那种）有反作弊封号风险，**本项目默认不做**。
- 枚举发声进程目前只覆盖**默认播放设备**上的会话（多设备切换待完善）。
- 目标程序如果是 DRM 保护音频路径，可能采不到声音（P1 实测确认）。
- **别在音量合成器里把目标程序静音**：进程回环采集发生在音量合成器"之后"，
  静音会直接切断信号（实测采集到精确的 0）；调小音量则会等比变小，
  程序里有数字增益可以补偿。
- `听显译.exe` 是**前端壳**，不是脱离 Python 的独立发行版；未签名，可能触发 SmartScreen 提示。
- 字幕准确率取决于模型与音源质量，实测数据都在 `docs/` 里，不做夸大。

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
| `docs/P1-实测记录.md` | 音频链路实测：进程隔离、会话音量、静音判定、电平表 |
| `docs/P2-ASR实测.md` | 各识别模型的中/英/日准确率与 RTF 实测；**含 2026-09 新旧模型横向对拍**（结论：新的不一定更好） |
| `docs/P4-翻译实测.md` | 翻译通道实测：LLM 批量、网页通道、官方 API 的可用性与耗时 |
| `docs/延迟调节.md` | 延迟旋钮说明：每个参数换多少延迟、换多少准确率 |
| `docs/显存与模型预算.md` | 显存占用、模型体积、量化档位、嵌入模型上下文与 KV 缓存预算 |
| `docs/窗口与退出行为.md` | 谁能关掉程序、托盘/设置窗/✕ 的行为，以及 Qt 的退出坑 |
| `docs/字幕窗尺寸与字号.md` | 拖边框缩放、自动高度（不留空行）、放不下时动态缩小字号 |
| `docs/静音阈值与电平表.md` | 静音阈值的实测参考、三个调节入口、设置里的实时电平表 |
| `docs/向导与重新配置.md` | 换模型/补下载的三个入口，以及"重跑不许把设置清回默认值"的约定 |
| `docs/识别模型选择.md` | 哪些模型能识别哪些语言、界面怎么限制、运行时怎么兜底 |
| `docs/第三方许可.md` | 依赖许可清单（LGPL/GPL 组件与商用注意点） |

## 目录结构（主要部分）

```
main.py              命令行入口（--selftest / --list-audio / --run / --meter / --settings / --wizard）
frontend.py          前端启动器源码（打包成 听显译.exe）
app/audio/           进程回环采集、音频管线、电平统计
app/asr/             识别引擎抽象、VAD、流式/离线引擎、语言路由
app/translate/       翻译适配器（官方 API / 网页通道 / OpenAI 兼容）、提示词、术语表
app/models/          模型注册表、下载器、硬件探测
app/ui/              悬浮字幕窗、控制窗、电平表、设置窗、向导、启动器
scripts/             验收脚本 + 出图预览 + 打包脚本
tests/               300+ 项 pytest（含 Qt offscreen 用例）
docs/                一手实测记录与专题说明
```

## 许可证

MIT，见 `LICENSE`。
第三方组件许可（PySide6 的 LGPL、libsndfile 等）见 `docs/第三方许可.md`。
