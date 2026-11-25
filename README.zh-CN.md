# Alice Native TTS

[English](README.md) · [项目介绍](#项目介绍) · [性能基准](#性能基准) · [使用指南](#使用指南)

## 项目介绍

Alice Native TTS 是支持 GPT-SoVITS、Irodori-TTS 和 Drama 3 的本地推理库和独立命令行工具。可以用
`alice-tts` 生成 WAV，也可以启动常驻 JSONL worker，边合成边接收 PCM。
Rust 应用可通过会话接口直接接收音频。请求支持取消，取消后可以继续使用会话。

| 输入格式 | 模型 | 音频交付 |
| --- | --- | --- |
| ABI-1 原生 ONNX 包 | v2Pro，含兼容的 v2ProPlus 权重导出 | 32 kHz PCM，有匹配的流式图时按语义 token 分块 |
| ABI-2 原生 ONNX 包 | v3、v4、V5 Turbo、V5 Dev | v3 为 24 kHz，其余为 48 kHz，按文本片段交付，V5 可开启提前出声 |
| Checkpoint 清单 | v1、v2、v2Pro、v2ProPlus、v3、v4、V5 Turbo、V5 Dev | 调用本机 GPT-SoVITS Python 运行时，采样率和流式方式随模型家族而定 |
| Irodori checkpoint 清单 | Irodori-TTS v4.1 Small | 日语合成，使用常驻本地 Python worker 按整句交付 PCM |
| Drama 3 GGUF 清单 | Drama 3 Q2 | 常驻 audio.cpp 会话按语义 token 分块输出 44.1 kHz PCM |

原生包由 Rust 和所选推理库执行。原生文本前端支持中文和英文，处理常见的
数字、日期、时间和金额格式。模型文件、参考音频及转写由使用者提供。

## 性能基准

本机使用 Windows 11、Core Ultra 9 285K 和桌面 RTX 5070 Ti 16 GB。各批为
单请求热态测量。**首包**从提交请求计到收到首个 PCM 块。**整句**计到
`done`。**RTF** 为整句耗时除以实际返回的音频时长。计时排除模型加载、
引擎构建和设备播放。

### Drama 3 Q2 · 2026-11-25

使用相同的本地 Q2_K GGUF 和 audio.cpp CUDA 运行包，每组通过 JSONL worker
计时 6 次。基准为 24 个 CPU 专家层、8 线程，调优组改用 16 层、6 线程和
`cuda_perf`。

| 中文标准句 | 首包 P50 | 估算可听起点 P50 | 整句 P50 | RTF |
| --- | ---: | ---: | ---: | ---: |
| 基准 | 179.2 ms | 449.9 ms | 3.69 s | 0.691 |
| CUDA 调优 | **150.3 ms** | **200.3 ms** | **2.82 s** | **0.490** |

调优配置多用约 2 GiB 显存。四组输入的整句加速为 1.16–1.45×，开头停顿
会影响实际出声时间。[完整输入、显存数据与复现方法](docs/performance-drama3-q2.md)

### Irodori-TTS v4.1 Small · 2026-11-25

RTX 5070 Ti 上每组测试 12 次热态请求，使用同一句日语、同一种子、
TensorRT FP16 去噪引擎和 40 步采样。两组均输出 5.04 秒的整句 PCM。

| 音频路径 | 首包 PCM | 整句完成 | RTF |
|---|---:|---:|---:|
| 原 RF 调度与 PyTorch DACVAE | 271.9 ms | 271.9 ms | 0.054 |
| 时间步复用、批量尾裁与 TensorRT DACVAE | **218.5 ms** | **218.5 ms** | **0.043** |

参考音频用例从 308.4 ms 降至 259.2 ms。四组用例的样本数和本地 Whisper 转写一致，相对 PCM RMS
误差为 0.22% 到 0.39%。[TensorRT 配置与构建](docs/standalone.md#irodori-tts-tensorrt-去噪与解码)

### V5 语义首窗 · 2026-11-25

用同一可执行文件、权重、参考音频和文本对比关闭与开启首窗。每组预热 2 次、
计时 6 次。Turbo 生成 16 个语义 token 后启动首个声学窗口，Dev 为 28 个。
Turbo 每窗 4 步，Dev 每窗 32 步、guidance 1.30。

| 模型 | 输入 | 首包 P50（ms）↓ | 开启后首包 P95（ms）↓ | 可听起点 P50（ms）↓ | 整句 P50（ms）↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| V5 Turbo | 短句 | 141.4 → **59.0** | 61.9 | 69.0 | 141.4 → 166.2 |
| V5 Turbo | 标准句 | 185.2 → **57.9** | 58.9 | 67.9 | 498.8 → 516.9 |
| V5 Turbo | 滚动长句 | 548.0 → **58.0** | 59.8 | 68.0 | 577.7 → 600.0 |
| V5 Dev | 短句 | 368.5 → **263.8** | 270.6 | 273.8 | 368.6 → 508.4 |
| V5 Dev | 标准句 | 418.6 → **262.1** | 267.0 | 272.1 | 1,186.1 → 1,382.1 |
| V5 Dev | 滚动长句 | 838.3 → **262.9** | 298.3 | 272.9 | 1,105.7 → 1,367.9 |

首包和整句按 **关闭 → 开启** 排列。六组首包 P50 加速 **1.40–9.44×**。
Turbo 首包含 300 ms 音频，Dev 含 790 ms。可听起点按 PCM 的 −45 dBFS
RMS 阈值估算。新增声学窗口会改变首个片段的波形，标准句整句分别增加
18 ms 和 196 ms。标准句后两个片段与基准逐样本一致。
[完整结果、PCM 检查与复现命令](docs/performance-v5-first-pcm.md)

### v3–V5 原生扩散 · 2026-11-24 至 25

这一批按文本片段出声，滚动长句可以在首个声学窗口结束后提前交付。各组使用
相同权重及参考音频，预热 2 次、计时 6 次。表中标准句按
**checkpoint TensorRT → 原生 TensorRT** 排列。前者由 GPT-SoVITS 调度，
后者使用 Rust 与 TensorRT 11.2，v3 的目标特征图使用 CUDA EP。

| 模型 | 首包 P50（ms）↓ | 整句 P50（ms）↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| v3 | 492 → **350** | 1,797 → 980 | 0.229 → 0.125 |
| v4 | 518 → **298** | 1,812 → 872 | 0.249 → 0.111 |
| V5 Turbo | 233 → **190** | 1,079 → 511 | 0.154 → 0.063 |
| V5 Dev | 748 → **432** | 2,781 → 1,235 | 0.397 → 0.152 |

四个模型、三种输入共 24 项首包与整句 P50，加速 **1.22–3.02×**。
[优化过程与复现](docs/performance-native-diffusion-tensorrt.md) ·
[同权重对照口径](docs/performance-native-diffusion.md) ·
[调度测试](docs/performance-native-diffusion-scheduling.md)

### Checkpoint CUDA 与 TensorRT · 2026-11-24

各家族在匹配的 GPT-SoVITS 运行时内比较 **PyTorch CUDA → checkpoint
TensorRT**。TensorRT 执行 GPT Transformer，v1 到 v2ProPlus 同时使用
TensorRT VITS 波形生成器，v3 到 V5 同时使用 TensorRT DiT Transformer
块与声码器。使用同一输入，每组预热 3 次、计时 12 次，复用已构建的引擎。

| 模型 | 首包 P50（ms）↓ | 整句 P50（ms）↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| v1 | 316 → 176 | 1,187 → 747 | 0.170 → 0.107 |
| v2 | 420 → 226 | 1,365 → 839 | 0.169 → 0.105 |
| v2Pro | 365 → 209 | 1,355 → 862 | 0.182 → 0.114 |
| v2ProPlus | 366 → 204 | 1,360 → 845 | 0.182 → 0.112 |
| v3 | 981 → 480 | 3,210 → 1,791 | 0.415 → 0.228 |
| v4 | 956 → 514 | 3,109 → 1,817 | 0.434 → 0.250 |
| V5 Turbo | 388 → 210 | 1,505 → 921 | 0.215 → 0.132 |
| V5 Dev | 1,491 → 732 | 4,751 → 2,757 | 0.679 → 0.394 |

[Checkpoint TensorRT 配置与图覆盖范围](docs/standalone.md#checkpoint-tensorrt)

### ABI-1 token 流式 · 2026-11-24

原生 32 kHz 包使用 TensorRT GPT、CUDA VITS 和独立的首段 VITS 会话。
每组预热 3 次、计时 12 次，首包均有 880 ms 音频。

| 模型包 | 首包 P50（ms）↓ | 整句 P50（ms）↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| 已有 v2Pro 包 | 94 | 804 | 0.121 |
| 新导出的 v2Pro 包 | 99 | 1,101 | 0.141 |
| 新导出的 v2ProPlus 权重包 | 103 | 1,052 | 0.135 |

早期实测还覆盖 CPU、DirectML、CUDA GPT 和新中文文本的前端开销。
在模型已加载、文本缓存未命中的 167-token 用例上，把 G2PW 与 BERT 切到
CUDA 后，前端从 139.7 ms 降到 13.1 ms，首段从 280.7 ms 降到 156.2 ms。
[CPU 与 DirectML](docs/performance.md) ·
[原生 TensorRT 与 CUDA VITS](docs/performance-r2.2.md) ·
[新文本的前端开销](docs/performance-r2.3.md)

### 同类 GPT-SoVITS 项目

同机实测 **gsv.cpp v0.1.3** 使用相同的文本、参考 WAV、扩散步数和
guidance。Alice 使用常驻 JSONL worker 和 V5 语义首窗。gsv.cpp 使用
Vulkan、FP16 AR 与 F32 声码器，通过本机 HTTP 长连接交付。首次读取含约
85 ms PCM。下表为标准句，按 **Alice / gsv.cpp** 排列。

| 模型 | 首包 P50（ms）↓ | 可听起点 P50（ms）↓ | 整句 P50（ms）↓ |
| --- | ---: | ---: | ---: |
| V5 Turbo | 57.9 / 243.6 | 67.9 / 403.6 | 516.9 / 732.8 |
| V5 Dev | 262.1 / 862.4 | 272.1 / 1,022.4 | 1,382.1 / 2,515.8 |

六组首包 P50 相对 gsv.cpp 加速 **3.08–9.36×**。滚动 Turbo 用例中，
Alice 用 600.0 ms 返回 9.12 秒音频，gsv.cpp 用 595.1 ms 返回 8.08 秒。
[同机对照报告](docs/performance-v5-first-pcm.md#同机-gsvcpp-对照)

以下公开成绩保留各项目的输入和计时口径。

- **AnifLive-TTS v1.5 · V5 Turbo** 在 RTX 5070 Ti 上测得 HTTP 长连接首包 P50 为 66.89 ms，可听起点 P50 为 80.93 ms。[HTTP 基准](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements)
- **GPT-SoVITS-cpp TensorRT 流式** 在 RTX 2080 Ti 上测得首包 460 ms、RTF 0.1205，输出约 19 秒音频。[SDK 基准](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks)
- **GPT-SoVITS 上游 v2ProPlus** 在 RTX 4090 上合成约 4 分钟音频，公开 RTF 为 0.014。[上游 README](https://github.com/RVC-Boss/GPT-SoVITS#features)

## 使用指南

### 构建命令行工具

安装稳定版 Rust。Windows 还需要 MSVC 构建工具。从仓库根目录构建。

```powershell
cargo build --locked --release --bin alice-tts
```

也可以用 `cargo install --locked --path . --bin alice-tts` 将工具安装到
命令搜索路径。

### 准备模型

原生推理使用模型包目录，直接运行 checkpoint 使用 JSON 清单。原生包包含
`pack.json`、ONNX 图、前端数据和参考数据。ABI-1 的 token 流式还需经过校验的
`sovits_stream.onnx` 和对应元数据。

- **准备 ABI-1**。使用兼容的导出文件与 GPT-SoVITS 安装生成模型包。[准备脚本](tools/prepare-native-tts-pack.py) · [流式图导出](tools/export-native-tts-stream.py)
- **导出 ABI-2**。提供本地 v3–V5 checkpoint、参考 WAV 和同权重 GPT 图。CPU 用 FP32，CUDA/TensorRT 可用 FP16。[原生扩散指南](docs/native-diffusion.md)
- **直接使用 checkpoint**。在清单里填写模型家族、权重路径、参考 WAV 和转写，再由命令行启动对应的本机运行时。[清单与依赖配置](docs/standalone.md#gpt-sovits-checkpoint)
- **运行 Irodori-TTS**。使用 v4.1 Small 权重和配套 DACVAE，可选参考 WAV 与风格描述。[Irodori-TTS 配置](docs/standalone.md#irodori-tts)
- **运行 Drama 3**。使用本地 Q2 GGUF 和 audio.cpp 0.9.4，支持 WAV 输出与 JSONL 流式输出。[Drama 3 配置](docs/standalone.md#drama-3-q2)

参考、文本及生成 token 共用原生 GPT 的 512 个缓存位置。合成长文时，按段
提交文本。

### 生成 WAV 或接收 PCM

按模型包和推理库选择原生设备。

| 原生设备 | 配置 |
| --- | --- |
| `cpu` | 默认构建，ABI-2 扩散使用 FP32 包 |
| `directml` | Windows 上的 ABI-1 实验 GPU 后端 |
| `cuda` | 用 `r1-cuda-ep` 构建并提供带 CUDA provider 的 ONNX Runtime |
| `tensor_rt_native` | 使用 TensorRT 和 CUDA runtime 构建，见下方 Windows 配置 |

以下示例使用支持 CPU 的模型包。

```powershell
./target/release/alice-tts inspect ./model-pack
./target/release/alice-tts synthesize ./model-pack --device cpu --text '你好，今天测试本地合成。' --output ./hello.wav
./target/release/alice-tts worker ./model-pack --device cpu --warmup-text '今天测试本地语音。'
```

收到 `ready` 后，从 stdin 发送 UTF-8 JSON，每行一个对象。

```json
{"op":"synthesize","request_id":"demo","text":"你好，今天测试本地合成。"}
```

stdout 的 `audio` 事件包含采样率和 Base64 编码的单声道 PCM16。每条请求
以 `done`、`cancelled` 或 `error` 结束。向当前 `request_id` 发送
`cancel` 可以取消合成。[JSONL 协议](docs/standalone.md#jsonl-协议)

通过 checkpoint 使用 CUDA PyTorch 时，将运行时路径指向本机 GPT-SoVITS。
使用 checkpoint TensorRT 时选择 `--device tensor_rt` 并准备对应的 Python
绑定和动态库。

```powershell
./target/release/alice-tts worker ./voice.json --gpt-sovits-root ./GPT-SoVITS --device cuda
```

### 原生 TensorRT 11 GPT（Windows x64）

配置 TensorRT SDK、CUDA runtime 和 CUDA NVCC 头文件目录。
`ORT_DYLIB_PATH` 指向支持 CUDA 的 ONNX Runtime DLL。将 TensorRT、
CUDA runtime、cuBLAS 与 ORT 对应的 DLL 目录加入 `PATH`。

```powershell
$env:ALICE_TRT_HOME = 'C:/SDK/TensorRT-11.2.1.2'
$env:ALICE_CUDA_HOME = 'C:/SDK/cuda_cudart-windows-x86_64-12.9.79-archive'
$env:ALICE_CUDA_NVCC_HOME = 'C:/SDK/cuda_nvcc-windows-x86_64-12.9.86-archive'
$env:ORT_DYLIB_PATH = 'C:/SDK/onnxruntime/onnxruntime.dll'
cargo build --locked --release --features r2-trt-native-cuda-vits --bin alice-tts
```

复现 V5 成绩前，准备混合 FP16 GPT opt5 缓存和声学 TensorRT plan。语义首窗
按需启用，Turbo 设为 16，Dev 设为 28。
[TensorRT plan 的构建与配置](docs/performance-native-diffusion-tensorrt.md#复现原生-tensorrt)

```powershell
$env:ALICE_NATIVE_TTS_V5_FIRST_PCM_TOKENS = '16'
./target/release/alice-tts worker ./packs/v5turbo --device tensor_rt_native
```

ABI-1 可单独导出 active-length GPT step 图，用
`ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX` 指定图路径。
[Active-length 导出脚本](tools/prepare-native-active-len.py) ·
[CUDA VITS 和独立首段会话](docs/standalone.md#原生-onnx-模型) ·
[CUDA 文本前端配置](docs/performance-r2.3.md)

### Rust 集成

使用 `alice_native_tts::session::{Model, Session, SessionOptions}` 打开模型并
加载到选定设备，从 `Session::synthesize` 的回调接收 `&[i16]`。采样率见
`Session::info()`。Project Alice 的应用适配器负责提供方选择和播放。
[会话 API](src/session.rs)

### 许可证与来源

推理代码使用 Apache-2.0 许可证，复用的前端资源保留上游署名。模型权重和
参考音频的来源与使用条款记录在包内的 `LICENSES.txt`。
[许可证](LICENSE) · [第三方资源声明](NOTICE.md)
