# Alice Native TTS

[English](README.md)

面向应用和命令行的本地 GPT-SoVITS 推理项目。Rust 原生引擎运行 v2Pro ABI-1
及 v3、v4、V5 Turbo、V5 Dev ABI-2 模型包。checkpoint 适配使用本机 GPT-SoVITS 运行时，支持 v1、v2、
v2Pro、v2ProPlus、v3、v4、V5 Turbo 和 V5 Dev。

仓库提供推理源码和模型包准备工具。模型权重和参考声线由模型包作者单独
提供，本仓库不包含这些文件。

## 独立命令行

用 `cargo build --locked --release --bin alice-tts` 构建命令行程序。它支持
模型检查、WAV 合成和常驻 JSONL 流式服务。Rust 应用可以通过
`alice_native_tts::session` 访问同一套模型加载和合成接口。使用方法、模型
清单和进程协议见[独立使用指南](docs/standalone.md)。

## 功能

- 使用版本化 ONNX 图进行原生 Rust 推理。
- 中文发音前端使用批量 G2PW；英文 G2P 使用随模型包提供的权重和 crate
  内的 GRU 推理代码。
- 支持若干数字、日期、时间、金额、百分比和温度格式的文本规范化。
- 默认使用 CPU；Windows 可显式选择实验性的 DirectML 后端。
- CUDA/TensorRT 的 GPT 路径保留 CPU 前端和声码器。另有 R1.2 文档说明的实验性流式 VITS CUDA EP 路径。GPT Step 始终禁止 CPU 回退；Encoder 默认严格放置，只有显式诊断模式才允许审计过的 CPU 辅助节点。
- 带有匹配流式图的 ABI-1 模型包按 token 块输出。ABI-2 扩散模型按文本片段输出 PCM。
- 在 ONNX 图运行之间协作式响应取消。

文本前端范围有限，不等同于 GPT-SoVITS Python 前端的完整实现。具体限制见
[当前限制](#当前限制)。

## 快速开始

安装稳定版 Rust 工具链，并准备一个兼容的模型包。下面的示例使用 ABI-1。
ABI-2 使用独立命令行或 Rust session 接口。准备方法见[模型包](#模型包)。

使用 CPU 合成 WAV：

```powershell
cargo run --release --example synthesize -- ./model-pack "你好，今天我们测试本地语音合成。" ./output.wav cpu
```

在 Windows 上显式选择 DirectML：

```powershell
cargo run --release --example synthesize -- ./model-pack "Hello, Alice." ./output.wav directml
```

CUDA 与 TensorRT EP 需要分别启用 Cargo feature；目前仅供实验，且需要与 ORT
匹配的 CUDA/TensorRT 运行库：

```powershell
cargo run --release --features r1-cuda-ep --example synthesize -- ./model-pack "你好。" ./output.wav cuda
cargo run --release --features r1-tensorrt-ep --example synthesize -- ./model-pack "你好。" ./output.wav tensor_rt
```

R1.1 CUDA GPT + CPU VITS 路径已在本机单一模型上通过性能及逐采样点音频一致性检查，但仍保持可选；
TensorRT 因缺少运行库而受阻。初始审计见 [R1.0 GPU 后端记录](docs/performance-r1-gpu.md)，
CUDA 1.30 运行时、常驻 KV Cache 与热态基准见 [R1.1 报告](docs/performance-r1.1.md)。
流式 VITS CUDA EP 的独立对照见 [R1.2 报告](docs/performance-r1.2.md)；该路径需要显式允许已审计的 CPU 辅助节点，默认仍使用 CPU VITS。

若要在指定毫秒数后取消，先传入后端，再传时长：

```powershell
cargo run --release --example synthesize -- ./model-pack "你好。" ./output.wav cpu 500
```

`stream_bench` 示例需要额外带流式图的模型包：

```powershell
cargo run --release --example stream_bench -- ./model-pack ./output cpu
```

## 模型包

ABI-1 的 `v2Pro` 模型包以 32 kHz 输出，GPT KV 容量为 512 token。
基础包包含 ONNX 图、前端数据、参考 WAV 及其文本，并须带有记录来源的
`LICENSES.txt`。`pack.json` 声明 ABI 和参考文件。启用 token 流式还需
`sovits_stream.onnx` 以及匹配的流式元数据。

可以使用 `tools/prepare-native-tts-pack.py`，从本地兼容模型导出文件和
GPT-SoVITS 安装准备模型包。Python 只用于这个离线准备步骤，运行时推理由
Rust 完成。可选流式图由 `tools/export-native-tts-stream.py` 导出并进行数值
对齐检查。

ABI-2 支持 v3、v4、V5 Turbo 和 V5 Dev。Rust 调度 GPT、目标特征、扩散步和
声码器，v3 输出 24 kHz，其余输出 48 kHz。Windows 实测路径使用原生 TensorRT
GPT 和 ONNX Runtime CUDA 声学图。离线导出、依赖及参考条件见[原生扩散指南](docs/native-diffusion.md)。

工具要求操作者自行提供模型文件和源码目录。请核对各上游模型的授权，并在
模型包中保留来源声明。本源码仓库不应放入模型权重、声线参考录音或生成的
模型包。

## API 与 Alice 集成

crate 对外提供 `Config`、`Backend`、`NativeTts`、模型包检查和文本规范化。
最简嵌入示例见 `examples/synthesize.rs`，可用 `cargo doc --open` 生成 API
文档。

Project Alice 的桌面适配器留在主项目中，负责提供方选择、共享缓存、事件、
播放队列和应用级取消；这些应用集成逻辑不属于本 crate。

## 性能说明

本机使用 Windows 11、Core Ultra 9 285K、桌面 RTX 5070 Ti 16 GB 和
TensorRT 11.2。通过常驻 `alice-tts` JSONL worker 用同一中文文本和参考音频
测试。首段和整句分别从发送请求计时到收到首个 `audio` 和 `done`。热态计时
排除模型加载、引擎构建和播放耗时。RTF 为整句耗时除以实际返回的音频时长。

### v3–V5 原生 TensorRT

标准句每组预热 2 次，计时 6 次，取 P50。每格依次为 **checkpoint TensorRT →
原生 TensorRT**，两条路径使用相同 GPT 和 SoVITS 权重。原生路径由 Rust
调度，v3 的目标特征使用 CUDA EP。V5 Turbo 扩散 4 步，V5 Dev 保持 32 步
和 guidance 1.30。

| 模型 | 首段 PCM（ms）↓ | 整句（ms）↓ | RTF ↓ |
| --- | ---: | ---: | ---: |
| v3 | 492 → **350** | 1,797 → **980** | 0.229 → **0.125** |
| v4 | 518 → **298** | 1,812 → **872** | 0.249 → **0.111** |
| V5 Turbo | 233 → **190** | 1,079 → **511** | 0.154 → **0.063** |
| V5 Dev | 748 → **432** | 2,781 → **1,235** | 0.397 → **0.152** |

V5 Turbo 短句首段 P50 为 **145 ms**。短句、标准句和滚动长句共 24 项首段与
整句 P50，相对 checkpoint TensorRT 加速 **1.22–3.02 倍**。详见
[优化结果与复现命令](docs/performance-native-diffusion-tensorrt.md)、
[checkpoint 测试口径](docs/performance-native-diffusion.md)和
[调度测试](docs/performance-native-diffusion-scheduling.md)。

V5 可选用语义首窗。标准句首包 P50，Turbo 从 185 ms 降至 **58 ms**，
Dev 从 419 ms 降至 **262 ms**。同机同输入下，gsv.cpp v0.1.3 分别为
244 ms 和 862 ms。此模式会改变首个片段的波形。见[V5 首包基准](docs/performance-v5-first-pcm.md)。

### ABI-1 token 流式

使用 TensorRT GPT 和 CUDA VITS，已有 v2Pro 模型包的标准句首段 P50 为
**94 ms**，整句为 **804 ms**。从 v2Pro 和 v2ProPlus checkpoint 导出的
原生包，首段 P50 分别为 **99 ms** 和 **103 ms**。三组首包均含 880 ms 的
音频，每组预热 3 次、计时 12 次。另见[早期 CPU 和 DirectML 结果](docs/performance.md)。

### 同类 GPT-SoVITS 项目

公开成绩保留各项目的输入场景和计时口径。

| 项目与模型 | 硬件与场景 | 首段 PCM 或首包（ms）↓ | 整句（ms）↓ | RTF ↓ | 来源 |
| --- | --- | ---: | ---: | ---: | --- |
| Alice Native TTS · V5 Turbo | RTX 5070 Ti，原生 TensorRT 语义首窗，JSONL 标准句 | 57.85 | 516.90 | 0.0649 | [本机基准](docs/performance-v5-first-pcm.md) |
| AnifLive-TTS v1.5 · V5 Turbo | RTX 5070 Ti，TensorRT 11 / Triton FP16，HTTP 长连接 PCM | 66.89 | 75.98 | 0.0473 | [AnifLive HTTP 基准](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements) |
| gsv.cpp v0.1.3 · V5 Turbo | RTX 5070 Ti，Vulkan，HTTP 长连接 PCM | 174.45 | 197.06 | 0.1120 | [AnifLive 的 gsv.cpp 实测](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements) |
| GPT-SoVITS-cpp | RTX 2080 Ti，TensorRT FP16 流式，约 19 秒混语音频 | 460 | — | 0.1205 | [SDK 基准](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS 上游 · v2ProPlus | RTX 4090，约 4 分钟音频 | — | — | 0.014 | [上游 README](https://github.com/RVC-Boss/GPT-SoVITS#features) |

三组 V5 Turbo 使用相同型号的 GPU，文本、声线和交付协议各异。Alice 的标准句
首包有 0.30 秒 PCM，整句有 7.96 秒音频。HTTP 基准每种模式测了 1,000 次
AnifLive 和 300 次 gsv.cpp 请求并取中位数，分别记录首包和可听起点。

## 当前限制

- 原生模型包支持中文、英文。
- ABI-1 导出图的语速固定为 1.0，声学噪声固定为 0.0。ABI-2 按模型家族固定扩散步数、初始噪声和 guidance。
- Alice 适配器的进程内缓存只保留一种基础声线配置；crate 本身将缓存所有权
  留给宿主程序。
- 取消只能在 ONNX 图运行之间协作处理，不能强制中断正在执行的 ONNX kernel。
- 中文轻声/儿化、依赖词性的英文同形异音词、日语和完整地区化数字语法尚未
  覆盖。
- 长文本长度受导出的 GPT KV 缓存容量限制。

## 许可证与来源

crate 使用 Apache-2.0。中文发音表、tokenizer 数据和拼音拆分代码的上游来源
见 [NOTICE.md](NOTICE.md)。模型包组件各自保有来源和授权信息，请查阅模型包的
`LICENSES.txt`。
