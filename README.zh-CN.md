<div align="center">

# Alice Native TTS

**在命令行或应用里运行本地 GPT-SoVITS。**

[快速开始](#快速开始) · [应用集成](#应用集成) · [性能实测](#性能实测) · [English](README.md)

</div>

Alice Native TTS 用 Rust 运行 GPT-SoVITS。`alice-tts` 可以写 WAV、通过管道向
应用交付音频，也可以常驻服务并复用已加载的模型。原生 v2Pro 路径用共享前端
资源直接读取原始 `.ckpt` 和 `.pth`。

## 快速开始

用稳定版 Rust 从当前源码安装 CLI。准备 `voice.json` 指向原始 v2Pro 权重，
或者使用包含 `pack.json` 的兼容模型包目录。[权重清单示例](docs/standalone.md#v2pro-原始权重直读)

```powershell
cargo install --locked --path . --bin alice-tts --features raw-checkpoint,server,pack-tools
alice-tts inspect ./voice.json
alice-tts synthesize ./voice.json --text "你好，我是 Alice。" --output ./speech.wav
```

直读 v2Pro 默认使用 CPU 和 FP32。CUDA 构建用 `--device cuda` 选显卡，
在清单中加入 `"native_precision": "fp16"` 可启用 FP16。

<details>
<summary>构建 Windows x64 CUDA 版</summary>

```powershell
cargo build --locked --release --features raw-checkpoint,r1-cuda-ep,server,pack-tools --bin alice-tts
./target/release/alice-tts.exe synthesize ./voice.json --device cuda --text "你好，我是 Alice。" --output ./speech-cuda.wav
```

运行时需要启用 CUDA 的 ONNX Runtime DLL，以及匹配的 CUDA 和 cuDNN 库。
Windows 实测使用 ONNX Runtime 1.30.0、CUDA 13 和 cuDNN 9。
[CUDA 运行配置](docs/standalone.md#cuda-直读)

</details>

## 应用集成

| 入口 | 应用拿到的数据 |
| --- | --- |
| `alice-tts synthesize MODEL` | 文件，或 stdout 上的 WAV / PCM 音频。`--text-file -` 从 stdin 读 UTF-8，收到 EOF 后合成。 |
| `alice-tts worker MODEL` | 常驻 JSONL 子进程，支持 `ready`、PCM 分块、完成和取消。[JSONL 协议](docs/standalone.md#jsonl-协议) |
| `alice-tts serve MODEL` | 一个已加载的模型提供 HTTP WAV 和 WebSocket PCM。[HTTP 与 WebSocket](docs/standalone.md#http-与-websocket) |
| `alice_native_tts::session` | 在 Rust 进程里使用 `Model` 和 `Session`，通过回调接收 PCM16。[Session API](src/session.rs) |

在 PowerShell 7.4 及以上版本，可将 WAV 直接送给 FFmpeg。

```powershell
alice-tts synthesize ./voice.json --text-file ./input.txt --output - |
  ffmpeg -i pipe:0 ./speech.flac
```

要边合成边读单声道 PCM16，给 `--output -` 加上 `--format s16le`。管道里的
stdout 是音频字节，诊断走 stderr，退出码表示合成是否完成。WAV 自带采样率，
完整头部写好后再交付。[命令行管道用法](docs/standalone.md#命令行管道)

## 性能实测

三轮测试都在同一张 RTX 5070 Ti（驱动 610.88）上使用相同的 v2Pro 权重、
参考音频和四条中文文本。每个后端跑两轮，每轮先预热一次，再计时四条。
Rust 使用 ONNX Runtime 1.30.0 和 CUDA 13，Python 使用 GPT-SoVITS、
PyTorch 2.7.0 和 CUDA 12.8。

从发送 JSONL 请求开始计时，到收到首个 PCM 或 `done` 结束。加载和预热
在计时前完成。表中耗时是中位数，单位为毫秒。“全部整段”汇总该后端的
八次请求。RTF 先按每条请求计算合成耗时除以实际音频时长，再取中位数。

### 最新同轮对照

| 后端 | 首包 | 短句整段 | 中等文本整段 | 全部整段 | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust CUDA FP16（优化前） | 749 | 1,598 | 3,308 | 2,432 | 0.534 |
| Rust CUDA FP16（优化后） | **390** | **733** | 2,017 | 1,318 | **0.277** |
| Rust CUDA FP32 | 413 | 747 | **1,843** | **1,287** | 0.278 |
| Python CUDA FP16 | 616 | 916 | 1,926 | 1,461 | 0.303 |

同轮测量中，FP16 的首包延迟下降 **48.0%**，全部整段耗时下降 **45.8%**。
这条流式 VITS 路径使用 cuDNN `DEFAULT` 卷积算法。对四条文本另行抓取的
Nsight trace 中，VITS 区间总耗时从 5.24 秒降到 1.65 秒，设备同步从
1,680 次降到零，GPU kernel 累计时间从 230 ms 增至 865 ms。

优化前后同为 FP16 时，最大时长差为 7.84 ms。逐窗对齐后，有声片段最低波形
相关系数为 0.9986。优化后的 FP16 与 Rust FP32 相比，两条中等文本的音频仍
分别长约 728 ms 和 188 ms。

### 之前的实测

下表是更早的两轮独立测量。旧 FP16 程序也在上方的最新对照里重新运行了一次。

| 测试轮次 | 后端 | 首包 | 短句整段 | 中等文本整段 | 全部整段 | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 首次 CUDA 直读 | Rust CUDA FP32 | 399 | 715 | 1,810 | 1,175 | 0.270 |
| 首次 CUDA 直读 | Python CUDA FP16 | 602 | 863 | 1,906 | 1,400 | 0.302 |
| 首次 FP16 测试 | Rust CUDA FP32 | 342 | 666 | 1,685 | 1,125 | 0.253 |
| 首次 FP16 测试 | Rust CUDA FP16 | 607 | 1,356 | 3,012 | 2,128 | 0.471 |
| 首次 FP16 测试 | Python CUDA FP16 | 446 | 716 | 1,357 | 1,074 | 0.236 |

<details>
<summary>四条测试文本</summary>

- 你好，今天的天气不错。
- 这杯茶已经凉了。
- 我把昨天整理的资料放在桌面上了，等你有空时，我们一起看一遍。
- 公交车还有十分钟就到。我们可以沿着这条路先走一段，到站牌旁边等。

</details>

其他路径的实测见 [ABI-1 CPU / DirectML](docs/performance.md)、
[CUDA GPT](docs/performance-r1.1.md)、[CUDA VITS](docs/performance-r1.2.md)、
[原生 TensorRT GPT](docs/performance-r2.1.md)、
[ABI-2 扩散模型](docs/performance-native-diffusion.md)及
[扩散调度](docs/performance-native-diffusion-scheduling.md)。

## 模型与运行时

| 模型输入 | 运行方式 | 音频输出 |
| --- | --- | --- |
| v2Pro 原始 `.ckpt` 和 `.pth` | Rust，启用 `raw-checkpoint` 并提供共享前端 | 32 kHz，按 token 分块 |
| v2Pro ABI-1 模型包 | Rust + ONNX Runtime | 32 kHz，有配套流式图时按 token 分块 |
| v3、v4、V5 Turbo、V5 Dev ABI-2 模型包 | Rust 调度 GPT、扩散和声码器 | 按文本片段交付，v3 为 24 kHz，其余为 48 kHz |
| v1、v2、v2Pro、v2ProPlus、v3、v4、V5 Turbo、V5 Dev checkpoint | 本机 GPT-SoVITS Python 适配器 | 流式方式和采样率由模型家族决定 |

`alice-tts pack` 从已导出的 ONNX 图组装 ABI-1 包，可读取 CMUdict、g2p_en
或已有的前端 JSON。`pack.json` 声明模型包与参考文件。使用者单独提供权重、
前端资源和参考音频。ABI-1 的 token 流式输出需要 `sovits_stream.onnx`
和匹配的元数据。[打包工具](docs/standalone.md#原生-abi-1-打包) · [流式图导出工具](tools/export-native-tts-stream.py) · [ABI-2 指南](docs/native-diffusion.md)

Windows 上的兼容 ONNX 模型包可以选择 DirectML。ORT TensorRT EP 使用
`r1-tensorrt-ep`，Windows x64 原生 TensorRT 11 GPT 使用 `r2-trt-native`，
搭配 CUDA 声码器可用 `r2-trt-native-cuda-vits`。
[原生 TensorRT 配置](docs/performance-r2.1.md) · [CUDA 声码器](docs/performance-r2.2.md) · [CUDA 文本前端](docs/performance-r2.3.md)

<details>
<summary>从宿主传入 TensorRT GPT 的 KV 长度</summary>

从 ABI-1 包生成 step 图，并在模型加载前选用它。

```powershell
python tools/prepare-native-active-len.py ./model-pack/gpt_step.onnx ./target/active-len/gpt_step.onnx
$env:ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX = (Resolve-Path ./target/active-len/gpt_step.onnx).Path
$env:ALICE_NATIVE_TTS_TRT_CACHE_DIR = Join-Path (Get-Location) 'target/active-len-engines'
cargo run --locked --release --features "r0-profiling r2-trt-native-cuda-vits" --example r0_profile -- ./model-pack ./target/active-len-profile tensor_rt_native --chunks=24 --warmups=5 --repeats=10 --cuda-vits --cuda-vits-allow-cpu
```

默认使用包内的 step 图，engine 缓存以图文件的指纹区分。

</details>

### 前端与适用范围

原生前端支持中文和英文。中文发音使用批量 G2PW，英文 G2P 使用 GRU。
文本规范化覆盖常用的数字、日期、时间、金额、百分比和温度格式。中文轻声、
儿化以及依赖词性的英文同形异音词仍处于有限覆盖范围。

ABI-1 的 GPT KV 容量为 512 token，语速为 1.0，声学噪声为零。ABI-2 使用
各家族的扩散设置。取消在推理边界协作处理。模型会话与缓存由宿主管理，
Project Alice 的桌面适配器负责提供方选择、播放队列和应用事件。

## 许可证

Apache-2.0。前端代码及数据的上游署名见 `NOTICE.md`，模型包来源和授权
记录在各自的 `LICENSES.txt`。[许可证](LICENSE) · [第三方署名](NOTICE.md)
