# GPT-SoVITS v3–V5 原生扩散包

ABI-2 由 Rust 调度 GPT、目标特征解码、扩散和声码器。CLI 的 `inspect`、
`synthesize`、`worker` 与 `alice_native_tts::session` 都可以加载该格式。
每段文本完成声码器推理后交付 PCM16。采样率由模型决定。

| `family` | 输出 | 去噪步数 | Guidance | 扩散窗口上限 |
| --- | ---: | ---: | ---: | ---: |
| `v3` | 24 kHz | 32 | 0 | 934 帧 |
| `v4` | 48 kHz | 32 | 0 | 1,000 帧 |
| `v5turbo` | 48 kHz | 4 | 0 | 1,000 帧 |
| `v5dev` | 48 kHz | 32 | 1.30，串行 CFG | 1,000 帧 |

## 离线导出

1. 按[独立使用指南](standalone.md#gpt-sovits-checkpoint)准备一个 checkpoint
   JSON、对应的 GPT-SoVITS 源码和 Python 环境。加载 V5 时使用支持相应变体的运行时。
   安装 `torch`、`onnx` 和该运行时所需的声码器、BERT 和 HuBERT 依赖。
2. 提供一个使用同一 GPT 权重导出的 ABI-1 图目录。导出器复用其 BERT、
   G2PW、英语词典和 GPT 图，校验 GPT 图学习参数的张量摘要，并生成接收
   host shape active length 的 step 图。已有 ABI-1 图仍须符合 512-token KV 接口。
3. 在新目录生成 ABI-2 包。运行时会使用包内的参考语义 token、说话人条件、
   提示 mel 和位置编码，换参考音频时重新导出。

```powershell
$GptSovitsPython = 'D:/GPT-SoVITS/runtime/python.exe'
& $GptSovitsPython -I -B tools/export-native-diffusion.py `
  --checkpoint ./v4.json --runtime D:/GPT-SoVITS `
  --gpt-pack ./gpt-graphs --output ./output/v4-native --precision fp16
```

`fp16` 使用 CUDA 进行导出，`fp32` 使用 CPU。导出时会用固定张量对照上游
目标特征和单步更新，`pack.json` 记录相对 L2、最大绝对误差和 GPT、SoVITS、
参考 WAV 的 SHA-256。四张声学图分别是 `target_features.onnx`、
`dit_condition.onnx`、`dit_step.onnx` 和 `vocoder.onnx`。保留运行时及模型
资产各自的许可与来源说明。

## 选择设备

| `--device` | 构建要求 | 推理路径 |
| --- | --- | --- |
| `cpu` | 默认构建，FP32 包 | ONNX Runtime CPU |
| `cuda` | `r1-cuda-ep`，ORT CUDA 运行库 | ONNX CPU GPT，ORT CUDA 声学图 |
| `tensor_rt_native` | Windows x64，`r2-trt-native-cuda-vits`，TensorRT 和 ORT CUDA 运行库 | 原生 TensorRT GPT，ORT CUDA 声学图 |

配置 SDK 和 DLL 路径的方法与 [ABI-1 TensorRT 构建](../README.md#native-tensorrt-11-gpt-windows-x64)相同。
ABI-2 使用包内的 active-length step 图。运行推理只需要 Rust 可执行文件、模型包和
对应运行库。

```powershell
cargo build --locked --release --features r2-trt-native-cuda-vits --bin alice-tts
./target/release/alice-tts inspect ./output/v4-native
./target/release/alice-tts synthesize ./output/v4-native --device tensor_rt_native `
  --text '今天测试本地语音。' --output ./output/v4-native.wav
./target/release/alice-tts worker ./output/v4-native --device tensor_rt_native
```

JSONL 握手中的 `streaming` 为 `text_fragments`，`default_language` 为
`auto`。原生前端支持中文和英文。每个片段的前缀、文本及生成 token 共用
512-token GPT KV 容量。若缓存先于结束 token 耗尽，该片段会返回错误。
窗口内的去噪状态在 CUDA 上复用，窗口间按家族更新提示条件。
取消发生在图执行边界，调用方可以继续使用同一会话。

复现真实 PCM 的方法见[调度测试](performance-native-diffusion-scheduling.md)。
首段延迟、整段时间、RTF 及 PCM 对比见[同权重 checkpoint 对比](performance-native-diffusion.md)。
