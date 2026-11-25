# Alice TTS 独立使用

构建 Rust 库和命令行程序。

```powershell
cargo build --locked --release --bin alice-tts
cargo install --locked --path . --bin alice-tts
```

Windows x64 运行包解压后可直接运行。系统需要 DirectML 和 Microsoft Visual C++ v14 x64 运行时。模型文件和 checkpoint 运行时由调用方提供。[DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) · [VC++ 运行时](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist#latest-supported-redistributable-version)

| 模型 | 推理路径 | 流式输出 |
| --- | --- | --- |
| ABI-1 v2Pro ONNX | Rust + ONNX Runtime，支持可选 GPU 后端 | 有匹配流式图时分 token 块输出 |
| ABI-2 v3、v4、V5 Turbo、V5 Dev ONNX | Rust + ONNX Runtime，可选原生 TensorRT GPT | 按文本片段输出，v3 为 24 kHz，其余为 48 kHz |
| GPT-SoVITS v1、v2、v2Pro、v2ProPlus checkpoint | 本机 GPT-SoVITS Python 子进程 | 分 token 块输出 |
| GPT-SoVITS v3、v4 checkpoint | 本机 GPT-SoVITS Python 子进程 | 按文本片段输出 |
| GPT-SoVITS V5 Turbo、V5 Dev checkpoint | 支持 V5 的本机 GPT-SoVITS Python 子进程 | 按文本片段输出 48 kHz PCM |
| Irodori-TTS v4.1 Small checkpoint | 本机 Irodori-TTS Python 子进程，支持 CPU 和 CUDA | 整句日语 PCM |
| Drama 3 Q2 GGUF | 本机 audio.cpp 0.9.4，支持 CPU 和 CUDA | 按语义 token 分块输出 44.1 kHz PCM |

## 原生 ONNX 模型

沿用已有 ABI-1 包目录。包有流式图时，命令行自动启用流式接口。ABI-2 使用同一套命令和 Rust session 接口，导出及运行依赖见[原生扩散指南](native-diffusion.md)。

```powershell
alice-tts inspect ./model-pack
alice-tts synthesize ./model-pack --text '你好，今天测试本地合成。' --output ./hello.wav
alice-tts worker ./model-pack --warmup-text '今天测试本地语音。'
```

默认设备为 CPU。原生 CUDA 构建使用 `r1-cuda-ep`，原生 TensorRT GPT + CUDA VITS 构建使用 `r2-trt-native-cuda-vits`。给命令加 `--device tensor_rt_native` 即可选择已编入的原生 TensorRT 路径。原生路径的 CUDA VITS 和首段 VITS 独立会话仍通过 `ALICE_NATIVE_TTS_CUDA_VITS=1` 和 `ALICE_NATIVE_TTS_FIRST_VITS_SESSION=1` 显式启用。

## GPT-SoVITS checkpoint

准备本机 GPT-SoVITS 安装与该版本所需的基础权重、分词模型和声码器。Alice TTS 调用安装里的 `GPT_SoVITS/TTS_infer_pack/TTS.py`。Windows 优先使用安装目录的 `runtime/python.exe`，也可以通过 `--python` 指定已经安装其依赖的解释器。[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

将模型和参考音频写进一个 JSON 清单。文件路径可以是绝对路径，也可以相对于清单所在目录。`bert` 和 `hubert` 可选，默认使用 GPT-SoVITS 安装自带的中文 BERT 与 HuBERT 目录。`family` 必须与实际加载的 SoVITS 模型一致。

```json
{
  "schema_version": 1,
  "family": "v4",
  "gpt": "./weights/s1v3.ckpt",
  "sovits": "./weights/s2Gv4.pth",
  "reference_audio": "./reference.wav",
  "reference_text": "这里填写参考音频的实际转写。",
  "reference_language": "zh",
  "default_language": "zh"
}
```

```powershell
alice-tts inspect ./voice.json
alice-tts synthesize ./voice.json --gpt-sovits-root 'D:/GPT-SoVITS' --device cuda --text '你好，今天测试本地合成。' --output ./hello-v4.wav
alice-tts worker ./voice.json --gpt-sovits-root 'D:/GPT-SoVITS' --device cuda
```

`--device cpu` 使用 PyTorch CPU。`--device cuda` 使用当前 Python 环境里的 CUDA PyTorch。v3 的典型输出为 24 kHz，v4 的典型输出为 48 kHz。调用方使用 `ready.model.sample_rate` 或 `audio.sample_rate` 确认实际采样率。

### Checkpoint TensorRT

使用 `--device tensor_rt` 启用 checkpoint TensorRT 后端。所选 Python 需要 CUDA PyTorch、ONNX 和 TensorRT Python 绑定，系统需要对应的 TensorRT、CUDA 运行库。Windows 上可让 `ALICE_TRT_HOME` 指向 SDK 目录。如果 Python 绑定放在独立目录，可将目录路径赋给 `ALICE_NATIVE_TTS_TRT_PYTHON_PATH`。

```powershell
$env:ALICE_TRT_HOME = 'C:\TensorRT-11.2.1.2'
alice-tts worker ./voice.json --gpt-sovits-root 'D:/GPT-SoVITS' --device tensor_rt
```

各家族使用 TensorRT 执行 GPT Transformer 的预填充和逐 token 解码。v1、v2、v2Pro 和 v2ProPlus 同时使用 TensorRT 执行 VITS 波形生成器。v3、v4 和 V5 同时使用 TensorRT 执行 DiT Transformer 块与声码器。文本和参考音频处理、GPT 采样、VITS 编码和扩散调度由本机 GPT-SoVITS 运行时执行。V5 的 `tensor_rt` 路径使用完整 DiT 块栈，并保留 Turbo 4 步和 Dev 32 步的采样设置。

首次加载时会从当前权重导出并构建引擎。构建时间计入首次启动，后续启动会复用与权重、GPU 和 TensorRT 版本匹配的引擎。默认缓存目录在 Windows 上为 `%LOCALAPPDATA%/alice-native-tts/tensorrt`。可通过 `ALICE_NATIVE_TTS_CHECKPOINT_TRT_CACHE_DIR` 指定缓存目录。引擎面向单请求合成，GPT 预填充最多支持 2,048 token，解码输入缓存最多支持 4,096 token，波形生成器最多支持 4,096 帧。

## GPT-SoVITS V5

V5 清单的 `family` 取值为 `v5turbo` 或 `v5dev`，分别对应 `s2Gv5turbo.pth` 和 `s2Gv5dev.pth`。GPT 路径可指向官方 `s1v3.ckpt`。将 V5 声码器放在 V5 安装的 `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth`，并把 `--gpt-sovits-root` 指向支持 V5 的 GPT-SoVITS 安装。[V5 权重](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained) · [GPT-SoVITS V5 说明](https://github.com/RVC-Boss/GPT-SoVITS#v5-release-notes)

```powershell
alice-tts worker ./voice-v5turbo.json --gpt-sovits-root 'D:/GPT-SoVITS-V5' --device cuda
alice-tts synthesize ./voice-v5dev.json --gpt-sovits-root 'D:/GPT-SoVITS-V5' --device cuda --text '你好，今天测试 V5。' --output ./hello-v5.wav
```

两种 V5 模型均输出 48 kHz PCM，按上游的文本片段接口交付音频。Turbo 使用上游的 4 步采样默认值，Dev 使用 32 步采样默认值。运行时加载后会核对清单里的家族与实际权重。


## Irodori-TTS

在本机准备 Irodori-TTS 源码和依赖。下面按上游的 Windows CUDA 安装方式配置
本次使用的源码版本。CPU 环境将 `--extra cu128` 改为 `--extra cpu`。
[上游安装说明](https://github.com/Aratako/Irodori-TTS/blob/f01c08cd40c0afa62a862439a5de12f6dcf62054/README.md)

```powershell
git clone https://github.com/Aratako/Irodori-TTS.git
git -C ./Irodori-TTS checkout f01c08cd40c0afa62a862439a5de12f6dcf62054
uv sync --project ./Irodori-TTS --python 3.11 --extra cu128
```

下载 checkpoint 和 codec 权重，在 `irodori.json` 中填写路径。将模型仓库的
`tokenizer/` 一并放在 `model.safetensors` 旁边。相对路径以清单所在目录为准。
省略 `codec_weights` 时由上游使用默认的 DACVAE Hub 模型。首次启动还会缓存上游
的 SilentCipher 音频水印权重。
[v4.1 Small 权重](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) ·
[日语 DACVAE 权重](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)

```json
{
  "schema_version": 1,
  "family": "irodori",
  "runtime_root": "./Irodori-TTS",
  "checkpoint": "./irodori-model/model.safetensors",
  "codec_weights": "./irodori-codec/weights.pth",
  "caption": "落ち着いた自然な声",
  "seed": 1234
}
```

风格描述 `caption` 和随机种子 `seed` 都可省略。可添加
`"reference_audio": "./reference.wav"` 使用本机参考音频，也可同时提供参考与
风格描述。程序优先使用 `runtime_root` 中的 `.venv`，可用 `--python` 指定
其他已配置的解释器。

```powershell
alice-tts inspect ./irodori.json
alice-tts synthesize ./irodori.json --device cuda --language ja --text 'こんにちは。今日は音声合成のテストです。' --output ./irodori.wav
alice-tts worker ./irodori.json --device cuda
```

worker 复用已加载的推理模型，`ready.model.streaming` 为 `utterance`。每次
合成返回一个单声道 PCM16 音频块，采样率由 codec 决定。支持 `ja` 和
`auto`。取消会结束本次 Python 子进程，下一次请求重新加载运行时。

### Irodori-TTS TensorRT 去噪与解码

可为 CUDA 清单添加 `"denoiser_plan": "./irodori-fp16.plan"`。它使用 TensorRT
执行 RF-DiT 前向，沿用官方时长预测、40 步采样、CFG、DACVAE 和 SilentCipher。
时长预测和采样复用同一次条件编码，常驻 worker 还可复用相同风格描述的编码。
有效条件的 K/V 在每次合成中打包复用。超出引擎形状范围的步骤走 PyTorch，
并在 stderr 记录回退。输出仍为整句 PCM。FP16 与 FP32 的引擎都可能产生
浮点舍入差异。

在目标 GPU 上用与运行时匹配的 TensorRT SDK 构建引擎。安装 SDK 对应的 Python
wheel，并将 SDK 与 CUDA 依赖的 DLL 目录加入 PATH。本机验证使用 TensorRT
11.2.1.2。导出时将 ONNX 依赖安装到单独目录，供导出命令通过 `PYTHONPATH`
加载，以保留 Irodori 音频依赖所需的 protobuf 版本。

```powershell
$exportDeps = Join-Path (Get-Location) 'irodori-export-deps'
python -m pip install --target $exportDeps onnx onnxscript
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $exportDeps
python tools/prepare-irodori-tensorrt.py --runtime-root ./Irodori-TTS --checkpoint ./irodori-model/model.safetensors --trtexec ./TensorRT/bin/trtexec.exe --output ./irodori-fp16.plan --precision fp16
$env:PYTHONPATH = $previousPythonPath
```

上面的 `python` 使用 Irodori 环境中的解释器。生成的 `.plan` 与同名 `.json`
需放在一起，并使用导出时的 checkpoint。选择 `--precision fp32` 会构建禁用
TF32 的 FP32 引擎。设置 `ALICE_IRODORI_PROFILE=1` 可查看阶段耗时及两种后端
各自执行的步数。

要用 TensorRT 解码，在同一 CUDA 清单加入 `"decoder_plan": "./dacvae-decoder.plan"`，
并提供构建该引擎的本地 `codec_weights`。去噪和解码均使用 TensorRT 时，RF
采样复用 GPU 上的时间步与步长，并批量检查尾部裁剪窗口。参考音频编码和
SilentCipher 水印使用官方实现。

```powershell
$env:PYTHONPATH = $exportDeps
python tools/prepare-irodori-codec.py --runtime-root ./Irodori-TTS --codec-weights ./irodori-codec/weights.pth --trtexec ./TensorRT/bin/trtexec.exe --output ./dacvae-decoder.plan
$env:PYTHONPATH = $previousPythonPath
```

解码引擎构建时会执行多个帧长度的实机测试，成功后写入 `.plan` 与同名
`.json`。本地显存较紧时可用 `--max-latent-frames` 缩小 profile，超出
解码 profile 的输入会由 PyTorch 解码。分项日志的 `codec_backend` 字段
记录实际使用的后端。

## Drama 3 Q2

准备 audio.cpp 的可执行文件及其随附的库，并在 PATH 中提供 Python 3。
`runtime_root` 指向含 `audiocpp_server.exe` 的目录，`checkpoint` 指向兼容的本地
Drama 3 GGUF。启动器在随机本机回环端口加载模型，并将它的流式 PCM 转为 JSONL。
[audio.cpp v0.9.4 运行包](https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.4) ·
[Drama 3 模型与运行选项](https://github.com/0xShug0/audio.cpp/blob/v0.9.4/docs/models/fish_drama3.md)

```json
{
  "schema_version": 1,
  "family": "drama3",
  "runtime_root": "./audio.cpp",
  "checkpoint": "./models/drama-3-backbone-8bit-experts-Q2_K.gguf"
}
```

```powershell
alice-tts inspect ./drama3.json
alice-tts synthesize ./drama3.json --device cuda --text '你好，欢迎收听。' --output ./drama3.wav
alice-tts worker ./drama3.json --device cuda
```

显式语言参数使用 `auto`。使用参考音色时，填写 `reference_audio` 和对应的
`reference_text` 转写。参考文件和模型路径均可相对清单填写。
`--python` 可指定 Python 解释器路径。

清单默认使用 8 个 CPU 线程、24 个 CPU 专家层和 2048 的上下文容量。
`session_options` 与 `request_options` 的值使用字符串。设定 `threads: 6`，
并添加下列 `session_options` 可以启用本次 RTX 5070 Ti 的 CUDA 调优配置。

```json
{
  "fish_drama3.cpu_expert_layers": "16",
  "fish_drama3.perf_mode": "cuda_perf"
}
```

这个配置增加 GPU 驻留专家数量，启用 audio.cpp 的融合 MoE 和设备采样，
可能改变同一种子的输出波形。应用与桌面共享显存。使用默认专家层数可以为其他
应用腾出显存，再按机器的显存余量调整。

## JSONL 协议

在 stdin 发送 UTF-8 JSON，每行一个对象。stdout 只输出 JSONL，运行日志走 stderr。模型加载成功且可选预热完成后，程序发送 `ready`。`ready` 包含 `protocol_version`、`device`、`pcm_format`、`channels` 和 `model`。

```jsonl
{"op":"synthesize","request_id":"one","text":"你好，今天测试本地合成。","language":"auto"}
{"op":"cancel","request_id":"one"}
{"op":"shutdown"}
```

- `synthesize` 接受一个非空 `request_id` 和 `text`。本地 Python 后端接受可选 `language`，原生模型按文本自动识别语言。
- `audio` 包含 `request_id`、`sample_rate` 和 `pcm`。`pcm` 是 Base64 编码的单声道小端 PCM16 样本。
- 每个已接受的请求以一个 `done`、`cancelled` 或 `error` 结束。`done.samples` 为总样本数。繁忙时新请求收到 `rejected`。
- `cancel` 针对当前 `request_id`。GPT-SoVITS checkpoint 和 Irodori 结束 Python 子进程，并在下一次合成时重新加载模型。Drama 3 关闭当前音频流并保留 audio.cpp 会话。原生 ONNX 模式在推理边界响应取消。
- stdin 到达 EOF 时，程序会处理完已接收的请求。`shutdown` 取消仍在进行的请求。

Rust 调用方使用 `alice_native_tts::session::{Model, Session, SessionOptions}`。读取 `Session::info()` 获取采样率，再由 `Session::synthesize()` 接收 `&[i16]`。回调里的样本属于当前调用。

## 实机验收

`tools/check-standalone.py` 从独立 Python 进程启动命令行程序，使用本机的真实模型和参考音频检查 PCM、分块、采样率、取消、恢复及 WAV 输出。可以指定 `--families` 选择本机已准备权重的家族。

```powershell
python tools/check-standalone.py --executable ./target/release/alice-tts.exe --native-pack ./model-pack --gpt-sovits-root 'D:/GPT-SoVITS' --output ./output/acceptance
```

使用 Drama 3 清单独立检查流式 PCM、取消、下一次请求和 WAV。

```powershell
python tools/check-standalone.py --executable ./target/release/alice-tts.exe --drama3-manifest ./drama3.json --checkpoint-device cuda --output ./output/drama3-acceptance
```
