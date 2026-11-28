# Alice TTS 独立使用

构建 Rust 库和命令行程序。

```powershell
cargo build --locked --release --bin alice-tts
cargo install --locked --path . --bin alice-tts
```

Windows x64 Release 提供 CPU 和 CUDA 13 两个 ZIP。两个版本都包含 `alice-tts.exe`，
支持 `pack`、`serve`、JSONL 和 v2Pro 原始 checkpoint 直读。解压后使用
`./alice-tts.exe --help` 查看命令。系统需要 Microsoft Visual C++ v14 x64 运行时。
CPU 包还使用 Windows 的 DirectML。CUDA 包包含 ONNX Runtime，系统需另外安装
CUDA 13 与 cuDNN 9 的运行库。模型权重、共享前端和参考音频由调用方提供。
[DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) · [VC++ 运行时](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist#latest-supported-redistributable-version)

| 模型 | 推理路径 | 流式输出 |
| --- | --- | --- |
| ABI-1 v2Pro ONNX | Rust + ONNX Runtime，支持可选 GPU 后端 | 有匹配流式图时分 token 块输出 |
| ABI-2 v3、v4、V5 Turbo、V5 Dev ONNX | Rust + ONNX Runtime，可选原生 TensorRT GPT | 按文本片段输出，v3 为 24 kHz，其余为 48 kHz |
| GPT-SoVITS v2Pro 原始 checkpoint（`native_assets`） | Rust 直读权重 + ONNX Runtime CPU 或 CUDA | 32 kHz，按 token 块输出 |
| GPT-SoVITS v1、v2、v2Pro、v2ProPlus checkpoint | 本机 GPT-SoVITS Python 子进程 | 分 token 块输出 |
| GPT-SoVITS v3、v4 checkpoint | 本机 GPT-SoVITS Python 子进程 | 按文本片段输出 |
| GPT-SoVITS V5 Turbo、V5 Dev checkpoint | 支持 V5 的本机 GPT-SoVITS Python 子进程 | 按文本片段输出 48 kHz PCM |

## 命令行管道

`synthesize` 接收 `--text TEXT` 或 `--text-file FILE`。令 `--text-file -`
即可从 stdin 读取 UTF-8 文本。子进程调用方写完一条文本后，关闭 stdin 以
发送 EOF，并持续读取 stdout 和 stderr。

| 选项 | 输出 |
| --- | --- |
| `--output ./speech.wav` | 写入新文件，stdout 返回一条 JSON 完成消息。 |
| `--output -` | stdout 返回完整 WAV。合成结束后填好 WAV 头，再从临时文件发送。 |
| `--output - --format s16le` | stdout 随模型逐块返回单声道小端 PCM16。 |

`--output -` 的 stdout 是音频字节。诊断走 stderr，失败时进程返回非零退出码。
`s16le` 每两个小端字节表示一个样本。原生模型的采样率见 `inspect MODEL`
的 `sample_rate`，v2Pro 直读为 32,000 Hz。需要由运行时确定采样率的模型可用
WAV 或读取 JSONL `ready.model.sample_rate`。

将 stdout 的 WAV 接入 FFmpeg。`input.txt` 使用 UTF-8 编码。

```powershell
alice-tts synthesize ./voice.json --text-file ./input.txt --output - |
  ffmpeg -i pipe:0 ./speech.flac
```

将 v2Pro 的流式 PCM 接入 FFmpeg。

```powershell
alice-tts synthesize ./voice.json --text-file ./input.txt --output - --format s16le |
  ffmpeg -f s16le -ar 32000 -ac 1 -i pipe:0 ./speech.wav
```

PowerShell 7.4 起，原生命令之间的管道保留音频字节。应用可直接打开子进程的
二进制 stdout。高频调用可使用常驻 `worker` 或 `serve`，让一次模型加载服务
多条请求。[PowerShell 二进制重定向](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_redirection#example-7-redirecting-binary-data-from-a-native-command) · [FFmpeg pipe 协议](https://ffmpeg.org/ffmpeg-protocols.html#pipe)

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

### v2Pro 原始权重直读

在清单中加入 `native_assets` 以选择 Rust 直读。程序从原始 `.ckpt` 和
`.pth` 读取张量，加载到内置的无权重架构中。它使用 CPU 或 CUDA 合成 32 kHz PCM，
支持中英文自动识别和 token 级流式输出。

```json
{
  "schema_version": 1,
  "family": "v2Pro",
  "gpt": "./voice.ckpt",
  "sovits": "./voice.pth",
  "reference_audio": "./reference.wav",
  "reference_text": "这里填写参考音频的实际转写。",
  "reference_language": "zh",
  "native_assets": "./shared-v2pro-assets"
}
```

`native_assets` 可以是现有 ABI-1 v2Pro 模型包目录，也可以是只包含以下共享
资源的目录。文件路径相对于清单所在目录。`config.json` 提供 v2Pro 的
音素表，BERT 和 HuBERT 由该目录提供。

```text
config.json
bert.onnx
ssl.onnx
spectrogram.onnx
sv_embedding.onnx
g2pw.onnx
cmudict.json
g2p-en.json
LICENSES.txt
```

```powershell
cargo build --locked --release --features raw-checkpoint,server --bin alice-tts
./target/release/alice-tts inspect ./voice.json
./target/release/alice-tts synthesize ./voice.json --text '你好，今天测试原始权重直读。' --output ./hello.wav
./target/release/alice-tts serve ./voice.json
```

`inspect` 和 `/healthz` 会报告 `native_checkpoint`。此模式支持标准 32 kHz
v2Pro 架构的 PyTorch ZIP checkpoint，包含使用 `05` 版本标记的 SoVITS 权重。
执行合成直接使用原文件，无需先导出声线模型或启动 Python。
`synthesize`、`worker`、HTTP 与 WebSocket 均可选择 CPU 或 CUDA。

#### CUDA 直读

启用 `r1-cuda-ep` 并选择 `--device cuda`。GPT 编码、逐 token 解码和流式
VITS 的主要算子使用 CUDA。逐 token 解码关闭 CPU 回退，编码和 VITS 的形状
辅助算子可以在 CPU 上运行。文本前端默认使用 CPU。

```powershell
$env:ORT_DYLIB_PATH = 'C:/path/to/onnxruntime.dll'
$env:PATH = 'C:/path/to/ort;C:/path/to/cuda-and-cudnn;' + $env:PATH
cargo build --locked --release --features raw-checkpoint,r1-cuda-ep,server --bin alice-tts
./target/release/alice-tts serve ./voice.json --device cuda
```

默认精度为 FP32。要试用 CUDA FP16，在 `voice.json` 顶层加入
`"native_precision": "fp16"`，再用 `--device cuda` 启动。GPT 编码、逐 token
解码、KV cache 和流式 VITS 的主要计算使用 FP16。参考音频编码、共享文本前端
和 PCM 输出接口仍使用 FP32。加载器在内存中按图需要的精度读取原始权重。
FP16 可能改变生成的 token、停顿和时长。

CUDA 运行包已包含 ONNX Runtime DLL。将 CUDA 13 和 cuDNN 9 的 DLL 目录加入
`PATH`，在解压目录运行 `./alice-tts.exe serve ./voice.json --device cuda`。
Windows x64 实测使用 ONNX Runtime 1.30.0、CUDA 13 和 cuDNN 9。
[ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

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

## HTTP 与 WebSocket

启用 `server` feature 后，`serve` 使用和 JSONL worker 相同的模型会话。

```powershell
cargo build --locked --release --features server --bin alice-tts
./target/release/alice-tts serve ./model-pack --bind 127.0.0.1:8090
```

默认监听 `127.0.0.1:8090`。`--device` 和 `--warmup-text` 也可用于服务模式。
原生 ABI-1 和 ABI-2 包，以及指定 `native_assets` 的 v2Pro 清单，都直接由 Rust
加载。Python 兼容模式使用本机的 GPT-SoVITS 安装和解释器。

| 接口 | 用法 |
| --- | --- |
| `GET /healthz` | 返回模型信息 `model`、`device`、`protocol_version` 和 `max_text_chars`。 |
| `POST /v1/tts` | 接收 JSON，返回 `audio/wav`，内含单声道 PCM16 和实际采样率。 |
| `GET /v1/tts/ws` | 升级为 WebSocket，逐块返回单声道小端 PCM16。 |

两个合成接口都接收 `{"text":"你好。","language":"auto"}`。`language` 可省略，
支持的值见 `/healthz` 中的 `model.languages`。JSON 最大 16 KiB，文本长度
以 `/healthz` 中的 `max_text_chars` 为准。ABI-1 原生模型上限为 180 字，服务的
通用上限为 4,096 字。HTTP 音频数据上限为 32 MiB，长音频可经 WebSocket
逐块接收。模型会话一次运行一个合成请求，繁忙时 HTTP 返回 503，WebSocket
返回 `code: "busy"` 的错误事件。

```powershell
$request = @{ text = '你好，我是 Alice。' } | ConvertTo-Json -Compress
Invoke-WebRequest -Uri http://127.0.0.1:8090/v1/tts -Method Post `
  -ContentType 'application/json; charset=utf-8' -Body $request -OutFile ./hello.wav
```

WebSocket 建连后发送一个 JSON 文本帧。服务先发送 `{"type":"start",...}`，
其中包含 `sample_rate`、`channels`、`pcm_format` 和 `streaming`。随后每个
二进制帧都是 PCM 样本，帧长随模型的输出块变化。成功时发送
`{"type":"done","samples":...}` 再关闭连接。发送 `{"op":"cancel"}` 可取消
当前合成，服务返回 `cancelled`。断开 WebSocket 也会触发协作式取消。
失败时会发送带有 `code` 和 `message` 的 `error` 事件。

用 Node.js 客户端接收原始 PCM 的示例。

```javascript
import { appendFileSync, writeFileSync } from "node:fs";

writeFileSync("hello.pcm", Buffer.alloc(0));
const socket = new WebSocket("ws://127.0.0.1:8090/v1/tts/ws");
socket.binaryType = "arraybuffer";
socket.onopen = () => socket.send(JSON.stringify({ text: "你好，我是 Alice。" }));
socket.onmessage = ({ data }) => {
  if (typeof data === "string") console.log(JSON.parse(data));
  else appendFileSync("hello.pcm", Buffer.from(data));
};
```

## 原生 ABI-1 打包

`pack` 读取已经导出的 v2Pro ONNX 图，用 Rust 转换 CMUdict 文本和
g2p_en 的 `checkpoint20.npz`。也可以直接传入现有的 `cmudict.json` 和
`g2p-en.json`。模型、参考音频及前端数据的来源和授权由 `--licenses` 指定。

```powershell
cargo build --locked --release --features pack-tools --bin alice-tts
./target/release/alice-tts pack `
  --exports ./exports --g2pw ./assets/g2pW.onnx `
  --cmudict ./assets/cmudict --g2p-en ./assets/checkpoint20.npz `
  --licenses ./assets/LICENSES.txt --reference ./voice.wav `
  --transcript '这里填写参考音频的实际转写。' --output ./model-pack
```

`--output` 指向新目录。需要 token 级流式合成时，传入
`--streaming-graph ./exports/sovits_stream.onnx`。打包器校验同目录下的
`sovits_stream.validation.json`，核对图的 SHA-256 及音频、潜变量的误差记录。
原始 Python 打包脚本和 PyTorch 模型导出脚本仍可使用。

## JSONL 协议

在 stdin 发送 UTF-8 JSON，每行一个对象。stdout 只输出 JSONL，运行日志走 stderr。模型加载成功且可选预热完成后，程序发送 `ready`。`ready` 包含 `protocol_version`、`device`、`pcm_format`、`channels` 和 `model`。

```jsonl
{"op":"synthesize","request_id":"one","text":"你好，今天测试本地合成。","language":"auto"}
{"op":"cancel","request_id":"one"}
{"op":"shutdown"}
```

- `synthesize` 接受一个非空 `request_id` 和 `text`。checkpoint 支持可选 `language`，原生模型按文本自动识别语言。
- `audio` 包含 `request_id`、`sample_rate` 和 `pcm`。`pcm` 是 Base64 编码的单声道小端 PCM16 样本。
- 每个已接受的请求以一个 `done`、`cancelled` 或 `error` 结束。`done.samples` 为总样本数。繁忙时新请求收到 `rejected`。
- `cancel` 针对当前 `request_id`。checkpoint 模式释放该请求的兼容子进程，并在下一次合成时重新加载模型。原生模式在推理边界响应取消。
- stdin 到达 EOF 时，程序会处理完已接收的请求。`shutdown` 取消仍在进行的请求。

Rust 调用方使用 `alice_native_tts::session::{Model, Session, SessionOptions}`。读取 `Session::info()` 获取采样率，再由 `Session::synthesize()` 接收 `&[i16]`。回调里的样本属于当前调用。

## 实机验收

`tools/check-standalone.py` 从独立 Python 进程启动命令行程序，使用本机的真实模型和参考音频检查 PCM、分块、采样率、取消、恢复及 WAV 输出。可以指定 `--families` 选择本机已准备权重的家族。

```powershell
python tools/check-standalone.py --executable ./target/release/alice-tts.exe --native-pack ./model-pack --gpt-sovits-root 'D:/GPT-SoVITS' --output ./output/acceptance
```
