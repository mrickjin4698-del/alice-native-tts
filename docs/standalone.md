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
| GPT-SoVITS v1、v2、v2Pro、v2ProPlus checkpoint | 本机 GPT-SoVITS Python 子进程 | 分 token 块输出 |
| GPT-SoVITS v3、v4 checkpoint | 本机 GPT-SoVITS Python 子进程 | 按文本片段输出 |
| GPT-SoVITS V5 Turbo、V5 Dev checkpoint | 支持 V5 的本机 GPT-SoVITS Python 子进程 | 按文本片段输出 48 kHz PCM |

## 原生 ONNX 模型

沿用已有 ABI-1 包目录。包有流式图时，命令行自动启用流式接口。

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

## GPT-SoVITS V5

V5 清单的 `family` 取值为 `v5turbo` 或 `v5dev`，分别对应 `s2Gv5turbo.pth` 和 `s2Gv5dev.pth`。GPT 路径可指向官方 `s1v3.ckpt`。将 V5 声码器放在 V5 安装的 `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth`，并把 `--gpt-sovits-root` 指向支持 V5 的 GPT-SoVITS 安装。[V5 权重](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained) · [GPT-SoVITS V5 说明](https://github.com/RVC-Boss/GPT-SoVITS#v5-release-notes)

```powershell
alice-tts worker ./voice-v5turbo.json --gpt-sovits-root 'D:/GPT-SoVITS-V5' --device cuda
alice-tts synthesize ./voice-v5dev.json --gpt-sovits-root 'D:/GPT-SoVITS-V5' --device cuda --text '你好，今天测试 V5。' --output ./hello-v5.wav
```

两种 V5 模型均输出 48 kHz PCM，按上游的文本片段接口交付音频。Turbo 使用上游的 4 步采样默认值，Dev 使用 32 步采样默认值。运行时加载后会核对清单里的家族与实际权重。


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
