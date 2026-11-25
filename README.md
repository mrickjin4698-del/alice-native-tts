# Alice Native TTS

[简体中文](README.zh-CN.md) · [Overview](#overview) · [Benchmarks](#benchmarks) · [Usage](#usage)

## Overview

Alice Native TTS runs GPT-SoVITS locally as a Rust library and a standalone
`alice-tts` executable. It can write a WAV, stream PCM from a persistent JSONL
worker, or deliver PCM chunks directly to a Rust application. A caller can
cancel a request and reuse the session.

| Model input | Supported families | Audio delivery |
| --- | --- | --- |
| Native ONNX pack, ABI-1 | v2Pro, including compatible v2ProPlus weight exports | 32 kHz PCM. Semantic-token streaming with a matching streaming graph |
| Native ONNX pack, ABI-2 | v3, v4, V5 Turbo, V5 Dev | Text fragments at 24 kHz for v3 and 48 kHz for the others. Optional early PCM for V5 |
| Checkpoint manifest | v1, v2, v2Pro, v2ProPlus, v3, v4, V5 Turbo, V5 Dev | Uses a local GPT-SoVITS Python runtime. The model family determines the sample rate and streaming mode |

Native packs use Rust and the selected inference libraries at runtime. The
native frontend handles Mandarin and English, including common number, date,
time and currency formats. Bring compatible model files and a reference
recording with its transcript.

## Benchmarks

Local runs used Windows 11, a Core Ultra 9 285K and a desktop RTX 5070 Ti with
16 GB VRAM. Each batch reports warm, single-request timings. **First PCM** is
the delay from request submission to the first audio chunk. **Completion** is
the delay to the `done` event. **RTF** is completion time divided by the duration
of the returned audio. Timings exclude model loading, engine builds and
device playback.

### V5 semantic-prefix streaming · November 25, 2026

Same executable, weights, reference audio and texts with the early PCM mode
off and on. Each case uses 2 warmups and 6 timed requests. The mode starts the
first acoustic window after 16 semantic tokens for Turbo or 28 for Dev. Turbo
uses 4 steps. Dev uses 32 steps and guidance 1.30 per window.

| Model | Input | First PCM P50, ms ↓ | First PCM P95 on, ms ↓ | Audible onset P50, ms ↓ | Completion P50, ms ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| V5 Turbo | Short | 141.4 → **59.0** | 61.9 | 69.0 | 141.4 → 166.2 |
| V5 Turbo | Standard | 185.2 → **57.9** | 58.9 | 67.9 | 498.8 → 516.9 |
| V5 Turbo | Rolling | 548.0 → **58.0** | 59.8 | 68.0 | 577.7 → 600.0 |
| V5 Dev | Short | 368.5 → **263.8** | 270.6 | 273.8 | 368.6 → 508.4 |
| V5 Dev | Standard | 418.6 → **262.1** | 267.0 | 272.1 | 1,186.1 → 1,382.1 |
| V5 Dev | Rolling | 838.3 → **262.9** | 298.3 | 272.9 | 1,105.7 → 1,367.9 |

First PCM and completion show **off → on**. All six first-PCM medians improve
by **1.40–9.44×**. The first packet carries 300 ms of audio for Turbo and 790 ms
for Dev. Audible onset is estimated from the PCM at a −45 dBFS RMS threshold.
The extra window changes the first fragment's waveform and adds work. In the
standard sentence, completion takes 18 ms more for Turbo and 196 ms more for
Dev, while both later fragments match the baseline sample for sample.
[Full results, PCM checks and reproduction](docs/performance-v5-first-pcm.md)

### v3–V5 native diffusion · November 24–25, 2026

This batch uses full text fragments and early rolling-window delivery on long
input. Each case uses 2 warmups and 6 timed requests with matching weights and
reference audio. The table shows **checkpoint TensorRT → native TensorRT**
for the standard sentence. The checkpoint adapter uses GPT-SoVITS to schedule
inference. The native path uses Rust and TensorRT 11.2 with CUDA EP for the v3
target feature graph.

| Model | First PCM P50, ms ↓ | Completion P50, ms ↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| v3 | 492 → **350** | 1,797 → 980 | 0.229 → 0.125 |
| v4 | 518 → **298** | 1,812 → 872 | 0.249 → 0.111 |
| V5 Turbo | 233 → **190** | 1,079 → 511 | 0.154 → 0.063 |
| V5 Dev | 748 → **432** | 2,781 → 1,235 | 0.397 → 0.152 |

Across all four models and three texts, the 24 first-PCM and completion
medians improve by **1.22–3.02×**.
[Optimization history and reproduction](docs/performance-native-diffusion-tensorrt.md) ·
[Matched checkpoint protocol](docs/performance-native-diffusion.md) ·
[Scheduling tests](docs/performance-native-diffusion-scheduling.md)

### Checkpoint CUDA vs. TensorRT · November 24, 2026

This batch compares **PyTorch CUDA → checkpoint TensorRT** inside each model
family's local GPT-SoVITS runtime. TensorRT runs GPT Transformer and the
VITS waveform generator for v1 through v2ProPlus. For v3 through V5 it runs
GPT Transformer, the DiT Transformer blocks and the vocoder. Each case uses
3 warmups and 12 timed requests on the same text, with cached engines.

| Model | First PCM P50, ms ↓ | Completion P50, ms ↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| v1 | 316 → 176 | 1,187 → 747 | 0.170 → 0.107 |
| v2 | 420 → 226 | 1,365 → 839 | 0.169 → 0.105 |
| v2Pro | 365 → 209 | 1,355 → 862 | 0.182 → 0.114 |
| v2ProPlus | 366 → 204 | 1,360 → 845 | 0.182 → 0.112 |
| v3 | 981 → 480 | 3,210 → 1,791 | 0.415 → 0.228 |
| v4 | 956 → 514 | 3,109 → 1,817 | 0.434 → 0.250 |
| V5 Turbo | 388 → 210 | 1,505 → 921 | 0.215 → 0.132 |
| V5 Dev | 1,491 → 732 | 4,751 → 2,757 | 0.679 → 0.394 |

[Checkpoint TensorRT setup and graph coverage](docs/standalone.md#checkpoint-tensorrt)

### ABI-1 token streaming · November 24, 2026

These native 32 kHz packs use TensorRT GPT, CUDA VITS and a dedicated first
VITS session. Each case uses 3 warmups and 12 timed requests. The first packet
contains 880 ms of audio.

| Pack | First PCM P50, ms ↓ | Completion P50, ms ↓ | RTF P50 ↓ |
| --- | ---: | ---: | ---: |
| Existing v2Pro pack | 94 | 804 | 0.121 |
| Exported v2Pro pack | 99 | 1,101 | 0.141 |
| Exported v2ProPlus weights | 103 | 1,052 | 0.135 |

Earlier measurements cover CPU and DirectML, CUDA GPT, and cold Mandarin text.
The cold-text run measured a frontend cache miss with the model already loaded.
On the 167-token case, moving G2PW and BERT from CPU to CUDA took the frontend
from 139.7 to 13.1 ms and first audio from 280.7 to 156.2 ms.
[CPU and DirectML](docs/performance.md) ·
[Native TensorRT and CUDA VITS](docs/performance-r2.2.md) ·
[Cold text](docs/performance-r2.3.md)

### Comparable GPT-SoVITS runtimes

The same-machine comparison with **gsv.cpp v0.1.3** uses the same texts,
reference WAV, steps and guidance. Alice uses a persistent JSONL worker and
the V5 semantic prefix. gsv.cpp uses Vulkan with FP16 AR and F32 vocoder over
local HTTP keep-alive. Its first read carries about 85 ms of PCM. Values below
are **Alice / gsv.cpp** on the standard sentence.

| Model | First PCM P50, ms ↓ | Audible onset P50, ms ↓ | Completion P50, ms ↓ |
| --- | ---: | ---: | ---: |
| V5 Turbo | 57.9 / 243.6 | 67.9 / 403.6 | 516.9 / 732.8 |
| V5 Dev | 262.1 / 862.4 | 272.1 / 1,022.4 | 1,382.1 / 2,515.8 |

The first-PCM speedup ranges from **3.08× to 9.36×** across the six cases. On the
rolling Turbo example, Alice finishes in 600.0 ms for 9.12 s of audio and
gsv.cpp in 595.1 ms for 8.08 s. [Controlled comparison](docs/performance-v5-first-pcm.md#同机-gsvcpp-对照)

Published measurements use each project's own inputs and timing protocol.

- **AnifLive-TTS v1.5, V5 Turbo** reported 66.89 ms first PCM and 80.93 ms audible onset P50 on an RTX 5070 Ti over HTTP keep-alive. [HTTP benchmark](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements)
- **GPT-SoVITS-cpp TensorRT streaming** reported a 460 ms first packet and 0.1205 RTF on an RTX 2080 Ti, with about 19 s of output. [SDK benchmark](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks)
- **Upstream GPT-SoVITS v2ProPlus** reported 0.014 RTF on an RTX 4090 for about four minutes of output. [Upstream README](https://github.com/RVC-Boss/GPT-SoVITS#features)

## Usage

### Build the CLI

Install a stable Rust toolchain and, on Windows, the MSVC build tools. Build
from the repository root.

```powershell
cargo build --locked --release --bin alice-tts
```

You can also install `alice-tts` on your path with
`cargo install --locked --path . --bin alice-tts`.

### Model packs

Choose a pack directory for native inference or a JSON manifest for an existing
GPT-SoVITS installation. A native pack contains `pack.json`, ONNX graphs,
frontend assets and reference data. ABI-1 token streaming adds a validated
`sovits_stream.onnx` and streaming metadata.

- **Prepare ABI-1** from compatible exports with [prepare-native-tts-pack.py](tools/prepare-native-tts-pack.py). Add a token-streaming graph with [export-native-tts-stream.py](tools/export-native-tts-stream.py).
- **Export ABI-2** from local v3–V5 checkpoints, the reference WAV and a matching GPT graph set. Choose FP32 for CPU or FP16 for the CUDA/TensorRT setup. [Native diffusion guide](docs/native-diffusion.md)
- **Use checkpoints directly** by putting the family, weight paths, reference WAV and transcript in a manifest. The CLI starts the matching local GPT-SoVITS runtime. [Manifest and dependency setup](docs/standalone.md#gpt-sovits-checkpoint)

The native GPT cache has 512 positions shared by the reference, input text
and generated tokens. Split long input into shorter passages when using a
native pack.

### Generate a WAV or stream PCM

Choose a native device for the pack and libraries you have.

| Native device | Setup |
| --- | --- |
| `cpu` | Default build. Use an FP32 ABI-2 pack for CPU diffusion |
| `directml` | Experimental Windows GPU path for ABI-1 |
| `cuda` | Build with `r1-cuda-ep` and provide ONNX Runtime with the CUDA provider |
| `tensor_rt_native` | Build with TensorRT and the CUDA runtime. See the Windows setup below |

Use a CPU-compatible pack for these commands.

```powershell
./target/release/alice-tts inspect ./model-pack
./target/release/alice-tts synthesize ./model-pack --device cpu --text '你好，今天测试本地合成。' --output ./hello.wav
./target/release/alice-tts worker ./model-pack --device cpu --warmup-text '今天测试本地语音。'
```

Wait for `ready`, then send a UTF-8 JSON object per line on stdin.

```json
{"op":"synthesize","request_id":"demo","text":"你好，今天测试本地合成。"}
```

Read `audio` events from stdout. Each event carries its sample rate and
Base64-encoded mono PCM16. Requests terminate with `done`, `cancelled`, or
`error`. Use `cancel` with the active request ID to stop synthesis.
[JSONL protocol](docs/standalone.md#jsonl-协议)

To run a checkpoint with CUDA PyTorch, point the CLI at your local installation.
For checkpoint TensorRT, use `--device tensor_rt` and install the matching Python
bindings and runtime libraries.

```powershell
./target/release/alice-tts worker ./voice.json --gpt-sovits-root ./GPT-SoVITS --device cuda
```

### Native TensorRT 11 GPT (Windows x64)

Set the TensorRT SDK, CUDA runtime and CUDA NVCC header directories. Point
`ORT_DYLIB_PATH` at the CUDA-enabled ONNX Runtime DLL. Add the corresponding
TensorRT, CUDA runtime, cuBLAS and ORT DLL directories to `PATH`.

```powershell
$env:ALICE_TRT_HOME = 'C:/SDK/TensorRT-11.2.1.2'
$env:ALICE_CUDA_HOME = 'C:/SDK/cuda_cudart-windows-x86_64-12.9.79-archive'
$env:ALICE_CUDA_NVCC_HOME = 'C:/SDK/cuda_nvcc-windows-x86_64-12.9.86-archive'
$env:ORT_DYLIB_PATH = 'C:/SDK/onnxruntime/onnxruntime.dll'
cargo build --locked --release --features r2-trt-native-cuda-vits --bin alice-tts
```

For the measured V5 configuration, prepare the mixed-FP16 GPT opt5 cache and
acoustic TensorRT plans before running the worker. The prefix setting is
optional. Use 16 for Turbo and 28 for Dev.
[TensorRT plan preparation](docs/performance-native-diffusion-tensorrt.md#复现原生-tensorrt)

```powershell
$env:ALICE_NATIVE_TTS_V5_FIRST_PCM_TOKENS = '16'
./target/release/alice-tts worker ./packs/v5turbo --device tensor_rt_native
```

For ABI-1, export the active-length GPT step separately and set
`ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX` to its path before loading the pack.
[Active-length export](tools/prepare-native-active-len.py) ·
[Optional CUDA VITS and first session](docs/standalone.md#原生-onnx-模型) ·
[CUDA frontend options](docs/performance-r2.3.md)

### Rust integration

Use `alice_native_tts::session::{Model, Session, SessionOptions}` to open a model,
load it on the selected device and receive `&[i16]` from `Session::synthesize`.
Read `Session::info()` for the sample rate. The crate is also used by Project
Alice, whose application adapter handles provider selection and playback.
[Session API](src/session.rs)

### License and provenance

The inference code is Apache-2.0. Adapted frontend resources retain their
upstream notices. Keep the source and license information for your weights
and reference assets in the pack's `LICENSES.txt`.
[License](LICENSE) · [Third-party notices](NOTICE.md)
