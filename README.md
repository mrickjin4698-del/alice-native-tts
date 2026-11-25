# Alice Native TTS

[简体中文](README.zh-CN.md)

Local GPT-SoVITS synthesis for applications and the command line. The native
Rust engine runs v2Pro ABI-1 packs and v3, v4, V5 Turbo and V5 Dev ABI-2 packs.
The checkpoint adapter runs v1, v2, v2Pro, v2ProPlus, v3, v4, V5 Turbo and V5 Dev
through an installed GPT-SoVITS runtime in a local process.

This repository contains the inference code and pack preparation tools. Model
weights and reference voices are supplied separately by each pack author and
are not included here.

## Standalone CLI

Build `alice-tts` with `cargo build --locked --release --bin alice-tts`. The CLI
inspects a model, synthesizes PCM16 WAV, or runs a persistent JSONL worker with
warmup, streaming and cancellation. Rust clients can use the same model loader
and synthesis interface through `alice_native_tts::session`.

See the [standalone guide](docs/standalone.md) for supported checkpoints, model
manifests, devices, installation and the JSONL protocol.

## Features

- Native Rust inference over versioned ONNX graphs.
- Mandarin pronunciation frontend with batched G2PW and English G2P using the
  included GRU inference code and pack-provided weights.
- Number, date, time, currency, percentage, and temperature normalization for
  supported display forms.
- CPU by default; DirectML is available on Windows as an experimental backend.
- The CUDA/TensorRT EP GPT path keeps the frontend and vocoder on CPU. A separate,
  experimental streaming VITS CUDA EP path is documented in R1.2. The optional
  native TensorRT 11 GPT backend is Windows x64-only and feature-gated; it keeps
  K/V cache on the GPU and retains CPU sampling. The two TensorRT routes use
  different runtime APIs.
- Optional token streaming with a matching ABI-1 streaming graph. ABI-2
  diffusion packs emit PCM after each text fragment.
- Cooperative cancellation between ONNX graph runs.

The text frontend is intentionally scoped. It is not full GPT-SoVITS Python
frontend parity; see [current limits](#current-limits).

## Quick start

Install a stable Rust toolchain and provide a compatible model pack in a
local directory. The example commands below use ABI-1. ABI-2 uses the
standalone CLI and Rust session interface. See [Model packs](#model-packs).

Synthesize a WAV with the CPU backend:

```powershell
cargo run --release --example synthesize -- ./model-pack "你好，今天我们测试本地语音合成。" ./output.wav cpu
```

On Windows, select DirectML explicitly:

```powershell
cargo run --release --example synthesize -- ./model-pack "Hello, Alice." ./output.wav directml
```

CUDA and TensorRT EPs are opt-in Cargo features and require compatible ORT plus
CUDA/TensorRT runtime libraries:

```powershell
cargo run --release --features r1-cuda-ep --example synthesize -- ./model-pack "Hello." ./output.wav cuda
cargo run --release --features r1-tensorrt-ep --example synthesize -- ./model-pack "Hello." ./output.wav tensor_rt
```

The R1.1 CUDA GPT + CPU VITS path passed performance and bit-identical audio
checks on one local host and remains opt-in. The ORT TensorRT EP still requires
its own matching runtime; it is separate from native TensorRT 11. See the [historical
R1.0 GPU audit](docs/performance-r1-gpu.md) and the [R1.1 CUDA runtime and
benchmark report](docs/performance-r1.1.md).
An opt-in streaming VITS CUDA EP experiment is measured separately in the
[R1.2 report](docs/performance-r1.2.md); it requires explicit CPU fallback for
the audited VITS helper nodes and does not change the default CPU VITS path.

### Native TensorRT 11 GPT (Windows x64)

This backend is experimental and is not the ORT TensorRT EP. Build it with the
TensorRT 11 SDK, CUDA runtime/NVCC headers, and a matching ONNX Runtime DLL
available locally:

```powershell
$env:ALICE_TRT_HOME = "C:\path\to\TensorRT-11.2.1.2"
$env:ALICE_CUDA_HOME = "C:\path\to\cuda_cudart-windows-x86_64-12.9.79-archive"
$env:ALICE_CUDA_NVCC_HOME = "C:\path\to\cuda_nvcc-windows-x86_64-12.9.86-archive"
$env:ORT_DYLIB_PATH = "C:\path\to\onnxruntime.dll"
# Add TensorRT, CUDA runtime and cuBLAS bin directories to PATH.
cargo run --release --features r2-trt-native --example synthesize -- ./model-pack "Hello." ./output.wav tensor_rt_native
```

The measured configuration and full reproduction notes are in the
[R2.1 performance report](docs/performance-r2.1.md). TensorRT 11 GPT inference
remains opt-in; CPU and DirectML defaults are unchanged.

An optional native TensorRT GPT + ORT CUDA VITS combination is available for
local benchmarking with the `r2-trt-native-cuda-vits` feature. It requires the
same native TensorRT SDK/runtime plus an ORT CUDA DLL and its CUDA dependencies.
Enable it explicitly with `--cuda-vits`; `--cuda-vits-allow-cpu` permits the
audited VITS helper/shape nodes to stay on CPU. The default backend is unchanged,
and the mixed route is still experimental pending listening and playback
validation. See the [R2.2 report](docs/performance-r2.2.md) for measured
results and the observed execution-provider node split.

### Cold Chinese text frontend

For a new Chinese text on a loaded `tensor_rt_native` or CUDA backend, the
optional CUDA G2PW + BERT frontend avoids the CPU cold-miss cost. Build with
`r1-cuda-ep` (the `r2-trt-native-cuda-vits` feature already includes it), then
set these variables before creating `NativeTts`:

```powershell
$env:ALICE_NATIVE_TTS_CUDA_FRONTEND = "1"
$env:ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU = "1"
```

The first variable selects CUDA for both G2PW and BERT unless a per-graph
`ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND` or
`ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND` override is present. The second allows
the audited shape/helper nodes to remain on CPU; the main graph math stays on
CUDA. Without that explicit helper setting, ORT rejects CPU-assigned nodes at
session creation instead of silently falling back. CPU and DirectML defaults
are unchanged; selecting CPU or DirectML keeps those Frontend paths. The
measured cold-text results and limits are in the
[R2.3 report](docs/performance-r2.3.md).

To cancel after a number of milliseconds, pass the backend first and duration second:

```powershell
cargo run --release --example synthesize -- ./model-pack "你好。" ./output.wav cpu 500
```

The `stream_bench` example requires a pack with the optional streaming graph:

```powershell
cargo run --release --example stream_bench -- ./model-pack ./output cpu
```

#### Host shape input for the native GPT step

On the native TensorRT backend, an ABI-1 step graph can take its active KV
length from a host shape input. Generate a separate graph from the pack's
original step and set the override before loading the model.

```powershell
python tools/prepare-native-active-len.py ./model-pack/gpt_step.onnx ./target/active-len/gpt_step.onnx
$env:ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX = (Resolve-Path ./target/active-len/gpt_step.onnx).Path
$env:ALICE_NATIVE_TTS_TRT_CACHE_DIR = Join-Path (Get-Location) 'target/active-len-engines'
cargo run --locked --release --features "r0-profiling r2-trt-native-cuda-vits" --example r0_profile -- ./model-pack ./target/active-len-profile tensor_rt_native --chunks=24 --warmups=5 --repeats=10 --cuda-vits --cuda-vits-allow-cpu
```

With the override unset, the runner loads the packed step graph from the
model directory. The engine cache keys include the model file fingerprint.

## Model packs

An ABI-1 `v2Pro` pack runs at 32 kHz with a 512-token GPT KV capacity.
A base pack contains the ONNX graphs, frontend data, a reference WAV
and its transcript, plus a `LICENSES.txt` provenance file. `pack.json` declares
the ABI and reference files. Optional token streaming also requires
`sovits_stream.onnx` and the matching streaming metadata.

Prepare a pack from local, compatible exports and an existing GPT-SoVITS
installation with `tools/prepare-native-tts-pack.py`. Python is needed only for
this offline preparation step; runtime inference is Rust. The optional
streaming graph is exported and parity-checked with
`tools/export-native-tts-stream.py`.

ABI-2 exports v3 at 24 kHz and v4, V5 Turbo and V5 Dev at 48 kHz. Rust drives
the GPT, target features, diffusion steps and vocoder. An FP32 pack can run on
CPU. The measured Windows GPU route uses native TensorRT GPT and ONNX Runtime
CUDA for the acoustic graphs. The [native diffusion guide](docs/native-diffusion.md)
covers pack preparation, backend selection and reference conditioning.

The tools expect model files and source installations to be provided by the
operator. Read each upstream model's terms and retain its notices in the pack.
Do not place model weights, a voice reference recording, or generated packs in
this source repository.

## API

The crate exports `Config`, `Backend`, `NativeTts`, pack validation, and text
normalization. A minimal embedding example is in `examples/synthesize.rs`;
Rust API documentation can be generated with:

```powershell
cargo doc --open
```

Alice's desktop integration adapter is maintained in Project Alice and is not
part of this crate. It handles application provider selection, shared caching,
event reporting, playback queues, and integration-level cancellation.

## Performance

Local tests use Windows 11, a Core Ultra 9 285K, a desktop RTX 5070 Ti 16 GB and
TensorRT 11.2. A persistent `alice-tts` JSONL worker runs the same Mandarin
text and reference recording. First PCM and completion measure from request
submission to `audio` and `done`, respectively. Warm timings exclude model
loading, engine builds and playback. RTF is completion time divided by the
duration of the returned audio.

### v3–V5 native TensorRT

Standard sentence P50 after 2 warmups and 6 measured requests per case. Each
cell shows **checkpoint TensorRT → native TensorRT** with matching GPT and
SoVITS weights. The native path runs in Rust and uses CUDA EP for v3 target
features. V5 Turbo uses 4 diffusion steps. V5 Dev keeps 32 steps and guidance 1.30.

| Model | First PCM (ms) ↓ | Completion (ms) ↓ | RTF ↓ |
| --- | ---: | ---: | ---: |
| v3 | 492 → **350** | 1,797 → **980** | 0.229 → **0.125** |
| v4 | 518 → **298** | 1,812 → **872** | 0.249 → **0.111** |
| V5 Turbo | 233 → **190** | 1,079 → **511** | 0.154 → **0.063** |
| V5 Dev | 748 → **432** | 2,781 → **1,235** | 0.397 → **0.152** |

V5 Turbo returned first PCM in **145 ms** P50 on the short sentence. Across
short, standard and rolling-window inputs, all 24 first-PCM and completion
P50 measurements improved by **1.22–3.02×** over checkpoint TensorRT. See the
[optimized results and reproduction commands](docs/performance-native-diffusion-tensorrt.md),
[checkpoint methodology](docs/performance-native-diffusion.md) and
[scheduling tests](docs/performance-native-diffusion-scheduling.md).

V5 also has an opt-in semantic-prefix mode. On the standard sentence, first
PCM P50 fell from 185 to **58 ms** for Turbo and from 419 to **262 ms** for
Dev. The same-machine gsv.cpp v0.1.3 comparison measured 244 ms and 862 ms,
respectively. The mode changes the first fragment's waveform. See
[V5 first PCM benchmarks](docs/performance-v5-first-pcm.md).

### ABI-1 token streaming

With TensorRT GPT and CUDA VITS, the existing v2Pro pack reached **94 ms** first
PCM and **804 ms** completion P50 on the standard sentence. The v2Pro and
v2ProPlus checkpoint exports reached **99 ms** and **103 ms** first PCM.
All three delivered 880 ms of audio in the first chunk, with 3 warmups and 12
measured requests per case. See the [earlier CPU and DirectML results](docs/performance.md).

### Other GPT-SoVITS runtimes

These published measurements retain each project's workload and timing boundary.

| Runtime and model | Hardware and workload | First PCM or packet (ms) ↓ | Completion (ms) ↓ | RTF ↓ | Source |
| --- | --- | ---: | ---: | ---: | --- |
| Alice Native TTS · V5 Turbo | RTX 5070 Ti, native TensorRT with a semantic prefix, standard sentence over JSONL | 57.85 | 516.90 | 0.0649 | [Local benchmark](docs/performance-v5-first-pcm.md) |
| AnifLive-TTS v1.5 · V5 Turbo | RTX 5070 Ti, TensorRT 11 / Triton FP16, HTTP keep-alive PCM | 66.89 | 75.98 | 0.0473 | [AnifLive HTTP benchmark](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements) |
| gsv.cpp v0.1.3 · V5 Turbo | RTX 5070 Ti, Vulkan, HTTP keep-alive PCM | 174.45 | 197.06 | 0.1120 | [AnifLive's gsv.cpp test](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements) |
| GPT-SoVITS-cpp | RTX 2080 Ti, TensorRT FP16 streaming, ~19 s of mixed-language audio | 460 | — | 0.1205 | [SDK benchmark](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS upstream · v2ProPlus | RTX 4090, ~4 minutes of audio | — | — | 0.014 | [Upstream README](https://github.com/RVC-Boss/GPT-SoVITS#features) |

The three V5 Turbo rows share a GPU model. Text, voices and delivery protocols
differ. Alice's standard sentence returns 0.30 s of audio in the first PCM
chunk and 7.96 s in total. The HTTP study reports medians from 1,000 AnifLive
requests and 300 gsv.cpp requests per mode. It measures audible onset separately
from PCM packet arrival.

## Current limits

- Native packs support Mandarin and English.
- ABI-1 exports fix speed at 1.0 and acoustic noise at 0.0. ABI-2 uses its
  family-specific step count, noise scale and guidance.
- The cache in the Alice adapter holds one base voice configuration per
  process. The crate itself leaves cache ownership to its host.
- Cancellation is cooperative between ONNX runs; a running ONNX kernel cannot
  be forcibly interrupted.
- The frontend does not yet cover Mandarin neutral-tone/erhua behavior,
  English part-of-speech-dependent homographs, Japanese, or every locale's
  number grammar.
- Long inputs are bounded to protect the exported GPT KV cache.

## License and provenance

The crate is distributed under Apache-2.0. Adapted Mandarin pronunciation
tables, tokenizer data, and pinyin splitting code retain their upstream
attribution in [NOTICE.md](NOTICE.md). Model-pack components have their own
provenance and licensing; see the pack's `LICENSES.txt`.
