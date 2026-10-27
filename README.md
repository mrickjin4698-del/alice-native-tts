# Alice Native TTS

[简体中文](README.zh-CN.md)

A Rust inference engine for GPT-SoVITS v2Pro ONNX model packs. The inference
library runs locally and does not start Python or an HTTP service. It supports
Mandarin and English text, CPU inference, optional Windows DirectML, clause
audio delivery, and an experimental token-streaming pack extension.

This repository contains the inference code and pack preparation tools. Model
weights and reference voices are supplied separately by each pack author and
are not included here.

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
- Optional token streaming when the model pack contains the matching ABI-1
  streaming graph.
- Cooperative cancellation between ONNX graph runs.

The text frontend is intentionally scoped. It is not full GPT-SoVITS Python
frontend parity; see [current limits](#current-limits).

## Quick start

Install a stable Rust toolchain and provide a compatible ABI-1 model pack in a
local directory. See [Model packs](#model-packs) for the required files and
preparation tools.

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

To cancel after a number of milliseconds, pass the backend first and duration second:

```powershell
cargo run --release --example synthesize -- ./model-pack "你好。" ./output.wav cpu 500
```

The `stream_bench` example requires a pack with the optional streaming graph:

```powershell
cargo run --release --example stream_bench -- ./model-pack ./output cpu
```

## Model packs

Inference requires an ABI-1 `v2Pro` pack at 32 kHz with a 512-token GPT KV
capacity. A base pack contains the ONNX graphs, frontend data, a reference WAV
and its transcript, plus a `LICENSES.txt` provenance file. `pack.json` declares
the ABI and reference files. Optional token streaming also requires
`sovits_stream.onnx` and the matching streaming metadata.

Prepare a pack from local, compatible exports and an existing GPT-SoVITS
installation with `tools/prepare-native-tts-pack.py`. Python is needed only for
this offline preparation step; runtime inference is Rust. The optional
streaming graph is exported and parity-checked with
`tools/export-native-tts-stream.py`.

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

This is an initial measurement snapshot from one machine, not a cross-device
benchmark or a performance guarantee. The implementation has not had a
systematic model-quantization, weight-deduplication, or broad hardware-tuning
pass. It does include targeted work such as batched pronunciation inference,
early clause delivery, optional token streaming, and DirectML support.

See [the benchmark notes](docs/performance.md) for hardware, timing definitions,
results, and reproduction commands.

## Current limits

- v2Pro packs only; Mandarin and English are supported.
- Speed is fixed at 1.0 and acoustic noise at 0.0 by the exported graphs.
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
