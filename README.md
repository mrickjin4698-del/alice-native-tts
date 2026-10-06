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
