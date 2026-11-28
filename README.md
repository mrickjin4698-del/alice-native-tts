<div align="center">

# Alice Native TTS

**Local GPT-SoVITS inference for the command line and your applications.**

[Quick start](#quick-start) · [Integration](#integration) · [Benchmarks](#benchmarks) · [简体中文](README.zh-CN.md)

</div>

Alice Native TTS runs GPT-SoVITS in Rust. Use `alice-tts` to write a WAV, pipe audio
into an application, or keep a model loaded for repeated requests. The native
v2Pro path reads original `.ckpt` and `.pth` files with shared frontend assets.

## Quick start

Install the CLI from this checkout with a stable Rust toolchain. Supply a v2Pro
checkpoint manifest as `voice.json`, or use a compatible directory containing `pack.json`.
[Checkpoint manifest](docs/standalone.md#v2pro-原始权重直读)

```powershell
cargo install --locked --path . --bin alice-tts --features raw-checkpoint,server,pack-tools
alice-tts inspect ./voice.json
alice-tts synthesize ./voice.json --text "你好，我是 Alice。" --output ./speech.wav
```

CPU and FP32 are the defaults for direct v2Pro inference. A CUDA build accepts
`--device cuda` and supports optional FP16 through `"native_precision": "fp16"`
in the manifest.

<details>
<summary>Build for CUDA on Windows x64</summary>

```powershell
cargo build --locked --release --features raw-checkpoint,r1-cuda-ep,server,pack-tools --bin alice-tts
./target/release/alice-tts.exe synthesize ./voice.json --device cuda --text "你好，我是 Alice。" --output ./speech-cuda.wav
```

Use a CUDA-enabled ONNX Runtime DLL and the matching CUDA and cuDNN libraries.
The Windows build was tested with ONNX Runtime 1.30.0, CUDA 13 and cuDNN 9.
[CUDA setup](docs/standalone.md#cuda-直读)

</details>

## Integration

| Entry point | What the application receives |
| --- | --- |
| `alice-tts synthesize MODEL` | A file or binary WAV or PCM over stdout. `--text-file -` reads UTF-8 from stdin up to EOF. |
| `alice-tts worker MODEL` | A persistent JSONL subprocess with `ready`, PCM chunks, completion and cancellation. [JSONL protocol](docs/standalone.md#jsonl-协议) |
| `alice-tts serve MODEL` | HTTP WAV and WebSocket PCM using one loaded model. [HTTP and WebSocket](docs/standalone.md#http-与-websocket) |
| `alice_native_tts::session` | `Model` and `Session` in your Rust process, with PCM16 callbacks. [Session API](src/session.rs) |

Pipe WAV into FFmpeg in PowerShell 7.4 or later.

```powershell
alice-tts synthesize ./voice.json --text-file ./input.txt --output - |
  ffmpeg -i pipe:0 ./speech.flac
```

Use `--format s16le` with `--output -` to receive mono PCM16 as it is synthesized.
Pipeline stdout contains audio bytes, diagnostics go to stderr, and the exit code
reports whether synthesis completed. WAV output includes the sample rate and
arrives with a finalized header after synthesis. [CLI pipe examples](docs/standalone.md#命令行管道)

## Benchmarks

All three runs used one RTX 5070 Ti with driver 610.88, the same v2Pro checkpoints
and reference audio, and the same four Mandarin inputs. Each backend ran two passes
with one warmup per process and four timed requests per pass. Rust used ONNX Runtime
1.30.0 with CUDA 13, and Python used GPT-SoVITS with PyTorch 2.7.0 and CUDA 12.8.

Times are medians in milliseconds from JSONL request submission to the first PCM
event or `done`. Model loading and warmup happen before timing. Overall medians
pool all eight requests per backend. For RTF, divide each request’s synthesis
time by its audio duration, then take the median.

### Latest run

| Backend | First PCM | Short total | Medium total | Overall total | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust CUDA FP16 (before) | 749 | 1,598 | 3,308 | 2,432 | 0.534 |
| Rust CUDA FP16 (optimized) | **390** | **733** | 2,017 | 1,318 | **0.277** |
| Rust CUDA FP32 | 413 | 747 | **1,843** | **1,287** | 0.278 |
| Python CUDA FP16 | 616 | 916 | 1,926 | 1,461 | 0.303 |

In the same run, the FP16 VITS change reduced first PCM latency by **48.0%** and overall completion time by **45.8%**. The streaming vocoder uses cuDNN `DEFAULT`
for this path. In separate Nsight captures of the four inputs, the VITS range
total fell from 5.24 s to 1.65 s and device synchronization calls fell from
1,680 to zero. GPU kernel time increased from 230 ms to 865 ms.

Across the optimized and earlier FP16 outputs, duration changed by at most 7.84 ms
and the lowest waveform correlation in aligned voiced windows was 0.9986.
The two medium inputs were about 728 ms and 188 ms longer in optimized FP16
than in Rust FP32.

### Earlier runs

These are independent measurements. The old FP16 binary was also rerun in the
latest comparison above.

| Run | Backend | First PCM | Short total | Medium total | Overall total | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| First direct CUDA run | Rust CUDA FP32 | 399 | 715 | 1,810 | 1,175 | 0.270 |
| First direct CUDA run | Python CUDA FP16 | 602 | 863 | 1,906 | 1,400 | 0.302 |
| First FP16 run | Rust CUDA FP32 | 342 | 666 | 1,685 | 1,125 | 0.253 |
| First FP16 run | Rust CUDA FP16 | 607 | 1,356 | 3,012 | 2,128 | 0.471 |
| First FP16 run | Python CUDA FP16 | 446 | 716 | 1,357 | 1,074 | 0.236 |

<details>
<summary>Benchmark inputs</summary>

- 你好，今天的天气不错。
- 这杯茶已经凉了。
- 我把昨天整理的资料放在桌面上了，等你有空时，我们一起看一遍。
- 公交车还有十分钟就到。我们可以沿着这条路先走一段，到站牌旁边等。

</details>

Other measured paths include [ABI-1 CPU and DirectML](docs/performance.md),
[CUDA GPT](docs/performance-r1.1.md), [CUDA VITS](docs/performance-r1.2.md),
[native TensorRT GPT](docs/performance-r2.1.md),
[ABI-2 diffusion](docs/performance-native-diffusion.md) and
[diffusion scheduling](docs/performance-native-diffusion-scheduling.md).

## Models and runtimes

| Model input | Runtime | Audio delivery |
| --- | --- | --- |
| Original v2Pro `.ckpt` and `.pth` | Rust with `raw-checkpoint` and the shared v2Pro frontend | 32 kHz, token chunks |
| v2Pro ABI-1 pack | Rust and ONNX Runtime | 32 kHz, token chunks with the matching streaming graph |
| v3, v4, V5 Turbo and V5 Dev ABI-2 packs | Rust schedules GPT, diffusion and vocoding | Text fragments at 24 kHz for v3 and 48 kHz for the others |
| v1, v2, v2Pro, v2ProPlus, v3, v4, V5 Turbo and V5 Dev checkpoints | Local GPT-SoVITS Python adapter | Streaming mode and sample rate follow the model family |

`alice-tts pack` assembles ABI-1 packs from exported ONNX graphs and accepts
CMUdict and g2p_en inputs or existing frontend JSON. `pack.json` identifies a
pack and its reference files. Supply the model weights, frontend assets and
reference audio separately. ABI-1 token streaming needs its own
`sovits_stream.onnx` and matching metadata.
[Pack tools](docs/standalone.md#原生-abi-1-打包) · [Streaming graph exporter](tools/export-native-tts-stream.py) · [ABI-2 guide](docs/native-diffusion.md)

DirectML is available on Windows for compatible ONNX packs. The ORT TensorRT EP
uses `r1-tensorrt-ep`. Native TensorRT 11 GPT on Windows x64 uses `r2-trt-native`,
with `r2-trt-native-cuda-vits` for the optional CUDA vocoder.
[Native TensorRT setup](docs/performance-r2.1.md) · [CUDA vocoder](docs/performance-r2.2.md) · [CUDA text frontend](docs/performance-r2.3.md)

<details>
<summary>Native TensorRT GPT with a host active-length input</summary>

Generate the step graph from an ABI-1 pack and select it before loading the model.

```powershell
python tools/prepare-native-active-len.py ./model-pack/gpt_step.onnx ./target/active-len/gpt_step.onnx
$env:ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX = (Resolve-Path ./target/active-len/gpt_step.onnx).Path
$env:ALICE_NATIVE_TTS_TRT_CACHE_DIR = Join-Path (Get-Location) 'target/active-len-engines'
cargo run --locked --release --features "r0-profiling r2-trt-native-cuda-vits" --example r0_profile -- ./model-pack ./target/active-len-profile tensor_rt_native --chunks=24 --warmups=5 --repeats=10 --cuda-vits --cuda-vits-allow-cpu
```

The packed step graph is used by default. Engine caches are keyed by the graph
fingerprint.

</details>

### Native frontend and scope

The native frontend handles Mandarin and English using batched G2PW and
GRU-based English G2P. Number, date, time, currency, percentage and temperature
normalization covers supported display forms. Pronunciation coverage is limited
for neutral tones, erhua and English homographs that depend on part of speech.

ABI-1 uses a 512-token GPT cache, fixed speed 1.0 and zero acoustic noise.
ABI-2 uses family-specific diffusion settings. Cancellation is cooperative at
inference boundaries. Hosts own their model sessions and caches. The Project
Alice desktop adapter manages provider selection, playback and app-level events.

## License

Apache-2.0. Upstream frontend attribution is recorded in `NOTICE.md`, and model
pack provenance belongs in each pack’s `LICENSES.txt`. [License](LICENSE) · [Third-party notices](NOTICE.md)
