# R1.1 CUDA GPT backend and resident KV cache

Measurement date: 2026-10-26. This report supersedes the CUDA status in the
[R1.0 GPU audit](performance-r1-gpu.md). It covers one Windows x64 host and
one local ABI-1 GPT-SoVITS v2Pro streaming pack. The model pack and reference
voice are not in this repository.

## Result

The CUDA path now completes real ABI-1 streaming synthesis on the RTX 5070 Ti.
GPT Step is strict CUDA-only, K/V cache tensors remain on CUDA between token
steps, and the CPU token sampler and CPU VITS path are unchanged. With identical
inputs, seed, tokens, and waveform, 20 measured CUDA samples after 30 warmups
improved median TTFA by 31.7%, GPT Step time by 51.5%, and elapsed time by
35.7% versus the CPU EP in the same ORT 1.30 runtime. CPU remains the default;
CUDA stays opt-in.

## Reproducible runtime combination

| Component | Verified local version |
|---|---|
| OS / architecture | Windows x64 |
| CPU | Intel Core Ultra 9 285K |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 (SM120), 16 GiB |
| Driver | 610.74 |
| Rust wrapper | `ort` 2.0.0-rc.12, dynamic loading enabled by `r1-cuda-ep` |
| ONNX Runtime | 1.30.0, build commit `f2c39fe2f` |
| CUDA runtime / cuBLAS | CUDA 13.0 (`torch 2.10.0+cu130` native DLLs) |
| cuDNN | 9.12.0.46 |

The ORT native DLLs came from the official `onnxruntime-gpu==1.30.0` Windows
wheel, SHA-256
`CF494DA233F3FC02BBEB2A3178E3ACE7361B6B4FEF7031B19C1E5AED5E9328CE`.
Only native DLLs were loaded by the Rust process; no Python inference process
or service was used. Set `ORT_DYLIB_PATH` to the extracted `onnxruntime.dll`
and put the ORT native DLL directory plus the CUDA 13/cuDNN DLL directory on
`PATH`. The CUDA wheel is deliberately not checked into the repository.

ORT's published CUDA matrix lists 1.30.x with CUDA 13.0 and cuDNN 9.x. NVIDIA
lists cuDNN 9.12.0 for CUDA 13.0, Blackwell, and compute capability 12.0; its
CUDA 13 table does not publish a Windows minimum driver value. CUDA 13 requires
a driver from R580 or later for minor-version compatibility, and this host is
on R610. These matrix facts align with the actual Windows execution results
below. See the [ORT CUDA EP matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[NVIDIA cuDNN 9.12 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.12.0/reference/support-matrix.html),
[CUDA driver compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/),
and [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus).

The previous ORT 1.24.2 runtime failed a one-node `Add` graph with CPU
fallback disabled: `cudaErrorNoKernelImageForDevice`. This was an SM120 kernel
compatibility failure before GPT graph execution. The ORT 1.30.0 CUDA 13
runtime passed the same strict Add test with output `[11, 22, 33, 44]`; its
ORT profile assigns the `Add` node to `CUDAExecutionProvider`.

### TensorRT status: BLOCKED

The ORT 1.30 wheel contains the TensorRT provider DLL, but this host has no
`nvinfer_10.dll` / TensorRT SDK runtime. The ORT TensorRT compatibility page
currently documents TensorRT 10.9 with CUDA 12.0–12.8 for `main` and 1.22; it
does not establish a 1.30/CUDA 13 combination. TensorRT was compile-checked
only and no TensorRT session or synthesis was attempted. This is an independent
blocker and does not affect the validated CUDA EP result. See the [ORT
TensorRT EP matrix](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

## Graph placement and K/V transfer proof

The minimal `Add` smoke used `with_disable_cpu_fallback()` and recorded
`CUDAExecutionProvider` as the executing provider. For the ABI-1 pack, an
explicit diagnostic opt-in allows audited helper nodes on CPU in GPT Encoder;
GPT Step remains CPU-fallback-disabled even in that diagnostic mode.

| Session | ORT trace evidence |
|---|---|
| GPT Encoder | 826 CUDA nodes, 272 CPU helper nodes; no CPU Gemm, MatMul, Conv, Attention, LayerNormalization, or RMSNormalization nodes. CPU helper operator kinds in the trace: Gather, Concat, Sqrt, Unsqueeze, Slice, Div, Cast, Reshape, Sub, Transpose. |
| GPT Step | 864/864 nodes assigned to CUDA; for 167 token steps, 144,288 CUDA node events (= 864 × 167), zero CPU node events. CPU fallback is disabled in code for this session. |
| VITS stream | CPU EP, unchanged. |

K/V cache handling uses two long-lived `IoBinding`s on the GPT Step session.
Each binding keeps its `k_cache_new` and `v_cache_new` outputs bound to CUDA
device 0; their output buffers are reused. Calls alternate bindings, so the
current input cache is read from the other binding while the selected binding
writes its output. The encoder likewise binds initial K/V outputs to CUDA.
Token IDs, indices, and sampling logits remain host-visible for the existing
CPU sampler.

For each Step call the code asserts CUDA device 0 for both K/V inputs and both
K/V outputs, then rejects input/output pointer aliasing. The 167-step measured
run recorded 167 resident-cache Steps and 668 endpoint residency assertions.
Because K/V inputs and outputs stay on the same CUDA device and every GPT Step
node is assigned to CUDA, there is no per-token K/V host/device round-trip.
This is an endpoint-placement proof, not an ORT byte counter; small token and
index inputs and top-k sampling outputs still cross the CPU/GPU boundary.

## Benchmark

Both backends used the same local ABI-1 pack, text
`今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。`,
seed 1234, model configuration, 8 GPT Step threads, streaming mode, and Chunk
24. Each process ran 30 warmups and 20 measured complete syntheses. ORT operator
profiling was disabled during latency measurement. TTFA is request start to
first audio callback; RTF is elapsed synthesis time divided by generated audio
duration.

| Measure | CUDA P50 / P95 | CPU P50 / P95 | CUDA change |
|---|---:|---:|---:|
| TTFA | 387 / 400 ms | 567 / 613 ms | −31.7% / −34.7% |
| GPT Step total | 914 / 933 ms | 1,885 / 1,989 ms | −51.5% / −53.1% |
| GPT Step per token, pooled over 3,340 steps | 5.29 / 6.46 ms | 11.35 / 12.60 ms | −53.3% / −48.7% |
| Total elapsed | 1,913 / 1,935 ms | 2,975 / 3,117 ms | −35.7% / −37.9% |
| RTF | 0.288 / 0.291 | 0.447 / 0.469 | −35.7% / −37.9% |

Warm CUDA phase P50/P95 was 6.27/8.90 ms for GPT Encoder, 914/933 ms for GPT
Step, and 849/853 ms for VITS. The largest remaining stages are GPT Step and
CPU VITS; text frontend processing was 137/140 ms. The cache-resident GPU path
helps GPT, while VITS remains a large share of total latency.

Process memory peak above the post-load baseline was 603 MiB private / 447 MiB
working set for CUDA and 464 MiB private / 381 MiB working set for CPU. VRAM
was sampled with `nvidia-smi` every 100 ms. It reported system-wide WDDM usage,
not process-attributed memory: the CUDA window ranged from 2,658 to 4,031 MiB
with a pre-run reading of 2,778 MiB; the CPU window ranged from 2,394 to 2,954
MiB with a pre-run reading of 2,430 MiB. Other GPU processes changed memory
usage between runs, so the delta is not a reliable TTS-only VRAM measurement.

## Numeric and audio consistency

Across all 20 samples, CPU and CUDA produced the same 167-token hash
`15228212952513995076` and the same 212,777-sample output length. The first
measured raw f32 audio file from each backend had the same SHA-256
`D479C7A6B5A57CBFAA77B36C83C97F65C901504608B00FABF91B1271F20FBEEC`; direct
sample comparison gave max absolute error 0, MAE 0, and cosine similarity 1.
This is bit-identical audio to the CPU reference for the fixed test input; no
separate human listening study was run.

No ONNX graph or weight changed. ABI-1 pack loading, sample seed, CPU sampler,
Chunk 24, callback ordering, and cancellation check locations remain unchanged.
CPU end-to-end synthesis and the CPU EP baseline completed. DirectML was not
re-run with this CUDA runtime package because it does not include
`onnxruntime_providers_dml.dll`; the existing DirectML route was not altered,
but its R1.1 runtime gate remains unverified. The R1.0 report contains an
earlier one-shot DirectML smoke on its separate ORT runtime.

## Changes and validation

- `Cargo.toml`, `Cargo.lock`: add opt-in dynamic ORT CUDA/TensorRT features for
  use with an explicitly selected native ORT runtime.
- `examples/cuda_smoke.rs`: add strict single-`Add` CUDA runtime smoke.
- `src/inference.rs`: add CUDA I/O Binding for Encoder outputs and GPT Step,
  persistent alternating output bindings, K/V device and alias assertions,
  and keep GPT Step strict against CPU placement.
- `src/profiling.rs`, `examples/r0_profile.rs`: report resident-cache steps and
  endpoint assertions, provider placement, latency percentiles, and which
  audited Encoder CPU-helper opt-in was used.
- `docs/performance-r1-gpu.md`, `docs/performance.md`, README links: mark the
  R1.0 GPU audit historical and point to this report.

Checks executed:

| Check | Result |
|---|---|
| Strict minimal CUDA Add on ORT 1.30 / CUDA 13 | Passed; CUDA node trace, exact output |
| ABI-1 CUDA E2E with strict GPT Step | Passed; 167 tokens, 7 audio callbacks |
| CUDA placement trace | Passed; 864/864 GPT Step nodes on CUDA; 826 Encoder nodes on CUDA; CPU helper nodes reported |
| 30 warmups + 20 CUDA samples | Passed |
| 30 warmups + 20 CPU samples | Passed |
| `cargo test --release --lib --features 'r0-profiling r1-cuda-ep'` | 14 passed, 0 failed, 2 ignored for model-pack fixtures |
| Default-feature `cargo check --release --features r0-profiling --example r0_profile` | Passed |
| TensorRT feature compile check | Passed; runtime remains BLOCKED as above |
| Default-feature `cargo test --release --lib` | Could not link: this host lacks static `onnxruntime.lib`; dynamic ORT test configuration passed |
| DirectML R1.1 E2E | Not run: the selected ORT 1.30 wheel has no DirectML provider DLL |

The smoke and benchmark artifacts remain local under the ignored `target/`
directory: `r1.1-cuda-add-final`, `r1.1-io-binding-smoke`,
`r1.1-cuda-bench`, and `r1.1-cpu-ort130-bench`. The 162 MB GPT Step ORT trace
is intentionally not copied into source control.

## Remaining work

1. Keep CUDA opt-in. The measured GPU result is faster on this single host and
   fixture, but the CPU path remains portable and no cross-hardware claim is
   supported.
2. Install a documented ORT/TensorRT combination that includes a matching
   TensorRT 10 runtime, then repeat strict graph placement, numerical checks,
   and 30+20 benchmarks before marking TensorRT usable.
3. Re-run the DirectML E2E on its matching ORT provider runtime to close the
   explicit R1.1 regression gate.
4. Profile CPU VITS and the request-level frontend next. Keep host sampling as
   the semantic reference until a separate GPU sampler equivalence test is
   designed.
