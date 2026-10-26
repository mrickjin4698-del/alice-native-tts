# R1.0 GPU backend audit: CUDA and TensorRT EP

> Historical audit of the initial ORT 1.24.2 runtime attempt. The later
> CUDA 13 / ORT 1.30.0 R1.1 implementation and measured results are in
> [performance-r1.1.md](performance-r1.1.md).

Audit date: 2026-10-26. This report covers the local ABI-1 GPT-SoVITS v2Pro
streaming pack and the Windows host recorded in [R1 GPT Step profiling](performance-r1.md).
The pack and reference voice remain outside this repository.

## Decision

CUDA and TensorRT are now opt-in build features, but neither backend passed the
runtime gate on this host. CPU remains the default. The experimental GPU path
targets only `gpt_encoder` and `gpt_step`; frontend/reference preparation and
the vocoder remain explicitly on CPU in this R1 scope. CPU EP fallback is
disabled for both GPT sessions, so graph placement problems are reported as
errors instead of appearing to be GPU inference.

No CUDA or TensorRT full inference completed. There are no GPU TTFA, RTF,
per-token latency, VRAM, or output-quality results, and no 20-sample hot
benchmark. No GPU performance gain is claimed.

## Runtime and compatibility audit

- Rust wrapper: `ort` 2.0.0-rc.12. The local prebuilt native runtime is ORT
  1.24.2 with the CUDA 13 provider package.
- GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 (SM120); driver
  610.74; CUDA UMD 13.3. The ORT CUDA provider DLL needs CUDA 13 user-mode
  libraries. They were not on the default `PATH`; for the provider-load smoke
  test, the local CUDA 13 DLL directory was added to `PATH`.
- TensorRT: the ORT TensorRT provider DLL is present, but its load fails because
  `nvinfer_10.dll` is not installed. The Python package probe also found no
  TensorRT SDK. This is a missing native runtime dependency, not a Python
  inference service failure.
- The current ONNX Runtime TensorRT EP compatibility table lists TensorRT 10.9
  with CUDA 12.0–12.8 for its documented current/main build. AnifLive-TTS
  reports TensorRT 11.2.1.2 with CUDA 12.8 and uses TensorRT `execute_async_v3`
  directly for its neural stages. Those are different runtime contracts; its
  SDK cannot be assumed to satisfy this ORT provider DLL. See the [ORT TensorRT
  compatibility and provider-order guidance](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
  and [AnifLive-TTS README](https://github.com/Hiruynk/AnifLive-TTS/blob/main/README.md).

AnifLive-TTS is useful as a design reference, not a directly comparable
benchmark: its published measurements use a different V2ProPlus pack and a
dedicated TensorRT 11 FP16 runtime. The relevant ideas are fitted persistent
GPT contexts, fixed GPU-resident KV buffers, GPU-resident reference tensors,
and reducing per-token launch/synchronization work. Its README also notes that
full GPT-step CUDA Graph capture remains constrained by TensorRT compatibility.

## Graph and transfer findings

The R0 CPU ORT trace contains 2,987 top-level nodes in `gpt_step.onnx`. The
fixed K/V cache inputs and outputs are each shaped `[24, 1, 512, 512]` at this
workload: 24 MiB per K or V tensor, 48 MiB together. The step graph runs 167
times for the 167 generated semantic tokens. Its largest traced operator
families are Gemm (96 calls/token, 1,045 ms total), Transpose (96/token,
366 ms), Slice (48/token, 329 ms), Split (26/token, 378 ms), Concat (2/token,
318 ms), and Reshape (291/token, 303 ms). These are cold/profiled operator
totals and rank the CPU graph; they are not GPU timing results.

The profiled GPT Encoder has one request-level run. For the R0 reference text
shape `[1, 179, 512]`, MatMul accounts for 59.5 ms across 145 calls, Pad 15.1 ms
across 2 calls (including cache padding from `[24, 1, 179, 512]` to
`[24, 1, 512, 512]`), and Add 9.3 ms across 172 calls. All those baseline trace
nodes ran on `CPUExecutionProvider`.

The existing Rust path uses ordinary `Session::run` with CPU `OrtValue`s. ORT
documents that CPU inputs are copied to the execution device and outputs are
copied back to CPU unless I/O Binding places them on the device. A CUDA GPT
step would therefore move about 48 MiB of K/V cache to the GPU and 48 MiB back
per token, or about 15.7 GiB of aggregate cache transfers over 167 steps, if
left on this path. [ORT I/O Binding guidance](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
describes device-bound inputs and reusable device outputs as the way to avoid
those copies.

`topk_values` and `topk_indices` are `[1, 50]`, together about 600 bytes per
step. The Rust sampler consumes the first 15 candidates and advances its
existing seed-1234 PRNG once per token. The transfer volume is small beside the
cache, but sampling still requires a host-visible result and synchronization
each step. Moving sampling onto CUDA would need a separate equivalence proof for
the current RNG and token sequence.

## Code changes

- `Cargo.toml`: add opt-in `r1-cuda-ep` and `r1-tensorrt-ep` features. The
  latter registers TensorRT before CUDA as its GPU fallback.
- `src/lib.rs`: expose feature-gated `Backend::Cuda` and `Backend::TensorRt`;
  existing CPU and DirectML variants and the default are unchanged.
- `src/inference.rs`: route GPT Encoder and the ABI-1 GPT Step through the
  selected EP; reject CPU-assigned nodes in both sessions. TF32 and TensorRT
  FP16 are disabled pending numeric and audio-quality validation. TensorRT
  engine/timing caches use `ALICE_NATIVE_TTS_TRT_CACHE_DIR`, or a per-user
  application cache location when unset. Frontend/reference preparation and
  VITS remain explicitly CPU in this GPU-only GPT step.
- `examples/r0_profile.rs`: include per-node execution-provider counts in ORT
  profile summaries so successful runs show actual placement rather than only
  the requested backend.
- `README.md`, `README.zh-CN.md`, and `docs/performance.md`: describe feature
  selection and link this report.

No ONNX graph, model weight, chunk size, sampler, seed, cancellation check, or
ABI-1 package contract was changed for this GPU attempt.

## Validation

Build and unit-test evidence:

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Passed |
| `cargo build --release --features 'r0-profiling r1-tensorrt-ep r1-gpt-step-v2' --example r0_profile` | Passed with `ORT_CUDA_VERSION=13` |
| `cargo test --release --lib --features 'r0-profiling r1-tensorrt-ep r1-gpt-step-v2'` | 14 passed, 0 failed, 2 ignored because external model-pack fixtures are required |

One-warmup/one-measurement CPU and DirectML smoke runs used chunk 24, the same
text, seed, configuration, and ABI-1 pack. These are regression smokes, not
P50/P95 benchmarks:

| Backend | TTFA | GPT Step | Elapsed | RTF | Tokens / samples | Wave SHA-256 |
|---|---:|---:|---:|---:|---|---|
| CPU | 677.9 ms | 2,374.4 ms | 3,514.4 ms | 0.529 | 167 / 212,777 | `45fdc080807296a1e9c9dfb4624fa94fb6f3bc0b16347dd6fba33620d9aacc36` |
| DirectML | 484.4 ms | 2,349.1 ms | 3,006.1 ms | 0.452 | 167 / 212,777 | `57f67b0318b1e4977713f944925800d3f29b9bb591bf8f0b131ceae7acc3f18a` |

Each smoke produced the same semantic-token hash,
`15228212952513995076`, as its own R0 backend baseline and a byte-identical
waveform to that baseline. CPU and DirectML waveforms are not expected to be
byte-identical to each other. The single sample per backend is not evidence of
a timing improvement or a statistically complete quality gate.

GPU attempts:

| Backend | Attempt | Result |
|---|---|---|
| CUDA EP | ABI-1, chunk 24, strict GPT placement, one smoke attempt | GPT Encoder session creation rejected because one or more nodes were assigned to the default CPU EP. No CPU fallback was allowed and no synthesis ran. |
| TensorRT EP | ABI-1, chunk 24, one smoke attempt | Provider load failed because `nvinfer_10.dll` is missing. No graph placement or synthesis ran. |

An earlier non-strict CUDA smoke on the experimental per-layer v2 graph with
the local ORT CUDA 13 package reached a `Slice_12` kernel and failed with
`cudaErrorNoKernelImageForDevice`, including with `CUDA_FORCE_PTX_JIT=1`; this
is recorded in the [R1 GPT Step report](performance-r1.md). The current strict
ABI-1 run fails earlier at GPT Encoder placement. Thus the local
ORT CUDA package is not yet a validated SM120 runtime for this model.

## Remaining work and next stage

1. Obtain a matched ORT/CUDA build that supports SM120 and a matched TensorRT EP
   SDK/runtime. Re-run GPT Encoder and GPT Step with CPU fallback disabled and
   collect the per-node provider trace.
2. Implement persistent device-resident K/V buffers and I/O Binding, likely
   using reusable ping-pong outputs so an output buffer is not aliased with the
   current input. This is the largest safe target by bytes transferred, but it
   changes tensor ownership and needs GPU equivalence tests.
3. Keep the current host sampler initially. Measure token synchronization
   separately before considering a CUDA sampler; preserve the seed and token
   sequence if it is later changed.
4. Evaluate CUDA Graph only after inputs, outputs, addresses, and dynamic context
   shapes satisfy ORT capture constraints. Do not copy an optimization claim
   from the different AnifLive-TTS model/runtime.
5. After a compatible provider runs end to end, compare CPU, CUDA, and TensorRT
   using at least 20 hot chunk-24 samples after warmup. Record GPT per-token and
   total latency, TTFA, RTF, P50/P95, process memory, VRAM, semantic tokens,
   waveform error, and listening/quality gates. Keep CPU as default unless the
   full performance and quality gates pass.

## Reproduction

For a compatible host/runtime, compile the profiler with the chosen EP feature
and keep the same local ABI-1 pack, text, seed, model settings, and chunk size:

```powershell
cargo run --release --features 'r0-profiling r1-cuda-ep' --example r0_profile -- `
  ./model-pack ./r1-cuda-placement cuda --chunks=24 --ort-profile

cargo run --release --features 'r0-profiling r1-tensorrt-ep' --example r0_profile -- `
  ./model-pack ./r1-tensorrt-placement tensor_rt --chunks=24 --ort-profile

cargo run --release --features 'r0-profiling r1-cuda-ep' --example r0_profile -- `
  ./model-pack ./r1-cuda-bench cuda --chunks=24 --warmups=30 --repeats=20

cargo run --release --features 'r0-profiling r1-tensorrt-ep' --example r0_profile -- `
  ./model-pack ./r1-tensorrt-bench tensor_rt --chunks=24 --warmups=30 --repeats=20
```

The first two commands collect placement traces; latency commands omit
`--ort-profile` so profiling overhead does not contaminate timing. Use at least
30 discarded warmups and 20 measured samples.
