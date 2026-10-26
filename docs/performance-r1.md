# R1 GPT Step and KV cache profiling

Measurement date: 2026-10-11. This report covers one local Windows host and one local ABI-1 GPT-SoVITS v2Pro streaming pack. The model pack and reference voice are not in this repository.

## Setup and measurement rules

- CPU: Intel Core Ultra 9 285K. GPU: NVIDIA GeForce RTX 5070 Ti, 16 GiB, driver 610.74.
- Rust `ort` 2.0.0-rc.12, ONNX Runtime 1.24.2 CUDA 13 build for the final paired CPU runs; the selected CPU backend used `CPUExecutionProvider`.
- Chunk size 24, fixed R0 test text, seed 1234, model config, and ABI-1 pack were unchanged.
- Each setting used 3 discarded warmups and 30 measured syntheses. P95 is nearest-rank. Per-token percentiles pool 5,010 GPT Step calls (30 runs × 167 tokens).
- TTFA is request start to the first audio callback. RTF is elapsed synthesis time divided by generated audio duration.
- Memory is process working set and private bytes sampled every 10 ms. It does not include GPU VRAM. CUDA did not complete a synthesis, so no CUDA memory result is available.
- The ORT operator trace is a separate instrumented run. Its node timings rank operators; they are not added to parent/session durations or to unprofiled latency results.

## Graph findings

The raw `gpt_step.onnx` contains 2,987 top-level nodes. Its K and V inputs each have shape `[24, batch, 512, 512]`; the outputs have the same fixed cache capacity. At runtime, batch is 1 and the context length grows inside that fixed 512-position buffer. Across 167 token steps, the trace shows:

| Operator | Calls | Trace time | Runtime shape evidence | Finding |
| --- | ---: | ---: | --- | --- |
| Gemm | 16,032 | 1,045 ms | 96 Gemm calls per token; common fused projections map `[1,512] → [1,2048]` and `[1,2048] → [1,512]` | Largest traced operator family; projections are repeated because the autoregressive token changes each step. |
| Split | 4,342 | 378 ms | Two cache Split nodes each run 167 times; each `[24,1,512,512]` input yields 24 `[1,1,512,512]` values | The two cache nodes account for about 344 ms. Other Split nodes account for the remaining trace time. |
| Transpose | 16,032 | 366 ms | Repeated attention transforms such as `[1,180..346,16,32] → [1,16,32,180..346]` | Attention/cache layout remains a large optimization area. |
| Slice | 8,016 | 329 ms | 48 calls per token; cache-sized `[1,512,512]` tensors produce valid-prefix tensors such as `[1,180..346,512]` | Repeatedly selects the active context from the fixed cache. |
| Concat | 334 | 318 ms | Two nodes per token combine 24 `[1,1,512,512]` layer values into `[24,1,512,512]` | Reassembles fixed-size K/V outputs on every step. |
| Reshape | 48,597 | 303 ms | About 291 calls per token; many reshape small projection tensors | Numerous shape operations; not all imply a data copy. |

The cache boundary tensors represent 24 MiB per K or V cache. The two Split nodes and two Concat nodes handle a nominal 96 MiB of cache tensor values per token step. This is a useful structural target, but the graph’s per-layer compute is not invariant across tokens.

All listed trace nodes ran on `CPUExecutionProvider`. The GPT Step remains the largest warmed stage: about 2.44 s of the ABI-1 30-run CPU baseline. The node trace is cold/profiled and should be used for ranking, not as a warmed timing comparison.

## Cache-layout graph experiment

I generated a local `gpt_step_v2.onnx` candidate that removes the two cache-boundary Split nodes and two Concat nodes, exposes 24 K tensors and 24 V tensors as graph inputs, and exposes the updated layers as 48 outputs. The layer computations and model weights were unchanged. This candidate was opt-in, and ABI-1 remained the default.

The candidate produced the same 167 semantic tokens and byte-identical waveform, but it was slower in 30 CPU samples:

| Measure | ABI-1, 4 threads | Per-layer cache graph, 4 threads | Change |
| --- | ---: | ---: | ---: |
| TTFA P50 | 662 ms | 832 ms | +25.7% |
| GPT Step P50 | 2,443 ms | 3,492 ms | +42.9% |
| Total elapsed P50 | 3,555 ms | 4,613 ms | +29.7% |
| RTF P50 | 0.535 | 0.694 | +29.7% |
| Peak private bytes | 3.72 GB | 3.78 GB | +51 MB |

The expanded 52-input/50-output interface removed the traced cache boundary operators but increased measured run time. The exact cause is not isolated; extra input/output handling and longer-lived graph values are candidates. The v2 graph was not made the default or included in the repository or model pack; the local test artifact stays outside the source tree.

## Selected CPU optimization: GPT Step thread count

The Gemm family dominates the trace, so I tested the ONNX Runtime intra-op thread count for the GPT Step session while leaving other sessions at four threads. A 5-sample screening at 1, 2, 8, and 16 threads favored 8. I then confirmed the setting with 30 post-warmup samples. The ABI-1 graph remained in use.

| Measure | ABI-1, 4 threads | ABI-1, 8 threads | Change |
| --- | ---: | ---: | ---: |
| TTFA P50 / P95 | 662 / 673 ms | 623 / 638 ms | −5.8% / −5.2% |
| GPT Step total P50 / P95 | 2,443 / 2,496 ms | 2,176 / 2,209 ms | −10.9% / −11.5% |
| Per-token GPT Step P50 / P95 | 14.63 / 15.98 ms | 12.99 / 14.08 ms | −11.2% / −11.9% |
| Total elapsed P50 / P95 | 3,555 / 3,619 ms | 3,265 / 3,296 ms | −8.2% / −8.9% |
| RTF P50 / P95 | 0.535 / 0.544 | 0.491 / 0.496 | −8.2% / −8.9% |
| Peak private bytes / working set | 3.72 / 3.59 GB | 3.73 / 3.59 GB | about +6 MB each |

On this machine, the no-override GPT Step thread count now uses `min(available_parallelism, 8)`. `ALICE_NATIVE_TTS_GPT_STEP_THREADS` can set a value from 1 to 64. The 30-run result used 8 explicitly; a no-override smoke run resolved to 8. Other ONNX sessions remain at four intra-op threads.

## Semantic and audio consistency

The 4-thread and 8-thread CPU runs each produced 167 semantic tokens with hash `15228212952513995076`. Both produced 212,777 audio samples. Their raw little-endian f32 waveform SHA-256 was identical: `45fdc080807296a1e9c9dfb4624fa94fb6f3bc0b16347dd6fba33620d9aacc36`. Numeric comparison reported max absolute error 0 and RMSE 0. Chunking, seed, and model configuration were unchanged. The v2 cache-layout experiment also matched the ABI-1 waveform exactly, despite its latency regression.

Cancellation checks remain at the same per-token and decode boundaries. Existing ABI-1 packs continue to load `gpt_step.onnx`; the experimental v2 loader requires an explicit feature and environment selection.

## CUDA EP attempt

The optional Rust ORT CUDA 13 provider compiled and registered. The first v2 CUDA smoke reached execution but failed at `Slice_12` with `cudaErrorNoKernelImageForDevice`; `CUDA_FORCE_PTX_JIT=1` produced the same error. No CUDA TTFA, RTF, or output-consistency sample was collected. The CUDA path remains opt-in and no GPU speedup is claimed. DirectML measurements are not used as a substitute for CUDA.

## Remaining work

- A compatible ORT CUDA kernel build for RTX 5070 Ti is needed before comparing CPU and CUDA EP latency or VRAM use.
- The dominant Gemm, attention Transpose, and valid-prefix Slice work remains. A cache rewrite that expands the model interface did not pay off; a future attempt should reduce cache traffic without creating dozens of graph I/O values.
- Results cover one local pack, text, CPU, and machine. The 8-thread setting still needs measurement on other CPU classes before making a cross-platform performance claim.

## Reproduction

Run against a local ABI-1 streaming pack. The harness writes JSON, raw f32 output, stage totals, and per-token GPT Step times. The model pack is not included in the repository.

```powershell
$env:ALICE_NATIVE_TTS_GPT_STEP_THREADS = '4'
cargo run --release --features 'r0-profiling' --example r0_profile -- `
  ./model-pack ./r1-cpu-4 cpu --chunks=24 --warmups=3 --repeats=30

$env:ALICE_NATIVE_TTS_GPT_STEP_THREADS = '8'
cargo run --release --features 'r0-profiling' --example r0_profile -- `
  ./model-pack ./r1-cpu-8 cpu --chunks=24 --warmups=3 --repeats=30
```