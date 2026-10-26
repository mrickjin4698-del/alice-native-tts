# R0 ONNX profiling and cache experiment

Measurement date: 2026-10-10. This is one local Windows host and one local
ABI-1 GPT-SoVITS v2Pro streaming pack. The pack and reference voice are not in
this repository.

## Setup and measurement rules

- CPU: Intel Core Ultra 9 285K, ORT configured for 4 intra-op and 1 inter-op
  thread.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GiB, driver 610.74.
- Rust release build, `ort` crate 2.0.0-rc.12.
- Text: `今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。`
- Each latency run discards one warmup synthesis, then measures 20 runs. ORT
  tracing and allocation capture use separate runs and are excluded from those
  latency samples.
- TTFA is request start to the first audio callback. RTF is elapsed time divided
  by generated audio duration. P95 uses nearest-rank selection from 20 samples.
- Memory is process working set and private bytes sampled every 10 ms. It does
  not include dedicated GPU VRAM. The Rust allocator probe excludes ORT's native
  C/C++ and DirectML allocations. Peak sampling spans the policy warmup, the 20
  latency runs, and the separate allocation probe.
- ORT event durations help rank graph work. Parent executor events overlap node
  events, so the two levels must not be added together. Tracing also increases
  first-run cost; use unprofiled runs for latency comparisons.

## Baseline bottlenecks

The warmed CPU chunk-24 run's first measured sample attributed about 2.44 s to
GPT Step, 0.90 s to streaming VITS, 0.14 s to the text frontend, and 0.073 s to
GPT Encoder. GPT Step called the ONNX graph 167 times for 167 semantic tokens;
VITS ran 7 times. A separate cold ORT trace measured 3.92 s across those GPT Step
calls and 1.05 s across VITS calls. The cold trace is for operator ranking, not
the warmed latency baseline.

| Rank | Phase | Representative warmed time | Evidence |
| ---: | --- | ---: | --- |
| 1 | GPT Step | 2,444 ms | 167 runs; each run consumes and returns the KV tensors |
| 2 | Streaming VITS | 900 ms | 7 runs; Conv 375 ms, FusedConv 321 ms, ConvTranspose 117 ms in the cold CPU trace |
| 3 | Text frontend | 143 ms | Request text is normalized and converted once per synthesis |
| 4 | GPT Encoder | 73 ms | One run; MatMul 59 ms and Pad 15 ms in the cold CPU trace |
| 5 | Rust-side input setup and copying | under 1 ms | GPT Step input 0.77 ms, host copies 0.48 ms, VITS input 0.04 ms in the cold trace run |

The CPU GPT Step trace's leading nodes were Gemm (1,045 ms across 16,032
events), Split (378 ms), Transpose (366 ms), Slice (329 ms), Concat (318 ms),
and Reshape (303 ms). The trace attributes those nodes to
`CPUExecutionProvider`. The graph returns `k_cache_new` and `v_cache_new` on
every token; the trace contains 8,016 Slice, 334 Concat, and 16,032 Transpose
events for one 167-token synthesis. This is the largest structural optimization
area. A cache-update graph redesign would need a versioned pack extension and
parity checks; no graph or model semantics were changed here.

DirectML did not move GPT Step to the GPU in this pack: its leading GPT Step
nodes, including Gemm, also ran on `CPUExecutionProvider`. VITS had
`DmlExecutionProvider` nodes. For warmed chunk-24 DirectML, the first measured
sample attributed about 2.47 s to GPT Step, 0.52 s to VITS, 0.056 s to the text
frontend, and 0.077 s to GPT Encoder.

The fixed reference spectrogram (`spec`) and speaker embedding (`sv`) are
already computed once at model load and passed as ORT values to every VITS
chunk. The phone/text condition was copied into a fresh tensor per chunk; the
experiment below caches that tensor for the request. DirectML's cold trace
reported 13.1 ms of `MemcpyFromHost` across 553 VITS events. A trial of
`IoBinding` with this graph's dynamic audio/latent outputs did not run: the
DirectML provider rejected the attempted CPU output binding. ORT's [I/O
binding guide](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
describes placing inputs and outputs on the target device to avoid transfers;
the [DirectML provider notes](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
also call out the benefit of known tensor shapes. The measured transfer time is
small next to GPT Step and VITS compute, so no binding path was retained.

## Chunk-size comparison

CPU ABI-1 streaming pack, before the cache change; each row has 20 warmed
samples. Generated semantic output was 167 tokens with the same hash for all
four chunk sizes.

| Chunk tokens | VITS calls | TTFA P50 / P95 (ms) | Elapsed P50 / P95 (ms) | RTF P50 / P95 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 21 | 370 / 389 | 4,235 / 4,288 | 0.640 / 0.648 |
| 16 | 11 | 518 / 530 | 3,740 / 3,783 | 0.562 / 0.569 |
| 24 | 7 | 660 / 673 | 3,574 / 3,648 | 0.538 / 0.549 |
| 32 | 6 | 817 / 829 | 3,581 / 3,640 | 0.537 / 0.546 |

Chunk 8 lowered TTFA P50 by 44% against chunk 24 and increased elapsed P50 by
18%. Audio output also changed: the four policies produced 211,698, 212,930,
212,777, and 213,287 samples, and each raw waveform had a different SHA-256.
The semantic token hash stayed the same. No listening or perceptual quality
study was run, so the default chunk size remains 24.

## Implemented request-scoped tensor reuse

The single retained optimization reuses invariant inputs without changing the
ABI-1 graphs or public synthesis API:

- GPT Step reuses the `x_len` and `y_len` ORT values returned by GPT Encoder
  instead of copying them into new Rust vectors and tensors on every token.
- Streaming VITS creates the request's `text_seq` tensor at its first chunk and
  reuses it for later chunks.

Before/after uses the same local pack, text, provider, and chunk size; each side
has 20 warmed samples. The allocation probe is a separate synthesis.

| Backend | Measure | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| CPU | TTFA P50 / P95 (ms) | 660 / 673 | 662 / 685 | +0.4% / +1.9% |
| CPU | Elapsed P50 / P95 (ms) | 3,574 / 3,648 | 3,594 / 3,642 | +0.5% / -0.2% |
| CPU | RTF P50 / P95 | 0.538 / 0.549 | 0.540 / 0.548 | +0.5% / -0.1% |
| DirectML | TTFA P50 / P95 (ms) | 483 / 525 | 486 / 573 | +0.6% / +9.3% |
| DirectML | Elapsed P50 / P95 (ms) | 3,137 / 3,287 | 3,200 / 3,258 | +2.0% / -0.9% |
| DirectML | RTF P50 / P95 | 0.472 / 0.494 | 0.481 / 0.490 | +2.0% / -0.9% |

The latency differences are within run-to-run variation; this change has no
confirmed TTFA or RTF gain. It reduced measured Rust allocation calls from
5,357 to 3,995 (25%), GPT Step input-phase allocations from 2,338 to 1,336
(43%), and host-copy-phase allocations from 368 to 28 (92%). Counted input copy
bytes fell from 399,472 to 393,680. The allocator's measured time did not move
consistently between backends, so allocation counts are the supported result.

For both CPU and DirectML, the before/after semantic token hash remained
`15228212952513995076`; sample count remained 212,777; and the raw waveform
SHA-256 was identical within that backend. CPU and DirectML waveforms were not
compared to each other.

Peak process memory was stable within the sampling resolution: about 3.73 GB
private bytes / 3.59 GB working set on CPU, and 4.49 GB private bytes / about
2.23 GB working set on DirectML. Model-load measurements were 5.7–6.0 s for CPU
and 5.85–5.88 s for DirectML. A separate cold, profiled DirectML run measured
2.82 s TTFA and 7.25 s elapsed; it includes first-use and profiler overhead and
is not comparable to the warmed distributions.

## Reproduction

Build the profiling harness and run against a local compatible ABI-1 streaming
pack. The harness writes JSON summaries and raw little-endian f32 audio under
the output directory. It does not include model files in the repository.

```powershell
cargo run --release --features r0-profiling --example r0_profile -- `
  ./model-pack ./r0-cpu cpu --chunks=8,16,24,32 --repeats=20

cargo run --release --features r0-profiling --example r0_profile -- `
  ./model-pack ./r0-directml directml --chunks=24 --repeats=20

cargo run --release --features r0-profiling --example r0_profile -- `
  ./model-pack ./r0-ort-trace cpu --chunks=24 --repeats=1 --ort-profile
```

For a fair before/after comparison, use the same pack, backend, text, chunk
policy, build mode, warmup count, and power profile. The R0 harness fixes the
sample text and reports TTFA, elapsed time, RTF, process memory, Rust allocator
counts, input/output copy bytes, semantic-token hash, stage timings, and ORT
node/provider events.
