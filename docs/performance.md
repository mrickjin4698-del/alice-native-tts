# Performance notes

These numbers are an engineering snapshot captured on 2026-10-03. They are not
a repeatable cross-platform benchmark and should not be read as a speed promise.
The current implementation has not had a systematic quantization, weight
deduplication, or broad hardware-tuning pass. It does contain targeted changes
for batching, earlier audio delivery, optional token streaming, and a Windows
DirectML backend.

## Test machine and measurement rules

- CPU: Intel Core Ultra 9 285K.
- GPU: NVIDIA RTX 5070 Ti, 16 GiB.
- Rust release build; ONNX Runtime 1.19.2, `ort` 2.0.0-rc.12.
- CPU sessions used 4 intra-op threads and 1 inter-op thread.
- The model pack used the local v2Pro voice. Model files and voice data are not
  included in this repository.
- Timings exclude audio-device playback. Warm synthesis results exclude model
  load. DirectML results vary with graph compilation and the first request.

## Warm clause synthesis

The table reports the second run in one process for each same-text CPU/DirectML
comparison. It measures completion of the full input by the native provider in
the Project Alice integration; it is not a comparison against Python GPT-SoVITS
or another TTS engine.

| Input | CPU | DirectML | Difference in this run |
| --- | ---: | ---: | ---: |
| Short Mandarin | 1,691 ms | 1,325 ms | 21.7% lower elapsed time |
| Polyphonic Mandarin | 2,347 ms | 2,062 ms | 12.2% lower |
| Short English | 1,201 ms | 1,053 ms | 12.3% lower |
| Longer Mandarin | 3,367 ms | 2,828 ms | 16.0% lower |

This small sample favored DirectML after warmup. The first DirectML inference
was slower on the same host: about 3,456 ms for the first complete Mandarin
sentence versus 1,704 ms on CPU, due in part to graph compilation. CPU remains
the default.

## Experimental token-streaming snapshot

The standalone crate's `stream_bench` example measured time from request start
to the first PCM callback. These figures include the first use of each backend
after model loading. DirectML's first Mandarin sample included its initial
graph compilation; later samples were faster.

| Input | CPU first PCM | DirectML first PCM | CPU complete | DirectML complete |
| --- | ---: | ---: | ---: | ---: |
| Short Mandarin | 665 ms | 2,530 ms | 1,945 ms | 3,881 ms |
| Mixed Mandarin/English | 643 ms | 730 ms | 3,154 ms | 3,235 ms |
| Short English | 546 ms | 464 ms | 1,372 ms | 1,358 ms |
| Longer Mandarin | 754 ms | 663 ms | 3,914 ms | 3,718 ms |

The table is a few fixed sentences in a sequential run, not a statistical
latency distribution. Each sentence produces multiple PCM callbacks when long
enough. The first token-streaming chunk is an early playable segment, not the
whole utterance; full completion still takes longer.

## Startup and model footprint

On the same host, loading the streaming pack took about 5.98 s on CPU and 6.30 s
with DirectML selected. First DirectML inference may add compilation latency.
The base model pack was about 3.09 GiB; the optional streaming pack was about
3.34 GiB.

An earlier proof-of-concept run measured about 4.87 GiB peak working set/RSS
while loading a 2.48 GiB subset of ONNX graphs. That measurement predates the
production pack and streaming integration, sampled every 10 ms, and is not a
current memory guarantee. Shorter peaks may have been missed.

## What is and is not optimized

The current code batches G2PW queries, can deliver complete clauses before the
rest of a response is synthesized, supports an optional VITS token-streaming
graph, and can run selected frontend/vocoder graphs through DirectML on Windows.
Those are targeted implementation choices, not evidence of broad model or
hardware optimization.

There has been no systematic model quantization, weight compression/deduplication,
cross-device tuning, or large multilingual quality/performance sweep. Results
do not cover all speech inputs, GPUs, drivers, OS versions, or CPUs. Clause
boundaries and streaming can change prosody and total work, so earlier first PCM
does not necessarily mean a faster or equivalent complete utterance.

## Reproduce

Use a compatible local pack and a release build:

```powershell
cargo run --release --example backend_bench -- ./model-pack ./bench-cpu cpu
cargo run --release --example backend_bench -- ./model-pack ./bench-directml directml
cargo run --release --example stream_bench -- ./streaming-pack ./stream-cpu cpu
cargo run --release --example stream_bench -- ./streaming-pack ./stream-directml directml
```

DirectML commands require Windows and a compatible GPU/driver. Keep the same
pack, text, build, power profile, and warmup procedure when comparing runs.
Record model-load time separately from warm synthesis and first-PCM latency.
