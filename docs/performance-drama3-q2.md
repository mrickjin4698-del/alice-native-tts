# Drama 3 Q2 on RTX 5070 Ti

Measured November 25, 2026 on a Core Ultra 9 285K, an RTX 5070 Ti with
16,303 MiB reported VRAM, and Windows 11. Both configurations use the same
`drama-3-backbone-8bit-experts-Q2_K.gguf` file and the audio.cpp v0.9.4 CUDA
13.3 build. The worker synthesizes mono PCM16 at 44.1 kHz. Each session uses a
context capacity of 2048, seed 123 and a 512-frame generation limit.

| Configuration | CPU expert layers | CPU threads | Decode preset |
| --- | ---: | ---: | --- |
| Baseline | 24 | 8 | Default |
| Tuned CUDA | 16 | 6 | `cuda_perf` |

The runs use ABBA order with one warmup and three measured requests per input
at each launch. First PCM is the delay to the first JSONL audio chunk.
Completion is the delay to `done`. Audible onset is estimated for immediate
playback using the first 10 ms PCM window at or above −45 dBFS RMS. Audio
sample counts and leading pauses can vary across synthesis configurations.

| Input | First PCM P50 (ms) | Estimated audible onset P50 (ms) | Completion P50 (ms) | RTF | Audio duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard Mandarin | 179.2 → 150.3 | 449.9 → 200.3 | 3,688 → 2,824 | 0.691 → 0.490 | 5.34 → 5.76 |
| Short Mandarin | 153.7 → 141.1 | 163.7 → 271.1 | 1,283 → 1,105 | 0.747 → 0.529 | 1.72 → 2.09 |
| Long Mandarin | 191.5 → 169.0 | 191.5 → 219.0 | 6,219 → 4,277 | 0.670 → 0.482 | 9.29 → 8.87 |
| English | 171.4 → 155.8 | 171.4 → 215.8 | 4,016 → 3,048 | 0.692 → 0.490 | 5.80 → 6.22 |

The standard input is “你好，欢迎收听。现在正在测试 Drama 三的本地语音合成。” Completion
improves 1.31× on that input, 1.45× on the long input and 1.32× on the English
input. First PCM improves 1.09–1.19×. RTF improves 1.39–1.41×. The baseline's
median estimated longest playback gap ranges from 28 to 35 ms. The tuned
configuration has a median of 0 ms on each input.

The faster configuration produces different PCM with the same seed. Local
Whisper large-v3-turbo transcribes the long Mandarin and English inputs
identically after punctuation normalization. Each configuration has one
“你”/“您” substitution among the two shorter Mandarin inputs. Both outputs
contain every clause. The measured samples have no full-scale clipping.

## GPU memory

`nvidia-smi` samples whole-device usage, including the Windows desktop.
The increase is the sampled peak minus usage before each session starts.

| Configuration | Before launch (MiB) | Sampled peak (MiB) | Increase (MiB) |
| --- | ---: | ---: | ---: |
| Baseline | 2,947–3,006 | 13,340–13,382 | 10,376–10,393 |
| Tuned CUDA | 2,991–3,011 | 15,398–15,404 | 12,393–12,407 |

A separate direct-server screening run measured a 10,065 MiB device peak
with 24 slots per layer in the pinned expert cache, up from 2,846 MiB.
First PCM was 315 ms and completion was 5.02 s for the standard Mandarin input.
That run uses `cpu_expert_layers=0`, `disk_experts=true`, `pin_experts=true`
and `expert_cache_size=24` in the `fish_drama3` session options.
[Drama 3 offload options](https://github.com/0xShug0/audio.cpp/blob/v0.9.4/docs/models/fish_drama3.md)

In a second ABBA run with four timed requests per input and configuration, 15
CPU expert layers reached 149.5 ms first PCM and 2.87 s completion for standard
Mandarin. Its same-run baseline was 175.3 ms and 3.61 s. The sampled GPU
increase was 12,450–12,651 MiB. It completed the long Mandarin and English
cases in 3.66 s and 2.66 s, with complete local Whisper transcripts.
The documented preset uses 16 CPU expert layers to keep more VRAM headroom
and retain the tested standard-sentence and short-sentence completion gains.

A separate three-request screening run tested 14 CPU expert layers with CUDA
codec and 12 layers with CPU codec. The 14-layer run completed in 2.30 s with
148 ms first PCM and a 15,399 MiB device peak. The same-run 16-layer result
was 2.71 s, 158 ms and 14,824 MiB. The CPU codec completed in 8.38 s with a
14,822 MiB peak. Both trials used six threads and `cuda_perf`.

## Reproduce

Create a baseline manifest and a copy with the tuned options.
[Manifest and runtime setup](standalone.md#drama-3-q2)

```powershell
python tools/benchmark-drama3.py --executable ./target/release/alice-tts.exe --baseline ./drama3.json --candidate ./drama3-cuda-perf.json --output ./output/drama3-abba
```

The benchmark writes `benchmark.json` with P50 and P95, `phases.json` with
per-request metrics and PCM hashes, and one WAV per input and phase.
