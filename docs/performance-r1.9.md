# R1.9 — Bounded Preview & Acoustic Refill

## Decision

R1.9 did not pass the production quality gate. Keep Chunk 24 as the default and do not enable bounded Preview in the public streaming path. The experiment is feature-gated and hidden from the documented API. The evidence does not justify more crossfade tuning: the full-prefix refill has no high-confidence, audible alignment for three short-text groups, and the only group that completed an actual early join was English short text, with a marginal correlation score and little playback-queue headroom.

The current experiment fails closed on an unsafe refill join. Because some audio has already been committed by then, it cannot safely switch back to a fresh Chunk 24 stream without repeating or cutting speech. This is an architectural limitation of publishing before validating the future join, not a reason to splice at offset zero.

## Method

- Runtime recorded by the benchmark: CUDA GPT + CUDA VITS, CPU token sampling, ABI-1. Chunk 24, seed 1234, model and text configuration were held constant within each group.
- Corpus: `zh_short`, `zh_poly_short`, `zh_long`, `zh_poly_long`, `en_short`, `en_long`, `mixed_short`, and `mixed_long`.
- Strategies: Chunk 24, R1.8 Preview17, and R1.9 onset-relative commits of 40/80/120 ms.
- Each group/strategy used 30 warmups followed by 20 measured runs: 1,200 warmups and 800 measured runs total.
- The corrected onset detector requires two adjacent 10 ms windows at RMS >= 1e-3, and vetoes a preview whose onset is later than 160 ms. The refill matcher requires correlation >= 0.70 plus both windows above `max(1e-4, 10% of the committed region's peak 10 ms RMS)`.
- TTFA is the first callback time. First-audible PCM is the first confirmed active sample across callbacks. RTF and elapsed time include the complete successful synthesis. Queue results simulate 32/64 ms playback buffers; they are not measurements from a real player.
- The first full run was invalidated after near-silence passed the onset detector. All figures below come from the corrected v2 run in `target/r1.9-bounded-bench-v2/results.json`.

The benchmark harness records the runtime, seed, percentile samples, callback timeline, token hash, PCM boundary jumps, refill diagnostics, and playback simulation. It does not record the GPU model identifier; the ASR environment reported an RTX 5070 Ti, but that is not a substitute for benchmark hardware metadata.

## Chunk 24 vs. Preview17

Values are P50/P95 in milliseconds. Preview17 is the comparison strategy from R1.8, not a production change from this task.

| Corpus | Chunk 24 callback TTFA | Preview17 callback TTFA | Chunk 24 first active PCM | Preview17 first active PCM |
|---|---:|---:|---:|---:|
| zh_short | 158.6/169.0 | 168.5/176.8 | 168.6/179.0 | 178.5/186.8 |
| zh_poly_short | 158.5/165.8 | 169.4/175.4 | 168.5/175.8 | 179.4/185.4 |
| zh_long | 208.9/221.1 | 169.6/176.6 | 428.9/441.1 | 389.6/396.6 |
| zh_poly_long | 204.2/206.9 | 164.6/170.5 | 474.2/476.9 | 434.6/440.5 |
| en_short | 203.6/209.3 | 167.8/192.4 | 203.6/209.3 | 167.8/192.4 |
| en_long | 201.2/205.7 | 164.4/170.8 | 561.2/565.7 | 524.4/530.8 |
| mixed_short | 154.5/160.0 | 162.8/170.6 | 154.5/160.0 | 162.8/170.6 |
| mixed_long | 203.1/209.4 | 163.8/169.3 | 463.1/469.4 | 423.8/429.3 |

Across the eight group medians, Preview17 improved callback TTFA by 20.2 ms on average, but increased average P50 elapsed time from 907.3 to 992.9 ms and average P50 RTF from 0.203 to 0.230. It was slower for the three short Chinese/mixed groups; long groups generally gained about 39–40 ms of first-audible time while taking longer overall. The improvement is not uniform enough to change the default.

| Corpus | Chunk 24 elapsed P50/P95 ms | Preview17 elapsed P50/P95 ms | Chunk 24 RTF P50/P95 | Preview17 RTF P50/P95 |
|---|---:|---:|---:|---:|
| zh_short | 451.3/461.2 | 567.1/589.5 | 0.221/0.226 | 0.278/0.289 |
| zh_poly_short | 479.9/501.2 | 596.5/612.2 | 0.211/0.220 | 0.262/0.269 |
| zh_long | 1292.5/1311.9 | 1362.2/1382.2 | 0.194/0.197 | 0.205/0.208 |
| zh_poly_long | 1679.9/1715.0 | 1748.2/1782.7 | 0.191/0.195 | 0.198/0.202 |
| en_short | 537.7/555.8 | 613.0/667.9 | 0.221/0.229 | 0.252/0.274 |
| en_long | 1181.2/1198.7 | 1245.0/1268.8 | 0.194/0.197 | 0.204/0.208 |
| mixed_short | 638.1/650.4 | 750.5/765.7 | 0.195/0.199 | 0.229/0.233 |
| mixed_long | 997.7/1024.2 | 1061.0/1087.8 | 0.197/0.202 | 0.209/0.215 |

## R1.9 bounded commit results

The table shows the 120 ms window. Callback/active values are P50/P95 ms; “complete” is the number of full streams. Queue counts are streams with simulated underruns at 32/64 ms. For an incomplete stream, an apparently empty underrun count does not pass playback acceptance.

| Corpus | TTFA P50/P95 | First active P50/P95 | Complete | Rejected | Join correlation P50 | Underrun 32/64 ms |
|---|---:|---:|---:|---:|---:|---:|
| zh_short | 168.6/176.9 | 178.6/186.9 | 0/20 | 20 | 0.967 | 1/1 |
| zh_poly_short | 167.6/176.5 | 177.6/186.5 | 0/20 | 20 | 0.802 | 0/0 |
| zh_long | 277.8/287.8 | 497.8/507.8 | 20/20 | 0 | — | 0/0 |
| zh_poly_long | 273.2/281.3 | 543.2/551.3 | 20/20 | 0 | — | 0/0 |
| en_short | 168.0/171.4 | 168.0/171.4 | 20/20 | 0 | 0.709 | 0/0 |
| en_long | 267.9/276.4 | 627.9/636.4 | 20/20 | 0 | — | 0/0 |
| mixed_short | 163.5/175.7 | 163.5/175.7 | 0/20 | 20 | 0.994 | 3/3 |
| mixed_long | 269.3/279.3 | 529.3/539.3 | 20/20 | 0 | — | 0/0 |

Across all three windows, 180/480 R1.9 runs failed refill matching in `zh_short`, `zh_poly_short`, and `mixed_short` after publishing 70–130 ms. The best correlation was sometimes high but came from low-energy windows: for example, `zh_short` at 120 ms had correlation 0.967, while its preview/refill RMS values (0.000156/0.000124) were both below the 0.000177 signal gate. `mixed_short` reached 0.994 correlation but had RMS values 0.000116/0.000110, below its 0.000515 gate. The resulting streams were incomplete; RTF and full-duration quality are therefore unavailable for these rows.

The onset gate vetoed preview publication for the four long groups in all 240 R1.9 samples: `en_long` onset was about 360 ms and `mixed_long` about 260 ms, beyond the 160 ms limit. These runs completed through the normal 24-token decode, but the discarded 17-token VITS preview made their first callbacks about 60–75 ms later than Chunk 24. This is a fallback in output behavior, but not a latency-neutral fallback.

`en_short` was the only group with accepted joins in all three windows. At 120 ms, its seam correlation was 0.709, just above the 0.70 gate; measured join jumps were about 1.0e-5 at the start and 3.2e-5 at the end. The 32/64 ms simulation had no underrun for 120 ms, but queue slack was only 16.1 ms at P50 and 10.7 ms at the worst sample. The 40/80 ms windows underflowed in 20/20 runs. Full-synthesis P50 elapsed time was 608.3 ms and RTF 0.250, versus 537.7 ms and 0.221 for Chunk 24. This is a candidate for listening only, not a pass.

## Semantic, audio, and cancellation checks

- The benchmark completed 620/800 samples. Every completed sample had the same semantic-token hash as its Chunk 24 control. The other 180 were incomplete refill-rejection streams; they are not counted as semantic or audio passes.
- All benchmark audio was finite. The harness saved full concatenated output and first-callback waveform comparisons. Those numeric comparisons do not establish perceptual equivalence.
- Whisper small screened all 40 anonymous benchmark tracks. English outputs were broadly consistent across strategies, while Chinese polyphone and mixed-language transcriptions varied; several rejected tracks were intentionally partial. This is machine screening only, not an ASR quality gate or an audio-quality pass.
- The isolated cancellation probe cancelled after one callback, observed `native TTS cancelled`, then completed a Chunk 24 recovery with the same semantic-token hash as the baseline.
- Human blind audition: **NOT RUN**. Real-device playback and underrun acceptance: **NOT RUN**. Thus no candidate passed the requested production quality gate.

## Files and artifacts

R1.9 changes are feature-gated; earlier R1.5–R1.8 dirty work in this checkout was preserved.

- `Cargo.toml` — experimental feature and benchmark example registration.
- `src/inference.rs` — hidden R1.9 experiment entrypoint and preview/refill timeline.
- `src/streaming.rs` — bounded matcher, signal/onset gates, seam diagnostics, and tests.
- `examples/r1_9_bounded_preview.rs` — probe, 8-group benchmark, cancellation probe, queue simulation, and anonymous export.
- `docs/performance-r1.9.md` — this report.

Local evidence artifacts (the `target` directory is ignored by Git):

- `target/r1.9-bounded-bench-v2/results.json` — all 800 measured rows and summary metrics.
- `target/r1.9-bounded-bench-v2/asr-screening-v2.json` — Whisper screening transcripts.
- `target/r1.9-bounded-bench-v2/r1.9-anonymous-listening.zip` — 40 anonymous WAVs, README, IDs, and 9-column ratings sheet. The private `blind-key.json` is excluded; the archive was checked for leakage.
- `target/r1.9-probe-v5/probe.json` — probe and cancellation-recovery record.

## Verification and next step

- `cargo check --release --lib` (default features): passed.
- `cargo check --release --examples --features 'r1-cuda-ep r1-r8-preview r1-r9-bounded-preview'`: passed.
- `cargo test --release --lib --features 'r1-cuda-ep r1-r8-preview r1-r9-bounded-preview' streaming::tests:: -- --nocapture`: 9 passed.
- Anonymous package: 40 WAV files; CSV parsed as 41 rows with 9 columns; no key or ASR file included.

Do not continue tuning the 40/80/120 ms windows or enlarge crossfades under ABI-1. If this direction is revisited, first add a decoder contract that can validate a future acoustic continuation before committing any preview samples, or provide a stable stateful acoustic continuation from the model. Until then, retain Chunk 24. No commit or push was made.
