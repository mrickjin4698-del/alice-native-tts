//! R1.9: bounded early publication with a full-prefix VITS refill.

use alice_native_tts::{Backend, ChunkInfo, Config, NativeTts};
use serde_json::{json, Value};
use std::{
    cell::Cell,
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

const SAMPLE_RATE: usize = 32_000;
const ACTIVE_WINDOW: usize = 320;
const ACTIVE_RMS_THRESHOLD: f64 = 1e-3;
const WARMUPS: usize = 30;
const SAMPLES: usize = 20;

const CORPUS: [(&str, &str); 8] = [
    ("zh_short", "你好。"),
    ("zh_poly_short", "行长来了。"),
    (
        "zh_long",
        "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
    ),
    (
        "zh_poly_long",
        "银行行长在重庆重看那段长长的乐章，朋友们还在行走；后来他着重介绍了重要的行程。",
    ),
    ("en_short", "Hello, Alice."),
    (
        "en_long",
        "The quick brown fox jumps over the lazy dog while Alice reads a book and welcomes everyone home.",
    ),
    ("mixed_short", "你好 Alice，请看这个 report。"),
    (
        "mixed_long",
        "你好 Alice，请 read the report about 银行行长 who will travel to 重庆 tomorrow.",
    ),
];

#[derive(Clone, Copy, Debug)]
enum StrategyKind {
    Chunk24,
    R18Preview17,
    BoundedAfterOnset(usize),
}

#[derive(Clone, Copy, Debug)]
struct Strategy {
    name: &'static str,
    kind: StrategyKind,
}

const STRATEGIES: [Strategy; 5] = [
    Strategy {
        name: "chunk24",
        kind: StrategyKind::Chunk24,
    },
    Strategy {
        name: "r18_preview17",
        kind: StrategyKind::R18Preview17,
    },
    Strategy {
        name: "bounded_after40ms",
        kind: StrategyKind::BoundedAfterOnset(40),
    },
    Strategy {
        name: "bounded_after80ms",
        kind: StrategyKind::BoundedAfterOnset(80),
    },
    Strategy {
        name: "bounded_after120ms",
        kind: StrategyKind::BoundedAfterOnset(120),
    },
];

#[derive(Clone, serde::Serialize)]
struct CallbackRecord {
    at_ms: f64,
    first_active_offset_ms: Option<f64>,
    semantic_tokens: usize,
    sample_count: usize,
    final_chunk: bool,
    boundary_jump_before: Option<f64>,
}

struct CapturedRun {
    row: Value,
    audio: Vec<f32>,
    first_callback_audio: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: r1_9_bounded_preview PACK OUTPUT_DIR [probe|bench]".into());
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let mode = args
        .get(3)
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "probe".to_owned());
    if mode != "probe" && mode != "bench" {
        return Err("mode must be probe or bench".into());
    }
    if output.exists() && fs::read_dir(&output)?.next().is_some() {
        return Err(format!(
            "refusing to overwrite non-empty output directory: {}",
            output.display()
        )
        .into());
    }
    fs::create_dir_all(&output)?;
    configure_runtime();

    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            backend: Backend::Cuda,
            streaming: true,
        },
        &|| false,
    )?;
    if mode == "probe" {
        run_probe(&mut model, &output)?;
    } else {
        run_benchmark(&mut model, &output)?;
    }
    Ok(())
}

fn configure_runtime() {
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND", "cuda");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND", "cuda");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_PARALLEL", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "1");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_CUDA_VITS", "1");
    env::set_var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_BINDING_MODE", "clear_rebind");
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH", "0");
    env::remove_var("ALICE_NATIVE_TTS_GPT_STEP");
}

fn run_probe(model: &mut NativeTts, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    for group_id in ["en_short", "en_long"] {
        let text = CORPUS
            .iter()
            .find(|(id, _)| *id == group_id)
            .map(|(_, text)| *text)
            .ok_or("missing R1.9 probe text")?;
        for strategy in STRATEGIES {
            let _ = synthesize(model, group_id, text, strategy)?;
        }
        for strategy in STRATEGIES {
            let run = synthesize(model, group_id, text, strategy)?;
            rows.push(run.row.clone());
            write_wav(
                &output.join(format!("{group_id}-{}.wav", strategy.name)),
                &run.audio,
            )?;
            write_wav(
                &output.join(format!("{group_id}-{}-first-callback.wav", strategy.name)),
                &run.first_callback_audio,
            )?;
            println!("{group_id}/{} {}", strategy.name, run.row);
        }
    }
    let cancellation = run_cancel_recovery_probe(model, &output)?;
    let report = json!({
        "schema": "alice-native-tts-r1.9-bounded-preview-probe-v1",
        "runtime": "CUDA GPT + CUDA VITS, CPU token sampling, ABI-1",
        "seed": 1234,
        "voice_onset_gate_ms": 160,
        "voice_onset_rms_threshold": ACTIVE_RMS_THRESHOLD,
        "voice_onset_confirmation_ms": 20,
        "join_min_rms_floor": 1e-4,
        "join_relative_rms_to_peak_10ms": 0.10,
        "join_correlation_gate": 0.70,
        "buffer_simulation_ms": [32, 64],
        "groups": rows,
        "cancellation_recovery": cancellation,
        "full_benchmark_started": false,
        "human_blind_audition": "NOT_RUN",
        "real_playback_acceptance": "NOT_RUN"
    });
    fs::write(
        output.join("probe.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

fn run_cancel_recovery_probe(
    model: &mut NativeTts,
    output: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let cancelled = Cell::new(false);
    let mut callback_count = 0usize;
    let cancel = || cancelled.get();
    let mut emit = |_: &[f32], _: ChunkInfo| {
        callback_count += 1;
        cancelled.set(true);
    };
    let result =
        model.synthesize_streaming_r1_9_bounded_experiment("Hello, Alice.", &cancel, 40, &mut emit);
    let cancellation_error = result.as_ref().err().map(ToString::to_string);
    let cancelled_after_callback = result.is_err() && callback_count == 1;

    let recovered = synthesize(model, "en_short", "Hello, Alice.", STRATEGIES[0])?;
    if !recovered.row["completed"].as_bool().unwrap_or(false) {
        return Err("Chunk-24 synthesis did not recover after R1.9 cancellation".into());
    }
    write_wav(
        &output.join("cancel-recovery-chunk24.wav"),
        &recovered.audio,
    )?;
    Ok(json!({
        "cancelled_after_first_callback": cancelled_after_callback,
        "callback_count_before_cancel": callback_count,
        "cancel_error": cancellation_error,
        "chunk24_recovered": recovered.row["completed"],
        "recovered_semantic_token_hash": recovered.row["semantic_token_hash"]
    }))
}

fn run_benchmark(model: &mut NativeTts, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut runs = Vec::new();
    let mut tracks = Vec::<(String, String, String, bool, Vec<f32>)>::new();
    let mut waveform_comparisons = Vec::new();

    for (group_id, text) in CORPUS {
        let mut by_strategy = vec![Vec::<Value>::new(); STRATEGIES.len()];
        let mut hashes = vec![None::<u64>; STRATEGIES.len()];
        let mut first_audio = vec![None::<Vec<f32>>; STRATEGIES.len()];
        let mut first_callbacks = vec![None::<Vec<f32>>; STRATEGIES.len()];

        for warmup in 0..WARMUPS {
            for strategy in STRATEGIES {
                let _ = synthesize(model, group_id, text, strategy)?;
            }
            if (warmup + 1) % 10 == 0 {
                eprintln!("R1.9 {group_id}: warmup {}/{}", warmup + 1, WARMUPS);
            }
        }

        for sample in 0..SAMPLES {
            let mut baseline_hash = None;
            for (index, strategy) in STRATEGIES.iter().copied().enumerate() {
                let run = synthesize(model, group_id, text, strategy)?;
                let complete = run.row["completed"].as_bool().unwrap_or(false);
                let token_hash = run.row["semantic_token_hash"].as_u64();
                if index == 0 {
                    baseline_hash = token_hash;
                }
                if complete {
                    if let Some(hash) = token_hash {
                        if let Some(previous) = hashes[index] {
                            if previous != hash {
                                return Err(format!(
                                    "semantic token hash changed within {group_id}/{}",
                                    strategy.name
                                )
                                .into());
                            }
                        } else {
                            hashes[index] = Some(hash);
                        }
                    }
                }
                let matches_baseline =
                    complete.then(|| token_hash.is_some() && token_hash == baseline_hash);
                let mut row = run.row.clone();
                row["semantic_tokens_match_chunk24"] = json!(matches_baseline);
                if sample == 0 {
                    first_audio[index] = Some(run.audio.clone());
                    first_callbacks[index] = Some(run.first_callback_audio.clone());
                }
                by_strategy[index].push(row.clone());
                runs.push(row);
            }
            if (sample + 1) % 5 == 0 {
                eprintln!("R1.9 {group_id}: measured {}/{}", sample + 1, SAMPLES);
            }
        }

        let baseline_hash = hashes[0];
        for row in runs.iter_mut().filter(|row| row["group_id"] == group_id) {
            if row["completed"].as_bool() == Some(true) {
                row["semantic_tokens_match_chunk24"] =
                    json!(row["semantic_token_hash"].as_u64() == baseline_hash);
            }
        }

        let baseline_callback = first_callbacks[0]
            .as_deref()
            .ok_or("missing Chunk-24 first callback")?;
        for (index, strategy) in STRATEGIES.iter().enumerate().skip(1) {
            let candidate = first_callbacks[index]
                .as_deref()
                .ok_or("missing candidate first callback")?;
            waveform_comparisons.push(json!({
                "group_id": group_id,
                "strategy": strategy.name,
                "scope": "common sample prefix of first callback; same sample index",
                "metrics": compare_waveforms(baseline_callback, candidate)
            }));
        }

        for (index, strategy) in STRATEGIES.iter().enumerate() {
            let audio = first_audio[index]
                .take()
                .ok_or("missing first measured waveform")?;
            let complete = by_strategy[index][0]["completed"]
                .as_bool()
                .unwrap_or(false);
            tracks.push((
                group_id.to_owned(),
                strategy.name.to_owned(),
                text.to_owned(),
                complete,
                audio,
            ));
        }
        for (index, strategy) in STRATEGIES.iter().enumerate() {
            println!(
                "R1.9 {group_id}/{} {}",
                strategy.name,
                summarize(&by_strategy[index])
            );
        }
    }

    let report = json!({
        "schema": "alice-native-tts-r1.9-bounded-preview-benchmark-v1",
        "runtime": "CUDA GPT + CUDA VITS, CPU token sampling, ABI-1",
        "seed": 1234,
        "chunk24_default_unchanged": true,
        "voice_onset_gate_ms": 160,
        "voice_onset_rms_threshold": ACTIVE_RMS_THRESHOLD,
        "voice_onset_confirmation_ms": 20,
        "join_min_rms_floor": 1e-4,
        "join_relative_rms_to_peak_10ms": 0.10,
        "minimum_commit_ms": 64,
        "maximum_commit_ms": 240,
        "after_onset_windows_ms": [40, 80, 120],
        "join_correlation_gate": 0.70,
        "buffer_simulation_ms": [32, 64],
        "warmups_per_group_strategy": WARMUPS,
        "samples_per_group_strategy": SAMPLES,
        "groups": summarize_all(&runs),
        "first_callback_waveform_comparison_vs_chunk24": waveform_comparisons,
        "runs": runs,
        "blind_package": "blind-listening/",
        "blind_key": "blind-key.json",
        "human_blind_audition": "NOT_RUN; anonymous package exported",
        "real_playback_acceptance": "NOT_RUN",
        "asr": "Local Whisper review is run separately on en_short/en_long failures."
    });
    fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    export_blind_package(output, tracks)?;
    Ok(())
}

fn synthesize(
    model: &mut NativeTts,
    group_id: &str,
    text: &str,
    strategy: Strategy,
) -> Result<CapturedRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut audio = Vec::<f32>::new();
    let mut first_callback_audio = Vec::<f32>::new();
    let mut callbacks = Vec::<CallbackRecord>::new();
    let mut callback = |samples: &[f32], info: ChunkInfo| {
        let at_ms = started.elapsed().as_secs_f64() * 1000.0;
        let chunk_activity = activity(samples);
        if callbacks.is_empty() {
            first_callback_audio.extend_from_slice(samples);
        }
        callbacks.push(CallbackRecord {
            at_ms,
            first_active_offset_ms: chunk_activity.first_active_offset_ms,
            semantic_tokens: info.semantic_tokens,
            sample_count: samples.len(),
            final_chunk: info.final_chunk,
            boundary_jump_before: audio
                .last()
                .zip(samples.first())
                .map(|(before, after)| f64::from((*before - *after).abs())),
        });
        audio.extend_from_slice(samples);
    };
    let (timings, diagnostics) = match strategy.kind {
        StrategyKind::Chunk24 => {
            let (timings, diagnostics) =
                model.synthesize_streaming_r1_8_experiment(text, &|| false, None, &mut callback)?;
            (Some(timings), diagnostics)
        }
        StrategyKind::R18Preview17 => {
            let (timings, diagnostics) = model.synthesize_streaming_r1_8_experiment(
                text,
                &|| false,
                Some(17),
                &mut callback,
            )?;
            (Some(timings), diagnostics)
        }
        StrategyKind::BoundedAfterOnset(after_onset_ms) => model
            .synthesize_streaming_r1_9_bounded_experiment(
                text,
                &|| false,
                after_onset_ms,
                &mut callback,
            )?,
    };
    drop(callback);

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let first_callback_ms = callbacks.first().map(|record| record.at_ms);
    let second_callback_ms = callbacks.get(1).map(|record| record.at_ms);
    let first_active_record = callbacks
        .iter()
        .enumerate()
        .find(|(_, record)| record.first_active_offset_ms.is_some());
    let first_active_pcm_ms = first_active_record.and_then(|(_, record)| {
        record
            .first_active_offset_ms
            .map(|offset| record.at_ms + offset)
    });
    let active = activity(&first_callback_audio);
    let completed = timings.is_some();
    let finite = audio.iter().all(|sample| sample.is_finite());
    if audio.is_empty() || !finite {
        return Err(format!("{group_id}/{} produced invalid PCM", strategy.name).into());
    }
    let rtf = if completed {
        timings
            .as_ref()
            .map(|_| elapsed_ms / (audio.len() as f64 / SAMPLE_RATE as f64 * 1000.0))
    } else {
        None
    };
    let simulations = json!([
        simulate_buffer(&callbacks, 32, Some(elapsed_ms)),
        simulate_buffer(&callbacks, 64, Some(elapsed_ms))
    ]);
    let boundary_jumps = callbacks
        .iter()
        .filter_map(|event| event.boundary_jump_before)
        .collect::<Vec<_>>();
    let row = json!({
        "group_id": group_id,
        "strategy": strategy.name,
        "completed": completed,
        "reject_reason": diagnostics["refill_rejected_reason"],
        "preview_fallback_reason": diagnostics["preview_fallback_reason"],
        "elapsed_ms": completed.then_some(elapsed_ms),
        "time_to_reject_ms": (!completed).then_some(elapsed_ms),
        "ttfa_callback_ms": first_callback_ms,
        "first_audible_pcm_ms": first_active_pcm_ms,
        "first_active_pcm_ms": first_active_pcm_ms,
        "first_active_callback_index": first_active_record.map(|(index, _)| index),
        "second_callback_ms": second_callback_ms,
        "first_callback_samples": first_callback_audio.len(),
        "first_segment_duration_ms": first_callback_audio.len() as f64 * 1000.0 / SAMPLE_RATE as f64,
        "first_segment_active_ms": active.active_ms,
        "first_segment_leading_silence_ms": active.leading_silence_ms,
        "preview_decoded_samples": diagnostics["preview_decoded_samples"],
        "preview_committed_samples": diagnostics["preview_committed_samples"],
        "preview_uncommitted_samples": diagnostics["preview_uncommitted_samples"],
        "preview_committed_ms": diagnostics["preview_committed_samples"].as_u64().map(|n| n as f64 * 1000.0 / SAMPLE_RATE as f64),
        "preview_uncommitted_ms": diagnostics["preview_uncommitted_samples"].as_u64().map(|n| n as f64 * 1000.0 / SAMPLE_RATE as f64),
        "preview_voice_onset_ms": diagnostics["preview_voice_onset_samples"].as_u64().map(|n| n as f64 * 1000.0 / SAMPLE_RATE as f64),
        "bounded_join": diagnostics["bounded_join"],
        "semantic_tokens": diagnostics["semantic_tokens"],
        "semantic_token_hash": diagnostics["semantic_token_hash"],
        "semantic_tokens_match_chunk24": Value::Null,
        "audio_chunks": timings.as_ref().map(|t| t.audio_chunks),
        "audio_samples": audio.len(),
        "audio_seconds": audio.len() as f64 / SAMPLE_RATE as f64,
        "rtf": rtf,
        "timings": timings,
        "callbacks": callbacks,
        "boundary_jumps": boundary_jumps,
        "playback_buffer_simulation": simulations,
        "audio_finite": finite
    });
    Ok(CapturedRun {
        row,
        audio,
        first_callback_audio,
    })
}

#[derive(Clone, Copy)]
struct Activity {
    leading_silence_ms: f64,
    first_active_offset_ms: Option<f64>,
    active_ms: f64,
}

fn activity(audio: &[f32]) -> Activity {
    let frame_rms = audio
        .chunks(ACTIVE_WINDOW)
        .map(|frame| {
            (frame
                .iter()
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>()
                / frame.len().max(1) as f64)
                .sqrt()
        })
        .collect::<Vec<_>>();
    let first_active = frame_rms
        .windows(2)
        .position(|pair| pair[0] >= ACTIVE_RMS_THRESHOLD && pair[1] >= ACTIVE_RMS_THRESHOLD);
    let active_windows = frame_rms
        .iter()
        .filter(|rms| **rms >= ACTIVE_RMS_THRESHOLD)
        .count();
    Activity {
        leading_silence_ms: first_active
            .map_or(audio.len() as f64 * 1000.0 / SAMPLE_RATE as f64, |index| {
                index as f64 * 10.0
            }),
        first_active_offset_ms: first_active.map(|index| index as f64 * 10.0),
        active_ms: active_windows as f64 * 10.0,
    }
}

fn simulate_buffer(
    callbacks: &[CallbackRecord],
    buffer_ms: usize,
    terminal_ms: Option<f64>,
) -> Value {
    let buffer_samples = SAMPLE_RATE * buffer_ms / 1000;
    let mut queued_samples = 0usize;
    let mut start_index = None;
    for (index, callback) in callbacks.iter().enumerate() {
        queued_samples += callback.sample_count;
        if queued_samples >= buffer_samples {
            start_index = Some(index);
            break;
        }
    }
    let Some(start_index) = start_index else {
        return json!({
            "buffer_ms": buffer_ms,
            "playback_started": false,
            "minimum_queue_slack_ms": Value::Null,
            "underrun_count": Value::Null
        });
    };
    let playback_start_ms = callbacks[start_index].at_ms;
    queued_samples = callbacks
        .iter()
        .take(start_index + 1)
        .map(|callback| callback.sample_count)
        .sum::<usize>();
    let mut cursor_ms = playback_start_ms;
    let mut minimum_slack_ms = queued_samples as f64 * 1000.0 / SAMPLE_RATE as f64;
    let mut underrun_count = 0usize;
    let mut underrun_ms = 0.0;
    for callback in callbacks.iter().skip(start_index + 1) {
        let elapsed_ms = (callback.at_ms - cursor_ms).max(0.0);
        let needed_samples = elapsed_ms * SAMPLE_RATE as f64 / 1000.0;
        let slack_samples = queued_samples as f64 - needed_samples;
        minimum_slack_ms = minimum_slack_ms.min(slack_samples * 1000.0 / SAMPLE_RATE as f64);
        if slack_samples < 0.0 {
            underrun_count += 1;
            underrun_ms += -slack_samples * 1000.0 / SAMPLE_RATE as f64;
            queued_samples = 0;
        } else {
            queued_samples = slack_samples as usize;
        }
        queued_samples += callback.sample_count;
        cursor_ms = callback.at_ms;
        minimum_slack_ms =
            minimum_slack_ms.min(queued_samples as f64 * 1000.0 / SAMPLE_RATE as f64);
    }
    if let Some(terminal_ms) = terminal_ms {
        let elapsed_ms = (terminal_ms - cursor_ms).max(0.0);
        let needed_samples = elapsed_ms * SAMPLE_RATE as f64 / 1000.0;
        let slack_samples = queued_samples as f64 - needed_samples;
        minimum_slack_ms = minimum_slack_ms.min(slack_samples * 1000.0 / SAMPLE_RATE as f64);
        if slack_samples < 0.0 {
            underrun_count += 1;
            underrun_ms += -slack_samples * 1000.0 / SAMPLE_RATE as f64;
            queued_samples = 0;
        } else {
            queued_samples = slack_samples as usize;
        }
    }
    json!({
        "buffer_ms": buffer_ms,
        "playback_started": true,
        "playback_start_ms": playback_start_ms,
        "minimum_queue_slack_ms": minimum_slack_ms,
        "underrun_count": underrun_count,
        "underrun_ms": underrun_ms,
        "queued_samples_after_last_callback": queued_samples
    })
}

fn summarize(rows: &[Value]) -> Value {
    let all_values = |key: &str| {
        rows.iter()
            .filter_map(|row| row[key].as_f64())
            .collect::<Vec<_>>()
    };
    let completed_values = |key: &str| {
        rows.iter()
            .filter(|row| row["completed"] == true)
            .filter_map(|row| row[key].as_f64())
            .collect::<Vec<_>>()
    };
    let p50p95 = |key: &str| {
        let items = all_values(key);
        json!({"p50": percentile(&items, 0.50), "p95": percentile(&items, 0.95)})
    };
    let p50p95_completed = |key: &str| {
        let items = completed_values(key);
        json!({"p50": percentile(&items, 0.50), "p95": percentile(&items, 0.95)})
    };
    let correlations = rows
        .iter()
        .filter_map(|row| row["bounded_join"]["correlation"].as_f64())
        .collect::<Vec<_>>();
    let queue_slack = rows
        .iter()
        .flat_map(|row| {
            row["playback_buffer_simulation"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter_map(|entry| entry["minimum_queue_slack_ms"].as_f64())
        .collect::<Vec<_>>();
    json!({
        "samples": rows.len(),
        "completed": rows.iter().filter(|row| row["completed"] == true).count(),
        "rejected": rows.iter().filter(|row| row["completed"] == false).count(),
        "ttfa_callback_ms": p50p95("ttfa_callback_ms"),
        "first_audible_pcm_ms": p50p95("first_audible_pcm_ms"),
        "second_callback_ms": p50p95("second_callback_ms"),
        "preview_committed_ms": p50p95("preview_committed_ms"),
        "preview_uncommitted_ms": p50p95("preview_uncommitted_ms"),
        "elapsed_ms": p50p95_completed("elapsed_ms"),
        "rtf": p50p95_completed("rtf"),
        "bounded_join_correlation": {
            "p50": percentile(&correlations, 0.50),
            "p05": percentile(&correlations, 0.05),
            "min": correlations.iter().copied().reduce(f64::min)
        },
        "minimum_queue_slack_ms": {
            "p50": percentile(&queue_slack, 0.50),
            "min": queue_slack.iter().copied().reduce(f64::min)
        },
        "underrun_runs_32ms": underrun_runs(rows, 32),
        "underrun_runs_64ms": underrun_runs(rows, 64),
        "failed_reasons": rows.iter().flat_map(|row| [row["reject_reason"].as_str(), row["preview_fallback_reason"].as_str()].into_iter().flatten()).collect::<Vec<_>>()
    })
}

fn underrun_runs(rows: &[Value], buffer_ms: usize) -> usize {
    rows.iter()
        .filter(|row| {
            row["playback_buffer_simulation"]
                .as_array()
                .and_then(|entries| entries.iter().find(|entry| entry["buffer_ms"] == buffer_ms))
                .and_then(|entry| entry["underrun_count"].as_u64())
                .unwrap_or(0)
                > 0
        })
        .count()
}

fn summarize_all(rows: &[Value]) -> Value {
    let mut groups = Vec::new();
    for (group_id, _) in CORPUS {
        for strategy in STRATEGIES {
            let selected = rows
                .iter()
                .filter(|row| row["group_id"] == group_id && row["strategy"] == strategy.name)
                .cloned()
                .collect::<Vec<_>>();
            groups.push(json!({
                "group_id": group_id,
                "strategy": strategy.name,
                "metrics": summarize(&selected)
            }));
        }
    }
    json!({"groups": groups})
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((quantile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

fn compare_waveforms(reference: &[f32], candidate: &[f32]) -> Value {
    let compared_samples = reference.len().min(candidate.len());
    if compared_samples == 0 {
        return json!({"valid": false, "compared_samples": 0});
    }
    let mut dot = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut absolute_error = 0.0f64;
    let mut max_error = 0.0f64;
    for (reference, candidate) in reference[..compared_samples]
        .iter()
        .zip(&candidate[..compared_samples])
    {
        let reference = f64::from(*reference);
        let candidate = f64::from(*candidate);
        let error = (reference - candidate).abs();
        dot += reference * candidate;
        reference_energy += reference * reference;
        candidate_energy += candidate * candidate;
        squared_error += error * error;
        absolute_error += error;
        max_error = max_error.max(error);
    }
    json!({
        "valid": true,
        "compared_samples": compared_samples,
        "cosine_similarity": dot / (reference_energy * candidate_energy).sqrt(),
        "rmse": (squared_error / compared_samples as f64).sqrt(),
        "mae": absolute_error / compared_samples as f64,
        "max_abs": max_error
    })
}

fn export_blind_package(
    output: &Path,
    tracks: Vec<(String, String, String, bool, Vec<f32>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let blind_dir = output.join("blind-listening");
    let tracks_dir = blind_dir.join("tracks");
    fs::create_dir_all(&tracks_dir)?;
    fs::write(
        blind_dir.join("README.txt"),
        "Anonymous R1.9 listening set. IDs do not encode group or strategy.\nPlease rate continuity, repeated/missing words, onset, prosody, timbre, pauses, and overall preference.\nSome candidates may be intentionally incomplete when a safe refill alignment was rejected.\nThe strategy key is stored separately as blind-key.json.\n",
    )?;
    let mut ids = (1..=tracks.len())
        .map(|index| format!("B{index:02}"))
        .collect::<Vec<_>>();
    let mut random = 0x9e37_79b9_7f4a_7c15u64;
    for index in (1..ids.len()).rev() {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        ids.swap(index, (random as usize) % (index + 1));
    }
    let mut key = Vec::new();
    let mut rating_rows = vec![
        "track_id,transcript,complete,continuity_1_to_5,repeated_or_missing_words,onset_prosody_timbre,pauses,overall_preference,notes".to_owned(),
    ];
    for (index, (group, strategy, text, complete, audio)) in tracks.into_iter().enumerate() {
        let track_id = ids[index].clone();
        write_wav(&tracks_dir.join(format!("{track_id}.wav")), &audio)?;
        key.push(json!({
            "track_id": track_id,
            "group_id": group,
            "strategy": strategy,
            "complete": complete,
            "text": text
        }));
        rating_rows.push(format!(
            "{},{},{},,,,,,",
            track_id,
            csv_escape(&text),
            complete
        ));
    }
    fs::write(blind_dir.join("track-ids.txt"), ids.join("\n") + "\n")?;
    fs::write(blind_dir.join("ratings.csv"), rating_rows.join("\n"))?;
    fs::write(
        output.join("blind-key.json"),
        serde_json::to_vec_pretty(&key)?,
    )?;
    Ok(())
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn write_wav(path: &Path, audio: &[f32]) -> Result<(), hound::Error> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in audio {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()
}
