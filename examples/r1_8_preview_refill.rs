//! R1.8 Preview -> Refill experiment on the ABI-1 streaming VITS graph.
//!
//! `probe` performs one matched sentence per strategy. `bench` performs the
//! requested eight-corpus 30 warmup + 20 sample sweep and exports anonymized
//! full-utterance WAVs for later human review.

use alice_native_tts::{Backend, ChunkInfo, Config, Error, NativeTts};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SAMPLE_RATE: usize = 32_000;
const ACTIVE_WINDOW: usize = 320; // 10 ms
const ACTIVE_RMS_THRESHOLD: f64 = 1e-4;
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
struct Strategy {
    name: &'static str,
    preview_tokens: Option<usize>,
}

const STRATEGIES: [Strategy; 3] = [
    Strategy {
        name: "chunk24",
        preview_tokens: None,
    },
    Strategy {
        name: "preview17_refill24",
        preview_tokens: Some(17),
    },
    Strategy {
        name: "preview20_refill24",
        preview_tokens: Some(20),
    },
];

#[derive(Clone, Serialize)]
struct CallbackRecord {
    at_ms: f64,
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
        return Err("usage: r1_8_preview_refill PACK OUTPUT_DIR [probe|bench]".into());
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
    let (text_id, text) = CORPUS[2];
    for strategy in STRATEGIES {
        let _ = synthesize(model, text_id, text, strategy)?;
    }
    let mut rows = Vec::new();
    let mut captured = Vec::new();
    let mut baseline_hash = None;
    for strategy in STRATEGIES {
        let run = synthesize(model, text_id, text, strategy)?;
        let token_hash = run.row["diagnostics"]["semantic_token_hash"].as_u64();
        if strategy.preview_tokens.is_none() {
            baseline_hash = token_hash;
        }
        let row = enrich_with_reference(run.row, baseline_hash, strategy.preview_tokens);
        write_f32le(
            &output.join(format!("probe-{}.f32le", strategy.name)),
            &run.audio,
        )?;
        write_wav(
            &output.join(format!("probe-{}.wav", strategy.name)),
            &run.audio,
        )?;
        captured.push((strategy.name, run.first_callback_audio.clone()));
        rows.push(row);
    }
    let baseline_audio = captured
        .iter()
        .find(|(name, _)| *name == "chunk24")
        .map(|(_, audio)| audio.as_slice())
        .ok_or("missing Chunk-24 probe waveform")?;
    let waveform_comparisons = captured
        .iter()
        .filter(|(name, _)| *name != "chunk24")
        .map(|(name, audio)| {
            json!({
                "strategy": name,
                "scope": "common sample prefix of first callback PCM",
                "metrics": compare_waveforms(baseline_audio, audio)
            })
        })
        .collect::<Vec<_>>();
    let cancellation_recovery = validate_cancellation_recovery(model)?;
    let report = json!({
        "schema": "alice-native-tts-r1.8-single-sentence-probe-v1",
        "mode": "single_sentence_probe",
        "runtime": "CUDA GPT + CUDA VITS, CPU token sampling, ABI-1, Chunk 24 control",
        "warmups_per_strategy": 1,
        "text_id": text_id,
        "text": text,
        "seed": 1234,
        "audio_activity_definition": "RMS >= 1e-4 in non-overlapping 10 ms windows",
        "strategies": rows,
        "first_callback_waveform_comparison_vs_chunk24": waveform_comparisons,
        "cancellation_recovery": cancellation_recovery,
        "full_sweep_started": false
    });
    fs::write(
        output.join("probe.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_benchmark(model: &mut NativeTts, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut runs = Vec::new();
    let mut waveform_comparisons = Vec::new();
    let mut blind_tracks = Vec::<(String, String, String, String, Vec<f32>)>::new();
    for (group_id, text) in CORPUS {
        let mut by_strategy: Vec<Vec<Value>> = STRATEGIES.iter().map(|_| Vec::new()).collect();
        let mut first_measured_audio = vec![None::<Vec<f32>>; STRATEGIES.len()];
        let mut first_measured_first_callback = vec![None::<Vec<f32>>; STRATEGIES.len()];
        let mut hashes = vec![None::<u64>; STRATEGIES.len()];

        for warmup in 0..WARMUPS {
            for strategy in STRATEGIES.iter().copied() {
                let _ = synthesize(model, group_id, text, strategy)?;
                if warmup == 0 {
                    eprintln!("R1.8 {group_id}: warmup {}/{}", warmup + 1, WARMUPS);
                }
            }
        }

        for sample_index in 0..SAMPLES {
            for (strategy_index, strategy) in STRATEGIES.iter().copied().enumerate() {
                let run = synthesize(model, group_id, text, strategy)?;
                let token_hash = run.row["diagnostics"]["semantic_token_hash"].as_u64();
                if let Some(hash) = token_hash {
                    if let Some(previous) = hashes[strategy_index] {
                        if previous != hash {
                            return Err(format!(
                                "semantic token hash changed within {group_id}/{}",
                                strategy.name
                            )
                            .into());
                        }
                    } else {
                        hashes[strategy_index] = Some(hash);
                    }
                }
                if sample_index == 0 {
                    first_measured_audio[strategy_index] = Some(run.audio.clone());
                    first_measured_first_callback[strategy_index] =
                        Some(run.first_callback_audio.clone());
                }
                let row = enrich_with_reference(run.row, hashes[0], strategy.preview_tokens);
                by_strategy[strategy_index].push(row.clone());
                runs.push(row);
            }
            if (sample_index + 1) % 5 == 0 {
                eprintln!(
                    "R1.8 {group_id}: measured {}/{} per strategy",
                    sample_index + 1,
                    SAMPLES
                );
            }
        }

        if hashes.iter().any(|hash| *hash != hashes[0]) {
            return Err(format!(
                "semantic tokens differ across R1.8 strategies for {group_id}: {hashes:?}"
            )
            .into());
        }

        let baseline_first_callback = first_measured_first_callback[0]
            .as_deref()
            .ok_or("missing Chunk-24 first callback waveform")?;
        for (index, strategy) in STRATEGIES.iter().enumerate().skip(1) {
            let candidate_first_callback = first_measured_first_callback[index]
                .as_deref()
                .ok_or("missing preview first callback waveform")?;
            let comparison = compare_waveforms(baseline_first_callback, candidate_first_callback);
            eprintln!(
                "R1.8 {group_id}/{} first-callback waveform {}",
                strategy.name, comparison
            );
            waveform_comparisons.push(json!({
                "group_id": group_id,
                "strategy": strategy.name,
                "scope": "common sample prefix of first callback PCM",
                "metrics": comparison
            }));
        }

        for (index, strategy) in STRATEGIES.iter().enumerate() {
            let audio = first_measured_audio[index]
                .take()
                .ok_or("missing measured audio for blind-listening track")?;
            blind_tracks.push((
                group_id.to_owned(),
                strategy.name.to_owned(),
                hashes[index].unwrap_or_default().to_string(),
                text.to_owned(),
                audio,
            ));
            let summary = summarize(&by_strategy[index]);
            println!("R1.8 {group_id}/{} {}", strategy.name, summary);
        }
    }

    let summary = summarize_all(&runs);
    let cancellation_recovery = validate_cancellation_recovery(model)?;
    let report = json!({
        "schema": "alice-native-tts-r1.8-preview-refill-benchmark-v1",
        "runtime": "CUDA GPT + CUDA VITS, CPU token sampling, ABI-1",
        "chunk_tokens": 24,
        "seed": 1234,
        "warmups_per_group_strategy": WARMUPS,
        "samples_per_group_strategy": SAMPLES,
        "buffer_simulation_ms": [32, 64],
        "audio_activity_definition": "RMS >= 1e-4 in non-overlapping 10 ms windows",
        "quality_gate": {
            "semantic_token_hashes_equal_within_each_group": true,
            "human_blind_audition": "NOT_RUN; package exported for review",
            "real_playback_acceptance": "NOT_RUN"
        },
        "first_callback_waveform_comparison_vs_chunk24": waveform_comparisons,
        "cancellation_recovery": cancellation_recovery,
        "summary": summary,
        "runs": runs,
        "blind_package": "blind-listening/",
        "blind_key": "blind-key.json"
    });
    fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    export_blind_package(output, blind_tracks)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn validate_cancellation_recovery(
    model: &mut NativeTts,
) -> Result<Value, Box<dyn std::error::Error>> {
    let text = CORPUS[2].1;
    let mut cases = Vec::new();
    for preview_tokens in [17, 20] {
        let cancel = AtomicBool::new(false);
        let mut callbacks = 0usize;
        let result = model.synthesize_streaming_r1_8_experiment(
            text,
            &|| cancel.load(Ordering::Acquire),
            Some(preview_tokens),
            |_, info| {
                if !info.final_chunk {
                    callbacks += 1;
                    cancel.store(true, Ordering::Release);
                }
            },
        );
        if !matches!(result, Err(Error::Cancelled)) || callbacks != 1 {
            return Err(format!(
                "R1.8 preview{preview_tokens} cancellation failed: callbacks={callbacks}, result={result:?}"
            )
            .into());
        }
        let mut recovered_audio = Vec::new();
        model.synthesize_streaming("你好。", &|| false, |audio, _| {
            recovered_audio.extend_from_slice(audio)
        })?;
        if recovered_audio.is_empty() || recovered_audio.iter().any(|sample| !sample.is_finite()) {
            return Err(
                format!("R1.8 preview{preview_tokens} did not recover after cancellation").into(),
            );
        }
        cases.push(json!({
            "preview_tokens": preview_tokens,
            "cancelled_after_callbacks": callbacks,
            "cancel_error_observed": true,
            "next_synthesis_recovered": true,
            "recovered_samples": recovered_audio.len()
        }));
    }
    Ok(json!({"cases": cases, "passed": true}))
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
        if callbacks.is_empty() {
            first_callback_audio.extend_from_slice(samples);
        }
        callbacks.push(CallbackRecord {
            at_ms,
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
    let (timings, diagnostics) = model.synthesize_streaming_r1_8_experiment(
        text,
        &|| false,
        strategy.preview_tokens,
        &mut callback,
    )?;
    drop(callback);
    if audio.is_empty() || audio.iter().any(|sample| !sample.is_finite()) {
        return Err(format!("{group_id}/{} produced invalid audio", strategy.name).into());
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let first_callback_ms = callbacks.first().map(|row| row.at_ms);
    let second_callback_ms = callbacks.get(1).map(|row| row.at_ms);
    let active = activity(&first_callback_audio);
    let rtf = elapsed_ms / (audio.len() as f64 / SAMPLE_RATE as f64 * 1000.0);
    let simulations = [
        simulate_buffer(&callbacks, 32),
        simulate_buffer(&callbacks, 64),
    ];
    let boundary_jumps = callbacks
        .iter()
        .filter_map(|event| event.boundary_jump_before)
        .collect::<Vec<_>>();
    let row = json!({
        "group_id": group_id,
        "strategy": strategy.name,
        "preview_tokens_requested": strategy.preview_tokens,
        "elapsed_ms": elapsed_ms,
        "ttfa_callback_ms": first_callback_ms,
        "first_audible_pcm_ms": first_callback_ms.map(|ms| ms + active.leading_silence_ms),
        "second_segment_arrival_ms": second_callback_ms,
        "boundary_jumps": boundary_jumps,
        "first_segment_samples": first_callback_audio.len(),
        "first_segment_duration_ms": first_callback_audio.len() as f64 * 1000.0 / SAMPLE_RATE as f64,
        "first_segment_active_duration_ms": active.active_ms,
        "first_segment_leading_silence_ms": active.leading_silence_ms,
        "rtf": rtf,
        "semantic_tokens": timings.semantic_tokens,
        "first_audio_tokens": timings.first_audio_tokens,
        "audio_chunks": timings.audio_chunks,
        "audio_samples": audio.len(),
        "audio_seconds": audio.len() as f64 / SAMPLE_RATE as f64,
        "callbacks": callbacks,
        "playback_buffer_simulation": simulations,
        "timings": timings,
        "diagnostics": diagnostics,
        "audio_finite": true
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
    active_ms: f64,
}

fn activity(audio: &[f32]) -> Activity {
    let mut first_active = None;
    let mut active_windows = 0usize;
    for (index, frame) in audio.chunks(ACTIVE_WINDOW).enumerate() {
        if frame.is_empty() {
            continue;
        }
        let rms = (frame
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>()
            / frame.len() as f64)
            .sqrt();
        if rms >= ACTIVE_RMS_THRESHOLD {
            first_active.get_or_insert(index);
            active_windows += 1;
        }
    }
    Activity {
        leading_silence_ms: first_active
            .map_or(audio.len() as f64 * 1000.0 / SAMPLE_RATE as f64, |index| {
                index as f64 * ACTIVE_WINDOW as f64 * 1000.0 / SAMPLE_RATE as f64
            }),
        active_ms: active_windows as f64 * ACTIVE_WINDOW as f64 * 1000.0 / SAMPLE_RATE as f64,
    }
}

fn compare_waveforms(reference: &[f32], candidate: &[f32]) -> Value {
    let compared_samples = reference.len().min(candidate.len());
    if compared_samples == 0 {
        return json!({"compared_samples": 0, "valid": false});
    }
    let mut dot = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut absolute_error = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut maximum_error = 0.0f64;
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
        absolute_error += error;
        squared_error += error * error;
        maximum_error = maximum_error.max(error);
    }
    let correlation = if reference_energy > 0.0 && candidate_energy > 0.0 {
        dot / (reference_energy * candidate_energy).sqrt()
    } else {
        0.0
    };
    json!({
        "valid": true,
        "compared_samples": compared_samples,
        "alignment": "same sample index from utterance start; no shift applied",
        "mae": absolute_error / compared_samples as f64,
        "rmse": (squared_error / compared_samples as f64).sqrt(),
        "max_abs": maximum_error,
        "cosine_similarity": correlation
    })
}

fn simulate_buffer(callbacks: &[CallbackRecord], buffer_ms: u64) -> Value {
    let threshold_samples = SAMPLE_RATE * buffer_ms as usize / 1000;
    let mut buffered_before_start = 0usize;
    let mut start_index = None;
    for (index, event) in callbacks.iter().enumerate() {
        buffered_before_start += event.sample_count;
        if buffered_before_start >= threshold_samples {
            start_index = Some(index);
            break;
        }
    }
    let Some(start_index) = start_index else {
        return json!({
            "buffer_ms": buffer_ms,
            "playback_started": false,
            "underrun_count": 0,
            "underrun_ms": 0.0
        });
    };
    let playback_start_ms = callbacks[start_index].at_ms + buffer_ms as f64;
    let mut queued_samples: f64 = callbacks
        .iter()
        .take_while(|event| event.at_ms <= playback_start_ms)
        .map(|event| event.sample_count as f64)
        .sum();
    let mut cursor_ms = playback_start_ms;
    let mut underrun_count = 0usize;
    let mut underrun_ms = 0.0;
    for event in callbacks
        .iter()
        .filter(|event| event.at_ms > playback_start_ms)
    {
        let needed = (event.at_ms - cursor_ms).max(0.0) * SAMPLE_RATE as f64 / 1000.0;
        if queued_samples < needed {
            let deficit = needed - queued_samples;
            underrun_count += 1;
            underrun_ms += deficit * 1000.0 / SAMPLE_RATE as f64;
            queued_samples = 0.0;
        } else {
            queued_samples -= needed;
        }
        queued_samples += event.sample_count as f64;
        cursor_ms = event.at_ms;
    }
    json!({
        "buffer_ms": buffer_ms,
        "playback_started": true,
        "playback_start_ms": playback_start_ms,
        "underrun_count": underrun_count,
        "underrun_ms": underrun_ms,
        "queued_samples_after_last_callback": queued_samples
    })
}

fn enrich_with_reference(
    mut row: Value,
    baseline_hash: Option<u64>,
    preview: Option<usize>,
) -> Value {
    if let Some(reference_hash) = baseline_hash {
        let candidate_hash = row["diagnostics"]["semantic_token_hash"].as_u64();
        row["semantic_tokens_match_chunk24"] = json!(candidate_hash == Some(reference_hash));
    } else {
        row["semantic_tokens_match_chunk24"] = Value::Null;
    }
    row["quality_candidate"] = json!(preview.is_some());
    row
}

fn summarize(rows: &[Value]) -> Value {
    let field = |row: &Value, key: &str| row[key].as_f64();
    let p50p95 = |key: &str| {
        let values = rows
            .iter()
            .filter_map(|row| field(row, key))
            .collect::<Vec<_>>();
        json!({"p50": percentile(&values, 0.50), "p95": percentile(&values, 0.95)})
    };
    let underruns = |buffer_ms: u64| {
        rows.iter()
            .filter(|row| {
                row["playback_buffer_simulation"][0]["buffer_ms"] == buffer_ms
                    || row["playback_buffer_simulation"][1]["buffer_ms"] == buffer_ms
            })
            .filter(|row| {
                row["playback_buffer_simulation"]
                    .as_array()
                    .and_then(|list| list.iter().find(|entry| entry["buffer_ms"] == buffer_ms))
                    .and_then(|entry| entry["underrun_count"].as_u64())
                    .unwrap_or(0)
                    > 0
            })
            .count()
    };
    let boundary_jumps = rows
        .iter()
        .filter_map(|row| row["boundary_jumps"].as_array())
        .flatten()
        .filter_map(Value::as_f64)
        .collect::<Vec<_>>();
    json!({
        "samples": rows.len(),
        "ttfa_callback_ms": p50p95("ttfa_callback_ms"),
        "first_audible_pcm_ms": p50p95("first_audible_pcm_ms"),
        "second_segment_arrival_ms": p50p95("second_segment_arrival_ms"),
        "first_segment_duration_ms": p50p95("first_segment_duration_ms"),
        "first_segment_active_duration_ms": p50p95("first_segment_active_duration_ms"),
        "elapsed_ms": p50p95("elapsed_ms"),
        "rtf": p50p95("rtf"),
        "boundary_jump": {
            "p50": percentile(&boundary_jumps, 0.50),
            "p95": percentile(&boundary_jumps, 0.95),
            "max": boundary_jumps.iter().copied().reduce(f64::max)
        },
        "underrun_runs_32ms": underruns(32),
        "underrun_runs_64ms": underruns(64)
    })
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
    json!({
        "groups": groups,
        "activity_threshold_rms": ACTIVE_RMS_THRESHOLD,
        "activity_window_ms": 10
    })
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

fn export_blind_package(
    output: &Path,
    tracks: Vec<(String, String, String, String, Vec<f32>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let blind_dir = output.join("blind-listening");
    let tracks_dir = blind_dir.join("tracks");
    fs::create_dir_all(&tracks_dir)?;
    fs::write(
        blind_dir.join("README.txt"),
        "Anonymous R1.8 listening set. Track IDs do not encode text group or strategy.\nPlease record clarity, continuity, repeated/missing words, pauses, and overall preference by track ID.\nThe mapping key is stored separately as blind-key.json.\n",
    )?;

    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
    seed ^= std::process::id() as u64;
    let mut ids = (0..tracks.len())
        .map(|index| format!("A{:02}", index + 1))
        .collect::<Vec<_>>();
    for index in (1..ids.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ids.swap(index, (seed as usize) % (index + 1));
    }

    let mut key = Vec::new();
    for (index, (group, strategy, token_hash, text, audio)) in tracks.into_iter().enumerate() {
        let id = ids[index].clone();
        write_wav(&tracks_dir.join(format!("{id}.wav")), &audio)?;
        key.push(json!({
            "track_id": id,
            "group_id": group,
            "strategy": strategy,
            "semantic_token_hash": token_hash,
            "text": text
        }));
    }
    fs::write(blind_dir.join("track-ids.txt"), ids.join("\n") + "\n")?;
    fs::write(
        output.join("blind-key.json"),
        serde_json::to_vec_pretty(&key)?,
    )?;
    Ok(())
}

fn write_f32le(path: &Path, audio: &[f32]) -> Result<(), std::io::Error> {
    let mut file = fs::File::create(path)?;
    for sample in audio {
        file.write_all(&sample.to_le_bytes())?;
    }
    file.flush()
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
