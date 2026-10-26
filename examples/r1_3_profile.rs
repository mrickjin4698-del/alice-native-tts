use alice_native_tts::{Backend, ChunkInfo, Config, NativeTts};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const CASES: [(&str, &str, &str, &str); 4] = [
    ("zh_short", "zh", "short", "你好。"),
    (
        "zh_long",
        "zh",
        "long",
        "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
    ),
    ("en_short", "en", "short", "Hello Alice, welcome back."),
    (
        "en_long",
        "en",
        "long",
        "The quick brown fox jumps over the lazy dog while Alice reads a book and welcomes everyone home.",
    ),
];
const THRESHOLDS: [usize; 4] = [12, 16, 20, 24];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: r1_3_profile PACK OUTPUT_DIR WARMUPS REPEATS [12,16,20,24] [case_ids]".into(),
        );
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    if warmups < 30 || repeats < 20 {
        return Err(
            "R1.3 requires at least 30 warmups and 20 measured runs per case/threshold".into(),
        );
    }
    let selected_thresholds = if let Some(value) = args.get(5) {
        let thresholds = value
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if thresholds.is_empty()
            || !thresholds.contains(&24)
            || thresholds.iter().any(|value| !THRESHOLDS.contains(value))
        {
            return Err("thresholds must be selected from 12,16,20,24 and include 24".into());
        }
        thresholds
    } else {
        THRESHOLDS.to_vec()
    };
    let selected_cases = if let Some(value) = args.get(6) {
        let requested = value.split(',').collect::<Vec<_>>();
        if requested.is_empty()
            || requested
                .iter()
                .any(|id| !CASES.iter().any(|case| case.0 == *id))
        {
            return Err("unknown R1.3 case id; use zh_short, zh_long, en_short, en_long".into());
        }
        CASES
            .into_iter()
            .filter(|case| requested.contains(&case.0))
            .collect::<Vec<_>>()
    } else {
        CASES.to_vec()
    };
    if std::env::var("ALICE_NATIVE_TTS_CUDA_VITS").ok().as_deref() != Some("1")
        || std::env::var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            != Some("1")
        || std::env::var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            != Some("1")
    {
        return Err("set the audited R1.2 CUDA GPT/VITS diagnostic opt-ins before running".into());
    }
    fs::create_dir_all(&output)?;
    let load_started = Instant::now();
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
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();

    for (case_id, language, length_class, text) in selected_cases.iter().copied() {
        let mut threshold_reports = Vec::new();
        let mut audio_variants = Vec::new();
        for &threshold in &selected_thresholds {
            eprintln!(
                "R1.3 starting case={case_id} first_chunk={threshold} warmups={warmups} samples={repeats}"
            );
            for _ in 0..warmups {
                run_once(&mut model, text, threshold, false)?;
            }
            let mut rows = Vec::with_capacity(repeats);
            for sample_index in 0..repeats {
                let (row, audio, chunks) = run_once(&mut model, text, threshold, true)?;
                if sample_index == 0 {
                    let audio_path = output.join(format!("{case_id}-first-{threshold}.wav"));
                    write_wav(&audio_path, &audio)?;
                    let semantic_tokens = row["r1_3_stages"]["semantic_tokens"]
                        .as_u64()
                        .unwrap_or_default();
                    let semantic_token_hash = row["r1_3_stages"]["semantic_token_hash"]
                        .as_u64()
                        .unwrap_or_default();
                    audio_variants.push((
                        threshold,
                        audio.clone(),
                        chunks.clone(),
                        semantic_tokens,
                        semantic_token_hash,
                    ));
                }
                rows.push(row);
            }
            threshold_reports.push(json!({
                "first_chunk_tokens": threshold,
                "subsequent_chunk_tokens": 24,
                "warmup_runs": warmups,
                "samples": rows,
                "summary": summarize(&rows, threshold),
                "quality_vs_24": null,
                "first_audio_file": format!("{case_id}-first-{threshold}.wav")
            }));
            eprintln!("R1.3 completed case={case_id} first_chunk={threshold}");
        }
        let reference_variant = audio_variants
            .iter()
            .find(|(threshold, _, _, _, _)| *threshold == 24)
            .ok_or("missing threshold-24 listening reference")?;
        let reference = &reference_variant.1;
        let reference_chunks = &reference_variant.2;
        for threshold_row in &mut threshold_reports {
            let threshold = threshold_row["first_chunk_tokens"].as_u64().unwrap() as usize;
            let (_, candidate, candidate_chunks, semantic_tokens, semantic_token_hash) =
                audio_variants
                    .iter()
                    .find(|(candidate_threshold, _, _, _, _)| *candidate_threshold == threshold)
                    .ok_or("missing first-chunk audio sample")?;
            let reference_semantic_tokens = reference_variant.3;
            let reference_semantic_token_hash = reference_variant.4;
            threshold_row.as_object_mut().unwrap().insert(
                "quality_vs_24".into(),
                if threshold == 24 {
                    json!({"reference": true})
                } else {
                    json!({
                        "reference_threshold": 24,
                        "semantic_token_count_equal": semantic_tokens == &reference_semantic_tokens,
                        "semantic_token_hash_equal": semantic_token_hash == &reference_semantic_token_hash,
                        "semantic_token_hash": semantic_token_hash,
                        "waveform": waveform_metrics(reference, candidate),
                        "callback_boundary_abs_jump": {
                            "reference": boundary_metrics(reference, reference_chunks),
                            "candidate": boundary_metrics(candidate, candidate_chunks)
                        }
                    })
                },
            );
        }
        cases.push(json!({
            "case_id": case_id,
            "language": language,
            "length_class": length_class,
            "text": text,
            "thresholds": threshold_reports
        }));
    }

    let cancellation_by_threshold = THRESHOLDS
        .iter()
        .map(|threshold| cancellation_check(&mut model, CASES[0].3, *threshold))
        .collect::<Result<Vec<_>, _>>()?;

    let report = json!({
        "stage": "alice-native-tts R1.3 first audio latency",
        "backend": "CUDA GPT + CUDA VITS EP, explicit audited CPU helper fallback",
        "model_load_ms": model_load_ms,
        "warmup_runs_per_case_threshold": warmups,
        "measured_runs_per_case_threshold": repeats,
        "first_chunk_thresholds": selected_thresholds,
        "subsequent_chunk_tokens": 24,
        "seed": 1234,
        "frontend_cache_enabled": std::env::var("ALICE_NATIVE_TTS_FRONTEND_CACHE")
            .ok()
            .as_deref()
            != Some("0"),
        "g2pw_batch_size": std::env::var("ALICE_NATIVE_TTS_G2PW_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4),
        "case_ids": selected_cases.iter().map(|case| case.0).collect::<Vec<_>>(),
        "cases": cases,
        "cancellation_recovery": cancellation_by_threshold,
        "notes": {
            "first_audio_timeline": "Each measured sample contains request-relative Frontend subphases, GPT prefill, every token sampling/Step phase through the first callback, first VITS subphases including SOLA, and first callback start/end.",
            "audio_files": "WAVs are PCM16 listening copies; numerical comparisons use the original f32 callback samples.",
            "ort_transfer_trace": "ORT operator profiling is a separate diagnostic run and is not mixed into hot benchmark timing."
        }
    });
    fs::write(
        output.join("r1.3-first-audio.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["cases"].as_array().unwrap().iter().map(|case| json!({"case_id":case["case_id"],"thresholds":case["thresholds"].as_array().unwrap().iter().map(|row| json!({"first_chunk_tokens":row["first_chunk_tokens"],"summary":row["summary"],"quality_vs_24":row["quality_vs_24"]})).collect::<Vec<_>>()})).collect::<Vec<_>>())?);
    Ok(())
}

fn run_once(
    model: &mut NativeTts,
    text: &str,
    threshold: usize,
    capture: bool,
) -> Result<(Value, Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut first_audio_ms = None;
    let mut audio = Vec::new();
    let mut chunk_lengths = Vec::new();
    let mut chunks = Vec::new();
    let (timings, profile) = model.synthesize_streaming_first_chunk_r1_3(
        text,
        &|| false,
        threshold,
        |samples, info: ChunkInfo| {
            let at_ms = started.elapsed().as_secs_f64() * 1000.0;
            first_audio_ms.get_or_insert(at_ms);
            chunk_lengths.push(samples.len());
            chunks.push(json!({
                "at_ms": at_ms,
                "semantic_tokens": info.semantic_tokens,
                "sample_count": samples.len(),
                "final": info.final_chunk
            }));
            if capture {
                audio.extend_from_slice(samples);
            }
        },
    )?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let sample_count = audio.len();
    let finite = !capture || audio.iter().all(|sample| sample.is_finite());
    if capture && (sample_count == 0 || !finite) {
        return Err("profile run returned empty or non-finite audio".into());
    }
    let audio_seconds = sample_count as f64 / 32_000.0;
    let row = json!({
        "ttfa_ms": first_audio_ms.unwrap_or(elapsed_ms),
        "elapsed_ms": elapsed_ms,
        "audio_seconds": audio_seconds,
        "rtf": if audio_seconds > 0.0 { elapsed_ms / (audio_seconds * 1000.0) } else { 0.0 },
        "sample_count": sample_count,
        "chunk_lengths": chunk_lengths,
        "chunks": chunks,
        "synthesis_timings": timings,
        "r1_3_stages": profile
    });
    Ok((row, audio, chunk_lengths))
}

fn summarize(rows: &[Value], threshold: usize) -> Value {
    let values = |mut select: Box<dyn FnMut(&Value) -> Option<f64>>| {
        let mut values = rows
            .iter()
            .filter_map(|row| select(row))
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        stats(&values)
    };
    let phase = |name: &'static str| {
        values(Box::new(move |row| {
            row["r1_3_stages"]["phases"]
                .as_array()?
                .iter()
                .find(|phase| phase["phase"] == name)?["elapsed_ms"]
                .as_f64()
        }))
    };
    let timeline = |name: &'static str, duration: bool| {
        values(Box::new(move |row| {
            let event = row["r1_3_stages"]["timeline"]
                .as_array()?
                .iter()
                .find(|event| event["event"] == name)?;
            if duration {
                Some(event["end_ms"].as_f64()? - event["start_ms"].as_f64()?)
            } else {
                event["start_ms"].as_f64()
            }
        }))
    };
    let mut token_steps = Vec::new();
    for token_index in 0..threshold {
        let mut durations = rows
            .iter()
            .filter_map(|row| {
                row["r1_3_stages"]["timeline"]
                    .as_array()?
                    .iter()
                    .find(|event| {
                        event["event"] == "gpt_step_run"
                            && event["index"].as_u64() == Some(token_index as u64)
                    })
                    .map(|event| {
                        event["end_ms"].as_f64().unwrap_or_default()
                            - event["start_ms"].as_f64().unwrap_or_default()
                    })
            })
            .collect::<Vec<_>>();
        durations.sort_by(f64::total_cmp);
        token_steps.push(json!({"token_index":token_index,"gpt_step_ms":stats(&durations)}));
    }
    json!({
        "sample_count": rows.len(),
        "ttfa_ms": values(Box::new(|row| row["ttfa_ms"].as_f64())),
        "elapsed_ms": values(Box::new(|row| row["elapsed_ms"].as_f64())),
        "rtf": values(Box::new(|row| row["rtf"].as_f64())),
        "frontend_total_ms": phase("frontend"),
        "frontend_breakdown_ms": {
            "normalize": phase("frontend_normalize"),
            "cache_lookup": phase("frontend_cache_lookup"),
            "cache_store": phase("frontend_cache_store"),
            "routing": phase("frontend_routing"),
            "tokenizer": phase("frontend_tokenizer"),
            "g2pw_prepare": phase("frontend_g2pw_prepare"),
            "g2pw_run": phase("frontend_g2pw_run"),
            "phoneme_map": phase("frontend_phoneme_map"),
            "bert_input": phase("frontend_bert_input"),
            "bert_run": phase("frontend_bert_run"),
            "feature_expand": phase("frontend_feature_expand"),
            "english_g2p": phase("frontend_english_g2p"),
            "join": phase("frontend_join")
        },
        "frontend_cache_hits": values(Box::new(|row| {
            row["r1_3_stages"]["frontend_cache_hits"].as_u64().map(|value| value as f64)
        })),
        "frontend_cache_misses": values(Box::new(|row| {
            row["r1_3_stages"]["frontend_cache_misses"].as_u64().map(|value| value as f64)
        })),
        "gpt_prefill_ms": timeline("gpt_prefill", true),
        "first_vits_total_ms": timeline("first_vits", true),
        "first_vits_components_ms": {
            "input": indexed_timeline_duration(rows, "vits_input", 0),
            "run": indexed_timeline_duration(rows, "vits_run", 0),
            "device_sync": indexed_timeline_duration(rows, "device_sync", 0),
            "output_validate": indexed_timeline_duration(rows, "vits_output", 0),
            "host_copy": indexed_timeline_duration(rows, "host_data_copy", 0),
            "sola": indexed_timeline_duration(rows, "sola", 0)
        },
        "first_callback_start_ms": timeline("audio_callback", false),
        "per_token_step_until_first_chunk": token_steps
    })
}

fn indexed_timeline_duration(rows: &[Value], name: &str, index: usize) -> Value {
    let mut durations = rows
        .iter()
        .filter_map(|row| {
            let events = row["r1_3_stages"]["timeline"].as_array()?;
            let matching = events
                .iter()
                .filter(|event| {
                    event["event"] == name && event["index"].as_u64() == Some(index as u64)
                })
                .collect::<Vec<_>>();
            (!matching.is_empty()).then(|| {
                matching
                    .iter()
                    .map(|event| {
                        event["end_ms"].as_f64().unwrap_or_default()
                            - event["start_ms"].as_f64().unwrap_or_default()
                    })
                    .sum::<f64>()
            })
        })
        .collect::<Vec<_>>();
    durations.sort_by(f64::total_cmp);
    stats(&durations)
}

fn stats(sorted: &[f64]) -> Value {
    if sorted.is_empty() {
        return json!({"p50": null, "p95": null});
    }
    json!({"p50": percentile(sorted, 0.50), "p95": percentile(sorted, 0.95)})
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn waveform_metrics(reference: &[f32], candidate: &[f32]) -> Value {
    let count = reference.len().min(candidate.len());
    if count == 0 {
        return json!({"reference_samples":reference.len(),"candidate_samples":candidate.len(),"comparable_samples":0});
    }
    let mut abs_sum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut signal_sum = 0.0f64;
    let mut dot = 0.0f64;
    let mut ref_square = 0.0f64;
    let mut cand_square = 0.0f64;
    for (&a, &b) in reference.iter().zip(candidate).take(count) {
        let difference = f64::from(a) - f64::from(b);
        abs_sum += difference.abs();
        square_sum += difference * difference;
        max_abs = max_abs.max(difference.abs());
        signal_sum += f64::from(a) * f64::from(a);
        dot += f64::from(a) * f64::from(b);
        ref_square += f64::from(a) * f64::from(a);
        cand_square += f64::from(b) * f64::from(b);
    }
    let rmse = (square_sum / count as f64).sqrt();
    let snr_db = 10.0 * (signal_sum / square_sum.max(f64::MIN_POSITIVE)).log10();
    let cosine = dot / (ref_square * cand_square).sqrt().max(f64::MIN_POSITIVE);
    json!({
        "reference_samples": reference.len(),
        "candidate_samples": candidate.len(),
        "comparable_samples": count,
        "sample_count_equal": reference.len() == candidate.len(),
        "mae": abs_sum / count as f64,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
        "snr_db": snr_db
    })
}

fn boundary_metrics(audio: &[f32], chunk_lengths: &[usize]) -> Value {
    let mut offset = 0usize;
    let mut jumps = Vec::new();
    for &length in chunk_lengths
        .iter()
        .take(chunk_lengths.len().saturating_sub(1))
    {
        offset = offset.saturating_add(length);
        if offset > 0 && offset < audio.len() {
            jumps.push((audio[offset] - audio[offset - 1]).abs() as f64);
        }
    }
    jumps.sort_by(f64::total_cmp);
    json!({"callback_boundaries":jumps.len(),"abs_jump_p50_p95_max":{
        "p50": jumps.get(percentile_index(jumps.len(),0.50)).copied(),
        "p95": jumps.get(percentile_index(jumps.len(),0.95)).copied(),
        "max": jumps.last().copied()
    }})
}

fn percentile_index(len: usize, p: f64) -> usize {
    ((len as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(len.saturating_sub(1))
}

fn cancellation_check(
    model: &mut NativeTts,
    text: &str,
    threshold: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let cancelled = AtomicBool::new(false);
    let mut callbacks = 0usize;
    let result = model.synthesize_streaming_first_chunk_r1_3(
        text,
        &|| cancelled.load(Ordering::Relaxed),
        threshold,
        |_, _| {
            callbacks += 1;
            cancelled.store(true, Ordering::Relaxed);
        },
    );
    let cancelled_as_expected = matches!(result, Err(alice_native_tts::Error::Cancelled));
    let followup = model.synthesize(text, &|| false)?;
    Ok(json!({
        "first_chunk_threshold": threshold,
        "cancelled_as_expected": cancelled_as_expected,
        "callbacks_before_cancel_return": callbacks,
        "recovered_followup_samples": followup.len(),
        "recovery_passed": !followup.is_empty() && followup.iter().all(|value| value.is_finite())
    }))
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 32_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
