use alice_native_tts::{Backend, ChunkInfo, Config, NativeTts};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
    time::Instant,
};

const CASES: [(&str, &str, &str, &str); 8] = [
    ("zh_short", "zh", "short", "你好。"),
    ("zh_poly_short", "zh", "polyphonic-short", "行长来了。"),
    (
        "zh_long",
        "zh",
        "long",
        "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
    ),
    (
        "zh_poly_long",
        "zh",
        "polyphonic-long",
        "银行行长在重庆重看那段长长的乐章，朋友们还在行走；后来他着重介绍了重要的行程。",
    ),
    ("en_short", "en", "short", "Hello, Alice."),
    (
        "en_long",
        "en",
        "long",
        "The quick brown fox jumps over the lazy dog while Alice reads a book and welcomes everyone home.",
    ),
    ("mixed_short", "mixed", "short", "你好 Alice，请看这个 report。"),
    (
        "mixed_long",
        "mixed",
        "long",
        "你好 Alice，请 read the report about 银行行长 who will travel to 重庆 tomorrow.",
    ),
];
const THRESHOLDS: [usize; 2] = [20, 24];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=8).contains(&args.len()) {
        return Err(
            "usage: r1_4_profile PACK OUTPUT_DIR WARMUPS REPEATS [20,24] [case_ids] [cold|hit]"
                .into(),
        );
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let warmups: usize = args[3].parse()?;
    let repeats: usize = args[4].parse()?;
    if warmups < 30 || repeats < 20 {
        return Err(
            "R1.4 requires at least 30 warmups and 20 measured runs per case/threshold".into(),
        );
    }
    let selected_thresholds = if let Some(value) = args.get(5) {
        let thresholds = value
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if !thresholds.contains(&20)
            || !thresholds.contains(&24)
            || thresholds.iter().any(|value| !THRESHOLDS.contains(value))
        {
            return Err("thresholds must be selected from 20,24 and include 24".into());
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
            return Err("unknown R1.4 case id; see the expanded corpus IDs".into());
        }
        CASES
            .into_iter()
            .filter(|case| requested.contains(&case.0))
            .collect::<Vec<_>>()
    } else {
        CASES.to_vec()
    };
    let cache_mode = args.get(7).map(String::as_str).unwrap_or("cold");
    match cache_mode {
        "cold" => {
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "1");
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS", "1");
        }
        "hit" => {
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "1");
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS", "0");
        }
        _ => return Err("cache mode must be cold or hit".into()),
    }
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
    fs::create_dir_all(output.join("blind-audio"))?;
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
    let memory_sampler = MemorySampler::start();
    let gpu_memory_after_load = gpu_memory_snapshot();
    let _ = fs::remove_file(output.join("blind-listening-key.jsonl"));
    let mut cases = Vec::new();

    for (case_id, language, length_class, text) in selected_cases.iter().copied() {
        let mut threshold_reports = Vec::new();
        let mut audio_variants = Vec::new();
        for &threshold in &selected_thresholds {
            eprintln!(
                "R1.4 starting case={case_id} first_chunk={threshold} cache={cache_mode} warmups={warmups} samples={repeats}"
            );
            memory_sampler.reset_peak();
            for _ in 0..warmups {
                run_once(&mut model, text, threshold, false)?;
            }
            let mut rows = Vec::with_capacity(repeats);
            let mut repeat_audio = None;
            for sample_index in 0..repeats {
                let (row, audio, chunks) = run_once(&mut model, text, threshold, true)?;
                if sample_index == 0 {
                    let audio_path = output.join(format!("{case_id}-first-{threshold}.wav"));
                    write_wav(&audio_path, &audio)?;
                    let semantic_tokens = row["r1_4_stages"]["semantic_tokens"]
                        .as_u64()
                        .unwrap_or_default();
                    let semantic_token_hash = row["r1_4_stages"]["semantic_token_hash"]
                        .as_u64()
                        .unwrap_or_default();
                    audio_variants.push((
                        threshold,
                        audio.clone(),
                        chunks.clone(),
                        semantic_tokens,
                        semantic_token_hash,
                    ));
                } else if sample_index == 1 {
                    let audio_path = output.join(format!("{case_id}-first-{threshold}-repeat.wav"));
                    write_wav(&audio_path, &audio)?;
                    repeat_audio = Some(audio);
                }
                rows.push(row);
            }
            let first_sample_audio = audio_variants
                .iter()
                .find(|(candidate_threshold, _, _, _, _)| *candidate_threshold == threshold)
                .map(|variant| variant.1.as_slice())
                .ok_or("missing first measured audio")?;
            let within_threshold_repeatability = repeat_audio
                .as_ref()
                .map(|repeated| waveform_metrics(first_sample_audio, repeated))
                .unwrap_or(Value::Null);
            let memory_after_group = json!({
                "after_group": current_memory(),
                "during_group_peak": memory_sampler.peak(),
                "nvidia_smi_snapshot": gpu_memory_snapshot()
            });
            threshold_reports.push(json!({
                "first_chunk_tokens": threshold,
                "subsequent_chunk_tokens": 24,
                "warmup_runs": warmups,
                "samples": rows,
                "summary": summarize(&rows, threshold),
                "boundary_multi_sample": boundary_summary(&rows),
                "within_threshold_repeatability_sample_0_vs_1": within_threshold_repeatability,
                "memory_after_group": memory_after_group,
                "quality_vs_24": null,
                "first_audio_file": format!("{case_id}-first-{threshold}.wav"),
                "repeat_audio_file": format!("{case_id}-first-{threshold}-repeat.wav")
            }));
            eprintln!("R1.4 completed case={case_id} first_chunk={threshold}");
        }
        let reference_variant = audio_variants
            .iter()
            .find(|(threshold, _, _, _, _)| *threshold == 24)
            .ok_or("missing threshold-24 listening reference")?;
        let reference = &reference_variant.1;
        let reference_chunks = &reference_variant.2;
        let reference_boundary_multi = threshold_reports
            .iter()
            .find(|report| report["first_chunk_tokens"] == 24)
            .and_then(|report| report["samples"].as_array())
            .map(|rows| boundary_summary(rows))
            .unwrap_or(Value::Null);
        let boundary_summaries = threshold_reports
            .iter()
            .filter_map(|report| {
                Some((
                    report["first_chunk_tokens"].as_u64()? as usize,
                    boundary_summary(report["samples"].as_array()?),
                ))
            })
            .collect::<Vec<_>>();
        let blind_pair_id = format!("P{:02}", cases.len() + 1);
        let blind_reverse = blind_order_reversed(case_id);
        let mut blind_mapping = json!({"case_id":case_id,"pair_id":blind_pair_id});
        for threshold_row in &mut threshold_reports {
            let threshold = threshold_row["first_chunk_tokens"].as_u64().unwrap() as usize;
            let (_, candidate, candidate_chunks, semantic_tokens, semantic_token_hash) =
                audio_variants
                    .iter()
                    .find(|(candidate_threshold, _, _, _, _)| *candidate_threshold == threshold)
                    .ok_or("missing first-chunk audio sample")?;
            let candidate_boundary_multi = boundary_summaries
                .iter()
                .find(|(candidate_threshold, _)| *candidate_threshold == threshold)
                .map(|(_, summary)| summary.clone())
                .unwrap_or(Value::Null);
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
                            "candidate": boundary_metrics(candidate, candidate_chunks),
                            "all_measured_reference": reference_boundary_multi.clone(),
                            "all_measured_candidate": candidate_boundary_multi
                        }
                    })
                },
            );
            let blind_label = if blind_reverse {
                if threshold == 20 {
                    "B"
                } else {
                    "A"
                }
            } else if threshold == 20 {
                "A"
            } else {
                "B"
            };
            let blind_path = format!("blind-audio/{blind_pair_id}-{blind_label}.wav");
            write_wav(&output.join(&blind_path), candidate)?;
            if threshold == 20 {
                blind_mapping["threshold_20_file"] = json!(blind_path);
            } else {
                blind_mapping["threshold_24_file"] = json!(blind_path);
            }
        }
        blind_mapping["order_reversed"] = json!(blind_reverse);
        blind_mapping["semantic_tokens_equal"] = json!(
            reference_variant.3
                == audio_variants
                    .iter()
                    .find(|(t, _, _, _, _)| *t == 20)
                    .map(|v| v.3)
                    .unwrap_or_default()
        );
        blind_mapping["semantic_token_hash_equal"] = json!(
            reference_variant.4
                == audio_variants
                    .iter()
                    .find(|(t, _, _, _, _)| *t == 20)
                    .map(|v| v.4)
                    .unwrap_or_default()
        );
        let mut blind_key = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output.join("blind-listening-key.jsonl"))?;
        use std::io::Write;
        writeln!(blind_key, "{}", serde_json::to_string(&blind_mapping)?)?;
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
        "stage": "alice-native-tts R1.4 cold frontend and streaming quality",
        "backend": "CUDA GPT + CUDA VITS EP, explicit audited CPU helper fallback",
        "model_load_ms": model_load_ms,
        "gpu_memory_after_model_load": gpu_memory_after_load,
        "warmup_runs_per_case_threshold": warmups,
        "measured_runs_per_case_threshold": repeats,
        "first_chunk_thresholds": selected_thresholds,
        "subsequent_chunk_tokens": 24,
        "seed": 1234,
        "frontend_cache_enabled": std::env::var("ALICE_NATIVE_TTS_FRONTEND_CACHE")
            .ok().as_deref() != Some("0"),
        "frontend_cache_force_miss": std::env::var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS")
            .ok().as_deref() == Some("1"),
        "cache_mode": cache_mode,
        "frontend_backends": {
            "g2pw": std::env::var("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND").unwrap_or_else(|_| "cpu".into()),
            "bert": std::env::var("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND").unwrap_or_else(|_| "cpu".into())
        },
        "frontend_parallel": std::env::var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL")
            .ok().as_deref() == Some("1"),
        "frontend_cuda_diagnostic_allow_cpu": std::env::var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU")
            .ok().as_deref() == Some("1"),
        "g2pw_batch_size": std::env::var("ALICE_NATIVE_TTS_G2PW_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4),
        "case_ids": selected_cases.iter().map(|case| case.0).collect::<Vec<_>>(),
        "cases": cases,
        "cancellation_recovery": cancellation_by_threshold,
        "notes": {
            "frontend_cache": if cache_mode == "cold" { "enabled with a profiling-only forced exact-key miss on each request; computes Frontend and records miss plus insert on warmup and measured requests, with model sessions hot after warmup" } else { "enabled: warmups populate exact-text entries; measured requests should report hits" },
            "first_audio_timeline": "Each measured sample contains request-relative Frontend subphases, GPT prefill, every token sampling/Step phase through first callback, first VITS subphases including SOLA, and first callback start/end.",
            "playback": "32/64 ms prebuffer underruns are an offline callback-arrival queue simulation; no audio device is driven by this crate.",
            "audio_files": "Threshold-tagged PCM16 listening copies plus anonymized 20-vs-24 blind pairs are exported; numerical metrics use original f32 callbacks.",
            "ort_transfer_trace": "ORT operator profiling is a separate diagnostic run and is not mixed into hot benchmark timing."
        }
    });
    fs::write(
        output.join("r1.4-first-audio.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut blind_prompt = String::from(
        "R1.4 blind pair listening sheet. Compare A and B for naturalness, pronunciation, timbre, boundary clicks, and continuity into later chunks. Do not infer threshold from filenames. Record preference and any audible defect per pair.\n",
    );
    for (index, case) in cases.iter().enumerate() {
        blind_prompt.push_str(&format!(
            "{}: pair P{:02} — A vs B; source case {} (language {}, {}).\n",
            index + 1,
            index + 1,
            case["case_id"].as_str().unwrap_or_default(),
            case["language"].as_str().unwrap_or_default(),
            case["length_class"].as_str().unwrap_or_default()
        ));
    }
    fs::write(output.join("blind-listening-prompts.txt"), blind_prompt)?;
    let concise = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            json!({
                "case_id": case["case_id"],
                "thresholds": case["thresholds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        let summary = &row["summary"];
                        json!({
                            "first_chunk_tokens": row["first_chunk_tokens"],
                            "ttfa_ms": summary["ttfa_ms"],
                            "elapsed_ms": summary["elapsed_ms"],
                            "rtf": summary["rtf"],
                            "frontend_ms": summary["frontend_total_ms"],
                            "first_chunk_audio_ms": summary["first_chunk_audio_ms"],
                            "playback_32ms_underrun_samples": summary["playback_prebuffer_32ms"]["samples_with_underruns"],
                            "playback_64ms_underrun_samples": summary["playback_prebuffer_64ms"]["samples_with_underruns"],
                            "semantic_token_hash_equal_to_24": row["quality_vs_24"]["semantic_token_hash_equal"],
                            "boundary_multi_sample": row["boundary_multi_sample"]["abs_jump_p50_p95_max"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string_pretty(&concise)?);
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
    let first_chunk_samples = chunk_lengths.first().copied().unwrap_or_default();
    let first_chunk_audio_ms = first_chunk_samples as f64 * 1000.0 / 32_000.0;
    let boundary = boundary_metrics(&audio, &chunk_lengths);
    let playback = json!({
        "prebuffer_32ms": playback_simulation(&chunks, 32.0),
        "prebuffer_64ms": playback_simulation(&chunks, 64.0)
    });
    let row = json!({
        "ttfa_ms": first_audio_ms.unwrap_or(elapsed_ms),
        "elapsed_ms": elapsed_ms,
        "audio_seconds": audio_seconds,
        "rtf": if audio_seconds > 0.0 { elapsed_ms / (audio_seconds * 1000.0) } else { 0.0 },
        "sample_count": sample_count,
        "first_chunk_samples": first_chunk_samples,
        "first_chunk_audio_ms": first_chunk_audio_ms,
        "audio_fingerprint_fnv1a64": audio_fingerprint(&audio),
        "chunk_lengths": chunk_lengths,
        "chunks": chunks,
        "boundary": boundary,
        "playback": playback,
        "synthesis_timings": timings,
        "r1_4_stages": profile
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
            row["r1_4_stages"]["phases"]
                .as_array()?
                .iter()
                .find(|phase| phase["phase"] == name)?["elapsed_ms"]
                .as_f64()
        }))
    };
    let timeline = |name: &'static str, duration: bool| {
        values(Box::new(move |row| {
            let event = row["r1_4_stages"]["timeline"]
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
    let measured_token_steps = rows
        .iter()
        .flat_map(|row| {
            row["r1_4_stages"]["timeline"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|event| event["event"] == "gpt_step_run")
                .filter_map(|event| event["index"].as_u64().map(|index| index as usize))
        })
        .max()
        .map_or(threshold, |index| index + 1);
    for token_index in 0..measured_token_steps {
        let mut durations = rows
            .iter()
            .filter_map(|row| {
                row["r1_4_stages"]["timeline"]
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
        "distinct_audio_fingerprints": distinct_u64(rows, "audio_fingerprint_fnv1a64"),
        "gpt_step_total_ms": phase("gpt_step_run"),
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
            "join": phase("frontend_join"),
            "parallel_wall": phase("frontend_parallel_run")
        },
        "frontend_cache_hits": values(Box::new(|row| {
            row["r1_4_stages"]["frontend_cache_hits"].as_u64().map(|value| value as f64)
        })),
        "frontend_cache_misses": values(Box::new(|row| {
            row["r1_4_stages"]["frontend_cache_misses"].as_u64().map(|value| value as f64)
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
        "first_chunk_audio_ms": values(Box::new(|row| row["first_chunk_audio_ms"].as_f64())),
        "total_audio_ms": values(Box::new(|row| row["audio_seconds"].as_f64().map(|v| v * 1000.0))),
        "playback_prebuffer_32ms": playback_summary(rows, "prebuffer_32ms"),
        "playback_prebuffer_64ms": playback_summary(rows, "prebuffer_64ms"),
        "per_token_step_ms": token_steps
    })
}

fn distinct_u64(rows: &[Value], field: &str) -> usize {
    let mut values = rows
        .iter()
        .filter_map(|row| row[field].as_u64())
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values.len()
}

fn indexed_timeline_duration(rows: &[Value], name: &str, index: usize) -> Value {
    let mut durations = rows
        .iter()
        .filter_map(|row| {
            let events = row["r1_4_stages"]["timeline"].as_array()?;
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
    json!({"callback_boundaries":jumps.len(),"abs_jump_values":&jumps,"abs_jump_p50_p95_max":{
        "p50": jumps.get(percentile_index(jumps.len(),0.50)).copied(),
        "p95": jumps.get(percentile_index(jumps.len(),0.95)).copied(),
        "max": jumps.last().copied()
    }})
}

fn boundary_summary(rows: &[Value]) -> Value {
    let jumps = rows
        .iter()
        .filter_map(|row| row["boundary"]["abs_jump_values"].as_array())
        .flatten()
        .filter_map(Value::as_f64)
        .collect::<Vec<_>>();
    let mut sorted = jumps;
    sorted.sort_by(f64::total_cmp);
    json!({
        "sample_count": rows.len(),
        "boundary_count": sorted.len(),
        "abs_jump_p50_p95_max": {
            "p50": sorted.get(percentile_index(sorted.len(), 0.50)).copied(),
            "p95": sorted.get(percentile_index(sorted.len(), 0.95)).copied(),
            "max": sorted.last().copied()
        }
    })
}

fn playback_simulation(chunks: &[Value], prebuffer_ms: f64) -> Value {
    let mut available_ms = 0.0f64;
    let mut previous_at_ms: Option<f64> = None;
    let mut started = false;
    let mut start_at_ms = None;
    let mut underrun_ms = 0.0f64;
    let mut underrun_count = 0usize;
    for chunk in chunks {
        let at_ms = chunk["at_ms"].as_f64().unwrap_or_default();
        let media_ms =
            chunk["sample_count"].as_u64().unwrap_or_default() as f64 * 1000.0 / 32_000.0;
        if let Some(previous) = previous_at_ms {
            if started {
                let consumed = (at_ms - previous).max(0.0);
                if consumed > available_ms {
                    underrun_ms += consumed - available_ms;
                    underrun_count += 1;
                    available_ms = media_ms;
                } else {
                    available_ms = available_ms - consumed + media_ms;
                }
            } else {
                available_ms += media_ms;
            }
        } else {
            available_ms += media_ms;
        }
        if !started && available_ms >= prebuffer_ms {
            started = true;
            start_at_ms = Some(at_ms);
        }
        previous_at_ms = Some(at_ms);
    }
    json!({
        "model": "offline callback-arrival queue simulation; no output device",
        "prebuffer_ms": prebuffer_ms,
        "playback_started": started,
        "simulated_first_playable_ms": start_at_ms,
        "underrun_count": underrun_count,
        "underrun_ms": underrun_ms,
        "ending_buffer_ms": available_ms
    })
}

fn playback_summary(rows: &[Value], name: &str) -> Value {
    let mut underruns = Vec::new();
    let mut counts = Vec::new();
    let mut first_playable = Vec::new();
    for row in rows {
        let Some(playback) = row["playback"].get(name) else {
            continue;
        };
        if let Some(value) = playback["underrun_ms"].as_f64() {
            underruns.push(value);
        }
        if let Some(value) = playback["underrun_count"].as_u64() {
            counts.push(value);
        }
        if let Some(value) = playback["simulated_first_playable_ms"].as_f64() {
            first_playable.push(value);
        }
    }
    underruns.sort_by(f64::total_cmp);
    first_playable.sort_by(f64::total_cmp);
    json!({
        "samples": rows.len(),
        "underrun_ms_p50_p95_max": {
            "p50": underruns.get(percentile_index(underruns.len(), 0.50)).copied(),
            "p95": underruns.get(percentile_index(underruns.len(), 0.95)).copied(),
            "max": underruns.last().copied()
        },
        "samples_with_underruns": counts.iter().filter(|count| **count > 0).count(),
        "simulated_first_playable_ms": stats(&first_playable)
    })
}

fn audio_fingerprint(audio: &[f32]) -> u64 {
    audio.iter().fold(0xcbf29ce484222325u64, |hash, sample| {
        (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x100000001b3)
    })
}

fn blind_order_reversed(case_id: &str) -> bool {
    case_id
        .bytes()
        .fold(0u8, |hash, byte| hash.wrapping_mul(31).wrapping_add(byte))
        & 1
        == 1
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
struct MemoryUsage {
    working_set_bytes: u64,
    private_bytes: u64,
}

#[cfg(windows)]
fn current_memory() -> MemoryUsage {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        values: [usize; 9],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCountersEx::default();
    counters.cb = std::mem::size_of::<ProcessMemoryCountersEx>() as u32;
    let size = counters.cb;
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return MemoryUsage::default();
    }
    MemoryUsage {
        working_set_bytes: counters.values[1] as u64,
        private_bytes: counters.values[8] as u64,
    }
}

#[cfg(not(windows))]
fn current_memory() -> MemoryUsage {
    MemoryUsage::default()
}

fn gpu_memory_snapshot() -> Value {
    use std::process::Command;
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let processes = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let capture = |result: std::io::Result<std::process::Output>| match result {
        Ok(output) => json!({
            "success": output.status.success(),
            "stdout": String::from_utf8_lossy(&output.stdout).trim(),
            "stderr": String::from_utf8_lossy(&output.stderr).trim()
        }),
        Err(error) => json!({"success":false,"error":error.to_string()}),
    };
    json!({"gpu":capture(gpu),"compute_processes":capture(processes)})
}

struct MemorySampler {
    stop: Arc<AtomicBool>,
    working_set_peak: Arc<AtomicU64>,
    private_bytes_peak: Arc<AtomicU64>,
    worker: Option<thread::JoinHandle<()>>,
}

impl MemorySampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let working_set_peak = Arc::new(AtomicU64::new(0));
        let private_bytes_peak = Arc::new(AtomicU64::new(0));
        let worker_stop = Arc::clone(&stop);
        let worker_working = Arc::clone(&working_set_peak);
        let worker_private = Arc::clone(&private_bytes_peak);
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                let usage = current_memory();
                worker_working.fetch_max(usage.working_set_bytes, Ordering::Relaxed);
                worker_private.fetch_max(usage.private_bytes, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
            }
        });
        let sampler = Self {
            stop,
            working_set_peak,
            private_bytes_peak,
            worker: Some(worker),
        };
        sampler.reset_peak();
        sampler
    }

    fn reset_peak(&self) {
        let usage = current_memory();
        self.working_set_peak
            .store(usage.working_set_bytes, Ordering::Relaxed);
        self.private_bytes_peak
            .store(usage.private_bytes, Ordering::Relaxed);
    }

    fn peak(&self) -> MemoryUsage {
        MemoryUsage {
            working_set_bytes: self.working_set_peak.load(Ordering::Relaxed),
            private_bytes: self.private_bytes_peak.load(Ordering::Relaxed),
        }
    }
}

impl Drop for MemorySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
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
