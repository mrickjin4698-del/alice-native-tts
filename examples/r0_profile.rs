//! Reproducible R0 latency, memory and ORT operator profiling harness.
use alice_native_tts::{
    profiling::{allocation_snapshot, enable_allocation_capture, reset_allocation_stats},
    Backend, Config, NativeTts,
};
use serde_json::{json, Value};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

struct AllocationTracker;

#[global_allocator]
static ALLOCATOR: AllocationTracker = AllocationTracker;

unsafe impl GlobalAlloc for AllocationTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !alice_native_tts::profiling::allocation_capture_enabled() {
            return unsafe { System.alloc(layout) };
        }
        let started = Instant::now();
        let pointer = unsafe { System.alloc(layout) };
        alice_native_tts::profiling::record_alloc(
            layout.size(),
            started.elapsed().as_nanos() as u64,
        );
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !alice_native_tts::profiling::allocation_capture_enabled() {
            return unsafe { System.alloc_zeroed(layout) };
        }
        let started = Instant::now();
        let pointer = unsafe { System.alloc_zeroed(layout) };
        alice_native_tts::profiling::record_alloc(
            layout.size(),
            started.elapsed().as_nanos() as u64,
        );
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if !alice_native_tts::profiling::allocation_capture_enabled() {
            return unsafe { System.realloc(pointer, layout, new_size) };
        }
        let started = Instant::now();
        let next = unsafe { System.realloc(pointer, layout, new_size) };
        alice_native_tts::profiling::record_realloc(
            layout.size(),
            new_size,
            started.elapsed().as_nanos() as u64,
        );
        next
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if !alice_native_tts::profiling::allocation_capture_enabled() {
            unsafe { System.dealloc(pointer, layout) };
            return;
        }
        let started = Instant::now();
        unsafe { System.dealloc(pointer, layout) };
        alice_native_tts::profiling::record_dealloc(
            layout.size(),
            started.elapsed().as_nanos() as u64,
        );
    }
}

const TEXT: &str = "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: r0_profile PACK OUTPUT_DIR cpu|directml|cuda|tensor_rt [--chunks=8,16,24,32] [--repeats=20] [--warmups=1] [--ort-profile] [--cuda-vits] [--cuda-vits-allow-cpu] [--gpu-diagnostic-allow-cpu]".into());
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let backend: Backend = serde_json::from_value(json!(args[3].as_str()))?;
    let mut chunks = vec![8usize, 16, 24, 32];
    let mut repeats = 20usize;
    let mut warmups = 1usize;
    let mut ort_profile = false;
    let mut cuda_vits = std::env::var("ALICE_NATIVE_TTS_CUDA_VITS").ok().as_deref() == Some("1");
    let mut cuda_vits_allow_cpu_nodes =
        std::env::var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            == Some("1");
    let mut gpu_diagnostic_allow_cpu = false;
    for arg in &args[4..] {
        if let Some(value) = arg.strip_prefix("--chunks=") {
            chunks = value
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<usize>, _>>()?;
        } else if let Some(value) = arg.strip_prefix("--repeats=") {
            repeats = value.parse()?;
        } else if let Some(value) = arg.strip_prefix("--warmups=") {
            warmups = value.parse()?;
        } else if arg == "--ort-profile" {
            ort_profile = true;
        } else if arg == "--cuda-vits" {
            cuda_vits = true;
        } else if arg == "--cuda-vits-allow-cpu" {
            cuda_vits_allow_cpu_nodes = true;
        } else if arg == "--gpu-diagnostic-allow-cpu" {
            gpu_diagnostic_allow_cpu = true;
        } else {
            return Err(format!("unknown argument: {arg}").into());
        }
    }
    if chunks.is_empty() || chunks.iter().any(|n| !(1..=128).contains(n)) {
        return Err("chunk sizes must be between 1 and 128 tokens".into());
    }
    if repeats == 0 {
        return Err("repeats must be at least 1".into());
    }
    if gpu_diagnostic_allow_cpu
        && (!ort_profile || !matches!(args[3].as_str(), "cuda" | "tensor_rt"))
    {
        return Err(
            "--gpu-diagnostic-allow-cpu requires a GPU backend and --ort-profile; it is only for audited diagnostics".into(),
        );
    }
    fs::create_dir_all(&output)?;
    if gpu_diagnostic_allow_cpu {
        std::env::set_var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU", "1");
        std::env::set_var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU", "1");
        eprintln!("GPU diagnostic mode: audited Encoder helper nodes and VITS nodes may run on CPU; GPT Step remains strict CUDA-only.");
    }
    if cuda_vits {
        if !matches!(args[3].as_str(), "cuda") {
            return Err("--cuda-vits requires the cuda backend".into());
        }
        std::env::set_var("ALICE_NATIVE_TTS_CUDA_VITS", "1");
    }
    if cuda_vits_allow_cpu_nodes {
        if !cuda_vits {
            return Err("--cuda-vits-allow-cpu requires --cuda-vits".into());
        }
        std::env::set_var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU", "1");
    }
    let gpu_encoder_cpu_helper_opt_in = std::env::var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU")
        .ok()
        .as_deref()
        == Some("1");
    let cuda_vits_cpu_fallback_opt_in =
        std::env::var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            == Some("1");
    let backend_value = serde_json::to_value(backend)?;
    let has_gpu_gpt_backend = matches!(backend_value.as_str(), Some("cuda" | "tensor_rt"));
    if ort_profile {
        std::env::set_var(
            "ALICE_NATIVE_TTS_ORT_PROFILE_DIR",
            output.join("ort-profile"),
        );
    }

    let step_graph = if cfg!(feature = "r1-gpt-step-v2")
        && std::env::var("ALICE_NATIVE_TTS_GPT_STEP").ok().as_deref() == Some("2")
        && pack.join("gpt_step_v2.onnx").is_file()
    {
        "gpt_step_v2/per-layer"
    } else {
        "gpt_step/abi1-packed"
    };
    let gpt_step_threads = std::env::var("ALICE_NATIVE_TTS_GPT_STEP_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |threads| threads.get().min(8))
        });
    let load_started = Instant::now();
    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            backend,
            streaming: true,
        },
        &|| false,
    )?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let memory = MemorySampler::start();
    let after_load = current_memory();
    let run_chunks = if ort_profile {
        &chunks[..1]
    } else {
        &chunks[..]
    };
    let mut all_reports = Vec::new();

    for &chunk_tokens in run_chunks {
        memory.reset_peak();
        let mut rows = Vec::new();
        if !ort_profile {
            for _ in 0..warmups {
                synthesize_once(&mut model, chunk_tokens, None)?;
            }
        }
        let sample_count = if ort_profile { 1 } else { repeats };
        for index in 0..sample_count {
            let wav_path = if index == 0 {
                Some(output.join(format!(
                    "{}-chunk-{chunk_tokens}.f32le",
                    backend_name(backend)
                )))
            } else {
                None
            };
            match synthesize_once(&mut model, chunk_tokens, wav_path.as_deref()) {
                Ok(row) => rows.push(row),
                Err(error) => {
                    if ort_profile {
                        if let Ok(profile_paths) = model.finish_ort_profiling() {
                            let mut partial_profiles = Vec::new();
                            for path in profile_paths.into_iter().filter(|path| path.is_file()) {
                                partial_profiles.push(summarize_ort_profile(&path)?);
                            }
                            if !partial_profiles.is_empty() {
                                fs::write(
                                    output.join("ort-profile-partial.json"),
                                    serde_json::to_vec_pretty(&partial_profiles)?,
                                )?;
                                eprintln!(
                                    "partial ORT provider trace: {}",
                                    output.join("ort-profile-partial.json").display()
                                );
                            }
                        }
                    }
                    return Err(error);
                }
            }
        }

        let allocation_probe = if !ort_profile {
            reset_allocation_stats();
            enable_allocation_capture(true);
            let result = synthesize_once(&mut model, chunk_tokens, None);
            enable_allocation_capture(false);
            let row = result?;
            Some(json!({"run": row, "rust_global_allocator": allocation_snapshot()}))
        } else {
            None
        };

        let memory_peak = memory.peak();
        let summary = summarize(&rows);
        let report = json!({
            "backend": backend,
            "chunk_tokens": chunk_tokens,
            "text": TEXT,
            "model_load_ms": load_ms,
            "warmup_runs": if ort_profile { 0 } else { warmups },
            "gpt_step_graph": step_graph,
            "gpt_step_intra_threads": gpt_step_threads,
            "gpu_diagnostic_allow_cpu": gpu_diagnostic_allow_cpu,
            "cuda_vits_requested": cuda_vits,
            "vits_backend": if cuda_vits { "cuda" } else { "cpu" },
            "gpu_encoder_cpu_helper_opt_in": gpu_encoder_cpu_helper_opt_in,
            "gpu_gpt_step_cpu_fallback_disabled": has_gpu_gpt_backend,
            "cuda_vits_cpu_fallback_opt_in": cuda_vits_cpu_fallback_opt_in,
            "cuda_vits_cpu_fallback_disabled": cuda_vits && !cuda_vits_cpu_fallback_opt_in,
            "samples": rows,
            "summary": summary,
            "allocation_probe": allocation_probe,
            "memory_bytes": {
                "after_load": after_load,
                "peak_during_this_chunk_policy": memory_peak,
                "peak_delta_from_after_load": {
                    "working_set": memory_peak.working_set.saturating_sub(after_load.working_set),
                    "private_bytes": memory_peak.private_bytes.saturating_sub(after_load.private_bytes)
                }
            },
            "measurement_notes": {
                "ort_operator_trace": ort_profile,
                "allocation_probe_is_excluded_from_latency_samples": true,
                "rust_allocator_does_not_include_onnx_runtime_native_c_allocations": true,
                "audio_output_is_raw_little_endian_f32": true,
                "cuda_cache_residency_is_asserted_each_step": has_gpu_gpt_backend,
                "cuda_cache_copy_measurement_scope": "K/V endpoint memory placement and strict GPT Step node placement; ORT internal device-to-device copies are not byte-counted"
            }
        });
        fs::write(
            output.join(format!(
                "{}-chunk-{chunk_tokens}.json",
                backend_name(backend)
            )),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "chunk_tokens": chunk_tokens,
                "summary": report["summary"],
                "memory_bytes": report["memory_bytes"]
            }))?
        );
        all_reports.push(report);
    }

    let ort_profiles = if ort_profile {
        let profile_paths = model.finish_ort_profiling()?;
        let mut sessions = Vec::new();
        for path in profile_paths {
            sessions.push(summarize_ort_profile(&path)?);
        }
        Some(sessions)
    } else {
        None
    };
    let final_report = json!({
        "backend": backend,
        "gpt_step_graph": step_graph,
        "gpt_step_intra_threads": gpt_step_threads,
        "gpu_diagnostic_allow_cpu": gpu_diagnostic_allow_cpu,
        "cuda_vits_requested": cuda_vits,
        "vits_backend": if cuda_vits { "cuda" } else { "cpu" },
        "gpu_encoder_cpu_helper_opt_in": gpu_encoder_cpu_helper_opt_in,
        "gpu_gpt_step_cpu_fallback_disabled": has_gpu_gpt_backend,
        "cuda_vits_cpu_fallback_opt_in": cuda_vits_cpu_fallback_opt_in,
        "cuda_vits_cpu_fallback_disabled": cuda_vits && !cuda_vits_cpu_fallback_opt_in,
        "model_load_ms": load_ms,
        "ort_profiles": ort_profiles,
        "chunk_reports": all_reports
    });
    fs::write(
        output.join(format!("{}-r0-summary.json", backend_name(backend))),
        serde_json::to_vec_pretty(&final_report)?,
    )?;
    Ok(())
}

fn synthesize_once(
    model: &mut NativeTts,
    chunk_tokens: usize,
    audio_file: Option<&Path>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut first_audio_ms = None;
    let mut samples = Vec::new();
    let mut chunks = Vec::new();
    let (synthesis, profile) =
        model.synthesize_streaming_r0(TEXT, &|| false, chunk_tokens, |audio, info| {
            let at_ms = started.elapsed().as_secs_f64() * 1000.0;
            first_audio_ms.get_or_insert(at_ms);
            chunks.push(json!({
                "at_ms": at_ms,
                "sample_count": audio.len(),
                "semantic_tokens": info.semantic_tokens,
                "final": info.final_chunk
            }));
            samples.extend_from_slice(audio);
        })?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err("profile run returned empty or non-finite audio".into());
    }
    if let Some(path) = audio_file {
        let mut bytes = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
        for sample in &samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        fs::write(path, bytes)?;
    }
    let audio_seconds = samples.len() as f64 / 32_000.0;
    Ok(json!({
        "elapsed_ms": elapsed_ms,
        "ttfa_ms": first_audio_ms.unwrap_or(elapsed_ms),
        "audio_seconds": audio_seconds,
        "rtf": elapsed_ms / (audio_seconds * 1000.0),
        "sample_count": samples.len(),
        "synthesis_timings": synthesis,
        "r0_stages": profile,
        "chunks": chunks
    }))
}

fn summarize(rows: &[Value]) -> Value {
    let field = |name: &str| -> Vec<f64> {
        let mut values = rows
            .iter()
            .filter_map(|row| row[name].as_f64())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values
    };
    let stats = |values: Vec<f64>| {
        if values.is_empty() {
            return json!({"p50": null, "p95": null});
        }
        let p50 = percentile(&values, 0.50);
        let p95 = percentile(&values, 0.95);
        json!({"p50": p50, "p95": p95})
    };
    let mut step_token_ms = rows
        .iter()
        .filter_map(|row| row["r0_stages"]["gpt_step_token_ms"].as_array())
        .flat_map(|values| values.iter().filter_map(Value::as_f64))
        .collect::<Vec<_>>();
    step_token_ms.sort_by(f64::total_cmp);
    let phase_values = |name: &str| -> Vec<f64> {
        let mut values = rows
            .iter()
            .filter_map(|row| row["r0_stages"]["phases"].as_array())
            .filter_map(|phases| {
                phases
                    .iter()
                    .find(|phase| phase["phase"] == name)
                    .and_then(|phase| phase["elapsed_ms"].as_f64())
            })
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values
    };
    let mut vits_call_run_ms = rows
        .iter()
        .filter_map(|row| row["r0_stages"]["vits_call_run_ms"].as_array())
        .flat_map(|values| values.iter().filter_map(Value::as_f64))
        .collect::<Vec<_>>();
    vits_call_run_ms.sort_by(f64::total_cmp);
    let mut vits_call_output_ms = rows
        .iter()
        .filter_map(|row| row["r0_stages"]["vits_call_output_ms"].as_array())
        .flat_map(|values| values.iter().filter_map(Value::as_f64))
        .collect::<Vec<_>>();
    vits_call_output_ms.sort_by(f64::total_cmp);
    let mut first_vits_call_ms = rows
        .iter()
        .filter_map(|row| {
            let stages = &row["r0_stages"];
            let run = stages["vits_call_run_ms"].as_array()?.first()?.as_f64()?;
            let output = stages["vits_call_output_ms"]
                .as_array()?
                .first()?
                .as_f64()?;
            let sync = stages["vits_call_device_sync_ms"]
                .as_array()?
                .first()?
                .as_f64()?;
            Some(run + output + sync)
        })
        .collect::<Vec<_>>();
    first_vits_call_ms.sort_by(f64::total_cmp);
    let mut gpt_step_ms = rows
        .iter()
        .filter_map(|row| row["r0_stages"]["phases"].as_array())
        .filter_map(|phases| {
            phases
                .iter()
                .find(|phase| phase["phase"] == "gpt_step_run")
                .and_then(|phase| phase["elapsed_ms"].as_f64())
        })
        .collect::<Vec<_>>();
    gpt_step_ms.sort_by(f64::total_cmp);
    json!({
        "sample_count": rows.len(),
        "ttfa_ms": stats(field("ttfa_ms")),
        "elapsed_ms": stats(field("elapsed_ms")),
        "rtf": stats(field("rtf")),
        "frontend_ms": stats(phase_values("frontend")),
        "gpt_encoder_input_ms": stats(phase_values("gpt_encoder_input")),
        "gpt_encoder_run_ms": stats(phase_values("gpt_encoder_run")),
        "gpt_encoder_output_ms": stats(phase_values("gpt_encoder_output")),
        "gpt_step_input_ms": stats(phase_values("gpt_step_input")),
        "gpt_step_output_ms": stats(phase_values("gpt_step_output")),
        "semantic_sampling_ms": stats(phase_values("semantic_sampling")),
        "host_data_copy_ms": stats(phase_values("host_data_copy")),
        "vits_input_ms": stats(phase_values("vits_input")),
        "vits_run_ms": stats(phase_values("vits_run")),
        "vits_output_ms": stats(phase_values("vits_output")),
        "device_sync_ms": stats(phase_values("device_sync")),
        "audio_callback_ms": stats(phase_values("audio_callback")),
        "vits_call_run_ms": stats(vits_call_run_ms),
        "vits_call_output_copy_ms": stats(vits_call_output_ms),
        "first_vits_call_ms": stats(first_vits_call_ms),
        "gpt_step_ms": stats(gpt_step_ms),
        "gpt_step_token_ms": stats(step_token_ms)
    })
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let index = ((values.len() as f64 * p).ceil() as usize).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

fn summarize_ort_profile(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    #[derive(Default)]
    struct TransferSummary {
        count: usize,
        total_us: f64,
        output_bytes: u64,
        durations_us: Vec<f64>,
        first_timestamp_us: Option<f64>,
        last_end_timestamp_us: Option<f64>,
    }
    let source = fs::read(path)?;
    let trace: Value = serde_json::from_slice(&source)?;
    let events = trace
        .as_array()
        .ok_or("ORT profile must be a Chrome trace event array")?;
    let mut provider_node_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut provider_event_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut seen_nodes = std::collections::BTreeSet::new();
    let mut node_event_counts: std::collections::BTreeMap<(String, String, String), usize> =
        std::collections::BTreeMap::new();
    let mut by_operator: std::collections::BTreeMap<(String, String), Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut runtime_events: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut transfer_events: std::collections::BTreeMap<(String, String), TransferSummary> =
        std::collections::BTreeMap::new();
    for event in events {
        if event["ph"] != "X" {
            continue;
        }
        let Some(duration_us) = event["dur"].as_f64() else {
            continue;
        };
        let duration_ms = duration_us / 1000.0;
        if let Some(operator) = event["args"]["op_name"].as_str() {
            let provider = event["args"]["provider"].as_str().unwrap_or("unknown");
            let event_name = event["name"].as_str().unwrap_or("unknown_node");
            let node_name = event_name
                .strip_suffix("_kernel_time")
                .unwrap_or(event_name);
            let node_key = (
                node_name.to_owned(),
                operator.to_owned(),
                provider.to_owned(),
            );
            *provider_event_counts
                .entry(provider.to_owned())
                .or_default() += 1;
            if seen_nodes.insert(node_key.clone()) {
                *provider_node_counts.entry(provider.to_owned()).or_default() += 1;
            }
            *node_event_counts.entry(node_key).or_default() += 1;
            by_operator
                .entry((operator.to_owned(), provider.to_owned()))
                .or_default()
                .push(duration_ms);
            if operator.contains("Memcpy") || operator.contains("Copy") {
                let summary = transfer_events
                    .entry((operator.to_owned(), provider.to_owned()))
                    .or_default();
                summary.count += 1;
                summary.total_us += duration_us;
                summary.durations_us.push(duration_us);
                let output_bytes = event["args"]["output_size"]
                    .as_u64()
                    .or_else(|| {
                        event["args"]["output_size"]
                            .as_str()
                            .and_then(|value| value.parse().ok())
                    })
                    .unwrap_or_default();
                summary.output_bytes = summary.output_bytes.saturating_add(output_bytes);
                if let Some(timestamp_us) = event["ts"].as_f64() {
                    summary.first_timestamp_us = Some(
                        summary
                            .first_timestamp_us
                            .map_or(timestamp_us, |previous| previous.min(timestamp_us)),
                    );
                    let end_us = timestamp_us + duration_us;
                    summary.last_end_timestamp_us = Some(
                        summary
                            .last_end_timestamp_us
                            .map_or(end_us, |previous| previous.max(end_us)),
                    );
                }
            }
        } else if let Some(name) = event["name"].as_str() {
            // Keep model/session/executor aggregates separate from graph nodes:
            // summing these with node durations would count the same work twice.
            runtime_events
                .entry(name.to_owned())
                .or_default()
                .push(duration_ms);
        }
    }
    let summarize = |events: std::collections::BTreeMap<String, Vec<f64>>| {
        let mut rows = events
            .into_iter()
            .map(|(name, mut durations)| {
                durations.sort_by(f64::total_cmp);
                let total_ms = durations.iter().sum::<f64>();
                json!({
                    "name": name,
                    "count": durations.len(),
                    "total_ms": total_ms,
                    "mean_ms": total_ms / durations.len().max(1) as f64,
                    "p50_ms": percentile(&durations, 0.50),
                    "p95_ms": percentile(&durations, 0.95)
                })
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            b["total_ms"]
                .as_f64()
                .unwrap_or_default()
                .total_cmp(&a["total_ms"].as_f64().unwrap_or_default())
        });
        rows
    };
    let mut operators = by_operator
        .into_iter()
        .map(|((operator, provider), mut durations)| {
            durations.sort_by(f64::total_cmp);
            let total_ms = durations.iter().sum::<f64>();
            json!({
                "operator": operator,
                "provider": provider,
                "count": durations.len(),
                "total_ms": total_ms,
                "mean_ms": total_ms / durations.len().max(1) as f64,
                "p50_ms": percentile(&durations, 0.50),
                "p95_ms": percentile(&durations, 0.95)
            })
        })
        .collect::<Vec<_>>();
    operators.sort_by(|a, b| {
        b["total_ms"]
            .as_f64()
            .unwrap_or_default()
            .total_cmp(&a["total_ms"].as_f64().unwrap_or_default())
    });
    let node_execution_providers = node_event_counts
        .into_iter()
        .map(|((node, operator, provider), event_count)| {
            json!({
                "node": node,
                "operator": operator,
                "provider": provider,
                "event_count": event_count
            })
        })
        .collect::<Vec<_>>();
    let cpu_primary_compute_nodes = node_execution_providers
        .iter()
        .filter(|node| {
            node["provider"] == "CPUExecutionProvider"
                && matches!(
                    node["operator"].as_str().unwrap_or_default(),
                    "Gemm"
                        | "MatMul"
                        | "Conv"
                        | "Attention"
                        | "LayerNormalization"
                        | "RMSNormalization"
                )
        })
        .cloned()
        .collect::<Vec<_>>();
    let cpu_assigned_nodes = node_execution_providers
        .iter()
        .filter(|node| node["provider"] == "CPUExecutionProvider")
        .cloned()
        .collect::<Vec<_>>();
    let transfer_events = transfer_events
        .into_iter()
        .map(|((operator, provider), mut summary)| {
            summary.durations_us.sort_by(f64::total_cmp);
            json!({
                "operator": operator,
                "provider": provider,
                "count": summary.count,
                "total_ms": summary.total_us / 1000.0,
                "output_bytes": summary.output_bytes,
                "duration_us_p50": percentile(&summary.durations_us, 0.50),
                "duration_us_p95": percentile(&summary.durations_us, 0.95),
                "first_timestamp_us": summary.first_timestamp_us,
                "last_end_timestamp_us": summary.last_end_timestamp_us
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "profile_file": path,
        "event_count": events.len(),
        "execution_provider_node_counts": provider_node_counts,
        "execution_provider_event_counts": provider_event_counts,
        "cpu_primary_compute_nodes": cpu_primary_compute_nodes,
        "cpu_assigned_nodes": cpu_assigned_nodes,
        "node_execution_providers": node_execution_providers,
        "operators": operators,
        "transfer_events": transfer_events,
        "runtime_events": summarize(runtime_events)
    }))
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
struct MemoryUsage {
    working_set: u64,
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
    let struct_size = counters.cb;
    let success = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, struct_size) };
    if success == 0 {
        return MemoryUsage::default();
    }
    MemoryUsage {
        working_set: counters.values[1] as u64,
        private_bytes: counters.values[8] as u64,
    }
}

#[cfg(not(windows))]
fn current_memory() -> MemoryUsage {
    MemoryUsage::default()
}

struct SharedMemory {
    working_set_peak: AtomicU64,
    private_bytes_peak: AtomicU64,
}

struct MemorySampler {
    stop: Arc<AtomicBool>,
    shared: Arc<SharedMemory>,
    worker: Option<thread::JoinHandle<()>>,
}

impl MemorySampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(SharedMemory {
            working_set_peak: AtomicU64::new(0),
            private_bytes_peak: AtomicU64::new(0),
        });
        let thread_stop = Arc::clone(&stop);
        let thread_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                record_peak(&thread_shared, current_memory());
                thread::sleep(Duration::from_millis(10));
            }
            record_peak(&thread_shared, current_memory());
        });
        let sampler = Self {
            stop,
            shared,
            worker: Some(worker),
        };
        sampler.reset_peak();
        sampler
    }

    fn reset_peak(&self) {
        let usage = current_memory();
        self.shared
            .working_set_peak
            .store(usage.working_set, Ordering::Relaxed);
        self.shared
            .private_bytes_peak
            .store(usage.private_bytes, Ordering::Relaxed);
    }

    fn peak(&self) -> MemoryUsage {
        MemoryUsage {
            working_set: self.shared.working_set_peak.load(Ordering::Relaxed),
            private_bytes: self.shared.private_bytes_peak.load(Ordering::Relaxed),
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

fn record_peak(shared: &SharedMemory, usage: MemoryUsage) {
    shared
        .working_set_peak
        .fetch_max(usage.working_set, Ordering::Relaxed);
    shared
        .private_bytes_peak
        .fetch_max(usage.private_bytes, Ordering::Relaxed);
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Directml => "directml",
        #[cfg(feature = "r1-cuda-ep")]
        Backend::Cuda => "cuda",
        #[cfg(feature = "r1-tensorrt-ep")]
        Backend::TensorRt => "tensor_rt",
    }
}
