//! R1.5 per-token CUDA GPT Step benchmark. The same local model pack, text,
//! reference audio, seed, frontend, and CUDA GPT/VITS path must be used for A/B.

use alice_native_tts::{Backend, ChunkInfo, Config, NativeTts};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

const TEXT: &str = "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。";
const CHUNK_TOKENS: usize = 24;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=10).contains(&args.len()) {
        return Err("usage: r1_5_profile PACK OUTPUT_DIR clear_rebind|reuse_static|clear_rebind_sync_probe [warmups=30] [repeats=20] [frontend=cpu|cuda] [cache=hit|miss] [text] [cuda_graph=off|on]".into());
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let mode = args[3].as_str();
    let warmups = args.get(4).map_or(Ok(30), |value| value.parse::<usize>())?;
    let repeats = args.get(5).map_or(Ok(20), |value| value.parse::<usize>())?;
    let frontend_backend = args.get(6).map_or("cuda", String::as_str);
    let frontend_cache_mode = args.get(7).map_or("hit", String::as_str);
    let text = args.get(8).map_or(TEXT, String::as_str);
    let cuda_graph_mode = args.get(9).map_or("off", String::as_str);
    if !["cpu", "cuda"].contains(&frontend_backend) {
        return Err("frontend must be cpu or cuda".into());
    }
    if !["hit", "miss"].contains(&frontend_cache_mode) {
        return Err("frontend cache mode must be hit or miss".into());
    }
    if !["off", "on"].contains(&cuda_graph_mode) {
        return Err("CUDA Graph mode must be off or on".into());
    }
    let allow_short_smoke = std::env::var("ALICE_NATIVE_TTS_R1_7_SHORT_SMOKE")
        .ok()
        .as_deref()
        == Some("1");
    if !allow_short_smoke && (warmups < 30 || repeats < 20) {
        return Err("R1.5 requires at least 30 warmups and 20 measured samples".into());
    }
    let (binding_mode, sync_probe) = match mode {
        "clear_rebind" => ("clear_rebind", false),
        "reuse_static" => ("reuse_static", false),
        "clear_rebind_sync_probe" => ("clear_rebind", true),
        _ => return Err("unknown R1.5 mode".into()),
    };
    fs::create_dir_all(&output)?;
    std::env::set_var("ALICE_NATIVE_TTS_GPT_STEP_BINDING_MODE", binding_mode);
    std::env::set_var(
        "ALICE_NATIVE_TTS_GPT_STEP_SYNC_PROBE",
        if sync_probe { "1" } else { "0" },
    );
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND", frontend_backend);
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND", frontend_backend);
    std::env::set_var(
        "ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH",
        if cuda_graph_mode == "on" { "1" } else { "0" },
    );
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_PARALLEL", "0");
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL", "0");
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "1");
    std::env::set_var(
        "ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS",
        if frontend_cache_mode == "miss" {
            "1"
        } else {
            "0"
        },
    );
    std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU", "1");
    std::env::set_var("ALICE_NATIVE_TTS_CUDA_VITS", "1");
    std::env::set_var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU", "1");
    std::env::set_var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU", "1");

    let gpu_before_load = gpu_memory_snapshot();
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
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let memory = MemorySampler::start();
    let gpu_after_load = gpu_memory_snapshot();
    let after_load = current_memory();

    for index in 0..warmups {
        let _ = run_once(&mut model, false, text)?;
        if (index + 1) % 10 == 0 {
            eprintln!("R1.5 warmup mode={mode} completed={}", index + 1);
        }
    }

    memory.reset_peak();
    let mut samples = Vec::with_capacity(repeats);
    let mut first_audio = None;
    for index in 0..repeats {
        let (row, audio) = run_once(&mut model, true, text)?;
        if index == 0 {
            let mut bytes = Vec::with_capacity(audio.len() * std::mem::size_of::<f32>());
            for sample in &audio {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            fs::write(output.join("first-audio.f32le"), bytes)?;
            write_wav(output.join("first-audio.wav"), &audio)?;
            first_audio = Some(audio_fingerprint(&audio));
        }
        samples.push(row);
        eprintln!(
            "R1.5 sample mode={mode} completed={}/{}",
            index + 1,
            repeats
        );
    }

    let after_run = current_memory();
    let peak = memory.peak();
    let gpu_after = gpu_memory_snapshot();
    let summary = summarize(&samples);
    let report = json!({
        "schema": "alice-native-tts-r1.5-gpt-execution-v1",
        "mode": mode,
        "binding_mode": binding_mode,
        "explicit_sync_probe": sync_probe,
        "cuda_graph_mode": cuda_graph_mode,
        "gpt_step_graph": if std::env::var("ALICE_NATIVE_TTS_GPT_STEP")
            .ok()
            .as_deref()
            == Some("2")
        {
            "gpt_step_v2/per-layer"
        } else {
            "abi1/packed"
        },
        "backend": format!("{} Frontend + CUDA GPT + CUDA VITS; audited CPU shape helpers only", frontend_backend.to_ascii_uppercase()),
        "frontend_backend": frontend_backend,
        "frontend_cache_mode": frontend_cache_mode,
        "text": text,
        "chunk_tokens": CHUNK_TOKENS,
        "seed": 1234,
        "warmups": warmups,
        "repeats": repeats,
        "ort_profiling_enabled": false,
        "model_load_ms": load_ms,
        "memory": {
            "after_load": after_load,
            "after_run": after_run,
            "during_measured_peak": peak,
            "gpu_before_load": gpu_before_load,
            "gpu_after_load": gpu_after_load,
            "gpu_after_run": gpu_after
        },
        "summary": summary,
        "first_audio_fingerprint_fnv1a64": first_audio,
        "samples": samples
    });
    fs::write(
        output.join("r1.5-gpt-execution.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": mode,
            "report": output.join("r1.5-gpt-execution.json"),
            "model_load_ms": load_ms,
            "summary": summary
        }))?
    );
    Ok(())
}

fn run_once(
    model: &mut NativeTts,
    capture: bool,
    text: &str,
) -> Result<(Value, Vec<f32>), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut ttfa_ms = None;
    let mut audio = Vec::new();
    let mut callbacks = Vec::new();
    let (timings, profile) = model.synthesize_streaming_r0(
        text,
        &|| false,
        CHUNK_TOKENS,
        |samples, info: ChunkInfo| {
            let callback_ms = started.elapsed().as_secs_f64() * 1000.0;
            ttfa_ms.get_or_insert(callback_ms);
            callbacks.push(json!({
                "at_ms": callback_ms,
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
    if capture && (audio.is_empty() || audio.iter().any(|sample| !sample.is_finite())) {
        return Err("R1.5 run produced empty or non-finite audio".into());
    }
    let audio_seconds = audio.len() as f64 / 32_000.0;
    let row = json!({
        "ttfa_ms": ttfa_ms.unwrap_or(elapsed_ms),
        "elapsed_ms": elapsed_ms,
        "rtf": if audio_seconds > 0.0 { elapsed_ms / (audio_seconds * 1000.0) } else { 0.0 },
        "audio_samples": audio.len(),
        "audio_fingerprint_fnv1a64": audio_fingerprint(&audio),
        "callbacks": callbacks,
        "timings": timings,
        "profile": profile
    });
    Ok((row, audio))
}

fn summarize(samples: &[Value]) -> Value {
    let values = |path: &[&str]| -> Vec<f64> {
        samples
            .iter()
            .filter_map(|sample| {
                let mut value = sample;
                for key in path {
                    value = value.get(*key)?;
                }
                value.as_f64()
            })
            .collect()
    };
    let metric = |path: &[&str]| distribution(values(path));
    let phase_metric = |phase_name: &'static str| {
        distribution(
            samples
                .iter()
                .filter_map(|sample| {
                    sample["profile"]["phases"]
                        .as_array()?
                        .iter()
                        .find(|phase| phase["phase"] == phase_name)
                        .and_then(|phase| phase["elapsed_ms"].as_f64())
                })
                .collect(),
        )
    };
    let token_phase_metric = |event_name: &'static str| {
        distribution(
            samples
                .iter()
                .filter_map(|sample| sample["profile"]["timeline"].as_array())
                .flatten()
                .filter(|event| event["event"] == event_name)
                .filter_map(|event| Some(event["end_ms"].as_f64()? - event["start_ms"].as_f64()?))
                .collect(),
        )
    };
    let token_steps = samples
        .iter()
        .filter_map(|sample| sample["profile"]["gpt_step_token_ms"].as_array())
        .flatten()
        .filter_map(Value::as_f64)
        .collect::<Vec<_>>();
    let tokens = samples
        .iter()
        .filter_map(|sample| sample["profile"]["semantic_tokens"].as_u64())
        .collect::<Vec<_>>();
    let hashes = samples
        .iter()
        .filter_map(|sample| sample["profile"]["semantic_token_hash"].as_u64())
        .collect::<Vec<_>>();
    json!({
        "ttfa_ms": metric(&["ttfa_ms"]),
        "elapsed_ms": metric(&["elapsed_ms"]),
        "rtf": metric(&["rtf"]),
        "gpt_step_total_phase_ms": phase_metric("gpt_step_run"),
        "gpt_step_binding_update_ms_per_token": token_phase_metric("gpt_step_binding_update"),
        "gpt_step_ort_run_ms_per_token": token_phase_metric("gpt_step_ort_run"),
        "gpt_step_sync_wait_ms_per_token": token_phase_metric("gpt_step_sync_wait"),
        "gpt_step_output_check_ms_per_token": token_phase_metric("gpt_step_output_check"),
        "gpt_step_request_bindings_ms": phase_metric("gpt_step_request_bindings"),
        "host_sampling_ms_per_token": token_phase_metric("semantic_sampling"),
        "host_sampling_total_ms": phase_metric("semantic_sampling"),
        "per_token_gpt_step_wall_ms_pooled": distribution(token_steps),
        "semantic_token_counts": tokens,
        "semantic_token_hashes": hashes,
        "audio_fingerprints": samples.iter().map(|sample| sample["audio_fingerprint_fnv1a64"].clone()).collect::<Vec<_>>(),
        "audio_sample_counts": samples.iter().map(|sample| sample["audio_samples"].clone()).collect::<Vec<_>>()
    })
}

fn distribution(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return json!({"count": 0});
    }
    values.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        values[((values.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)]
    };
    json!({
        "count": values.len(),
        "min": values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": values[values.len() - 1],
        "mean": values.iter().sum::<f64>() / values.len() as f64
    })
}

fn audio_fingerprint(audio: &[f32]) -> u64 {
    audio.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, sample| {
        (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn write_wav(path: PathBuf, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
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

#[derive(Clone, Copy, Default, serde::Serialize)]
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
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => json!({
            "success": output.status.success(),
            "stdout": String::from_utf8_lossy(&output.stdout).trim(),
            "stderr": String::from_utf8_lossy(&output.stderr).trim()
        }),
        Err(error) => json!({"success": false, "error": error.to_string()}),
    }
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
