//! Warmed, short Nsight capture for R1.7 cache-copy comparisons.
//!
//! Run this under `nsys profile --capture-range=cudaProfilerApi`. It warms the
//! same model in-process, then stops CUDA collection at the first audio
//! callback while allowing synthesis to finish for output verification.

use alice_native_tts::{Backend, ChunkInfo, Config, NativeTts};
use serde_json::json;
use std::{
    env,
    ffi::{c_void, OsStr},
    fs,
    path::PathBuf,
    time::Instant,
};

const TEXT: &str = "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。";
const DEFAULT_WARMUPS: usize = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: r1_7_copy_trace PACK OUTPUT_DIR [warmups=2]".into());
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let warmups = args.get(3).map_or(Ok(DEFAULT_WARMUPS), |value| {
        value.to_string_lossy().parse::<usize>()
    })?;
    if warmups == 0 {
        return Err("R1.7 short Nsight capture requires at least one warmup".into());
    }
    fs::create_dir_all(&output)?;

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
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_SYNC_PROBE", "0");
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH", "0");
    env::set_var("ALICE_NATIVE_TTS_NSIGHT_NVTX", "1");

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
    for index in 0..warmups {
        model.synthesize_streaming(TEXT, &|| false, |_, _| {})?;
        eprintln!("R1.7 trace warmup {}/{}", index + 1, warmups);
    }

    let profiler = CudaProfiler::load()?;
    let mut audio = Vec::<f32>::new();
    let mut chunks = Vec::<(usize, usize, bool)>::new();
    let mut first_callback_ms = None::<f64>;
    profiler.start()?;
    let request_started = Instant::now();
    let mut capture_active = true;
    let mut capture_stop_error = None::<String>;
    let synthesis = model.synthesize_streaming(TEXT, &|| false, |samples, info: ChunkInfo| {
        let elapsed_ms = request_started.elapsed().as_secs_f64() * 1000.0;
        first_callback_ms.get_or_insert(elapsed_ms);
        chunks.push((info.semantic_tokens, samples.len(), info.final_chunk));
        audio.extend_from_slice(samples);
        if capture_active {
            if let Err(error) = profiler.stop() {
                capture_stop_error = Some(error.to_string());
            }
            capture_active = false;
        }
    });
    if capture_active {
        if let Err(error) = profiler.stop() {
            capture_stop_error.get_or_insert_with(|| error.to_string());
        }
    }
    if let Some(error) = capture_stop_error {
        return Err(std::io::Error::other(error).into());
    }
    let timings = synthesis?;
    let elapsed_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    if audio.is_empty() || audio.iter().any(|sample| !sample.is_finite()) {
        return Err("R1.7 trace synthesis produced empty or non-finite audio".into());
    }
    let wav_path = output.join("trace-audio.wav");
    write_wav(&wav_path, &audio)?;
    let summary = json!({
        "schema": "alice-native-tts-r1.7-first-chunk-trace-v1",
        "text": TEXT,
        "seed": 1234,
        "chunk_tokens": 24,
        "warmups": warmups,
        "profiled_requests": 1,
        "trace_stopped_at": "first_audio_callback",
        "gpt_step_graph": if env::var("ALICE_NATIVE_TTS_GPT_STEP").ok().as_deref() == Some("2") { "gpt_step_v2/per-layer" } else { "abi1/packed" },
        "cuda_graph": false,
        "binding_mode": "clear_rebind",
        "ttfa_ms": first_callback_ms,
        "elapsed_ms": elapsed_ms,
        "audio_samples": audio.len(),
        "timings": timings,
        "chunks": chunks.iter().map(|(tokens, samples, final_chunk)| json!({
            "semantic_tokens": tokens,
            "samples": samples,
            "final": final_chunk
        })).collect::<Vec<_>>(),
        "wav": wav_path
    });
    fs::write(
        output.join("r1.7-copy-trace.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<(), hound::Error> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 32_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()
}

#[cfg(windows)]
struct CudaProfiler {
    module: *mut c_void,
    start_fn: unsafe extern "C" fn() -> i32,
    stop_fn: unsafe extern "C" fn() -> i32,
}

#[cfg(windows)]
impl CudaProfiler {
    fn load() -> Result<Self, Box<dyn std::error::Error>> {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryW(path: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        }
        let dll = env::var_os("ALICE_NATIVE_TTS_CUDART_DLL")
            .unwrap_or_else(|| OsStr::new("cudart64_13.dll").to_os_string());
        let wide: Vec<u16> = OsStr::new(&dll).encode_wide().chain(Some(0)).collect();
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        if module.is_null() {
            return Err(format!("failed to load CUDA runtime DLL: {dll:?}").into());
        }
        let start = unsafe { GetProcAddress(module, b"cudaProfilerStart\0".as_ptr()) };
        let stop = unsafe { GetProcAddress(module, b"cudaProfilerStop\0".as_ptr()) };
        if start.is_null() || stop.is_null() {
            return Err("CUDA runtime does not export cudaProfilerStart/Stop".into());
        }
        Ok(Self {
            module,
            start_fn: unsafe { std::mem::transmute(start) },
            stop_fn: unsafe { std::mem::transmute(stop) },
        })
    }

    fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let status = unsafe { (self.start_fn)() };
        if status != 0 {
            return Err(format!("cudaProfilerStart returned {status}").into());
        }
        Ok(())
    }

    fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let status = unsafe { (self.stop_fn)() };
        if status != 0 {
            return Err(format!("cudaProfilerStop returned {status}").into());
        }
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for CudaProfiler {
    fn drop(&mut self) {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn FreeLibrary(module: *mut c_void) -> i32;
        }
        unsafe { FreeLibrary(self.module) };
    }
}

#[cfg(not(windows))]
struct CudaProfiler;

#[cfg(not(windows))]
impl CudaProfiler {
    fn load() -> Result<Self, Box<dyn std::error::Error>> {
        Err("R1.7 CUDA profiler control is Windows-only".into())
    }
    fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        Err("R1.7 CUDA profiler control is Windows-only".into())
    }
    fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        Err("R1.7 CUDA profiler control is Windows-only".into())
    }
}
