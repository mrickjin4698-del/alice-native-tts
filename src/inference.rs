#[cfg(feature = "r2-trt-native")]
use crate::native_trt::NativeTrtGpt;
use crate::{
    audio, check,
    frontend::Frontend,
    invalid,
    profiling::{ProfileCollector, ProfilePhase},
    streaming::{
        find_bounded_preview_join, first_voice_sample, preview_fallback_reason,
        stitch_bounded_preview, BoundedPreviewJoin, SolaAlignment, StreamDecoder, CHUNK_TOKENS,
    },
    validate_pack, validate_streaming_pack, Backend, Config, Pack, Result,
};
#[cfg(feature = "r1-cuda-ep")]
use ort::value::DynTensorValueType;
#[cfg(feature = "r1-cuda-ep")]
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, RunOptions},
};
use ort::{
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor},
};
#[cfg(feature = "r0-profiling")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

const R1_9_MAX_VOICE_ONSET_MS: usize = 160;
const R1_9_MIN_COMMIT_MS: usize = 64;
const R1_9_MAX_COMMIT_MS: usize = 240;

#[cfg(feature = "r2-0-first-audible")]
struct LoadTrace {
    name: &'static str,
    started: Option<Instant>,
}

#[cfg(feature = "r2-0-first-audible")]
impl LoadTrace {
    fn begin(name: &'static str) -> Self {
        let enabled = std::env::var("ALICE_NATIVE_TTS_LOAD_TRACE").ok().as_deref() == Some("1");
        if enabled {
            eprintln!("[alice-native-tts load] BEGIN {name}");
        }
        Self {
            name,
            started: enabled.then(Instant::now),
        }
    }
}

#[cfg(feature = "r2-0-first-audible")]
impl Drop for LoadTrace {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            eprintln!(
                "[alice-native-tts load] END {} elapsed_ms={:.1}",
                self.name,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}

#[cfg(not(feature = "r2-0-first-audible"))]
struct LoadTrace;

#[cfg(not(feature = "r2-0-first-audible"))]
impl LoadTrace {
    #[inline(always)]
    fn begin(_name: &'static str) -> Self {
        Self
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SynthesisTimings {
    pub frontend_ms: f64,
    pub gpt_ms: f64,
    pub vocoder_ms: f64,
    pub semantic_tokens: usize,
    pub first_audio_tokens: usize,
    pub audio_chunks: usize,
}

#[derive(Default, serde::Serialize)]
struct PreviewRefillDiagnostics {
    preview_tokens: Option<usize>,
    refill_target_tokens: Option<usize>,
    preview_callback_tokens: Option<usize>,
    preview_fallback_reason: Option<&'static str>,
    preview_decoded_samples: Option<usize>,
    preview_committed_samples: Option<usize>,
    preview_uncommitted_samples: Option<usize>,
    preview_voice_onset_samples: Option<usize>,
    bounded_join: Option<BoundedPreviewJoin>,
    refill_rejected_reason: Option<&'static str>,
    full_prefix_refill: bool,
    semantic_tokens: usize,
    semantic_token_hash: u64,
    sola_alignments: Vec<SolaAlignment>,
}

#[derive(Clone, Debug)]
struct BoundedPreviewState {
    decoded_pcm: Vec<f32>,
    committed_samples: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ChunkInfo {
    pub semantic_tokens: usize,
    pub final_chunk: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GptStepCacheLayout {
    Abi1Packed,
    #[allow(dead_code)]
    V2PerLayer,
}

enum GptKvCache {
    Abi1Packed {
        key: DynValue,
        value: DynValue,
    },
    #[allow(dead_code)]
    CudaAbi1PingPong {
        key: DynValue,
        value: DynValue,
        next_binding: usize,
    },
    #[allow(dead_code)]
    CudaV2Initial {
        key: DynValue,
        value: DynValue,
    },
    #[allow(dead_code)]
    CudaV2PerLayer {
        key: Vec<DynValue>,
        value: Vec<DynValue>,
        next_binding: usize,
    },
    V2PerLayer {
        key: Vec<DynValue>,
        value: Vec<DynValue>,
    },
    #[cfg(feature = "r2-trt-native")]
    TensorRtNative,
}

const GPT_STEP_CACHE_LAYERS: usize = 24;

pub(crate) fn tf(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Tensor<f32>> {
    Ok(Tensor::from_array((shape.into(), data))?)
}
pub(crate) fn ti(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Result<Tensor<i64>> {
    Ok(Tensor::from_array((shape.into(), data))?)
}
pub(crate) fn open(dir: &Path, name: &str) -> Result<Session> {
    open_on(dir, name, Backend::Cpu)
}
pub(crate) fn open_on(dir: &Path, name: &str, backend: Backend) -> Result<Session> {
    open_path(&dir.join(format!("{name}.onnx")), name, backend)
}

// Experimental bounded cache: one extra session dedicated to the ordinary
// 24-token first chunk. Models, tensor shapes and audio output stay unchanged.
fn open_first_vits(dir: &Path, streaming: bool, backend: Backend) -> Result<Option<Session>> {
    #[cfg(feature = "r1-cuda-ep")]
    if streaming
        && backend == Backend::Cuda
        && std::env::var("ALICE_NATIVE_TTS_FIRST_VITS_SESSION")
            .ok()
            .as_deref()
            == Some("1")
    {
        return Ok(Some(open_on(dir, "sovits_stream", backend)?));
    }
    let _ = (dir, streaming, backend);
    Ok(None)
}

fn open_path(path: &Path, name: &str, backend: Backend) -> Result<Session> {
    let _load_trace = match name {
        "bert" => Some(LoadTrace::begin("session:bert")),
        "g2pw" => Some(LoadTrace::begin("session:g2pw")),
        "ssl" => Some(LoadTrace::begin("session:ssl")),
        "vq_encoder" => Some(LoadTrace::begin("session:vq_encoder")),
        "spectrogram" => Some(LoadTrace::begin("session:spectrogram")),
        "sv_embedding" => Some(LoadTrace::begin("session:sv_embedding")),
        "gpt_encoder" => Some(LoadTrace::begin("session:gpt_encoder")),
        "gpt_step" => Some(LoadTrace::begin("session:gpt_step")),
        "gpt_step_init_v2" => Some(LoadTrace::begin("session:gpt_step_init_v2")),
        "sovits" => Some(LoadTrace::begin("session:sovits")),
        "sovits_stream" => Some(LoadTrace::begin("session:sovits_stream")),
        _ => None,
    };
    #[cfg(feature = "r2-0-first-audible")]
    if std::env::var("ALICE_NATIVE_TTS_LOAD_TRACE").ok().as_deref() == Some("1") {
        let bytes = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        eprintln!("[alice-native-tts load] SESSION name={name} backend={backend:?} bytes={bytes}");
    }
    #[allow(unused_mut)]
    let mut strict_gpu_placement = false;
    #[allow(unused_variables)]
    let allow_cpu_nodes_for_gpu_diagnostics =
        std::env::var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            == Some("1");
    #[allow(unused_variables)]
    let allow_cpu_nodes_for_cuda_vits_diagnostics =
        std::env::var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            == Some("1");
    #[allow(unused_variables)]
    let allow_cpu_nodes_for_cuda_frontend_diagnostics =
        std::env::var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU")
            .ok()
            .as_deref()
            == Some("1");
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda {
        strict_gpu_placement = match name {
            // The ABI-1 encoder graph has a small number of audited CPU helper nodes.
            // Keep that diagnostic escape hatch isolated from the autoregressive step.
            "gpt_encoder" => !allow_cpu_nodes_for_gpu_diagnostics,
            "gpt_step" | "gpt_step_init_v2" => true,
            "sovits" | "sovits_stream"
                if std::env::var("ALICE_NATIVE_TTS_CUDA_VITS").ok().as_deref() == Some("1") =>
            {
                !allow_cpu_nodes_for_cuda_vits_diagnostics
            }
            "bert" | "g2pw" => !allow_cpu_nodes_for_cuda_frontend_diagnostics,
            _ => false,
        };
    }
    #[cfg(feature = "r1-tensorrt-ep")]
    if backend == Backend::TensorRt {
        strict_gpu_placement = match name {
            "gpt_encoder" => !allow_cpu_nodes_for_gpu_diagnostics,
            "gpt_step" | "gpt_step_init_v2" => true,
            _ => false,
        };
    }
    let intra_threads = if name == "gpt_step" {
        match std::env::var_os("ALICE_NATIVE_TTS_GPT_STEP_THREADS") {
            Some(value) => {
                let threads = value
                    .to_string_lossy()
                    .parse::<usize>()
                    .map_err(|_| invalid("ALICE_NATIVE_TTS_GPT_STEP_THREADS must be an integer"))?;
                if !(1..=64).contains(&threads) {
                    return Err(invalid(
                        "ALICE_NATIVE_TTS_GPT_STEP_THREADS must be between 1 and 64",
                    ));
                }
                threads
            }
            None => std::thread::available_parallelism().map_or(4, |threads| threads.get().min(8)),
        }
    } else {
        4
    };
    let _trace = LoadTrace::begin("ort_session_builder_create");
    let builder = Session::builder()?;
    drop(_trace);
    let _trace = LoadTrace::begin("ort_session_set_intra_threads");
    let builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|e| invalid(e.to_string()))?;
    drop(_trace);
    let _trace = LoadTrace::begin("ort_session_set_inter_threads");
    let mut builder = builder
        .with_inter_threads(1)
        .map_err(|e| invalid(e.to_string()))?;
    drop(_trace);
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda
        && matches!(name, "gpt_step" | "gpt_step_init_v2")
        && std::env::var("ALICE_NATIVE_TTS_ORT_VERBOSE")
            .ok()
            .as_deref()
            == Some("1")
    {
        builder = builder
            .with_log_level(ort::logging::LogLevel::Verbose)
            .map_err(|e| invalid(e.to_string()))?
            .with_log_verbosity(2)
            .map_err(|e| invalid(e.to_string()))?
            .with_logger(std::sync::Arc::new(
                |level, category, id, location, message| {
                    eprintln!("[ORT {level:?}] {category} [{id}] {location}: {message}");
                },
            ))
            .map_err(|e| invalid(e.to_string()))?;
    }
    if backend == Backend::Directml {
        #[cfg(windows)]
        {
            // Required by DirectML. Explicit selection must report a failure,
            // rather than silently presenting a CPU session as GPU inference.
            builder = builder
                .with_memory_pattern(false)
                .map_err(|e| invalid(e.to_string()))?
                .with_parallel_execution(false)
                .map_err(|e| invalid(e.to_string()))?
                .with_execution_providers([ort::ep::DirectML::default()
                    .with_performance_preference(
                        ort::ep::directml::PerformancePreference::HighPerformance,
                    )
                    .build()
                    .error_on_failure()])
                .map_err(|e| invalid(e.to_string()))?;
        }
        #[cfg(not(windows))]
        return Err(invalid("DirectML TTS requires Windows; select CPU"));
    }
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda {
        let enable_gpt_step_cuda_graph = name == "gpt_step"
            && std::env::var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH")
                .ok()
                .as_deref()
                == Some("1");
        let target_tuning = (name == "target_features")
            .then(|| std::env::var("ALICE_NATIVE_TTS_DIFFUSION_CUDA_TARGET").unwrap_or_default());
        let tf32 = matches!(target_tuning.as_deref(), Some("tf32" | "tf32_nc1d"));
        let mut provider = ort::ep::CUDA::default().with_tf32(tf32);
        if matches!(target_tuning.as_deref(), Some("nc1d" | "tf32_nc1d")) {
            provider = provider.with_conv1d_pad_to_nc1d(true);
        }
        if enable_gpt_step_cuda_graph {
            provider = provider.with_cuda_graph(true);
        }
        let cuda_provider = provider.build().error_on_failure();
        builder = builder
            .with_execution_providers([cuda_provider])
            .map_err(|e| invalid(e.to_string()))?;
    }
    #[cfg(feature = "r1-tensorrt-ep")]
    if backend == Backend::TensorRt {
        let cache_dir = tensorrt_cache_dir()?;
        builder = builder
            .with_execution_providers([
                ort::ep::TensorRT::default()
                    // Keep FP32 until numerical and audio-quality gates pass.
                    .with_fp16(false)
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache_dir.to_string_lossy())
                    .with_engine_cache_prefix(name)
                    .with_timing_cache(true)
                    .with_timing_cache_path(
                        cache_dir
                            .join(format!("{name}.timing.cache"))
                            .to_string_lossy(),
                    )
                    .build()
                    .error_on_failure(),
                ort::ep::CUDA::default()
                    .with_tf32(false)
                    .build()
                    .error_on_failure(),
            ])
            .map_err(|e| invalid(e.to_string()))?;
    }
    if strict_gpu_placement {
        builder = builder
            .with_disable_cpu_fallback()
            .map_err(|e| invalid(e.to_string()))?;
    }
    #[cfg(feature = "r0-profiling")]
    if let Some(path) = profile_prefix(name)? {
        builder = builder
            .with_profiling(path)
            .map_err(|e| invalid(e.to_string()))?;
    }
    let _trace = LoadTrace::begin("ort_commit_from_file");
    let result = builder.commit_from_file(path).map_err(|error| {
        if strict_gpu_placement {
            invalid(format!(
                "strict GPU placement rejected CPU-assigned nodes in {name}: {error}"
            ))
        } else {
            invalid(error.to_string())
        }
    });
    drop(_trace);
    result
}

fn non_gpt_backend(backend: Backend) -> Backend {
    #[allow(unused_mut)]
    let mut backend = backend;
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda {
        backend = Backend::Cpu;
    }
    #[cfg(feature = "r1-tensorrt-ep")]
    if backend == Backend::TensorRt {
        backend = Backend::Cpu;
    }
    #[cfg(feature = "r2-trt-native")]
    if backend == Backend::TensorRtNative {
        backend = Backend::Cpu;
    }
    backend
}

fn frontend_execution_backend(backend: Backend) -> Result<Backend> {
    let fallback = non_gpt_backend(backend);
    if std::env::var("ALICE_NATIVE_TTS_CUDA_FRONTEND")
        .ok()
        .as_deref()
        != Some("1")
    {
        return Ok(fallback);
    }

    #[cfg(feature = "r1-cuda-ep")]
    {
        #[cfg(feature = "r2-trt-native")]
        let native_trt = backend == Backend::TensorRtNative;
        #[cfg(not(feature = "r2-trt-native"))]
        let native_trt = false;
        if is_cuda_backend(backend) || native_trt {
            return Ok(Backend::Cuda);
        }
        return Ok(fallback);
    }

    #[cfg(not(feature = "r1-cuda-ep"))]
    {
        #[cfg(feature = "r2-trt-native")]
        if backend == Backend::TensorRtNative {
            return Err(invalid(
                "ALICE_NATIVE_TTS_CUDA_FRONTEND=1 requires the r1-cuda-ep feature",
            ));
        }
        Ok(fallback)
    }
}

fn gpt_backend(backend: Backend) -> Backend {
    match backend {
        #[cfg(feature = "r1-cuda-ep")]
        Backend::Cuda => Backend::Cuda,
        #[cfg(feature = "r1-tensorrt-ep")]
        Backend::TensorRt => Backend::TensorRt,
        #[cfg(feature = "r2-trt-native")]
        Backend::TensorRtNative => Backend::Cpu,
        _ => Backend::Cpu,
    }
}

fn is_cuda_backend(backend: Backend) -> bool {
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda {
        return true;
    }
    #[cfg(feature = "r1-tensorrt-ep")]
    if backend == Backend::TensorRt {
        return true;
    }
    let _ = backend;
    false
}

fn cuda_vits_requested(backend: Backend) -> bool {
    #[cfg(feature = "r1-cuda-ep")]
    if backend == Backend::Cuda {
        return std::env::var("ALICE_NATIVE_TTS_CUDA_VITS").ok().as_deref() == Some("1");
    }
    #[cfg(all(feature = "r2-trt-native", feature = "r1-cuda-ep"))]
    if backend == Backend::TensorRtNative {
        return std::env::var("ALICE_NATIVE_TTS_CUDA_VITS").ok().as_deref() == Some("1");
    }
    let _ = backend;
    false
}

#[cfg(feature = "r1-cuda-ep")]
fn ensure_cuda_value(value: &DynValue, name: &str) -> Result<()> {
    let tensor = value.downcast_ref::<DynTensorValueType>()?;
    if tensor.memory_info().allocation_device() != AllocationDevice::CUDA
        || tensor.memory_info().device_id() != 0
    {
        return Err(invalid(format!(
            "{name} is not stored on CUDA device 0: {:?}",
            tensor.memory_info()
        )));
    }
    Ok(())
}

#[cfg(feature = "r1-cuda-ep")]
fn tensor_data_ptr(value: &DynValue, name: &str) -> Result<usize> {
    let tensor = value.downcast_ref::<DynTensorValueType>()?;
    if !matches!(value.dtype(), ort::value::ValueType::Tensor { .. }) {
        return Err(invalid(format!("{name} is not an ONNX tensor")));
    }
    Ok(tensor.data_ptr() as usize)
}

#[cfg(feature = "r1-tensorrt-ep")]
fn tensorrt_cache_dir() -> Result<PathBuf> {
    let path = std::env::var_os("ALICE_NATIVE_TTS_TRT_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .map(|root| root.join("AliceNativeTts").join("OrtTensorRtCache"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("alice-native-tts-ort-tensorrt-cache"));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn open_step(dir: &Path, backend: Backend) -> Result<(Session, GptStepCacheLayout)> {
    let backend = gpt_backend(backend);
    #[cfg(feature = "r1-gpt-step-v2")]
    if std::env::var("ALICE_NATIVE_TTS_GPT_STEP").ok().as_deref() == Some("2") {
        let graph = dir.join("gpt_step_v2.onnx");
        if !graph.is_file() {
            return Err(invalid(
                "gpt_step_v2 was selected but gpt_step_v2.onnx is missing",
            ));
        }
        return Ok((
            open_path(&graph, "gpt_step", backend)?,
            GptStepCacheLayout::V2PerLayer,
        ));
    }
    Ok((
        open_on(dir, "gpt_step", backend)?,
        GptStepCacheLayout::Abi1Packed,
    ))
}

#[cfg(feature = "r0-profiling")]
fn profile_prefix(name: &str) -> Result<Option<PathBuf>> {
    static NEXT_PROFILE: AtomicU64 = AtomicU64::new(0);
    if !matches!(
        name,
        "bert"
            | "g2pw"
            | "gpt_encoder"
            | "gpt_step"
            | "gpt_step_init_v2"
            | "sovits"
            | "sovits_stream"
    ) {
        return Ok(None);
    }
    let Some(directory) = std::env::var_os("ALICE_NATIVE_TTS_ORT_PROFILE_DIR") else {
        return Ok(None);
    };
    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory)?;
    let serial = NEXT_PROFILE.fetch_add(1, Ordering::Relaxed);
    Ok(Some(directory.join(format!("{name}-{serial}"))))
}
pub(crate) fn take(out: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    out.remove(name)
        .ok_or_else(|| invalid(format!("incompatible ONNX graph: missing {name}")))
}

fn validate_gpt_lengths(
    x_len: &DynValue,
    y_len: &DynValue,
    text_len: usize,
    prompt_tokens: usize,
) -> Result<()> {
    if x_len.try_extract_tensor::<i64>()?.1 != [text_len as i64]
        || y_len.try_extract_tensor::<i64>()?.1 != [prompt_tokens as i64]
    {
        return Err(invalid("GPT encoder has fixed fixture lengths"));
    }
    Ok(())
}

#[cfg(feature = "r1-cuda-ep")]
struct GpuStepBinding {
    binding: IoBinding,
    run_options: Option<RunOptions>,
}

#[cfg(all(feature = "r1-cuda-ep", feature = "r0-profiling"))]
#[derive(Default)]
struct GptStepCudaSpans {
    binding_update: Option<(Instant, Instant)>,
    ort_run: Option<(Instant, Instant)>,
    sync_wait: Option<(Instant, Instant)>,
    output_check: Option<(Instant, Instant)>,
}

#[cfg(all(feature = "r1-cuda-ep", not(feature = "r0-profiling")))]
#[derive(Default)]
struct GptStepCudaSpans;

#[cfg(feature = "r1-cuda-ep")]
impl GpuStepBinding {
    fn new(
        session: &Session,
        graph_id: Option<usize>,
        cache_layout: GptStepCacheLayout,
    ) -> Result<Self> {
        let cpu_output = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )?;
        let cuda_output = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let mut binding = session.create_binding()?;
        binding.bind_output_to_device("topk_values", &cpu_output)?;
        binding.bind_output_to_device("topk_indices", &cpu_output)?;
        match cache_layout {
            GptStepCacheLayout::Abi1Packed => {
                binding.bind_output_to_device("k_cache_new", &cuda_output)?;
                binding.bind_output_to_device("v_cache_new", &cuda_output)?;
            }
            GptStepCacheLayout::V2PerLayer => {
                for layer in 0..GPT_STEP_CACHE_LAYERS {
                    binding.bind_output_to_device(
                        &format!("k_cache_new_layer_{layer:02}"),
                        &cuda_output,
                    )?;
                    binding.bind_output_to_device(
                        &format!("v_cache_new_layer_{layer:02}"),
                        &cuda_output,
                    )?;
                }
            }
        }
        let run_options = if let Some(graph_id) = graph_id {
            let mut options = RunOptions::new()?;
            options.add_config_entry("gpu_graph_id", graph_id.to_string())?;
            Some(options)
        } else {
            None
        };
        Ok(Self {
            binding,
            run_options,
        })
    }

    fn bind_request_inputs(&mut self, x_len: &DynValue, y_len: &DynValue) -> Result<()> {
        self.binding.clear_inputs();
        self.binding.bind_input("x_len", x_len)?;
        self.binding.bind_input("y_len", y_len)?;
        Ok(())
    }
}

#[cfg(feature = "r1-cuda-ep")]
fn run_gpt_encoder_cuda(
    session: &mut Session,
    phoneme_ids: Tensor<i64>,
    prompts: &Tensor<i64>,
    bert_feature: Tensor<f32>,
) -> Result<(DynValue, DynValue, DynValue, DynValue, DynValue, DynValue)> {
    let cuda_output = MemoryInfo::new(
        AllocationDevice::CUDA,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let cpu_output = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?;
    let mut binding = session.create_binding()?;
    binding.bind_input("phoneme_ids", &phoneme_ids)?;
    binding.bind_input("prompts", prompts)?;
    binding.bind_input("bert_feature", &bert_feature)?;
    binding.bind_output_to_device("topk_values", &cpu_output)?;
    binding.bind_output_to_device("topk_indices", &cpu_output)?;
    binding.bind_output_to_device("x_len", &cpu_output)?;
    binding.bind_output_to_device("y_len", &cpu_output)?;
    binding.bind_output_to_device("k_cache", &cuda_output)?;
    binding.bind_output_to_device("v_cache", &cuda_output)?;
    let mut outputs = session.run_binding(&binding)?;
    let topk_values = take(&mut outputs, "topk_values")?;
    let topk_indices = take(&mut outputs, "topk_indices")?;
    let x_len = take(&mut outputs, "x_len")?;
    let y_len = take(&mut outputs, "y_len")?;
    let k_cache = take(&mut outputs, "k_cache")?;
    let v_cache = take(&mut outputs, "v_cache")?;
    ensure_cuda_value(&k_cache, "gpt_encoder.k_cache")?;
    ensure_cuda_value(&v_cache, "gpt_encoder.v_cache")?;
    Ok((topk_values, topk_indices, x_len, y_len, k_cache, v_cache))
}

#[cfg(feature = "r1-cuda-ep")]
fn run_gpt_step_cuda(
    session: &mut Session,
    binding: &mut GpuStepBinding,
    sample: Tensor<i64>,
    idx: Tensor<i64>,
    x_len: &DynValue,
    y_len: &DynValue,
    key: &DynValue,
    value: &DynValue,
    reuse_static_inputs: bool,
    sync_probe: bool,
) -> Result<(DynValue, DynValue, DynValue, DynValue, GptStepCudaSpans)> {
    #[cfg(windows)]
    let _token_nvtx = crate::nvtx::Range::push(b"gpt_step_token\0");

    #[cfg(feature = "r0-profiling")]
    let mut spans = GptStepCudaSpans::default();
    #[cfg(not(feature = "r0-profiling"))]
    let spans = GptStepCudaSpans;

    #[cfg(feature = "r0-profiling")]
    let binding_started = Instant::now();
    ensure_cuda_value(key, "gpt_step.k_cache input")?;
    ensure_cuda_value(value, "gpt_step.v_cache input")?;

    let io = &mut binding.binding;
    // Keep this binding's device-side output buffers alive and reusable.
    // Only the small CPU inputs change from token to token.
    if !reuse_static_inputs {
        io.clear_inputs();
        io.bind_input("x_len", x_len)?;
        io.bind_input("y_len", y_len)?;
    }
    io.bind_input("samples", &sample)?;
    io.bind_input("idx", &idx)?;
    io.bind_input("k_cache", key)?;
    io.bind_input("v_cache", value)?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.binding_update = Some((binding_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let run_started = Instant::now();
    let outputs_result = {
        #[cfg(windows)]
        let _ort_nvtx = crate::nvtx::Range::push(b"gpt_step_ort_run\0");
        if let Some(run_options) = &binding.run_options {
            session.run_binding_with_options(io, run_options)
        } else {
            session.run_binding(io)
        }
    };
    let mut outputs = outputs_result?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.ort_run = Some((run_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let sync_started = sync_probe.then(Instant::now);
    if sync_probe {
        // Diagnostic-only fence: the normal path relies on the CPU-bound
        // top-k output contract and does not add an explicit stream wait.
        io.synchronize_outputs()?;
    }
    #[cfg(feature = "r0-profiling")]
    if let Some(sync_started) = sync_started {
        spans.sync_wait = Some((sync_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let output_started = Instant::now();
    let topk_values = take(&mut outputs, "topk_values")?;
    let topk_indices = take(&mut outputs, "topk_indices")?;
    let key_cache = take(&mut outputs, "k_cache_new")?;
    let value_cache = take(&mut outputs, "v_cache_new")?;
    ensure_cuda_value(&key_cache, "gpt_step.k_cache output")?;
    ensure_cuda_value(&value_cache, "gpt_step.v_cache output")?;
    if tensor_data_ptr(key, "gpt_step.k_cache input")?
        == tensor_data_ptr(&key_cache, "gpt_step.k_cache output")?
        || tensor_data_ptr(value, "gpt_step.v_cache input")?
            == tensor_data_ptr(&value_cache, "gpt_step.v_cache output")?
    {
        return Err(invalid(
            "GPT Step ping-pong cache buffers alias their inputs",
        ));
    }
    let topk_values_tensor = topk_values.downcast_ref::<DynTensorValueType>()?;
    let topk_indices_tensor = topk_indices.downcast_ref::<DynTensorValueType>()?;
    let topk_values_info = topk_values_tensor.memory_info();
    let topk_indices_info = topk_indices_tensor.memory_info();
    if topk_values_info.allocation_device() != AllocationDevice::CPU
        || topk_indices_info.allocation_device() != AllocationDevice::CPU
    {
        return Err(invalid(
            "GPT Step sampling outputs must remain host-accessible for the existing sampler",
        ));
    }
    #[cfg(feature = "r0-profiling")]
    {
        spans.output_check = Some((output_started, Instant::now()));
    }
    Ok((topk_values, topk_indices, key_cache, value_cache, spans))
}

#[cfg(feature = "r1-cuda-ep")]
fn take_cuda_cache_layers(
    outputs: &mut ort::session::SessionOutputs<'_>,
    prefix: &str,
) -> Result<Vec<DynValue>> {
    (0..GPT_STEP_CACHE_LAYERS)
        .map(|layer| {
            let value = take(outputs, &format!("{prefix}_{layer:02}"))?;
            ensure_cuda_value(&value, &format!("gpt_step_v2.{prefix}_{layer:02}"))?;
            Ok(value)
        })
        .collect()
}

#[cfg(feature = "r1-cuda-ep")]
fn check_cuda_v2_cache_outputs(
    key: &[DynValue],
    value: &[DynValue],
    key_new: &[DynValue],
    value_new: &[DynValue],
) -> Result<()> {
    if [key.len(), value.len(), key_new.len(), value_new.len()]
        .iter()
        .any(|len| *len != GPT_STEP_CACHE_LAYERS)
    {
        return Err(invalid("GPT Step v2 CUDA cache layer count mismatch"));
    }
    for layer in 0..GPT_STEP_CACHE_LAYERS {
        if tensor_data_ptr(&key[layer], "gpt_step_v2.k_cache input")?
            == tensor_data_ptr(&key_new[layer], "gpt_step_v2.k_cache output")?
            || tensor_data_ptr(&value[layer], "gpt_step_v2.v_cache input")?
                == tensor_data_ptr(&value_new[layer], "gpt_step_v2.v_cache output")?
        {
            return Err(invalid(
                "GPT Step v2 CUDA ping-pong cache buffers alias their inputs",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "r1-cuda-ep")]
fn check_cuda_sampling_outputs(topk_values: &DynValue, topk_indices: &DynValue) -> Result<()> {
    let values = topk_values.downcast_ref::<DynTensorValueType>()?;
    let indices = topk_indices.downcast_ref::<DynTensorValueType>()?;
    if values.memory_info().allocation_device() != AllocationDevice::CPU
        || indices.memory_info().allocation_device() != AllocationDevice::CPU
    {
        return Err(invalid(
            "GPT Step sampling outputs must remain host-accessible for the existing sampler",
        ));
    }
    Ok(())
}

#[cfg(feature = "r1-cuda-ep")]
fn run_gpt_step_v2_cuda_init(
    session: &mut Session,
    binding: &mut GpuStepBinding,
    sample: Tensor<i64>,
    idx: Tensor<i64>,
    x_len: &DynValue,
    y_len: &DynValue,
    key: &DynValue,
    value: &DynValue,
    sync_probe: bool,
) -> Result<(
    DynValue,
    DynValue,
    Vec<DynValue>,
    Vec<DynValue>,
    GptStepCudaSpans,
)> {
    #[cfg(windows)]
    let _token_nvtx = crate::nvtx::Range::push(b"gpt_step_token\0");
    #[cfg(feature = "r0-profiling")]
    let mut spans = GptStepCudaSpans::default();
    #[cfg(not(feature = "r0-profiling"))]
    let spans = GptStepCudaSpans;

    ensure_cuda_value(key, "gpt_step_init_v2.k_cache input")?;
    ensure_cuda_value(value, "gpt_step_init_v2.v_cache input")?;
    #[cfg(feature = "r0-profiling")]
    let binding_started = Instant::now();
    let io = &mut binding.binding;
    io.clear_inputs();
    io.bind_input("x_len", x_len)?;
    io.bind_input("y_len", y_len)?;
    io.bind_input("samples", &sample)?;
    io.bind_input("idx", &idx)?;
    io.bind_input("k_cache", key)?;
    io.bind_input("v_cache", value)?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.binding_update = Some((binding_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let run_started = Instant::now();
    let outputs_result = {
        #[cfg(windows)]
        let _ort_nvtx = crate::nvtx::Range::push(b"gpt_step_ort_run\0");
        if let Some(run_options) = &binding.run_options {
            session.run_binding_with_options(io, run_options)
        } else {
            session.run_binding(io)
        }
    };
    let mut outputs = outputs_result?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.ort_run = Some((run_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let sync_started = sync_probe.then(Instant::now);
    if sync_probe {
        io.synchronize_outputs()?;
    }
    #[cfg(feature = "r0-profiling")]
    if let Some(sync_started) = sync_started {
        spans.sync_wait = Some((sync_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let output_started = Instant::now();
    let topk_values = take(&mut outputs, "topk_values")?;
    let topk_indices = take(&mut outputs, "topk_indices")?;
    let key_new = take_cuda_cache_layers(&mut outputs, "k_cache_new_layer")?;
    let value_new = take_cuda_cache_layers(&mut outputs, "v_cache_new_layer")?;
    let packed_key_ptr = tensor_data_ptr(key, "gpt_step_init_v2.k_cache input")?;
    let packed_value_ptr = tensor_data_ptr(value, "gpt_step_init_v2.v_cache input")?;
    for cache in key_new.iter().chain(&value_new) {
        let ptr = tensor_data_ptr(cache, "gpt_step_init_v2 cache output")?;
        if ptr == packed_key_ptr || ptr == packed_value_ptr {
            return Err(invalid(
                "GPT Step v2 initial cache outputs alias their packed inputs",
            ));
        }
    }
    check_cuda_sampling_outputs(&topk_values, &topk_indices)?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.output_check = Some((output_started, Instant::now()));
    }
    Ok((topk_values, topk_indices, key_new, value_new, spans))
}

#[cfg(feature = "r1-cuda-ep")]
fn run_gpt_step_v2_cuda(
    session: &mut Session,
    binding: &mut GpuStepBinding,
    sample: Tensor<i64>,
    idx: Tensor<i64>,
    x_len: &DynValue,
    y_len: &DynValue,
    key: &[DynValue],
    value: &[DynValue],
    reuse_static_inputs: bool,
    sync_probe: bool,
) -> Result<(
    DynValue,
    DynValue,
    Vec<DynValue>,
    Vec<DynValue>,
    GptStepCudaSpans,
)> {
    if key.len() != GPT_STEP_CACHE_LAYERS || value.len() != GPT_STEP_CACHE_LAYERS {
        return Err(invalid("GPT Step v2 CUDA expects 24 K/V cache layers"));
    }
    for layer in 0..GPT_STEP_CACHE_LAYERS {
        ensure_cuda_value(&key[layer], "gpt_step_v2.k_cache input")?;
        ensure_cuda_value(&value[layer], "gpt_step_v2.v_cache input")?;
    }
    #[cfg(windows)]
    let _token_nvtx = crate::nvtx::Range::push(b"gpt_step_token\0");
    #[cfg(feature = "r0-profiling")]
    let mut spans = GptStepCudaSpans::default();
    #[cfg(not(feature = "r0-profiling"))]
    let spans = GptStepCudaSpans;

    #[cfg(feature = "r0-profiling")]
    let binding_started = Instant::now();
    let io = &mut binding.binding;
    if !reuse_static_inputs {
        io.clear_inputs();
        io.bind_input("x_len", x_len)?;
        io.bind_input("y_len", y_len)?;
    }
    io.bind_input("samples", &sample)?;
    io.bind_input("idx", &idx)?;
    for layer in 0..GPT_STEP_CACHE_LAYERS {
        io.bind_input(&format!("k_cache_layer_{layer:02}"), &key[layer])?;
        io.bind_input(&format!("v_cache_layer_{layer:02}"), &value[layer])?;
    }
    #[cfg(feature = "r0-profiling")]
    {
        spans.binding_update = Some((binding_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let run_started = Instant::now();
    let outputs_result = {
        #[cfg(windows)]
        let _ort_nvtx = crate::nvtx::Range::push(b"gpt_step_ort_run\0");
        if let Some(run_options) = &binding.run_options {
            session.run_binding_with_options(io, run_options)
        } else {
            session.run_binding(io)
        }
    };
    let mut outputs = outputs_result?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.ort_run = Some((run_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let sync_started = sync_probe.then(Instant::now);
    if sync_probe {
        io.synchronize_outputs()?;
    }
    #[cfg(feature = "r0-profiling")]
    if let Some(sync_started) = sync_started {
        spans.sync_wait = Some((sync_started, Instant::now()));
    }

    #[cfg(feature = "r0-profiling")]
    let output_started = Instant::now();
    let topk_values = take(&mut outputs, "topk_values")?;
    let topk_indices = take(&mut outputs, "topk_indices")?;
    let key_new = take_cuda_cache_layers(&mut outputs, "k_cache_new_layer")?;
    let value_new = take_cuda_cache_layers(&mut outputs, "v_cache_new_layer")?;
    check_cuda_v2_cache_outputs(key, value, &key_new, &value_new)?;
    check_cuda_sampling_outputs(&topk_values, &topk_indices)?;
    #[cfg(feature = "r0-profiling")]
    {
        spans.output_check = Some((output_started, Instant::now()));
    }
    Ok((topk_values, topk_indices, key_new, value_new, spans))
}

pub struct NativeTts {
    model_dir: PathBuf,
    vocoder_backend: Backend,
    vocoder_fallback_backend: Backend,
    cuda_vits: bool,
    cuda_gpt: bool,
    #[cfg(feature = "r1-cuda-ep")]
    cuda_step_bindings: Option<[GpuStepBinding; 2]>,
    #[cfg(feature = "r1-cuda-ep")]
    cuda_step_init_binding: Option<GpuStepBinding>,
    warmed: bool,
    frontend: Frontend,
    enc: Option<Session>,
    step: Option<Session>,
    #[cfg(feature = "r2-trt-native")]
    native_trt_gpt: Option<NativeTrtGpt>,
    #[cfg(feature = "r1-cuda-ep")]
    step_init: Option<Session>,
    step_cache_layout: GptStepCacheLayout,
    sovits: Session,
    first_sovits: Option<Session>,
    prompts: Tensor<i64>,
    #[cfg(feature = "r2-trt-native")]
    prompt_ids: Vec<i64>,
    spec: DynValue,
    sv: DynValue,
    reference_phones: Vec<i64>,
    reference_features: Vec<f32>,
    prompt_tokens: usize,
    streaming: bool,
    #[cfg(feature = "r0-profiling")]
    ort_profiling: bool,
}

impl NativeTts {
    fn native_trt_gpt_active(&self) -> bool {
        #[cfg(feature = "r2-trt-native")]
        {
            self.native_trt_gpt.is_some()
        }
        #[cfg(not(feature = "r2-trt-native"))]
        {
            false
        }
    }

    fn cuda_vits_active(&self) -> bool {
        #[cfg(feature = "r1-cuda-ep")]
        {
            self.cuda_vits && self.vocoder_backend == Backend::Cuda
        }
        #[cfg(not(feature = "r1-cuda-ep"))]
        {
            false
        }
    }

    pub fn load(config: &Config, cancel: &impl Fn() -> bool) -> Result<Self> {
        let _trace = LoadTrace::begin("validate_model_pack");
        validate_pack(&config.model_dir)?;
        if config.streaming {
            validate_streaming_pack(&config.model_dir)?;
        }
        drop(_trace);
        check(cancel)?;
        let dir = &config.model_dir;
        let _trace = LoadTrace::begin("read_pack_and_reference_audio");
        let pack: Pack = serde_json::from_slice(&std::fs::read(dir.join("pack.json"))?)?;
        if config.reference_audio.is_some() != config.reference_text.is_some() {
            return Err(invalid(
                "custom reference WAV and transcript must be configured together",
            ));
        }
        let reference_path = config
            .reference_audio
            .clone()
            .unwrap_or_else(|| dir.join(pack.reference_audio));
        let reference_text = config
            .reference_text
            .as_deref()
            .unwrap_or(&pack.reference_text);
        let (wav16, wav32) = audio::reference(&reference_path)?;
        drop(_trace);
        let frontend_backend = non_gpt_backend(config.backend);
        let frontend_session_backend = frontend_execution_backend(config.backend)?;
        #[cfg(feature = "r2-trt-native")]
        let native_trt_enabled = config.backend == Backend::TensorRtNative;
        #[cfg(not(feature = "r2-trt-native"))]
        let native_trt_enabled = false;
        let cuda_vits = cuda_vits_requested(config.backend);
        let vocoder_backend = if cuda_vits && config.streaming {
            #[cfg(feature = "r1-cuda-ep")]
            {
                Backend::Cuda
            }
            #[cfg(not(feature = "r1-cuda-ep"))]
            {
                frontend_backend
            }
        } else {
            frontend_backend
        };
        let _trace = LoadTrace::begin("frontend_session_load");
        let mut frontend = Frontend::load(dir, frontend_session_backend)?;
        drop(_trace);
        check(cancel)?;
        let _trace = LoadTrace::begin("frontend_reference_process");
        let (reference_phones, reference_features) = frontend.process(reference_text, cancel)?;
        drop(_trace);
        let _trace = LoadTrace::begin("reference_cpu_session_load");
        let mut ssl = open(dir, "ssl")?;
        check(cancel)?;
        let mut vq = open(dir, "vq_encoder")?;
        check(cancel)?;
        let mut spectrogram = open(dir, "spectrogram")?;
        check(cancel)?;
        let mut speaker = open(dir, "sv_embedding")?;
        drop(_trace);
        check(cancel)?;
        let mut padded = wav16.clone();
        padded.resize(padded.len() + 4800, 0.0);
        let _trace = LoadTrace::begin("reference_ssl_inference");
        let mut ssl_out = ssl.run(ort::inputs! {"audio"=>tf(vec![1,padded.len()],padded)?})?;
        let content = take(&mut ssl_out, "last_hidden_state")?;
        drop(_trace);
        check(cancel)?;
        let _trace = LoadTrace::begin("reference_vq_inference");
        let vq_out = vq.run(ort::inputs! {"ssl_content"=>&content})?;
        let (_, codes) = vq_out
            .get("codes")
            .ok_or_else(|| invalid("incompatible VQ output"))?
            .try_extract_tensor::<i64>()?;
        let prompt_tokens = codes.len();
        let prompt_ids = codes.to_vec();
        let prompts = ti(vec![1, prompt_tokens], prompt_ids.clone())?;
        drop(_trace);
        let _trace = LoadTrace::begin("reference_spectrogram_inference");
        let spec = take(
            &mut spectrogram.run(ort::inputs! {"audio"=>tf(vec![1,wav32.len()],wav32)?})?,
            "spectrogram",
        )?;
        drop(_trace);
        check(cancel)?;
        let _trace = LoadTrace::begin("reference_speaker_inference");
        let sv = take(
            &mut speaker.run(ort::inputs! {"audio"=>tf(vec![1,wav16.len()],wav16)?})?,
            "sv_embedding",
        )?;
        drop(_trace);
        check(cancel)?;
        // Reference-only sessions can release their weights before GPT/VITS load.
        drop(vq_out);
        drop(ssl_out);
        drop(content);
        drop((ssl, vq, spectrogram, speaker));
        #[cfg(feature = "r2-trt-native")]
        let native_trt_gpt = if native_trt_enabled {
            check(cancel)?;
            let _trace = LoadTrace::begin("native_tensorrt_gpt_load");
            let runner = NativeTrtGpt::load(dir)?;
            drop(_trace);
            Some(runner)
        } else {
            None
        };
        let (enc, step, step_cache_layout) = if native_trt_enabled {
            (None, None, GptStepCacheLayout::Abi1Packed)
        } else {
            let encoder_backend = gpt_backend(config.backend);
            let _trace = LoadTrace::begin("gpt_encoder_session_load");
            let enc = open_on(dir, "gpt_encoder", encoder_backend)?;
            drop(_trace);
            check(cancel)?;
            let _trace = LoadTrace::begin("gpt_step_session_load");
            let (step, step_cache_layout) = open_step(dir, config.backend)?;
            drop(_trace);
            (Some(enc), Some(step), step_cache_layout)
        };
        let cuda_gpt = is_cuda_backend(config.backend);
        #[cfg(feature = "r1-cuda-ep")]
        let step_init = if cuda_gpt && step_cache_layout == GptStepCacheLayout::V2PerLayer {
            if std::env::var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH")
                .ok()
                .as_deref()
                == Some("1")
            {
                return Err(invalid(
                    "CUDA Graph and the R1.7 per-layer cache experiment must be tested separately",
                ));
            }
            let graph = dir.join("gpt_step_init_v2.onnx");
            if !graph.is_file() {
                return Err(invalid(
                    "CUDA GPT Step v2 requires gpt_step_init_v2.onnx in the model pack",
                ));
            }
            let _trace = LoadTrace::begin("gpt_step_init_v2_session_load");
            let session = open_path(&graph, "gpt_step_init_v2", config.backend)?;
            drop(_trace);
            Some(session)
        } else {
            None
        };
        #[cfg(feature = "r1-cuda-ep")]
        let _trace = LoadTrace::begin("gpt_step_io_binding_setup");
        #[cfg(feature = "r1-cuda-ep")]
        let cuda_step_bindings = if cuda_gpt {
            let cuda_graph = std::env::var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH")
                .ok()
                .as_deref()
                == Some("1");
            Some([
                GpuStepBinding::new(
                    step.as_ref()
                        .ok_or_else(|| invalid("CUDA GPT Step Session is missing"))?,
                    cuda_graph.then_some(0),
                    step_cache_layout,
                )?,
                GpuStepBinding::new(
                    step.as_ref()
                        .ok_or_else(|| invalid("CUDA GPT Step Session is missing"))?,
                    cuda_graph.then_some(1),
                    step_cache_layout,
                )?,
            ])
        } else {
            None
        };
        #[cfg(feature = "r1-cuda-ep")]
        let cuda_step_init_binding = if let Some(session) = step_init.as_ref() {
            Some(GpuStepBinding::new(
                session,
                None,
                GptStepCacheLayout::V2PerLayer,
            )?)
        } else {
            None
        };
        #[cfg(feature = "r1-cuda-ep")]
        drop(_trace);
        check(cancel)?;
        let _trace = LoadTrace::begin("vits_session_load");
        let sovits = open_on(
            dir,
            if config.streaming {
                "sovits_stream"
            } else {
                "sovits"
            },
            vocoder_backend,
        )?;
        drop(_trace);
        check(cancel)?;
        let first_sovits = open_first_vits(dir, config.streaming, vocoder_backend)?;
        check(cancel)?;
        if reference_phones.len() + prompt_tokens + 32 >= 512 {
            return Err(invalid("reference exceeds KV cache budget"));
        }
        Ok(Self {
            model_dir: dir.to_owned(),
            vocoder_backend,
            vocoder_fallback_backend: frontend_backend,
            cuda_vits,
            cuda_gpt,
            #[cfg(feature = "r1-cuda-ep")]
            cuda_step_bindings,
            #[cfg(feature = "r1-cuda-ep")]
            cuda_step_init_binding,
            warmed: false,
            frontend,
            enc,
            step,
            #[cfg(feature = "r2-trt-native")]
            native_trt_gpt,
            #[cfg(feature = "r1-cuda-ep")]
            step_init,
            step_cache_layout,
            sovits,
            first_sovits,
            prompts,
            #[cfg(feature = "r2-trt-native")]
            prompt_ids,
            spec,
            sv,
            reference_phones,
            reference_features,
            prompt_tokens,
            streaming: config.streaming,
            #[cfg(feature = "r0-profiling")]
            ort_profiling: std::env::var_os("ALICE_NATIVE_TTS_ORT_PROFILE_DIR").is_some(),
        })
    }

    /// Explicitly closes the opt-in ORT profiles and returns their file paths.
    /// This is available only in the profiling build.
    #[cfg(feature = "r0-profiling")]
    pub fn finish_ort_profiling(&mut self) -> Result<Vec<PathBuf>> {
        if !self.ort_profiling {
            return Ok(Vec::new());
        }
        let mut paths = Vec::with_capacity(5);
        if let Some(encoder) = self.enc.as_mut() {
            paths.push(PathBuf::from(encoder.end_profiling()?));
        }
        if let Some(step) = self.step.as_mut() {
            paths.push(PathBuf::from(step.end_profiling()?));
        }
        #[cfg(feature = "r1-cuda-ep")]
        if let Some(step_init) = self.step_init.as_mut() {
            paths.push(PathBuf::from(step_init.end_profiling()?));
        }
        paths.push(PathBuf::from(self.sovits.end_profiling()?));
        if let Some(first) = self.first_sovits.as_mut() {
            paths.push(PathBuf::from(first.end_profiling()?));
        }
        paths.extend(self.frontend.finish_ort_profiling()?);
        self.ort_profiling = false;
        Ok(paths)
    }

    /// Change only the vocoder. Keep GPT, frontend and reference tensors;
    /// failed/cancelled preparation leaves the original decoder usable.
    pub fn set_streaming(&mut self, enabled: bool, cancel: &impl Fn() -> bool) -> Result<()> {
        check(cancel)?;
        if enabled == self.streaming {
            return Ok(());
        }
        if enabled {
            validate_streaming_pack(&self.model_dir)?;
        }
        let next_backend = if enabled && self.cuda_vits {
            #[cfg(feature = "r1-cuda-ep")]
            {
                Backend::Cuda
            }
            #[cfg(not(feature = "r1-cuda-ep"))]
            {
                self.vocoder_fallback_backend
            }
        } else {
            self.vocoder_fallback_backend
        };
        let next = open_on(
            &self.model_dir,
            if enabled { "sovits_stream" } else { "sovits" },
            next_backend,
        )?;
        check(cancel)?;
        let next_first = open_first_vits(&self.model_dir, enabled, next_backend)?;
        check(cancel)?;
        self.sovits = next;
        self.first_sovits = next_first;
        self.vocoder_backend = next_backend;
        self.streaming = enabled;
        self.warmed = false;
        Ok(())
    }

    /// Run representative inference without sending PCM to any consumer.
    /// Returns whether inference warmup was already completed in this mode.
    pub fn prepare(&mut self, cancel: &impl Fn() -> bool) -> Result<bool> {
        check(cancel)?;
        let already_warm = self.warmed;
        if !already_warm {
            self.synthesize_streaming("你好，欢迎回来。", cancel, |_, _| {})?;
        }
        check(cancel)?;
        Ok(already_warm)
    }

    /// Collects the selected decoder's audio. Cancellation is checked at every
    /// semantic token and after VITS, so interrupted output is never published.
    pub fn synthesize(&mut self, text: &str, cancel: &impl Fn() -> bool) -> Result<Vec<f32>> {
        self.synthesize_profiled(text, cancel)
            .map(|(audio, _)| audio)
    }

    pub fn synthesize_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
    ) -> Result<(Vec<f32>, SynthesisTimings)> {
        let mut samples = Vec::new();
        let timings =
            self.synthesize_streaming(text, cancel, |audio, _| samples.extend_from_slice(audio))?;
        Ok((samples, timings))
    }

    /// In token mode, emits complete audio fragments during GPT generation.
    /// In compatibility mode, delivers one complete sentence. Every fragment
    /// is checked for cancellation before and after the caller's FIFO write.
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<SynthesisTimings> {
        let mut profile = ProfileCollector::disabled();
        self.synthesize_streaming_inner(
            text,
            cancel,
            CHUNK_TOKENS,
            CHUNK_TOKENS,
            None,
            None,
            None,
            &mut profile,
            emit,
        )
    }

    /// Profiling-only entrypoint. It preserves the regular inference path while
    /// allowing the benchmark to sweep the VITS streaming chunk threshold.
    #[cfg(feature = "r0-profiling")]
    pub fn synthesize_streaming_r0(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        chunk_tokens: usize,
        emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<(SynthesisTimings, crate::profiling::R0Timings)> {
        if !(1..=128).contains(&chunk_tokens) {
            return Err(invalid("R0 chunk size must be between 1 and 128 tokens"));
        }
        let mut profile = ProfileCollector::enabled();
        let timings = self.synthesize_streaming_inner(
            text,
            cancel,
            chunk_tokens,
            chunk_tokens,
            None,
            None,
            None,
            &mut profile,
            emit,
        )?;
        Ok((timings, profile.finish()))
    }

    /// Profiling-only entrypoint for changing the first audio threshold while
    /// keeping subsequent streaming chunks at the ABI-1 default of 24 tokens.
    #[cfg(feature = "r0-profiling")]
    #[doc(hidden)]
    pub fn synthesize_streaming_first_chunk_r1_3(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        first_chunk_tokens: usize,
        emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<(SynthesisTimings, crate::profiling::R0Timings)> {
        if ![12, 16, 20, 24].contains(&first_chunk_tokens) {
            return Err(invalid(
                "R1.3 first chunk threshold must be 12, 16, 20, or 24 tokens",
            ));
        }
        let mut profile = ProfileCollector::enabled();
        let timings = self.synthesize_streaming_inner(
            text,
            cancel,
            first_chunk_tokens,
            CHUNK_TOKENS,
            None,
            None,
            None,
            &mut profile,
            emit,
        )?;
        Ok((timings, profile.finish()))
    }

    /// Feature-gated R1.8 experiment and matched Chunk-24 control. A preview
    /// is emitted after 17 or 20 semantic tokens, then refilled at 24 tokens.
    /// `None` runs the unchanged Chunk-24 strategy through the same harness.
    #[cfg(feature = "r1-r8-preview")]
    #[doc(hidden)]
    pub fn synthesize_streaming_r1_8_experiment(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        preview_tokens: Option<usize>,
        emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<(SynthesisTimings, serde_json::Value)> {
        if preview_tokens.is_some_and(|tokens| ![17, 20].contains(&tokens)) {
            return Err(invalid("R1.8 preview must use 17 or 20 semantic tokens"));
        }
        if !self.streaming {
            return Err(invalid("R1.8 preview requires the ABI-1 streaming graph"));
        }
        let mut profile = ProfileCollector::disabled();
        let mut diagnostics = PreviewRefillDiagnostics {
            preview_tokens,
            refill_target_tokens: preview_tokens.map(|_| CHUNK_TOKENS),
            ..PreviewRefillDiagnostics::default()
        };
        let timings = self.synthesize_streaming_inner(
            text,
            cancel,
            preview_tokens.unwrap_or(CHUNK_TOKENS),
            CHUNK_TOKENS,
            preview_tokens.map(|_| CHUNK_TOKENS),
            Some(&mut diagnostics),
            None,
            &mut profile,
            emit,
        )?;
        if preview_tokens.is_some()
            && diagnostics.preview_callback_tokens.is_none()
            && diagnostics.preview_fallback_reason.is_none()
        {
            diagnostics.preview_fallback_reason = Some("semantic_eos_before_preview_threshold");
        }
        Ok((timings, serde_json::to_value(diagnostics)?))
    }

    /// R1.9 feasibility experiment. It emits only a short, onset-relative
    /// prefix at token 17, then re-decodes the full semantic prefix at token
    /// 24. Unsafe joins return diagnostics with no silent splice fallback.
    #[cfg(feature = "r1-r9-bounded-preview")]
    #[doc(hidden)]
    pub fn synthesize_streaming_r1_9_bounded_experiment(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        publish_after_onset_ms: usize,
        emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<(Option<SynthesisTimings>, serde_json::Value)> {
        if ![40, 80, 120].contains(&publish_after_onset_ms) {
            return Err(invalid(
                "R1.9 publish window must be 40, 80, or 120 ms after voice onset",
            ));
        }
        if !self.streaming {
            return Err(invalid("R1.9 preview requires the ABI-1 streaming graph"));
        }
        let mut profile = ProfileCollector::disabled();
        let mut diagnostics = PreviewRefillDiagnostics {
            preview_tokens: Some(17),
            refill_target_tokens: Some(CHUNK_TOKENS),
            ..PreviewRefillDiagnostics::default()
        };
        let result = self.synthesize_streaming_inner(
            text,
            cancel,
            17,
            CHUNK_TOKENS,
            Some(CHUNK_TOKENS),
            Some(&mut diagnostics),
            Some(publish_after_onset_ms),
            &mut profile,
            emit,
        );
        match result {
            Ok(timings) => Ok((Some(timings), serde_json::to_value(diagnostics)?)),
            Err(_) if diagnostics.refill_rejected_reason.is_some() => {
                Ok((None, serde_json::to_value(diagnostics)?))
            }
            Err(error) => Err(error),
        }
    }

    fn synthesize_streaming_inner(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        first_chunk_tokens: usize,
        chunk_tokens: usize,
        preview_refill_at: Option<usize>,
        mut preview_diagnostics: Option<&mut PreviewRefillDiagnostics>,
        bounded_preview_after_onset_ms: Option<usize>,
        profile: &mut ProfileCollector,
        mut emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<SynthesisTimings> {
        check(cancel)?;
        if text.chars().count() > 180 {
            return Err(invalid("native TTS sentence exceeds 180 characters"));
        }
        let started = Instant::now();
        let (phones, features) = profile
            .measure_with_profile(ProfilePhase::Frontend, |profile| {
                self.frontend.process_profiled(text, cancel, profile)
            })?;
        let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
        let gpt_prefill_event = profile.start_event();
        let started = Instant::now();
        let n = self.reference_phones.len() + phones.len();
        let budget = 512usize
            .checked_sub(n + self.prompt_tokens)
            .ok_or_else(|| {
                invalid("sentence exceeds KV cache budget; split into shorter sentences")
            })?;
        if budget < 32 {
            return Err(invalid(
                "sentence leaves insufficient semantic token budget",
            ));
        }
        let mut all_phones = profile.measure(ProfilePhase::GptEncoderInput, || {
            self.reference_phones.clone()
        });
        profile.measure(ProfilePhase::HostDataCopy, || {
            all_phones.extend_from_slice(&phones);
        });
        let mut all_features =
            profile.measure(ProfilePhase::GptEncoderInput, || vec![0.0; 1024 * n]);
        profile.measure(ProfilePhase::HostDataCopy, || {
            for row in 0..1024 {
                let r = self.reference_phones.len();
                let t = phones.len();
                all_features[row * n..row * n + r]
                    .copy_from_slice(&self.reference_features[row * r..(row + 1) * r]);
                all_features[row * n + r..(row + 1) * n]
                    .copy_from_slice(&features[row * t..(row + 1) * t]);
            }
        });
        #[cfg(feature = "r0-profiling")]
        profile.add_input_copy_bytes(
            (self.reference_phones.len() + phones.len()) * std::mem::size_of::<i64>()
                + (self.reference_features.len() + features.len()) * std::mem::size_of::<f32>(),
        );
        check(cancel)?;
        let (mut values, mut ids, x_len, y_len, mut kv_cache) = if self.native_trt_gpt_active() {
            #[cfg(feature = "r2-trt-native")]
            {
                let runner = self
                    .native_trt_gpt
                    .as_mut()
                    .ok_or_else(|| invalid("native TensorRT GPT runner was not initialized"))?;
                runner.reset();
                let (raw_values, raw_ids, raw_x_len, raw_y_len) = {
                    profile.measure(ProfilePhase::GptEncoderRun, || {
                        runner.run_encoder(&all_phones, &self.prompt_ids, &all_features)
                    })?
                };
                let (values, ids, x_len, y_len) =
                    profile.measure(ProfilePhase::GptEncoderOutput, || {
                        let count = raw_values.len();
                        Ok::<_, crate::Error>((
                            tf(vec![1, count], raw_values)?.into_dyn(),
                            ti(vec![1, count], raw_ids)?.into_dyn(),
                            ti(vec![1], vec![raw_x_len])?.into_dyn(),
                            ti(vec![1], vec![raw_y_len])?.into_dyn(),
                        ))
                    })?;
                validate_gpt_lengths(&x_len, &y_len, n, self.prompt_tokens)?;
                (values, ids, x_len, y_len, GptKvCache::TensorRtNative)
            }
            #[cfg(not(feature = "r2-trt-native"))]
            {
                return Err(invalid(
                    "native TensorRT GPT inference requires the r2-trt-native feature",
                ));
            }
        } else if self.cuda_gpt {
            #[cfg(feature = "r1-cuda-ep")]
            {
                let phoneme_ids = profile
                    .measure(ProfilePhase::GptEncoderInput, || ti(vec![1, n], all_phones))?;
                let bert_feature = profile.measure(ProfilePhase::GptEncoderInput, || {
                    tf(vec![1, 1024, n], all_features)
                })?;
                let (values, ids, x_len, y_len, k, v) =
                    profile.measure(ProfilePhase::GptEncoderRun, || {
                        run_gpt_encoder_cuda(
                            self.enc
                                .as_mut()
                                .ok_or_else(|| invalid("CUDA GPT Encoder Session is missing"))?,
                            phoneme_ids,
                            &self.prompts,
                            bert_feature,
                        )
                    })?;
                validate_gpt_lengths(&x_len, &y_len, n, self.prompt_tokens)?;
                let cache = match self.step_cache_layout {
                    GptStepCacheLayout::Abi1Packed => GptKvCache::CudaAbi1PingPong {
                        key: k,
                        value: v,
                        next_binding: 0,
                    },
                    GptStepCacheLayout::V2PerLayer => {
                        GptKvCache::CudaV2Initial { key: k, value: v }
                    }
                };
                (values, ids, x_len, y_len, cache)
            }
            #[cfg(not(feature = "r1-cuda-ep"))]
            {
                return Err(invalid(
                    "CUDA GPT inference was requested without the r1-cuda-ep feature",
                ));
            }
        } else {
            let phoneme_ids =
                profile.measure(ProfilePhase::GptEncoderInput, || ti(vec![1, n], all_phones))?;
            let bert_feature = profile.measure(ProfilePhase::GptEncoderInput, || {
                tf(vec![1, 1024, n], all_features)
            })?;
            let enc = self
                .enc
                .as_mut()
                .ok_or_else(|| invalid("ORT GPT Encoder Session is missing"))?;
            let mut out = profile.measure(ProfilePhase::GptEncoderRun, || {
                enc.run(ort::inputs! {
                "phoneme_ids" => phoneme_ids,
                "prompts" => &self.prompts,
                "bert_feature" => bert_feature
                })
            })?;
            // These lengths stay fixed for the whole autoregressive request.
            let (x_len, y_len) = profile.measure(ProfilePhase::GptEncoderOutput, || {
                Ok::<_, crate::Error>((take(&mut out, "x_len")?, take(&mut out, "y_len")?))
            })?;
            validate_gpt_lengths(&x_len, &y_len, n, self.prompt_tokens)?;
            let (values, ids, k, v) = profile.measure(ProfilePhase::GptEncoderOutput, || {
                Ok::<_, crate::Error>((
                    take(&mut out, "topk_values")?,
                    take(&mut out, "topk_indices")?,
                    take(&mut out, "k_cache")?,
                    take(&mut out, "v_cache")?,
                ))
            })?;
            let cache = match self.step_cache_layout {
                GptStepCacheLayout::Abi1Packed => GptKvCache::Abi1Packed { key: k, value: v },
                GptStepCacheLayout::V2PerLayer => GptKvCache::V2PerLayer {
                    key: split_cache_layers(k, "k_cache")?,
                    value: split_cache_layers(v, "v_cache")?,
                },
            };
            drop(out);
            (values, ids, x_len, y_len, cache)
        };
        profile.finish_event("gpt_prefill", gpt_prefill_event, None);
        #[cfg(feature = "r1-cuda-ep")]
        let reuse_cuda_step_bindings = self.cuda_gpt
            && std::env::var("ALICE_NATIVE_TTS_GPT_STEP_BINDING_MODE")
                .ok()
                .as_deref()
                == Some("reuse_static");
        #[cfg(feature = "r1-cuda-ep")]
        if reuse_cuda_step_bindings {
            profile.measure(ProfilePhase::GptStepRequestBindings, || {
                let bindings = self
                    .cuda_step_bindings
                    .as_mut()
                    .ok_or_else(|| invalid("CUDA GPT Step I/O Binding was not initialized"))?;
                for binding in bindings {
                    binding.bind_request_inputs(&x_len, &y_len)?;
                }
                Ok::<_, crate::Error>(())
            })?;
        }
        #[cfg(feature = "r1-cuda-ep")]
        let sync_probe = std::env::var("ALICE_NATIVE_TTS_GPT_STEP_SYNC_PROBE")
            .ok()
            .as_deref()
            == Some("1");
        let mut random = 1234u64;
        let mut tokens = Vec::new();
        let mut eos = false;
        let mut decoder = StreamDecoder::default();
        if preview_refill_at.is_some() && bounded_preview_after_onset_ms.is_none() {
            decoder.enable_preview_refill_mode();
        }
        let mut bounded_preview_state: Option<BoundedPreviewState> = None;
        let cuda_vits_active = self.cuda_vits_active();
        let mut vocoder_ms = 0.0;
        let mut delivery_ms = 0.0;
        let mut first_audio_tokens = 0;
        let mut audio_chunks = 0;
        let mut next_audio_tokens = first_chunk_tokens;
        let mut preview_pending = preview_refill_at.is_some();
        for idx in 0..budget.min(350) {
            check(cancel)?;
            let token = profile.measure(ProfilePhase::SemanticSampling, || {
                pick(&values, &ids, &mut random)
            })?;
            if token == 1024 {
                eos = true;
                break;
            }
            if !(0..1024).contains(&token) {
                return Err(invalid("invalid semantic token"));
            }
            tokens.push(token);
            if self.native_trt_gpt_active() {
                #[cfg(feature = "r2-trt-native")]
                {
                    let (raw_values, raw_ids) = profile.measure_gpt_step(|| {
                        self.native_trt_gpt
                            .as_mut()
                            .ok_or_else(|| {
                                invalid("native TensorRT GPT runner was not initialized")
                            })?
                            .run_step(token, idx)
                    })?;
                    (values, ids) = profile.measure(ProfilePhase::GptStepOutput, || {
                        let count = raw_values.len();
                        Ok::<_, crate::Error>((
                            tf(vec![1, count], raw_values)?.into_dyn(),
                            ti(vec![1, count], raw_ids)?.into_dyn(),
                        ))
                    })?;
                }
                #[cfg(not(feature = "r2-trt-native"))]
                {
                    return Err(invalid(
                        "native TensorRT GPT inference requires the r2-trt-native feature",
                    ));
                }
            } else {
                let (sample, idx_input) = profile.measure(ProfilePhase::GptStepInput, || {
                    Ok::<_, crate::Error>((
                        ti(vec![1, 1], vec![token])?,
                        ti(vec![1], vec![idx as i64])?,
                    ))
                })?;
                #[cfg(feature = "r1-cuda-ep")]
                if let GptKvCache::CudaAbi1PingPong {
                    key,
                    value,
                    next_binding,
                } = &mut kv_cache
                {
                    let slot = *next_binding;
                    let bindings = self
                        .cuda_step_bindings
                        .as_mut()
                        .ok_or_else(|| invalid("CUDA GPT Step I/O Binding was not initialized"))?;
                    let (next_values, next_ids, next_key, next_value, _spans) = profile
                        .measure_gpt_step(|| {
                            run_gpt_step_cuda(
                                self.step
                                    .as_mut()
                                    .ok_or_else(|| invalid("CUDA GPT Step Session is missing"))?,
                                &mut bindings[slot],
                                sample,
                                idx_input,
                                &x_len,
                                &y_len,
                                key,
                                value,
                                reuse_cuda_step_bindings,
                                sync_probe,
                            )
                        })?;
                    #[cfg(feature = "r0-profiling")]
                    {
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepBindingUpdate,
                            _spans.binding_update,
                        );
                        profile.record_gpt_step_event(ProfilePhase::GptStepOrtRun, _spans.ort_run);
                        profile
                            .record_gpt_step_event(ProfilePhase::GptStepSyncWait, _spans.sync_wait);
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepOutputCheck,
                            _spans.output_check,
                        );
                    }
                    (values, ids) = profile.measure(ProfilePhase::GptStepOutput, || {
                        Ok::<_, crate::Error>((next_values, next_ids))
                    })?;
                    *key = next_key;
                    *value = next_value;
                    *next_binding = 1 - slot;
                    #[cfg(feature = "r0-profiling")]
                    profile.add_cuda_kv_cache_step();
                } else if matches!(&kv_cache, GptKvCache::CudaV2Initial { .. }) {
                    let (next_values, next_ids, next_key, next_value, _spans) = {
                        let GptKvCache::CudaV2Initial { key, value } = &kv_cache else {
                            unreachable!("cache variant checked above")
                        };
                        let binding = self.cuda_step_init_binding.as_mut().ok_or_else(|| {
                            invalid("CUDA GPT Step v2 init binding was not initialized")
                        })?;
                        let session = self.step_init.as_mut().ok_or_else(|| {
                            invalid("CUDA GPT Step v2 init session was not initialized")
                        })?;
                        profile.measure_gpt_step(|| {
                            run_gpt_step_v2_cuda_init(
                                session, binding, sample, idx_input, &x_len, &y_len, key, value,
                                sync_probe,
                            )
                        })?
                    };
                    #[cfg(feature = "r0-profiling")]
                    {
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepBindingUpdate,
                            _spans.binding_update,
                        );
                        profile.record_gpt_step_event(ProfilePhase::GptStepOrtRun, _spans.ort_run);
                        profile
                            .record_gpt_step_event(ProfilePhase::GptStepSyncWait, _spans.sync_wait);
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepOutputCheck,
                            _spans.output_check,
                        );
                    }
                    (values, ids) = profile.measure(ProfilePhase::GptStepOutput, || {
                        Ok::<_, crate::Error>((next_values, next_ids))
                    })?;
                    kv_cache = GptKvCache::CudaV2PerLayer {
                        key: next_key,
                        value: next_value,
                        next_binding: 0,
                    };
                    #[cfg(feature = "r0-profiling")]
                    profile.add_cuda_kv_cache_step();
                } else if let GptKvCache::CudaV2PerLayer {
                    key,
                    value,
                    next_binding,
                } = &mut kv_cache
                {
                    let slot = *next_binding;
                    let bindings = self.cuda_step_bindings.as_mut().ok_or_else(|| {
                        invalid("CUDA GPT Step v2 I/O Binding was not initialized")
                    })?;
                    let (next_values, next_ids, next_key, next_value, _spans) = profile
                        .measure_gpt_step(|| {
                            run_gpt_step_v2_cuda(
                                self.step
                                    .as_mut()
                                    .ok_or_else(|| invalid("CUDA GPT Step Session is missing"))?,
                                &mut bindings[slot],
                                sample,
                                idx_input,
                                &x_len,
                                &y_len,
                                key,
                                value,
                                reuse_cuda_step_bindings,
                                sync_probe,
                            )
                        })?;
                    #[cfg(feature = "r0-profiling")]
                    {
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepBindingUpdate,
                            _spans.binding_update,
                        );
                        profile.record_gpt_step_event(ProfilePhase::GptStepOrtRun, _spans.ort_run);
                        profile
                            .record_gpt_step_event(ProfilePhase::GptStepSyncWait, _spans.sync_wait);
                        profile.record_gpt_step_event(
                            ProfilePhase::GptStepOutputCheck,
                            _spans.output_check,
                        );
                    }
                    (values, ids) = profile.measure(ProfilePhase::GptStepOutput, || {
                        Ok::<_, crate::Error>((next_values, next_ids))
                    })?;
                    *key = next_key;
                    *value = next_value;
                    *next_binding = 1 - slot;
                    #[cfg(feature = "r0-profiling")]
                    profile.add_cuda_kv_cache_step();
                } else {
                    (values, ids, kv_cache) = profile.measure_gpt_step(|| {
                        run_gpt_step_cpu(
                            self.step
                                .as_mut()
                                .ok_or_else(|| invalid("ORT GPT Step Session is missing"))?,
                            sample,
                            idx_input,
                            &x_len,
                            &y_len,
                            &kv_cache,
                            self.step_cache_layout,
                        )
                    })?;
                }
                #[cfg(not(feature = "r1-cuda-ep"))]
                {
                    (values, ids, kv_cache) = profile.measure_gpt_step(|| {
                        run_gpt_step_cpu(
                            self.step
                                .as_mut()
                                .ok_or_else(|| invalid("ORT GPT Step Session is missing"))?,
                            sample,
                            idx_input,
                            &x_len,
                            &y_len,
                            &kv_cache,
                            self.step_cache_layout,
                        )
                    })?;
                }
            }
            #[cfg(feature = "r0-profiling")]
            profile.add_gpt_step();
            if self.streaming && tokens.len() >= next_audio_tokens {
                let is_preview_attempt = preview_pending;
                let decode_started = Instant::now();
                let first_vits_event = (audio_chunks == 0).then(|| profile.start_event()).flatten();
                let audio = if let Some(bounded_state) = bounded_preview_state.take() {
                    decoder.reset_unemitted_preview();
                    let refilled = decoder.decode(
                        &mut self.sovits,
                        &tokens,
                        &phones,
                        &self.spec,
                        &self.sv,
                        false,
                        cuda_vits_active,
                        cancel,
                        profile,
                    )?;
                    let join = find_bounded_preview_join(
                        &bounded_state.decoded_pcm,
                        &refilled,
                        bounded_state.committed_samples,
                    );
                    if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                        diagnostics.full_prefix_refill = true;
                        diagnostics.bounded_join = join.clone();
                    }
                    let Some(mut join) = join.filter(|join| join.accepted) else {
                        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                            diagnostics.refill_rejected_reason =
                                Some("no_high_confidence_uncommitted_tail_alignment");
                            diagnostics.semantic_tokens = tokens.len();
                            diagnostics.semantic_token_hash = semantic_token_hash(&tokens);
                        }
                        return Err(invalid(
                            "R1.9 bounded Preview Refill rejected: no high-confidence join",
                        ));
                    };
                    let stitched = stitch_bounded_preview(
                        &bounded_state.decoded_pcm,
                        &refilled,
                        bounded_state.committed_samples,
                        &mut join,
                    );
                    if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                        diagnostics.full_prefix_refill = true;
                        diagnostics.bounded_join = Some(join);
                    }
                    stitched
                } else {
                    let use_first = cuda_vits_active
                        && audio_chunks == 0
                        && first_chunk_tokens == CHUNK_TOKENS
                        && chunk_tokens == CHUNK_TOKENS
                        && tokens.len() == CHUNK_TOKENS
                        && preview_refill_at.is_none()
                        && self.first_sovits.is_some();
                    let session = if use_first {
                        self.first_sovits
                            .as_mut()
                            .expect("checked first VITS session")
                    } else {
                        &mut self.sovits
                    };
                    let audio = decoder.decode(
                        session,
                        &tokens,
                        &phones,
                        &self.spec,
                        &self.sv,
                        false,
                        cuda_vits_active,
                        cancel,
                        profile,
                    )?;
                    if use_first {
                        // I/O bindings belong to their session; retain latent/tail
                        // audio but recreate binding for the ordinary continuation.
                        decoder.release_session_binding();
                    }
                    audio
                };
                profile.finish_event("first_vits", first_vits_event, Some(audio_chunks));
                vocoder_ms += decode_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
                let mut bounded_callback_audio: Option<Vec<f32>> = None;
                if is_preview_attempt {
                    if let Some(reason) = preview_fallback_reason(&audio) {
                        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                            diagnostics.preview_fallback_reason = Some(reason);
                            diagnostics.preview_decoded_samples =
                                Some(audio.len() + decoder.pending_audio_tail().len());
                            diagnostics.preview_committed_samples = Some(0);
                            diagnostics.preview_uncommitted_samples =
                                diagnostics.preview_decoded_samples;
                        }
                        decoder.reset_unemitted_preview();
                        preview_pending = false;
                        next_audio_tokens = preview_refill_at
                            .unwrap_or_else(|| next_audio_tokens.saturating_add(chunk_tokens));
                        continue;
                    }
                    if let Some(publish_after_onset_ms) = bounded_preview_after_onset_ms {
                        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                            diagnostics.preview_decoded_samples =
                                Some(audio.len() + decoder.pending_audio_tail().len());
                            diagnostics.preview_committed_samples = Some(0);
                            diagnostics.preview_uncommitted_samples =
                                diagnostics.preview_decoded_samples;
                        }
                        let onset = first_voice_sample(&audio);
                        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                            diagnostics.preview_voice_onset_samples = onset;
                        }
                        let Some(onset) =
                            onset.filter(|sample| *sample <= R1_9_MAX_VOICE_ONSET_MS * 32)
                        else {
                            if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                                diagnostics.preview_fallback_reason =
                                    Some("voice_onset_missing_or_after_160ms");
                            }
                            decoder.reset_unemitted_preview();
                            preview_pending = false;
                            next_audio_tokens = preview_refill_at
                                .unwrap_or_else(|| next_audio_tokens.saturating_add(chunk_tokens));
                            continue;
                        };
                        let requested_commit = onset.saturating_add(publish_after_onset_ms * 32);
                        if requested_commit > R1_9_MAX_COMMIT_MS * 32 {
                            if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                                diagnostics.preview_fallback_reason =
                                    Some("onset_relative_publish_window_exceeds_240ms");
                            }
                            decoder.reset_unemitted_preview();
                            preview_pending = false;
                            next_audio_tokens = preview_refill_at
                                .unwrap_or_else(|| next_audio_tokens.saturating_add(chunk_tokens));
                            continue;
                        }
                        let commit_samples =
                            requested_commit.max(R1_9_MIN_COMMIT_MS * 32).div_ceil(320) * 320;
                        if commit_samples >= audio.len() {
                            if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                                diagnostics.preview_fallback_reason =
                                    Some("publish_window_consumes_preview");
                            }
                            decoder.reset_unemitted_preview();
                            preview_pending = false;
                            next_audio_tokens = preview_refill_at
                                .unwrap_or_else(|| next_audio_tokens.saturating_add(chunk_tokens));
                            continue;
                        }
                        let mut decoded_pcm = audio.clone();
                        decoded_pcm.extend_from_slice(decoder.pending_audio_tail());
                        bounded_preview_state = Some(BoundedPreviewState {
                            decoded_pcm,
                            committed_samples: commit_samples,
                        });
                        bounded_callback_audio = Some(audio[..commit_samples].to_vec());
                        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                            diagnostics.preview_decoded_samples =
                                Some(audio.len() + decoder.pending_audio_tail().len());
                            diagnostics.preview_committed_samples = Some(commit_samples);
                            diagnostics.preview_uncommitted_samples = Some(
                                audio.len() + decoder.pending_audio_tail().len() - commit_samples,
                            );
                        }
                    }
                    if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                        diagnostics.preview_callback_tokens = Some(tokens.len());
                    }
                    preview_pending = false;
                }
                if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
                    if let Some(alignment) = decoder.take_sola_alignment() {
                        diagnostics.sola_alignments.push(alignment);
                    }
                }
                if first_audio_tokens == 0 {
                    first_audio_tokens = tokens.len();
                }
                audio_chunks += 1;
                let delivery_started = Instant::now();
                profile.measure(ProfilePhase::AudioCallback, || {
                    emit(
                        bounded_callback_audio.as_deref().unwrap_or(&audio),
                        ChunkInfo {
                            semantic_tokens: tokens.len(),
                            final_chunk: false,
                        },
                    )
                });
                profile.mark_first_audio();
                delivery_ms += delivery_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
                next_audio_tokens = if is_preview_attempt {
                    preview_refill_at
                        .unwrap_or_else(|| next_audio_tokens.saturating_add(chunk_tokens))
                } else {
                    next_audio_tokens.saturating_add(chunk_tokens)
                };
            }
        }
        check(cancel)?;
        if !eos || tokens.is_empty() {
            return Err(invalid(
                "semantic generation did not finish within the cache budget; shorten the sentence",
            ));
        }
        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
            diagnostics.semantic_tokens = tokens.len();
            diagnostics.semantic_token_hash = semantic_token_hash(&tokens);
        }
        #[cfg(feature = "r0-profiling")]
        profile.record_semantic_tokens(&tokens);
        let gpt_ms = (started.elapsed().as_secs_f64() * 1000.0 - vocoder_ms - delivery_ms).max(0.0);
        let semantic_tokens = tokens.len();
        let started = Instant::now();
        let samples = if self.streaming {
            decoder.decode(
                &mut self.sovits,
                &tokens,
                &phones,
                &self.spec,
                &self.sv,
                true,
                cuda_vits_active,
                cancel,
                profile,
            )?
        } else {
            let (pred_semantic, text_seq, noise_scale) =
                profile.measure(ProfilePhase::VitsInput, || {
                    Ok::<_, crate::Error>((
                        ti(vec![1, 1, tokens.len()], tokens)?,
                        ti(vec![1, phones.len()], phones)?,
                        tf(vec![1], vec![0.0])?,
                    ))
                })?;
            let audio = profile.measure(ProfilePhase::VitsRun, || {
                self.sovits.run(ort::inputs! {
                    "pred_semantic" => pred_semantic,
                    "text_seq" => text_seq,
                    "refer_spec" => &self.spec,
                    "sv_emb" => &self.sv,
                    "noise_scale" => noise_scale
                })
            })?;
            let (sample_count, samples) = profile.measure(ProfilePhase::VitsOutput, || {
                let (_, samples) = audio
                    .get("audio")
                    .ok_or_else(|| invalid("incompatible VITS output"))?
                    .try_extract_tensor::<f32>()?;
                Ok::<_, crate::Error>((samples.len(), samples.to_vec()))
            })?;
            #[cfg(feature = "r0-profiling")]
            {
                profile.add_output_copy_bytes(sample_count * std::mem::size_of::<f32>());
                profile.add_vits_call();
            }
            let _ = sample_count;
            samples
        };
        if let Some(diagnostics) = preview_diagnostics.as_deref_mut() {
            if let Some(alignment) = decoder.take_sola_alignment() {
                diagnostics.sola_alignments.push(alignment);
            }
        }
        vocoder_ms += started.elapsed().as_secs_f64() * 1000.0;
        check(cancel)?;
        if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid synthesized audio"));
        }
        if first_audio_tokens == 0 {
            first_audio_tokens = semantic_tokens;
        }
        audio_chunks += 1;
        profile.measure(ProfilePhase::AudioCallback, || {
            emit(
                &samples,
                ChunkInfo {
                    semantic_tokens,
                    final_chunk: true,
                },
            )
        });
        profile.mark_first_audio();
        check(cancel)?;
        self.warmed = true;
        Ok(SynthesisTimings {
            frontend_ms,
            gpt_ms,
            vocoder_ms,
            semantic_tokens,
            first_audio_tokens,
            audio_chunks,
        })
    }
}

fn semantic_token_hash(tokens: &[i64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn split_cache_layers(value: DynValue, cache_name: &str) -> Result<Vec<DynValue>> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    if shape.len() != 4
        || shape[0] != GPT_STEP_CACHE_LAYERS as i64
        || shape.iter().any(|dimension| *dimension <= 0)
        || data.len() % GPT_STEP_CACHE_LAYERS != 0
    {
        return Err(invalid(format!(
            "incompatible GPT Step v2 {cache_name} shape: {shape:?}"
        )));
    }
    let layer_size = data.len() / GPT_STEP_CACHE_LAYERS;
    let mut layer_shape = shape[1..]
        .iter()
        .map(|dimension| *dimension as usize)
        .collect::<Vec<_>>();
    layer_shape.insert(0, 1);
    let mut layers = Vec::with_capacity(GPT_STEP_CACHE_LAYERS);
    for layer in 0..GPT_STEP_CACHE_LAYERS {
        let start = layer * layer_size;
        layers.push(
            Tensor::from_array((
                layer_shape.clone(),
                data[start..start + layer_size].to_vec(),
            ))?
            .into_dyn(),
        );
    }
    Ok(layers)
}

fn run_gpt_step_v2<'s>(
    session: &'s mut Session,
    sample: Tensor<i64>,
    idx: Tensor<i64>,
    x_len: &DynValue,
    y_len: &DynValue,
    key: &[DynValue],
    value: &[DynValue],
) -> Result<ort::session::SessionOutputs<'s>> {
    if key.len() != GPT_STEP_CACHE_LAYERS || value.len() != GPT_STEP_CACHE_LAYERS {
        return Err(invalid("GPT Step v2 expects 24 K/V cache layers"));
    }
    let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
        ("samples".to_owned(), sample.into()),
        ("idx".to_owned(), idx.into()),
        ("x_len".to_owned(), x_len.into()),
        ("y_len".to_owned(), y_len.into()),
    ];
    for layer in 0..GPT_STEP_CACHE_LAYERS {
        inputs.push((format!("k_cache_layer_{layer:02}"), (&key[layer]).into()));
        inputs.push((format!("v_cache_layer_{layer:02}"), (&value[layer]).into()));
    }
    Ok(session.run(inputs)?)
}

fn take_cache_layers(
    outputs: &mut ort::session::SessionOutputs<'_>,
    prefix: &str,
) -> Result<Vec<DynValue>> {
    (0..GPT_STEP_CACHE_LAYERS)
        .map(|layer| take(outputs, &format!("{prefix}_{layer:02}")))
        .collect()
}

fn run_gpt_step_cpu(
    session: &mut Session,
    sample: Tensor<i64>,
    idx: Tensor<i64>,
    x_len: &DynValue,
    y_len: &DynValue,
    cache: &GptKvCache,
    layout: GptStepCacheLayout,
) -> Result<(DynValue, DynValue, GptKvCache)> {
    let mut outputs = match cache {
        GptKvCache::Abi1Packed { key, value } => session.run(ort::inputs! {
            "samples" => sample,
            "k_cache" => key,
            "v_cache" => value,
            "x_len" => x_len,
            "y_len" => y_len,
            "idx" => idx
        })?,
        GptKvCache::V2PerLayer { key, value } => {
            run_gpt_step_v2(session, sample, idx, x_len, y_len, key, value)?
        }
        GptKvCache::CudaAbi1PingPong { .. } => {
            return Err(invalid("CUDA KV cache reached the CPU Session::run path"));
        }
        GptKvCache::CudaV2Initial { .. } | GptKvCache::CudaV2PerLayer { .. } => {
            return Err(invalid(
                "CUDA GPT Step v2 cache reached the CPU Session::run path",
            ));
        }
        #[cfg(feature = "r2-trt-native")]
        GptKvCache::TensorRtNative => {
            return Err(invalid(
                "native TensorRT cache reached the ORT Session::run path",
            ));
        }
    };
    let values = take(&mut outputs, "topk_values")?;
    let ids = take(&mut outputs, "topk_indices")?;
    let cache = match layout {
        GptStepCacheLayout::Abi1Packed => GptKvCache::Abi1Packed {
            key: take(&mut outputs, "k_cache_new")?,
            value: take(&mut outputs, "v_cache_new")?,
        },
        GptStepCacheLayout::V2PerLayer => GptKvCache::V2PerLayer {
            key: take_cache_layers(&mut outputs, "k_cache_new_layer")?,
            value: take_cache_layers(&mut outputs, "v_cache_new_layer")?,
        },
    };
    Ok((values, ids, cache))
}

fn pick(values: &DynValue, indices: &DynValue, state: &mut u64) -> Result<i64> {
    let (_, v) = values.try_extract_tensor::<f32>()?;
    let (_, ids) = indices.try_extract_tensor::<i64>()?;
    pick_topk(v, ids, state)
}

pub(crate) fn pick_topk(v: &[f32], ids: &[i64], state: &mut u64) -> Result<i64> {
    if v.is_empty() || v.len() != ids.len() || v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid semantic logits"));
    }
    let count = 15.min(v.len());
    let peak = v[..count].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let weights: Vec<f64> = v[..count]
        .iter()
        .map(|x| (*x as f64 - peak).exp())
        .collect();
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let target = (*state >> 11) as f64 / ((1u64 << 53) as f64) * weights.iter().sum::<f64>();
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return Ok(ids[i]);
        }
    }
    Ok(ids[count - 1])
}
