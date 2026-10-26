use crate::{
    audio, check,
    frontend::Frontend,
    invalid,
    profiling::{ProfileCollector, ProfilePhase},
    streaming::{StreamDecoder, CHUNK_TOKENS},
    validate_pack, validate_streaming_pack, Backend, Config, Pack, Result,
};
#[cfg(feature = "r1-cuda-ep")]
use ort::value::DynTensorValueType;
#[cfg(feature = "r1-cuda-ep")]
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::IoBinding,
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

#[derive(Debug, serde::Serialize)]
pub struct SynthesisTimings {
    pub frontend_ms: f64,
    pub gpt_ms: f64,
    pub vocoder_ms: f64,
    pub semantic_tokens: usize,
    pub first_audio_tokens: usize,
    pub audio_chunks: usize,
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
    V2PerLayer {
        key: Vec<DynValue>,
        value: Vec<DynValue>,
    },
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

fn open_path(path: &Path, name: &str, backend: Backend) -> Result<Session> {
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
            "gpt_step" => true,
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
            "gpt_step" => true,
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
    let mut builder = Session::builder()?
        .with_intra_threads(intra_threads)
        .map_err(|e| invalid(e.to_string()))?
        .with_inter_threads(1)
        .map_err(|e| invalid(e.to_string()))?;
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
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_tf32(false)
                .build()
                .error_on_failure()])
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
    builder.commit_from_file(path).map_err(|error| {
        if strict_gpu_placement {
            invalid(format!(
                "strict GPU placement rejected CPU-assigned nodes in {name}: {error}"
            ))
        } else {
            invalid(error.to_string())
        }
    })
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
    backend
}

fn gpt_backend(backend: Backend) -> Backend {
    match backend {
        #[cfg(feature = "r1-cuda-ep")]
        Backend::Cuda => Backend::Cuda,
        #[cfg(feature = "r1-tensorrt-ep")]
        Backend::TensorRt => Backend::TensorRt,
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
        "bert" | "g2pw" | "gpt_encoder" | "gpt_step" | "sovits" | "sovits_stream"
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
fn take(out: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<DynValue> {
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
}

#[cfg(feature = "r1-cuda-ep")]
impl GpuStepBinding {
    fn new(session: &Session) -> Result<Self> {
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
        binding.bind_output_to_device("k_cache_new", &cuda_output)?;
        binding.bind_output_to_device("v_cache_new", &cuda_output)?;
        Ok(Self { binding })
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
) -> Result<(DynValue, DynValue, DynValue, DynValue)> {
    ensure_cuda_value(key, "gpt_step.k_cache input")?;
    ensure_cuda_value(value, "gpt_step.v_cache input")?;

    let io = &mut binding.binding;
    // Keep this binding's device-side output buffers alive and reusable.
    // Only the small CPU inputs change from token to token.
    io.clear_inputs();
    io.bind_input("samples", &sample)?;
    io.bind_input("idx", &idx)?;
    io.bind_input("x_len", x_len)?;
    io.bind_input("y_len", y_len)?;
    io.bind_input("k_cache", key)?;
    io.bind_input("v_cache", value)?;
    let mut outputs = session.run_binding(io)?;
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
    Ok((topk_values, topk_indices, key_cache, value_cache))
}

pub struct NativeTts {
    model_dir: PathBuf,
    vocoder_backend: Backend,
    vocoder_fallback_backend: Backend,
    cuda_vits: bool,
    cuda_gpt: bool,
    #[cfg(feature = "r1-cuda-ep")]
    cuda_step_bindings: Option<[GpuStepBinding; 2]>,
    warmed: bool,
    frontend: Frontend,
    enc: Session,
    step: Session,
    step_cache_layout: GptStepCacheLayout,
    sovits: Session,
    prompts: Tensor<i64>,
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
        validate_pack(&config.model_dir)?;
        if config.streaming {
            validate_streaming_pack(&config.model_dir)?;
        }
        check(cancel)?;
        let dir = &config.model_dir;
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
        let frontend_backend = non_gpt_backend(config.backend);
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
        let mut frontend = Frontend::load(dir, frontend_backend)?;
        check(cancel)?;
        let (reference_phones, reference_features) = frontend.process(reference_text, cancel)?;
        let mut ssl = open(dir, "ssl")?;
        check(cancel)?;
        let mut vq = open(dir, "vq_encoder")?;
        check(cancel)?;
        let mut spectrogram = open(dir, "spectrogram")?;
        check(cancel)?;
        let mut speaker = open(dir, "sv_embedding")?;
        check(cancel)?;
        let mut padded = wav16.clone();
        padded.resize(padded.len() + 4800, 0.0);
        let mut ssl_out = ssl.run(ort::inputs! {"audio"=>tf(vec![1,padded.len()],padded)?})?;
        let content = take(&mut ssl_out, "last_hidden_state")?;
        check(cancel)?;
        let vq_out = vq.run(ort::inputs! {"ssl_content"=>&content})?;
        let (_, codes) = vq_out
            .get("codes")
            .ok_or_else(|| invalid("incompatible VQ output"))?
            .try_extract_tensor::<i64>()?;
        let prompt_tokens = codes.len();
        let prompts = ti(vec![1, prompt_tokens], codes.to_vec())?;
        let spec = take(
            &mut spectrogram.run(ort::inputs! {"audio"=>tf(vec![1,wav32.len()],wav32)?})?,
            "spectrogram",
        )?;
        check(cancel)?;
        let sv = take(
            &mut speaker.run(ort::inputs! {"audio"=>tf(vec![1,wav16.len()],wav16)?})?,
            "sv_embedding",
        )?;
        check(cancel)?;
        // Reference-only sessions can release their weights before GPT/VITS load.
        drop(vq_out);
        drop(ssl_out);
        drop(content);
        drop((ssl, vq, spectrogram, speaker));
        let encoder_backend = gpt_backend(config.backend);
        let enc = open_on(dir, "gpt_encoder", encoder_backend)?;
        check(cancel)?;
        let (step, step_cache_layout) = open_step(dir, config.backend)?;
        let cuda_gpt = is_cuda_backend(config.backend);
        if cuda_gpt && step_cache_layout != GptStepCacheLayout::Abi1Packed {
            return Err(invalid(
                "CUDA-resident KV cache currently requires the ABI-1 packed GPT Step graph",
            ));
        }
        #[cfg(feature = "r1-cuda-ep")]
        let cuda_step_bindings = if cuda_gpt {
            Some([GpuStepBinding::new(&step)?, GpuStepBinding::new(&step)?])
        } else {
            None
        };
        check(cancel)?;
        let sovits = open_on(
            dir,
            if config.streaming {
                "sovits_stream"
            } else {
                "sovits"
            },
            vocoder_backend,
        )?;
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
            warmed: false,
            frontend,
            enc,
            step,
            step_cache_layout,
            sovits,
            prompts,
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
        paths.push(PathBuf::from(self.enc.end_profiling()?));
        paths.push(PathBuf::from(self.step.end_profiling()?));
        paths.push(PathBuf::from(self.sovits.end_profiling()?));
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
        self.sovits = next;
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
            &mut profile,
            emit,
        )?;
        Ok((timings, profile.finish()))
    }

    fn synthesize_streaming_inner(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        first_chunk_tokens: usize,
        chunk_tokens: usize,
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
        let phoneme_ids =
            profile.measure(ProfilePhase::GptEncoderInput, || ti(vec![1, n], all_phones))?;
        let bert_feature = profile.measure(ProfilePhase::GptEncoderInput, || {
            tf(vec![1, 1024, n], all_features)
        })?;
        let (mut values, mut ids, x_len, y_len, mut kv_cache) = if self.cuda_gpt {
            #[cfg(feature = "r1-cuda-ep")]
            {
                let (values, ids, x_len, y_len, k, v) =
                    profile.measure(ProfilePhase::GptEncoderRun, || {
                        run_gpt_encoder_cuda(
                            &mut self.enc,
                            phoneme_ids,
                            &self.prompts,
                            bert_feature,
                        )
                    })?;
                validate_gpt_lengths(&x_len, &y_len, n, self.prompt_tokens)?;
                let cache = GptKvCache::CudaAbi1PingPong {
                    key: k,
                    value: v,
                    next_binding: 0,
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
            let mut out = profile.measure(ProfilePhase::GptEncoderRun, || {
                self.enc.run(ort::inputs! {
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
        let mut random = 1234u64;
        let mut tokens = Vec::new();
        let mut eos = false;
        let mut decoder = StreamDecoder::default();
        let cuda_vits_active = self.cuda_vits_active();
        let mut vocoder_ms = 0.0;
        let mut delivery_ms = 0.0;
        let mut first_audio_tokens = 0;
        let mut audio_chunks = 0;
        let mut next_audio_tokens = first_chunk_tokens;
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
                let (next_values, next_ids, next_key, next_value) =
                    profile.measure_gpt_step(|| {
                        run_gpt_step_cuda(
                            &mut self.step,
                            &mut bindings[slot],
                            sample,
                            idx_input,
                            &x_len,
                            &y_len,
                            key,
                            value,
                        )
                    })?;
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
                        &mut self.step,
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
                        &mut self.step,
                        sample,
                        idx_input,
                        &x_len,
                        &y_len,
                        &kv_cache,
                        self.step_cache_layout,
                    )
                })?;
            }
            #[cfg(feature = "r0-profiling")]
            profile.add_gpt_step();
            if self.streaming && tokens.len() >= next_audio_tokens {
                let decode_started = Instant::now();
                let first_vits_event = (audio_chunks == 0).then(|| profile.start_event()).flatten();
                let audio = decoder.decode(
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
                profile.finish_event("first_vits", first_vits_event, Some(audio_chunks));
                vocoder_ms += decode_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
                if first_audio_tokens == 0 {
                    first_audio_tokens = tokens.len();
                }
                audio_chunks += 1;
                let delivery_started = Instant::now();
                profile.measure(ProfilePhase::AudioCallback, || {
                    emit(
                        &audio,
                        ChunkInfo {
                            semantic_tokens: tokens.len(),
                            final_chunk: false,
                        },
                    )
                });
                profile.mark_first_audio();
                delivery_ms += delivery_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
                next_audio_tokens = next_audio_tokens.saturating_add(chunk_tokens);
            }
        }
        check(cancel)?;
        if !eos || tokens.is_empty() {
            return Err(invalid(
                "semantic generation did not finish within the cache budget; shorten the sentence",
            ));
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
