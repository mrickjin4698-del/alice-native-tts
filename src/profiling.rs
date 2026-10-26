//! Opt-in profiling support used by the R0 benchmark example.
//! The normal build does not enable ORT tracing or collect stage timings.

use std::cell::Cell;

#[cfg(feature = "r0-profiling")]
use std::time::Instant;

#[cfg(feature = "r0-profiling")]
use serde::Serialize;

const PHASE_COUNT: usize = 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ProfilePhase {
    Unscoped = 0,
    Frontend = 1,
    GptEncoderInput = 2,
    GptEncoderRun = 3,
    GptEncoderOutput = 4,
    GptStepInput = 5,
    GptStepRun = 6,
    GptStepOutput = 7,
    SemanticSampling = 8,
    VitsInput = 9,
    VitsRun = 10,
    VitsOutput = 11,
    HostDataCopy = 12,
    AudioCallback = 13,
    DeviceSync = 14,
    FrontendNormalize = 15,
    FrontendRouting = 16,
    FrontendTokenizer = 17,
    FrontendG2pwPrepare = 18,
    FrontendG2pwRun = 19,
    FrontendPhonemeMap = 20,
    FrontendBertInput = 21,
    FrontendBertRun = 22,
    FrontendFeatureExpand = 23,
    FrontendEnglishG2p = 24,
    FrontendJoin = 25,
    Sola = 26,
    FrontendCacheLookup = 27,
    FrontendCacheStore = 28,
    FrontendParallelRun = 29,
}

impl ProfilePhase {
    pub const ALL: [Self; PHASE_COUNT] = [
        Self::Unscoped,
        Self::Frontend,
        Self::GptEncoderInput,
        Self::GptEncoderRun,
        Self::GptEncoderOutput,
        Self::GptStepInput,
        Self::GptStepRun,
        Self::GptStepOutput,
        Self::SemanticSampling,
        Self::VitsInput,
        Self::VitsRun,
        Self::VitsOutput,
        Self::HostDataCopy,
        Self::AudioCallback,
        Self::DeviceSync,
        Self::FrontendNormalize,
        Self::FrontendRouting,
        Self::FrontendTokenizer,
        Self::FrontendG2pwPrepare,
        Self::FrontendG2pwRun,
        Self::FrontendPhonemeMap,
        Self::FrontendBertInput,
        Self::FrontendBertRun,
        Self::FrontendFeatureExpand,
        Self::FrontendEnglishG2p,
        Self::FrontendJoin,
        Self::Sola,
        Self::FrontendCacheLookup,
        Self::FrontendCacheStore,
        Self::FrontendParallelRun,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Unscoped => "unscoped",
            Self::Frontend => "frontend",
            Self::GptEncoderInput => "gpt_encoder_input",
            Self::GptEncoderRun => "gpt_encoder_run",
            Self::GptEncoderOutput => "gpt_encoder_output",
            Self::GptStepInput => "gpt_step_input",
            Self::GptStepRun => "gpt_step_run",
            Self::GptStepOutput => "gpt_step_output",
            Self::SemanticSampling => "semantic_sampling",
            Self::VitsInput => "vits_input",
            Self::VitsRun => "vits_run",
            Self::VitsOutput => "vits_output",
            Self::HostDataCopy => "host_data_copy",
            Self::AudioCallback => "audio_callback",
            Self::DeviceSync => "device_sync",
            Self::FrontendNormalize => "frontend_normalize",
            Self::FrontendRouting => "frontend_routing",
            Self::FrontendTokenizer => "frontend_tokenizer",
            Self::FrontendG2pwPrepare => "frontend_g2pw_prepare",
            Self::FrontendG2pwRun => "frontend_g2pw_run",
            Self::FrontendPhonemeMap => "frontend_phoneme_map",
            Self::FrontendBertInput => "frontend_bert_input",
            Self::FrontendBertRun => "frontend_bert_run",
            Self::FrontendFeatureExpand => "frontend_feature_expand",
            Self::FrontendEnglishG2p => "frontend_english_g2p",
            Self::FrontendJoin => "frontend_join",
            Self::Sola => "sola",
            Self::FrontendCacheLookup => "frontend_cache_lookup",
            Self::FrontendCacheStore => "frontend_cache_store",
            Self::FrontendParallelRun => "frontend_parallel_run",
        }
    }
}

thread_local! {
    static CURRENT_PHASE: Cell<usize> = const { Cell::new(ProfilePhase::Unscoped as usize) };
}

pub(crate) struct PhaseGuard(usize);

impl PhaseGuard {
    fn enter(phase: ProfilePhase) -> Self {
        CURRENT_PHASE.with(|current| Self(current.replace(phase as usize)))
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        CURRENT_PHASE.with(|current| current.set(self.0));
    }
}

#[cfg(feature = "r0-profiling")]
mod allocation {
    use super::{ProfilePhase, PHASE_COUNT};
    use serde::Serialize;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static CAPTURE: AtomicBool = AtomicBool::new(false);
    static ALLOC_CALLS: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];
    static REALLOC_CALLS: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];
    static DEALLOC_CALLS: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];
    static ALLOCATED_BYTES: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];
    static FREED_BYTES: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];
    static ALLOCATOR_NS: [AtomicU64; PHASE_COUNT] = [const { AtomicU64::new(0) }; PHASE_COUNT];

    #[derive(Clone, Debug, Serialize)]
    pub struct PhaseAllocation {
        pub phase: &'static str,
        pub alloc_calls: u64,
        pub realloc_calls: u64,
        pub dealloc_calls: u64,
        pub allocated_bytes: u64,
        pub freed_bytes: u64,
        pub allocator_time_ms: f64,
    }

    #[derive(Clone, Debug, Serialize)]
    pub struct AllocationSnapshot {
        /// Rust global allocator only; ONNX Runtime's native C allocations are not included.
        pub phases: Vec<PhaseAllocation>,
    }

    pub fn enabled() -> bool {
        CAPTURE.load(Ordering::Relaxed)
    }

    pub fn reset() {
        for i in 0..PHASE_COUNT {
            ALLOC_CALLS[i].store(0, Ordering::Relaxed);
            REALLOC_CALLS[i].store(0, Ordering::Relaxed);
            DEALLOC_CALLS[i].store(0, Ordering::Relaxed);
            ALLOCATED_BYTES[i].store(0, Ordering::Relaxed);
            FREED_BYTES[i].store(0, Ordering::Relaxed);
            ALLOCATOR_NS[i].store(0, Ordering::Relaxed);
        }
    }

    pub fn set_enabled(enabled: bool) {
        CAPTURE.store(enabled, Ordering::Relaxed);
    }

    pub fn record_alloc(phase: usize, bytes: usize, ns: u64) {
        let phase = phase.min(PHASE_COUNT - 1);
        ALLOC_CALLS[phase].fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES[phase].fetch_add(bytes as u64, Ordering::Relaxed);
        ALLOCATOR_NS[phase].fetch_add(ns, Ordering::Relaxed);
    }

    pub fn record_realloc(phase: usize, old_bytes: usize, new_bytes: usize, ns: u64) {
        let phase = phase.min(PHASE_COUNT - 1);
        REALLOC_CALLS[phase].fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES[phase].fetch_add(new_bytes as u64, Ordering::Relaxed);
        FREED_BYTES[phase].fetch_add(old_bytes as u64, Ordering::Relaxed);
        ALLOCATOR_NS[phase].fetch_add(ns, Ordering::Relaxed);
    }

    pub fn record_dealloc(phase: usize, bytes: usize, ns: u64) {
        let phase = phase.min(PHASE_COUNT - 1);
        DEALLOC_CALLS[phase].fetch_add(1, Ordering::Relaxed);
        FREED_BYTES[phase].fetch_add(bytes as u64, Ordering::Relaxed);
        ALLOCATOR_NS[phase].fetch_add(ns, Ordering::Relaxed);
    }

    pub fn current_phase() -> usize {
        super::CURRENT_PHASE.with(|phase| phase.get())
    }

    pub fn snapshot() -> AllocationSnapshot {
        use Ordering::Relaxed;
        AllocationSnapshot {
            phases: ProfilePhase::ALL
                .iter()
                .enumerate()
                .map(|(i, phase)| PhaseAllocation {
                    phase: phase.label(),
                    alloc_calls: ALLOC_CALLS[i].load(Relaxed),
                    realloc_calls: REALLOC_CALLS[i].load(Relaxed),
                    dealloc_calls: DEALLOC_CALLS[i].load(Relaxed),
                    allocated_bytes: ALLOCATED_BYTES[i].load(Relaxed),
                    freed_bytes: FREED_BYTES[i].load(Relaxed),
                    allocator_time_ms: ALLOCATOR_NS[i].load(Relaxed) as f64 / 1_000_000.0,
                })
                .collect(),
        }
    }
}

#[cfg(feature = "r0-profiling")]
pub use allocation::{AllocationSnapshot, PhaseAllocation};

#[cfg(feature = "r0-profiling")]
pub fn reset_allocation_stats() {
    allocation::set_enabled(false);
    allocation::reset();
}

#[cfg(feature = "r0-profiling")]
pub fn enable_allocation_capture(enabled: bool) {
    allocation::set_enabled(enabled);
}

#[cfg(feature = "r0-profiling")]
pub fn allocation_snapshot() -> AllocationSnapshot {
    allocation::snapshot()
}

#[cfg(feature = "r0-profiling")]
#[doc(hidden)]
pub fn allocation_capture_enabled() -> bool {
    allocation::enabled()
}

#[cfg(feature = "r0-profiling")]
#[doc(hidden)]
pub fn record_alloc(bytes: usize, ns: u64) {
    allocation::record_alloc(allocation::current_phase(), bytes, ns);
}

#[cfg(feature = "r0-profiling")]
#[doc(hidden)]
pub fn record_realloc(old_bytes: usize, new_bytes: usize, ns: u64) {
    allocation::record_realloc(allocation::current_phase(), old_bytes, new_bytes, ns);
}

#[cfg(feature = "r0-profiling")]
#[doc(hidden)]
pub fn record_dealloc(bytes: usize, ns: u64) {
    allocation::record_dealloc(allocation::current_phase(), bytes, ns);
}

#[cfg(feature = "r0-profiling")]
#[derive(Clone, Debug, Serialize)]
pub struct PhaseTiming {
    pub phase: &'static str,
    pub elapsed_ms: f64,
}

#[cfg(feature = "r0-profiling")]
#[derive(Clone, Debug, Default, Serialize)]
pub struct R0Timings {
    pub phases: Vec<PhaseTiming>,
    /// Request-relative spans for each profiled stage and autoregressive token.
    pub timeline: Vec<TimelineEvent>,
    pub input_copy_bytes: u64,
    pub output_copy_bytes: u64,
    pub gpt_step_iterations: u64,
    pub gpt_step_token_ms: Vec<f64>,
    pub cuda_kv_cache_steps: u64,
    pub cuda_kv_cache_device_endpoint_assertions: u64,
    pub vits_calls: u64,
    pub vits_call_run_ms: Vec<f64>,
    pub vits_call_output_ms: Vec<f64>,
    pub vits_call_device_sync_ms: Vec<f64>,
    /// Static reference/text condition payload bound once per CUDA VITS utterance.
    /// This is not a measured DMA counter.
    pub vits_static_condition_payload_bytes: u64,
    /// Dynamic semantic/result/overlap input payload across CUDA VITS calls.
    /// This is not a measured DMA counter.
    pub vits_dynamic_input_payload_bytes: u64,
    /// Sum of latent and audio output payload sizes returned from VITS.
    pub vits_output_tensor_payload_bytes: u64,
    pub semantic_tokens: usize,
    pub semantic_token_hash: u64,
    pub frontend_cache_hits: u64,
    pub frontend_cache_misses: u64,
}

#[cfg(feature = "r0-profiling")]
#[derive(Clone, Debug, Serialize)]
pub struct TimelineEvent {
    pub event: String,
    pub start_ms: f64,
    pub end_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

pub(crate) struct ProfileCollector {
    enabled: bool,
    #[cfg(feature = "r0-profiling")]
    elapsed_ns: [u64; PHASE_COUNT],
    #[cfg(feature = "r0-profiling")]
    input_copy_bytes: u64,
    #[cfg(feature = "r0-profiling")]
    output_copy_bytes: u64,
    #[cfg(feature = "r0-profiling")]
    gpt_step_iterations: u64,
    #[cfg(feature = "r0-profiling")]
    gpt_step_token_ns: Vec<u64>,
    #[cfg(feature = "r0-profiling")]
    cuda_kv_cache_steps: u64,
    #[cfg(feature = "r0-profiling")]
    cuda_kv_cache_device_endpoint_assertions: u64,
    #[cfg(feature = "r0-profiling")]
    vits_calls: u64,
    #[cfg(feature = "r0-profiling")]
    vits_call_run_ns: Vec<u64>,
    #[cfg(feature = "r0-profiling")]
    vits_call_output_ns: Vec<u64>,
    #[cfg(feature = "r0-profiling")]
    vits_call_device_sync_ns: Vec<u64>,
    #[cfg(feature = "r0-profiling")]
    vits_static_condition_payload_bytes: u64,
    #[cfg(feature = "r0-profiling")]
    vits_dynamic_input_payload_bytes: u64,
    #[cfg(feature = "r0-profiling")]
    vits_output_tensor_payload_bytes: u64,
    #[cfg(feature = "r0-profiling")]
    semantic_tokens: usize,
    #[cfg(feature = "r0-profiling")]
    semantic_token_hash: u64,
    #[cfg(feature = "r0-profiling")]
    frontend_cache_hits: u64,
    #[cfg(feature = "r0-profiling")]
    frontend_cache_misses: u64,
    #[cfg(feature = "r0-profiling")]
    request_started: Instant,
    #[cfg(feature = "r0-profiling")]
    timeline: Vec<TimelineEvent>,
    #[cfg(feature = "r0-profiling")]
    timeline_stopped: bool,
}

impl ProfileCollector {
    #[cfg(feature = "r0-profiling")]
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            #[cfg(feature = "r0-profiling")]
            elapsed_ns: [0; PHASE_COUNT],
            #[cfg(feature = "r0-profiling")]
            input_copy_bytes: 0,
            #[cfg(feature = "r0-profiling")]
            output_copy_bytes: 0,
            #[cfg(feature = "r0-profiling")]
            gpt_step_iterations: 0,
            #[cfg(feature = "r0-profiling")]
            gpt_step_token_ns: Vec::new(),
            #[cfg(feature = "r0-profiling")]
            cuda_kv_cache_steps: 0,
            #[cfg(feature = "r0-profiling")]
            cuda_kv_cache_device_endpoint_assertions: 0,
            #[cfg(feature = "r0-profiling")]
            vits_calls: 0,
            #[cfg(feature = "r0-profiling")]
            vits_call_run_ns: Vec::new(),
            #[cfg(feature = "r0-profiling")]
            vits_call_output_ns: Vec::new(),
            #[cfg(feature = "r0-profiling")]
            vits_call_device_sync_ns: Vec::new(),
            #[cfg(feature = "r0-profiling")]
            vits_static_condition_payload_bytes: 0,
            #[cfg(feature = "r0-profiling")]
            vits_dynamic_input_payload_bytes: 0,
            #[cfg(feature = "r0-profiling")]
            vits_output_tensor_payload_bytes: 0,
            #[cfg(feature = "r0-profiling")]
            semantic_tokens: 0,
            #[cfg(feature = "r0-profiling")]
            semantic_token_hash: 0,
            #[cfg(feature = "r0-profiling")]
            frontend_cache_hits: 0,
            #[cfg(feature = "r0-profiling")]
            frontend_cache_misses: 0,
            #[cfg(feature = "r0-profiling")]
            request_started: Instant::now(),
            #[cfg(feature = "r0-profiling")]
            timeline: Vec::new(),
            #[cfg(feature = "r0-profiling")]
            timeline_stopped: false,
        }
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn enabled() -> Self {
        Self {
            enabled: true,
            elapsed_ns: [0; PHASE_COUNT],
            input_copy_bytes: 0,
            output_copy_bytes: 0,
            gpt_step_iterations: 0,
            gpt_step_token_ns: Vec::new(),
            cuda_kv_cache_steps: 0,
            cuda_kv_cache_device_endpoint_assertions: 0,
            vits_calls: 0,
            vits_call_run_ns: Vec::new(),
            vits_call_output_ns: Vec::new(),
            vits_call_device_sync_ns: Vec::new(),
            vits_static_condition_payload_bytes: 0,
            vits_dynamic_input_payload_bytes: 0,
            vits_output_tensor_payload_bytes: 0,
            semantic_tokens: 0,
            semantic_token_hash: 0,
            frontend_cache_hits: 0,
            frontend_cache_misses: 0,
            request_started: Instant::now(),
            timeline: Vec::new(),
            timeline_stopped: false,
        }
    }

    pub(crate) fn measure<T>(&mut self, phase: ProfilePhase, f: impl FnOnce() -> T) -> T {
        if !self.enabled {
            return f();
        }
        let _guard = PhaseGuard::enter(phase);
        #[cfg(feature = "r0-profiling")]
        let started = Instant::now();
        let output = f();
        #[cfg(feature = "r0-profiling")]
        {
            let ended = Instant::now();
            self.elapsed_ns[phase as usize] += ended.duration_since(started).as_nanos() as u64;
            let index = match phase {
                ProfilePhase::SemanticSampling
                | ProfilePhase::GptStepInput
                | ProfilePhase::GptStepRun
                | ProfilePhase::GptStepOutput => Some(self.gpt_step_iterations as usize),
                ProfilePhase::VitsInput
                | ProfilePhase::VitsRun
                | ProfilePhase::VitsOutput
                | ProfilePhase::DeviceSync
                | ProfilePhase::HostDataCopy
                | ProfilePhase::Sola => self.vits_calls.checked_sub(1).map(|n| n as usize),
                _ => None,
            };
            if !self.timeline_stopped {
                self.timeline.push(TimelineEvent {
                    event: phase.label().to_owned(),
                    start_ms: started.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    end_ms: ended.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    index,
                });
            }
        }
        output
    }

    pub(crate) fn record_external_event(
        &mut self,
        phase: ProfilePhase,
        started: std::time::Instant,
        ended: std::time::Instant,
    ) {
        #[cfg(not(feature = "r0-profiling"))]
        let _ = (phase, started, ended);
        if !self.enabled {
            return;
        }
        #[cfg(feature = "r0-profiling")]
        {
            self.elapsed_ns[phase as usize] += ended.duration_since(started).as_nanos() as u64;
            if !self.timeline_stopped {
                self.timeline.push(TimelineEvent {
                    event: phase.label().to_owned(),
                    start_ms: started.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    end_ms: ended.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    index: None,
                });
            }
        }
    }

    pub(crate) fn measure_with_profile<T>(
        &mut self,
        phase: ProfilePhase,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        if !self.enabled {
            return f(self);
        }
        let _guard = PhaseGuard::enter(phase);
        #[cfg(feature = "r0-profiling")]
        let started = Instant::now();
        let output = f(self);
        #[cfg(feature = "r0-profiling")]
        {
            let ended = Instant::now();
            self.elapsed_ns[phase as usize] += ended.duration_since(started).as_nanos() as u64;
            if !self.timeline_stopped {
                self.timeline.push(TimelineEvent {
                    event: phase.label().to_owned(),
                    start_ms: started.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    end_ms: ended.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    index: None,
                });
            }
        }
        output
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn start_event(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[cfg(not(feature = "r0-profiling"))]
    pub(crate) fn start_event(&self) -> Option<std::time::Instant> {
        None
    }

    pub(crate) fn finish_event(
        &mut self,
        event: &str,
        started: Option<std::time::Instant>,
        index: Option<usize>,
    ) {
        #[cfg(feature = "r0-profiling")]
        if let Some(started) = started {
            let ended = Instant::now();
            if !self.timeline_stopped {
                self.timeline.push(TimelineEvent {
                    event: event.to_owned(),
                    start_ms: started.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    end_ms: ended.duration_since(self.request_started).as_secs_f64() * 1000.0,
                    index,
                });
            }
        }
        #[cfg(not(feature = "r0-profiling"))]
        let _ = (event, started, index);
    }

    pub(crate) fn mark_first_audio(&mut self) {
        #[cfg(feature = "r0-profiling")]
        if self.enabled {
            self.timeline_stopped = true;
        }
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn add_input_copy_bytes(&mut self, bytes: usize) {
        self.input_copy_bytes = self.input_copy_bytes.saturating_add(bytes as u64);
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn add_output_copy_bytes(&mut self, bytes: usize) {
        self.output_copy_bytes = self.output_copy_bytes.saturating_add(bytes as u64);
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn add_gpt_step(&mut self) {
        self.gpt_step_iterations = self.gpt_step_iterations.saturating_add(1);
    }

    #[cfg(all(feature = "r0-profiling", feature = "r1-cuda-ep"))]
    pub(crate) fn add_cuda_kv_cache_step(&mut self) {
        self.cuda_kv_cache_steps = self.cuda_kv_cache_steps.saturating_add(1);
        // Both cache inputs and outputs are checked for CUDA device 0 residency
        // on every bound Step invocation before values are handed to ORT.
        self.cuda_kv_cache_device_endpoint_assertions = self
            .cuda_kv_cache_device_endpoint_assertions
            .saturating_add(4);
    }

    pub(crate) fn measure_gpt_step<T>(&mut self, f: impl FnOnce() -> T) -> T {
        #[cfg(feature = "r0-profiling")]
        {
            if !self.enabled {
                return f();
            }
            let started = Instant::now();
            let output = self.measure(ProfilePhase::GptStepRun, f);
            self.gpt_step_token_ns
                .push(started.elapsed().as_nanos() as u64);
            output
        }
        #[cfg(not(feature = "r0-profiling"))]
        {
            self.measure(ProfilePhase::GptStepRun, f)
        }
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn add_vits_call(&mut self) {
        self.vits_calls = self.vits_calls.saturating_add(1);
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn record_vits_call_stages(
        &mut self,
        run_ns: u64,
        output_ns: u64,
        device_sync_ns: u64,
        static_condition_payload_bytes: usize,
        dynamic_input_payload_bytes: usize,
        output_payload_bytes: usize,
    ) {
        if !self.enabled {
            return;
        }
        self.vits_call_run_ns.push(run_ns);
        self.vits_call_output_ns.push(output_ns);
        self.vits_call_device_sync_ns.push(device_sync_ns);
        self.vits_static_condition_payload_bytes = self
            .vits_static_condition_payload_bytes
            .saturating_add(static_condition_payload_bytes as u64);
        self.vits_dynamic_input_payload_bytes = self
            .vits_dynamic_input_payload_bytes
            .saturating_add(dynamic_input_payload_bytes as u64);
        self.vits_output_tensor_payload_bytes = self
            .vits_output_tensor_payload_bytes
            .saturating_add(output_payload_bytes as u64);
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn record_semantic_tokens(&mut self, tokens: &[i64]) {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for token in tokens {
            for byte in token.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        self.semantic_tokens = tokens.len();
        self.semantic_token_hash = hash;
    }

    pub(crate) fn record_frontend_cache_hit(&mut self) {
        #[cfg(feature = "r0-profiling")]
        if self.enabled {
            self.frontend_cache_hits = self.frontend_cache_hits.saturating_add(1);
        }
    }

    pub(crate) fn record_frontend_cache_miss(&mut self) {
        #[cfg(feature = "r0-profiling")]
        if self.enabled {
            self.frontend_cache_misses = self.frontend_cache_misses.saturating_add(1);
        }
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn finish(mut self) -> R0Timings {
        self.timeline
            .sort_by(|left, right| left.start_ms.total_cmp(&right.start_ms));
        R0Timings {
            phases: ProfilePhase::ALL
                .iter()
                .map(|phase| PhaseTiming {
                    phase: phase.label(),
                    elapsed_ms: self.elapsed_ns[*phase as usize] as f64 / 1_000_000.0,
                })
                .collect(),
            timeline: self.timeline,
            input_copy_bytes: self.input_copy_bytes,
            output_copy_bytes: self.output_copy_bytes,
            gpt_step_iterations: self.gpt_step_iterations,
            gpt_step_token_ms: self
                .gpt_step_token_ns
                .into_iter()
                .map(|ns| ns as f64 / 1_000_000.0)
                .collect(),
            cuda_kv_cache_steps: self.cuda_kv_cache_steps,
            cuda_kv_cache_device_endpoint_assertions: self.cuda_kv_cache_device_endpoint_assertions,
            vits_calls: self.vits_calls,
            vits_call_run_ms: self
                .vits_call_run_ns
                .into_iter()
                .map(|ns| ns as f64 / 1_000_000.0)
                .collect(),
            vits_call_output_ms: self
                .vits_call_output_ns
                .into_iter()
                .map(|ns| ns as f64 / 1_000_000.0)
                .collect(),
            vits_call_device_sync_ms: self
                .vits_call_device_sync_ns
                .into_iter()
                .map(|ns| ns as f64 / 1_000_000.0)
                .collect(),
            vits_static_condition_payload_bytes: self.vits_static_condition_payload_bytes,
            vits_dynamic_input_payload_bytes: self.vits_dynamic_input_payload_bytes,
            vits_output_tensor_payload_bytes: self.vits_output_tensor_payload_bytes,
            semantic_tokens: self.semantic_tokens,
            semantic_token_hash: self.semantic_token_hash,
            frontend_cache_hits: self.frontend_cache_hits,
            frontend_cache_misses: self.frontend_cache_misses,
        }
    }
}
