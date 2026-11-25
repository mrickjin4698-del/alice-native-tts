//! v2Pro latent overlap and SOLA stitching, independent for each utterance.
use crate::{
    check,
    inference::{tf, ti},
    invalid,
    profiling::{ProfileCollector, ProfilePhase},
    Result,
};
use ort::{
    memory::Allocator,
    session::{Session, SessionOutputs},
    value::{DynValue, Tensor},
};
#[cfg(feature = "r1-cuda-ep")]
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::IoBinding,
};

pub(crate) const CHUNK_TOKENS: usize = 24;
const CHANNELS: usize = 192;
const SAMPLES_PER_TOKEN: usize = 1280;
const OVERLAP_TOKENS: usize = 2;
const OVERLAP_SAMPLES: usize = OVERLAP_TOKENS * SAMPLES_PER_TOKEN;
const PREVIEW_MIN_PCM_SAMPLES: usize = 2_048; // 64 ms at 32 kHz
const PREVIEW_SILENCE_PEAK: f32 = 1e-5;
const SOLA_MIN_CORRELATION: f64 = 0.20;
const BOUNDED_JOIN_MIN_CORRELATION: f64 = 0.70;
const BOUNDED_JOIN_MIN_RMS_FLOOR: f64 = 1e-4;
const BOUNDED_JOIN_RELATIVE_RMS: f64 = 0.10;
const R1_9_VOICE_RMS_THRESHOLD: f64 = 1e-3;
const BOUNDED_JOIN_SEARCH_SAMPLES: usize = 320;
const BOUNDED_JOIN_COARSE_STEP: usize = 16;
const BOUNDED_JOIN_BOUNDARY_STEP: usize = 320;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct BoundedPreviewJoin {
    pub(crate) preview_boundary_samples: usize,
    pub(crate) refill_start_samples: usize,
    pub(crate) correlation: f64,
    pub(crate) preview_rms: f64,
    pub(crate) refill_rms: f64,
    pub(crate) minimum_rms_gate: f64,
    pub(crate) accepted: bool,
    pub(crate) fallback_reason: Option<&'static str>,
    pub(crate) seam_start_jump: Option<f64>,
    pub(crate) seam_end_jump: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SolaAlignment {
    pub(crate) best_offset_samples: usize,
    pub(crate) applied_offset_samples: usize,
    pub(crate) correlation: f64,
    pub(crate) previous_rms: f64,
    pub(crate) incoming_rms: f64,
    pub(crate) fallback_reason: Option<&'static str>,
}

#[derive(Default)]
pub(crate) struct StreamDecoder {
    latent: Vec<f32>,
    latent_frames: usize,
    tokens: usize,
    tail: Vec<f32>,
    text_seq: Option<Tensor<i64>>,
    #[cfg(feature = "r1-cuda-ep")]
    cuda_io_binding: Option<IoBinding>,
    preview_refill_mode: bool,
    last_sola_alignment: Option<SolaAlignment>,
}

#[cfg(feature = "r1-cuda-ep")]
fn new_cuda_vits_binding(
    session: &Session,
    spec: &DynValue,
    sv: &DynValue,
    text_seq: &Tensor<i64>,
) -> Result<IoBinding> {
    let mut binding = session.create_binding()?;
    // Reference-only inputs live with the binding for the whole utterance.
    binding.bind_input("refer_spec", spec)?;
    binding.bind_input("sv_emb", sv)?;
    binding.bind_input("text_seq", text_seq)?;
    Ok(binding)
}

#[cfg(feature = "r1-cuda-ep")]
fn bind_cuda_vits_cpu_outputs(binding: &mut IoBinding) -> Result<()> {
    // The streaming output length changes at the final chunk. Recreate the
    // dynamic CPU output allocations per invocation instead of reusing a
    // buffer with the previous chunk's shape.
    binding.clear_outputs();
    let cpu_output = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?;
    binding.bind_output_to_device("latent", &cpu_output)?;
    binding.bind_output_to_device("audio", &cpu_output)?;
    Ok(())
}

fn validate_streaming_outputs(outputs: &SessionOutputs<'_>, result_tokens: usize) -> Result<usize> {
    let (shape, latent) = outputs
        .get("latent")
        .ok_or_else(|| invalid("missing streaming latent output"))?
        .try_extract_tensor::<f32>()?;
    if shape.as_ref() != [1, CHANNELS as i64, (result_tokens * 2) as i64]
        || latent.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid streaming latent output"));
    }
    let (shape, samples) = outputs
        .get("audio")
        .ok_or_else(|| invalid("missing streaming audio output"))?
        .try_extract_tensor::<f32>()?;
    if shape.as_ref() != [1, 1, (result_tokens * SAMPLES_PER_TOKEN) as i64]
        || samples.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid streaming audio length/data"));
    }
    Ok((latent.len() + samples.len()) * std::mem::size_of::<f32>())
}

fn copy_streaming_outputs(
    outputs: &SessionOutputs<'_>,
    overlap_tokens: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (_, latent) = outputs
        .get("latent")
        .ok_or_else(|| invalid("missing streaming latent output"))?
        .try_extract_tensor::<f32>()?;
    let (_, samples) = outputs
        .get("audio")
        .ok_or_else(|| invalid("missing streaming audio output"))?
        .try_extract_tensor::<f32>()?;
    let latent = latent.to_vec();
    let extra = overlap_tokens.saturating_sub(OVERLAP_TOKENS) * SAMPLES_PER_TOKEN;
    let audio = samples[extra..].to_vec();
    Ok((latent, audio))
}

fn measure_profiled<T>(
    profile: &mut ProfileCollector,
    phase: ProfilePhase,
    f: impl FnOnce() -> Result<T>,
) -> Result<(T, u64)> {
    #[cfg(feature = "r0-profiling")]
    let started = profile.is_enabled().then(std::time::Instant::now);
    let output = profile.measure(phase, f)?;
    #[cfg(feature = "r0-profiling")]
    let elapsed_ns = started.map_or(0, |started| started.elapsed().as_nanos() as u64);
    #[cfg(not(feature = "r0-profiling"))]
    let elapsed_ns = 0;
    Ok((output, elapsed_ns))
}

impl StreamDecoder {
    pub(crate) fn release_session_binding(&mut self) {
        #[cfg(feature = "r1-cuda-ep")]
        {
            self.cuda_io_binding = None;
        }
    }

    pub(crate) fn enable_preview_refill_mode(&mut self) {
        self.preview_refill_mode = true;
    }

    pub(crate) fn reset_unemitted_preview(&mut self) {
        self.latent.clear();
        self.latent_frames = 0;
        self.tokens = 0;
        self.tail.clear();
        self.last_sola_alignment = None;
    }

    pub(crate) fn take_sola_alignment(&mut self) -> Option<SolaAlignment> {
        self.last_sola_alignment.take()
    }

    pub(crate) fn pending_audio_tail(&self) -> &[f32] {
        &self.tail
    }

    pub fn decode(
        &mut self,
        session: &mut Session,
        tokens: &[i64],
        phones: &[i64],
        spec: &DynValue,
        sv: &DynValue,
        final_chunk: bool,
        cuda_vits: bool,
        cancel: &impl Fn() -> bool,
        profile: &mut ProfileCollector,
    ) -> Result<Vec<f32>> {
        self.last_sola_alignment = None;
        check(cancel)?;
        let added = tokens
            .len()
            .checked_sub(self.tokens)
            .ok_or_else(|| invalid("streaming token count moved backwards"))?;
        if added == 0 {
            return if final_chunk {
                Ok(std::mem::take(&mut self.tail))
            } else {
                Err(invalid("empty non-final semantic chunk"))
            };
        }
        #[cfg(feature = "r0-profiling")]
        profile.add_vits_call();
        // Original decode_streaming keeps at least ten new/context tokens
        // for the vocoder when EOS leaves a short last chunk.
        let overlap = if self.tokens == 0 {
            0
        } else {
            OVERLAP_TOKENS + 10usize.saturating_sub(added)
        };
        let frames = overlap * 2;
        if frames > self.latent_frames {
            return Err(invalid("insufficient streaming latent context"));
        }
        let mut previous = profile.measure(ProfilePhase::HostDataCopy, || {
            let mut previous = Vec::with_capacity(CHANNELS * frames);
            for channel in 0..CHANNELS {
                let end = (channel + 1) * self.latent_frames;
                previous.extend_from_slice(&self.latent[end - frames..end]);
            }
            previous
        });
        #[cfg(feature = "r0-profiling")]
        profile.add_input_copy_bytes(previous.len() * std::mem::size_of::<f32>());
        #[cfg(feature = "r0-profiling")]
        let overlap_input_bytes = previous.len() * std::mem::size_of::<f32>();
        let result = added + overlap;
        // from_array rejects zero-sized dimensions; ORT's allocator supports
        // the empty context required by the first streaming chunk.
        let previous = profile.measure(ProfilePhase::VitsInput, || {
            if frames == 0 {
                Ok(Tensor::<f32>::new(
                    &Allocator::default(),
                    [1_i64, CHANNELS as i64, 0],
                )?)
            } else {
                tf(vec![1, CHANNELS, frames], std::mem::take(&mut previous))
            }
        })?;
        let semantic_input = profile.measure(ProfilePhase::HostDataCopy, || tokens.to_vec());
        if self.text_seq.is_none() {
            // The phone sequence is identical for every chunk of this request.
            let text_data = profile.measure(ProfilePhase::HostDataCopy, || phones.to_vec());
            self.text_seq = Some(profile.measure(ProfilePhase::VitsInput, || {
                ti(vec![1, phones.len()], text_data)
            })?);
            #[cfg(feature = "r0-profiling")]
            profile.add_input_copy_bytes(phones.len() * std::mem::size_of::<i64>());
        }
        #[cfg(feature = "r0-profiling")]
        profile.add_input_copy_bytes(semantic_input.len() * std::mem::size_of::<i64>());
        let (semantic_input, result_input) = profile.measure(ProfilePhase::VitsInput, || {
            Ok::<_, crate::Error>((
                ti(vec![1, 1, tokens.len()], semantic_input)?,
                ti(vec![1], vec![result as i64])?,
            ))
        })?;
        let text_seq = self
            .text_seq
            .as_ref()
            .ok_or_else(|| invalid("missing cached VITS text condition"))?;
        #[cfg(all(feature = "r1-cuda-ep", feature = "r0-profiling"))]
        let creating_cuda_vits_binding = cuda_vits && self.cuda_io_binding.is_none();
        #[cfg(feature = "r1-cuda-ep")]
        let (
            new_latent,
            audio,
            output_tensor_payload_bytes,
            vits_run_ns,
            device_sync_ns,
            vits_output_ns,
        ) = if cuda_vits {
            if self.cuda_io_binding.is_none() {
                self.cuda_io_binding = Some(new_cuda_vits_binding(session, spec, sv, text_seq)?);
            }
            let binding = self
                .cuda_io_binding
                .as_mut()
                .ok_or_else(|| invalid("missing CUDA VITS I/O Binding"))?;
            binding.bind_input("pred_semantic", &semantic_input)?;
            binding.bind_input("result_length", &result_input)?;
            binding.bind_input("overlap_latent", &previous)?;
            bind_cuda_vits_cpu_outputs(binding)?;
            let (out, vits_run_ns) = measure_profiled(profile, ProfilePhase::VitsRun, || {
                Ok(session.run_binding(binding)?)
            })?;
            let (_, device_sync_ns) = measure_profiled(profile, ProfilePhase::DeviceSync, || {
                binding.synchronize_outputs()?;
                Ok(())
            })?;
            let (output_tensor_payload_bytes, output_validation_ns) =
                measure_profiled(profile, ProfilePhase::VitsOutput, || {
                    validate_streaming_outputs(&out, result)
                })?;
            check(cancel)?;
            let ((new_latent, audio), output_copy_ns) =
                measure_profiled(profile, ProfilePhase::HostDataCopy, || {
                    copy_streaming_outputs(&out, overlap)
                })?;
            let vits_output_ns = output_validation_ns.saturating_add(output_copy_ns);
            (
                new_latent,
                audio,
                output_tensor_payload_bytes,
                vits_run_ns,
                device_sync_ns,
                vits_output_ns,
            )
        } else {
            let (out, vits_run_ns) = measure_profiled(profile, ProfilePhase::VitsRun, || {
                Ok(session.run(ort::inputs! {
                        "pred_semantic" => &semantic_input,
                        "text_seq" => text_seq,
                        "refer_spec" => spec,
                        "sv_emb" => sv,
                        "result_length" => &result_input,
                        "overlap_latent" => &previous
                })?)
            })?;
            let (output_tensor_payload_bytes, output_validation_ns) =
                measure_profiled(profile, ProfilePhase::VitsOutput, || {
                    validate_streaming_outputs(&out, result)
                })?;
            check(cancel)?;
            let ((new_latent, audio), output_copy_ns) =
                measure_profiled(profile, ProfilePhase::HostDataCopy, || {
                    copy_streaming_outputs(&out, overlap)
                })?;
            let vits_output_ns = output_validation_ns.saturating_add(output_copy_ns);
            (
                new_latent,
                audio,
                output_tensor_payload_bytes,
                vits_run_ns,
                0,
                vits_output_ns,
            )
        };
        #[cfg(not(feature = "r1-cuda-ep"))]
        let (
            new_latent,
            audio,
            output_tensor_payload_bytes,
            vits_run_ns,
            device_sync_ns,
            vits_output_ns,
        ) = {
            let _ = cuda_vits;
            let (out, vits_run_ns) = measure_profiled(profile, ProfilePhase::VitsRun, || {
                Ok(session.run(ort::inputs! {
                        "pred_semantic" => &semantic_input,
                        "text_seq" => text_seq,
                        "refer_spec" => spec,
                        "sv_emb" => sv,
                        "result_length" => &result_input,
                        "overlap_latent" => &previous
                })?)
            })?;
            let (output_tensor_payload_bytes, output_validation_ns) =
                measure_profiled(profile, ProfilePhase::VitsOutput, || {
                    validate_streaming_outputs(&out, result)
                })?;
            check(cancel)?;
            let ((new_latent, audio), output_copy_ns) =
                measure_profiled(profile, ProfilePhase::HostDataCopy, || {
                    copy_streaming_outputs(&out, overlap)
                })?;
            let vits_output_ns = output_validation_ns.saturating_add(output_copy_ns);
            (
                new_latent,
                audio,
                output_tensor_payload_bytes,
                vits_run_ns,
                0,
                vits_output_ns,
            )
        };
        #[cfg(not(feature = "r0-profiling"))]
        let _ = (
            output_tensor_payload_bytes,
            vits_run_ns,
            device_sync_ns,
            vits_output_ns,
        );
        #[cfg(feature = "r0-profiling")]
        if profile.is_enabled() {
            let output_copy_bytes = (new_latent.len() + audio.len()) * std::mem::size_of::<f32>();
            #[cfg(feature = "r1-cuda-ep")]
            let static_condition_payload_bytes = if creating_cuda_vits_binding {
                phones.len() * std::mem::size_of::<i64>()
                    + spec.try_extract_tensor::<f32>()?.1.len() * std::mem::size_of::<f32>()
                    + sv.try_extract_tensor::<f32>()?.1.len() * std::mem::size_of::<f32>()
            } else {
                0
            };
            #[cfg(not(feature = "r1-cuda-ep"))]
            let static_condition_payload_bytes = 0;
            let dynamic_input_payload_bytes = tokens.len() * std::mem::size_of::<i64>()
                + std::mem::size_of::<i64>()
                + overlap_input_bytes;
            profile.record_vits_call_stages(
                vits_run_ns,
                vits_output_ns,
                device_sync_ns,
                if cuda_vits {
                    static_condition_payload_bytes
                } else {
                    0
                },
                if cuda_vits {
                    dynamic_input_payload_bytes
                } else {
                    0
                },
                if cuda_vits {
                    output_tensor_payload_bytes
                } else {
                    0
                },
            );
            profile.add_output_copy_bytes(output_copy_bytes);
        }
        self.latent = new_latent;
        self.latent_frames = result * 2;
        self.tokens = tokens.len();
        let audio = profile.measure(ProfilePhase::Sola, || {
            let mut audio = audio;
            if !self.tail.is_empty() {
                if self.preview_refill_mode {
                    let (stitched, alignment) = sola_preview_refill(&self.tail, &audio);
                    self.last_sola_alignment = Some(alignment);
                    audio = stitched;
                } else {
                    audio = sola(&self.tail, &audio);
                }
            }
            self.tail.clear();
            if !final_chunk {
                if audio.len() <= OVERLAP_SAMPLES {
                    return Err(invalid("streaming chunk is shorter than its overlap"));
                }
                self.tail = audio.split_off(audio.len() - OVERLAP_SAMPLES);
            }
            Ok::<_, crate::Error>(audio)
        })?;
        Ok(audio)
    }
}

/// Align at most 10 ms and blend the withheld 80 ms with a periodic Hann
/// window, matching GPT-SoVITS' SOLA rule. Never repeat already-emitted PCM.
fn sola(tail: &[f32], audio: &[f32]) -> Vec<f32> {
    let overlap = tail.len();
    let search = 320.min(audio.len().saturating_sub(overlap));
    let mut best = f64::NEG_INFINITY;
    let mut offset = 0;
    for candidate in 0..=search {
        let segment = &audio[candidate..candidate + overlap];
        let dot: f64 = tail
            .iter()
            .zip(segment)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let energy: f64 = segment.iter().map(|x| f64::from(*x).powi(2)).sum();
        let score = dot / (energy + 1e-8).sqrt();
        if score > best {
            best = score;
            offset = candidate;
        }
    }
    crossfade_at_offset(tail, audio, offset)
}

fn sola_with_alignment(tail: &[f32], audio: &[f32]) -> (Vec<f32>, SolaAlignment) {
    let overlap = tail.len();
    let search = 320.min(audio.len().saturating_sub(overlap));
    let mut best = f64::NEG_INFINITY;
    let mut offset = 0;
    let tail_energy: f64 = tail.iter().map(|x| f64::from(*x).powi(2)).sum();
    let mut candidate_energy = 0.0;
    for candidate in 0..=search {
        let segment = &audio[candidate..candidate + overlap];
        let dot: f64 = tail
            .iter()
            .zip(segment)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let energy: f64 = segment.iter().map(|x| f64::from(*x).powi(2)).sum();
        let score = dot / (energy + 1e-8).sqrt();
        if score > best {
            best = score;
            offset = candidate;
            candidate_energy = energy;
        }
    }
    let correlation = best_cosine(tail_energy, candidate_energy, best, tail.len());
    let alignment = SolaAlignment {
        best_offset_samples: offset,
        applied_offset_samples: offset,
        correlation,
        previous_rms: rms_from_energy(tail_energy, tail.len()),
        incoming_rms: rms_from_energy(candidate_energy, overlap),
        fallback_reason: None,
    };
    (crossfade_at_offset(tail, audio, offset), alignment)
}

fn sola_preview_refill(tail: &[f32], audio: &[f32]) -> (Vec<f32>, SolaAlignment) {
    let (legacy_audio, mut alignment) = sola_with_alignment(tail, audio);
    let fallback_reason = if alignment.previous_rms < f64::from(PREVIEW_SILENCE_PEAK)
        && alignment.incoming_rms < f64::from(PREVIEW_SILENCE_PEAK)
    {
        Some("both_overlap_windows_silent")
    } else if alignment.previous_rms < f64::from(PREVIEW_SILENCE_PEAK)
        || alignment.incoming_rms < f64::from(PREVIEW_SILENCE_PEAK)
    {
        Some("one_overlap_window_silent")
    } else if !alignment.correlation.is_finite() || alignment.correlation < SOLA_MIN_CORRELATION {
        Some("low_waveform_correlation")
    } else {
        None
    };
    if let Some(reason) = fallback_reason {
        alignment.applied_offset_samples = 0;
        alignment.fallback_reason = Some(reason);
        (crossfade_at_offset(tail, audio, 0), alignment)
    } else {
        (legacy_audio, alignment)
    }
}

fn crossfade_at_offset(tail: &[f32], audio: &[f32], offset: usize) -> Vec<f32> {
    let overlap = tail.len();
    if offset.saturating_add(overlap) > audio.len() {
        return Vec::new();
    }
    let mut result = audio[offset..].to_vec();
    for (i, previous) in tail.iter().enumerate() {
        let weight = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / overlap as f64).cos();
        result[i] = (weight * f64::from(result[i]) + (1.0 - weight) * f64::from(*previous)) as f32;
    }
    result
}

fn best_cosine(tail_energy: f64, candidate_energy: f64, legacy_score: f64, len: usize) -> f64 {
    if tail_energy <= 0.0 || candidate_energy <= 0.0 || len == 0 {
        return 0.0;
    }
    // The legacy score is dot / sqrt(candidate_energy + eps). Recover its dot
    // product and normalize both windows only for diagnostics/preview gating.
    let dot = legacy_score * (candidate_energy + 1e-8).sqrt();
    dot / (tail_energy * candidate_energy).sqrt()
}

fn rms_from_energy(energy: f64, len: usize) -> f64 {
    if len == 0 {
        0.0
    } else {
        (energy / len as f64).sqrt()
    }
}

pub(crate) fn preview_fallback_reason(audio: &[f32]) -> Option<&'static str> {
    if audio.len() < PREVIEW_MIN_PCM_SAMPLES {
        return Some("preview_pcm_below_64ms");
    }
    if audio
        .iter()
        .fold(0.0f32, |peak, sample| peak.max(sample.abs()))
        < PREVIEW_SILENCE_PEAK
    {
        return Some("preview_pcm_silent");
    }
    None
}

pub(crate) fn first_voice_sample(audio: &[f32]) -> Option<usize> {
    let frames = audio
        .chunks(320)
        .map(|frame| {
            rms_from_energy(
                frame.iter().map(|sample| f64::from(*sample).powi(2)).sum(),
                frame.len(),
            )
        })
        .collect::<Vec<_>>();
    frames
        .windows(2)
        .position(|pair| pair[0] >= R1_9_VOICE_RMS_THRESHOLD && pair[1] >= R1_9_VOICE_RMS_THRESHOLD)
        .map(|index| index * 320)
}

/// Find a high-confidence join entirely inside audio that has not yet been
/// committed. A missing or weak match is reported to the caller; it must not
/// be converted into a zero-offset crossfade.
pub(crate) fn find_bounded_preview_join(
    preview: &[f32],
    refill: &[f32],
    committed_samples: usize,
) -> Option<BoundedPreviewJoin> {
    let overlap = OVERLAP_SAMPLES.min(preview.len()).min(refill.len());
    if overlap < OVERLAP_SAMPLES || committed_samples.saturating_add(overlap) > preview.len() {
        return None;
    }
    let first_boundary = committed_samples + overlap;
    let last_boundary = preview.len().min(refill.len());
    if first_boundary > last_boundary {
        return None;
    }
    let committed_peak_frame_rms = preview[..committed_samples]
        .chunks(320)
        .map(|frame| {
            rms_from_energy(
                frame.iter().map(|sample| f64::from(*sample).powi(2)).sum(),
                frame.len(),
            )
        })
        .fold(0.0, f64::max);
    let minimum_rms_gate =
        (committed_peak_frame_rms * BOUNDED_JOIN_RELATIVE_RMS).max(BOUNDED_JOIN_MIN_RMS_FLOOR);

    let mut best_signal: Option<BoundedPreviewJoin> = None;
    let mut best_any: Option<BoundedPreviewJoin> = None;
    let mut boundary = first_boundary;
    while boundary <= last_boundary {
        let preview_start = boundary - overlap;
        let search_start = preview_start.saturating_sub(BOUNDED_JOIN_SEARCH_SAMPLES);
        let search_end = preview_start
            .saturating_add(BOUNDED_JOIN_SEARCH_SAMPLES)
            .min(refill.len() - overlap);
        let mut coarse_signal: Option<(usize, f64, f64, f64)> = None;
        let mut coarse_any: Option<(usize, f64, f64, f64)> = None;
        let mut candidate = search_start;
        while candidate <= search_end {
            let (correlation, preview_rms, refill_rms) = compare_windows(
                &preview[preview_start..boundary],
                &refill[candidate..candidate + overlap],
            );
            let item = (candidate, correlation, preview_rms, refill_rms);
            if coarse_any
                .as_ref()
                .map_or(true, |(_, score, _, _)| correlation > *score)
            {
                coarse_any = Some(item);
            }
            if preview_rms >= minimum_rms_gate
                && refill_rms >= minimum_rms_gate
                && coarse_signal
                    .as_ref()
                    .map_or(true, |(_, score, _, _)| correlation > *score)
            {
                coarse_signal = Some(item);
            }
            candidate = candidate.saturating_add(BOUNDED_JOIN_COARSE_STEP);
        }

        for coarse_offset in [
            coarse_signal.map(|(offset, _, _, _)| offset),
            coarse_any.map(|(offset, _, _, _)| offset),
        ]
        .into_iter()
        .flatten()
        {
            let refine_start = coarse_offset.saturating_sub(BOUNDED_JOIN_COARSE_STEP);
            let refine_end = coarse_offset
                .saturating_add(BOUNDED_JOIN_COARSE_STEP)
                .min(search_end);
            for candidate in refine_start.max(search_start)..=refine_end {
                let (correlation, preview_rms, refill_rms) = compare_windows(
                    &preview[preview_start..boundary],
                    &refill[candidate..candidate + overlap],
                );
                let has_signal = preview_rms >= minimum_rms_gate && refill_rms >= minimum_rms_gate;
                let replace = |best: &mut Option<BoundedPreviewJoin>| {
                    if best
                        .as_ref()
                        .map_or(true, |current| correlation > current.correlation)
                    {
                        let accepted = has_signal && correlation >= BOUNDED_JOIN_MIN_CORRELATION;
                        *best = Some(BoundedPreviewJoin {
                            preview_boundary_samples: boundary,
                            refill_start_samples: candidate,
                            correlation,
                            preview_rms,
                            refill_rms,
                            minimum_rms_gate,
                            accepted,
                            fallback_reason: if accepted {
                                None
                            } else if !has_signal {
                                Some("alignment_window_below_minimum_rms")
                            } else {
                                Some("no_high_confidence_alignment")
                            },
                            seam_start_jump: None,
                            seam_end_jump: None,
                        });
                    }
                };
                replace(&mut best_any);
                if has_signal {
                    replace(&mut best_signal);
                }
            }
        }

        if let Some(join) = best_signal.as_ref().filter(|join| join.accepted) {
            return Some(join.clone());
        }
        boundary = boundary.saturating_add(BOUNDED_JOIN_BOUNDARY_STEP);
    }
    best_signal.or(best_any)
}

pub(crate) fn stitch_bounded_preview(
    preview: &[f32],
    refill: &[f32],
    committed_samples: usize,
    join: &mut BoundedPreviewJoin,
) -> Vec<f32> {
    let overlap_start = join
        .preview_boundary_samples
        .saturating_sub(OVERLAP_SAMPLES);
    let mut output = preview[committed_samples..overlap_start].to_vec();
    let overlap = crossfade_at_offset(
        &preview[overlap_start..join.preview_boundary_samples],
        refill,
        join.refill_start_samples,
    );
    join.seam_start_jump = output
        .last()
        .or_else(|| {
            committed_samples
                .checked_sub(1)
                .and_then(|index| preview.get(index))
        })
        .zip(overlap.first())
        .map(|(before, after)| f64::from((*before - *after).abs()));
    join.seam_end_jump = overlap
        .get(OVERLAP_SAMPLES - 1)
        .zip(refill.get(join.refill_start_samples + OVERLAP_SAMPLES))
        .map(|(before, after)| f64::from((*before - *after).abs()));
    output.extend(overlap);
    output
}

fn compare_windows(left: &[f32], right: &[f32]) -> (f64, f64, f64) {
    if left.len() != right.len() || left.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut dot = 0.0f64;
    let mut left_energy = 0.0f64;
    let mut right_energy = 0.0f64;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    let left_rms = rms_from_energy(left_energy, left.len());
    let right_rms = rms_from_energy(right_energy, right.len());
    let correlation = if left_energy > 0.0 && right_energy > 0.0 {
        dot / (left_energy * right_energy).sqrt()
    } else {
        0.0
    };
    (correlation, left_rms, right_rms)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_chunk_accepts_empty_context_tensor() {
        let tensor = Tensor::<f32>::new(&Allocator::default(), [1_i64, 192, 0]).unwrap();
        let (shape, data) = tensor.try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.as_ref(), [1, 192, 0]);
        assert!(data.is_empty());
    }
    #[test]
    fn sola_recovers_known_offset_without_repeating_previous_audio() {
        let tail: Vec<f32> = (0..2560)
            .map(|i| (i as f32 * 0.13).sin() * (i as f32 * 0.071).cos())
            .collect();
        let mut audio = vec![0.0; 79];
        audio.extend_from_slice(&tail);
        audio.extend(vec![0.25; 1280]);
        let joined = sola(&tail, &audio);
        assert_eq!(joined.len(), 3840);
        assert!(joined[..2560]
            .iter()
            .zip(&tail)
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert_eq!(&joined[2560..], vec![0.25; 1280]);
    }
    #[test]
    fn silent_overlap_uses_zero_offset_and_remains_finite() {
        let joined = sola(&vec![0.0; 2560], &vec![0.0; 3840]);
        assert_eq!(joined, vec![0.0; 3840]);
    }

    #[test]
    fn preview_fallback_rejects_short_and_silent_pcm() {
        assert_eq!(
            preview_fallback_reason(&vec![0.1; 2047]),
            Some("preview_pcm_below_64ms")
        );
        assert_eq!(
            preview_fallback_reason(&vec![0.0; 4096]),
            Some("preview_pcm_silent")
        );
        assert_eq!(preview_fallback_reason(&vec![0.1; 4096]), None);
    }

    #[test]
    fn preview_sola_uses_zero_offset_when_overlap_match_is_weak() {
        let tail: Vec<f32> = (0..2560)
            .map(|index| (index as f32 * 0.171).sin())
            .collect();
        let unrelated: Vec<f32> = (0..3840)
            .map(|index| (index as f32 * 0.113).cos() * 0.1)
            .collect();
        let (joined, alignment) = sola_preview_refill(&tail, &unrelated);
        assert!(alignment.correlation < SOLA_MIN_CORRELATION);
        assert_eq!(alignment.applied_offset_samples, 0);
        assert_eq!(alignment.fallback_reason, Some("low_waveform_correlation"));
        assert_eq!(joined.len(), unrelated.len());
        assert!(joined.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn bounded_preview_join_rejects_unrelated_audio_without_zero_offset_fallback() {
        let preview = (0..24_000)
            .map(|index| ((index as f32) * 0.017).sin())
            .collect::<Vec<_>>();
        let refill = (0..32_000)
            .map(|index| ((index as f32) * 0.031).cos())
            .collect::<Vec<_>>();
        let join = find_bounded_preview_join(&preview, &refill, 4_000).unwrap();
        assert!(!join.accepted);
        assert_eq!(join.fallback_reason, Some("no_high_confidence_alignment"));
    }

    #[test]
    fn bounded_preview_join_finds_same_timeline_audio() {
        let preview = (0..24_000)
            .map(|index| {
                let time = index as f32;
                (time * 0.017).sin() * 0.6 + (time * 0.0031).cos() * 0.4
            })
            .collect::<Vec<_>>();
        let mut refill = vec![0.0; 32_000];
        refill[300..(300 + preview.len())].copy_from_slice(&preview);
        let mut join = find_bounded_preview_join(&preview, &refill, 4_000).unwrap();
        assert!(join.accepted, "{join:?}");
        assert!(join.correlation >= BOUNDED_JOIN_MIN_CORRELATION);
        assert!(
            join.refill_start_samples
                .abs_diff(join.preview_boundary_samples - OVERLAP_SAMPLES)
                <= BOUNDED_JOIN_SEARCH_SAMPLES
        );
        let stitched = stitch_bounded_preview(&preview, &refill, 4_000, &mut join);
        assert!(!stitched.is_empty());
        assert!(join.seam_start_jump.unwrap() < 0.01, "{join:?}");
        let end_sample = join.refill_start_samples + OVERLAP_SAMPLES;
        let expected_end_jump = f64::from((refill[end_sample - 1] - refill[end_sample]).abs());
        let actual_end_jump = join.seam_end_jump.unwrap();
        assert!(
            (actual_end_jump - expected_end_jump).abs() < 1e-4,
            "{join:?}; expected={expected_end_jump}; actual={actual_end_jump}"
        );
    }

    #[test]
    fn bounded_preview_join_rejects_correlated_but_inaudible_windows() {
        let preview = (0..24_000)
            .map(|index| ((index as f32) * 0.017).sin() * 1e-5)
            .collect::<Vec<_>>();
        let mut refill = vec![0.0; 32_000];
        refill[300..(300 + preview.len())].copy_from_slice(&preview);
        let join = find_bounded_preview_join(&preview, &refill, 4_000).unwrap();
        assert!(join.correlation > 0.99, "{join:?}");
        assert!(join.preview_rms < join.minimum_rms_gate, "{join:?}");
        assert_eq!(
            join.fallback_reason,
            Some("alignment_window_below_minimum_rms")
        );
        assert!(!join.accepted);
    }

    #[test]
    fn first_voice_sample_uses_ten_millisecond_rms_windows() {
        let mut audio = vec![0.0; 640];
        audio.extend(std::iter::repeat_n(0.1, 640));
        assert_eq!(first_voice_sample(&audio), Some(640));
        assert_eq!(first_voice_sample(&vec![0.0; 3_200]), None);
        assert_eq!(first_voice_sample(&vec![1e-4; 3_200]), None);
    }
}
