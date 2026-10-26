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

#[derive(Default)]
pub(crate) struct StreamDecoder {
    latent: Vec<f32>,
    latent_frames: usize,
    tokens: usize,
    tail: Vec<f32>,
    text_seq: Option<Tensor<i64>>,
    #[cfg(feature = "r1-cuda-ep")]
    cuda_io_binding: Option<IoBinding>,
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
                audio = sola(&self.tail, &audio);
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
    let mut result = audio[offset..].to_vec();
    for (i, previous) in tail.iter().enumerate() {
        let weight = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / overlap as f64).cos();
        result[i] = (weight * f64::from(result[i]) + (1.0 - weight) * f64::from(*previous)) as f32;
    }
    result
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
}
