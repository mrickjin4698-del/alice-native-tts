//! ABI-2 GPT-SoVITS diffusion graphs and native per-fragment scheduling.

use super::ModelFamily;
use crate::{
    check,
    frontend::Frontend,
    inference::{open_on, pick_topk, take, tf, ti},
    invalid, Backend, Result,
};
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session as OrtSession,
    value::DynValue,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, time::Instant};

const MEL_CHANNELS: usize = 100;
const BERT_CHANNELS: usize = 1024;
const KV_CAPACITY: usize = 512;
const EOS: i64 = 1024;

const FILES: &[&str] = &[
    "config.json",
    "bert.onnx",
    "g2pw.onnx",
    "cmudict.json",
    "g2p-en.json",
    "gpt_encoder.onnx",
    "gpt_step.onnx",
    "gpt_step_active_len.onnx",
    "target_features.onnx",
    "dit_condition.onnx",
    "dit_step.onnx",
    "vocoder.onnx",
    "reference_bert.f32",
    "reference_features.f32",
    "reference_mel.f32",
    "reference_speaker.f32",
    "rope.f32",
];

#[derive(Deserialize)]
pub(super) struct Pack {
    abi: u32,
    pub(super) family: ModelFamily,
    pub(super) sample_rate: u32,
    precision: Precision,
    reference_phones: Vec<i64>,
    reference_codes: Vec<i64>,
    reference_frames: usize,
    feature_channels: usize,
    speaker_channels: usize,
    rope_channels: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Precision {
    Fp32,
    Fp16,
}

#[derive(Clone, Copy)]
struct Schedule {
    sample_rate: u32,
    samples_per_frame: usize,
    steps: usize,
    cfg: bool,
    temperature: f32,
    reference_limit: usize,
    frame_budget: usize,
    v5: bool,
}

impl Schedule {
    fn for_family(family: ModelFamily) -> Result<Self> {
        let (sample_rate, samples_per_frame, reference_limit, frame_budget) = match family {
            ModelFamily::V3 => (24_000, 256, 468, 934),
            ModelFamily::V4 | ModelFamily::V5Turbo | ModelFamily::V5Dev => (48_000, 480, 500, 1000),
            _ => {
                return Err(invalid(
                    "ABI-2 diffusion packs require v3, v4, v5turbo or v5dev",
                ))
            }
        };
        let v5 = matches!(family, ModelFamily::V5Turbo | ModelFamily::V5Dev);
        Ok(Self {
            sample_rate,
            samples_per_frame,
            steps: if family == ModelFamily::V5Turbo {
                4
            } else {
                32
            },
            cfg: family == ModelFamily::V5Dev,
            temperature: if v5 { 0.875 } else { 1.0 },
            reference_limit,
            frame_budget,
            v5,
        })
    }

    fn target_frames(self, reference_frames: usize) -> usize {
        let available = self.frame_budget - reference_frames;
        if self.v5 {
            available.min(640)
        } else {
            available
        }
    }
}

impl Pack {
    pub(super) fn open(dir: &Path, manifest: serde_json::Value) -> Result<Self> {
        let pack: Self = serde_json::from_value(manifest)?;
        let schedule = Schedule::for_family(pack.family)?;
        if pack.abi != 2
            || pack.sample_rate != schedule.sample_rate
            || pack.reference_frames == 0
            || pack.reference_frames > schedule.reference_limit
            || !(1..=2048).contains(&pack.feature_channels)
            || !(1..=2048).contains(&pack.speaker_channels)
            || !(1..=256).contains(&pack.rope_channels)
            || pack.reference_phones.is_empty()
            || pack.reference_codes.is_empty()
            || pack
                .reference_phones
                .len()
                .saturating_add(pack.reference_codes.len())
                >= KV_CAPACITY
            || pack
                .reference_phones
                .iter()
                .any(|id| !(0..732).contains(id))
            || pack.reference_codes.iter().any(|id| !(0..EOS).contains(id))
        {
            return Err(invalid("invalid native diffusion pack dimensions"));
        }
        for file in FILES {
            if !dir.join(file).is_file() {
                return Err(invalid(format!("native diffusion pack is missing {file}")));
            }
        }
        Ok(pack)
    }
}

fn read_f32(dir: &Path, name: &str, count: usize) -> Result<Vec<f32>> {
    let path = dir.join(name);
    if path.metadata()?.len() != (count * 4) as u64 {
        return Err(invalid(format!("invalid {name} byte count")));
    }
    let bytes = fs::read(path)?;
    if bytes.len() != count * 4 {
        return Err(invalid(format!("{name} changed while loading")));
    }
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("non-finite values in {name}")));
    }
    Ok(values)
}

struct Frames {
    channels: usize,
    len: usize,
    values: Vec<f32>,
}

impl Frames {
    fn read(dir: &Path, name: &str, channels: usize, len: usize) -> Result<Self> {
        Ok(Self {
            channels,
            len,
            values: read_f32(dir, name, channels * len)?,
        })
    }

    fn from_output(value: &DynValue, channels: usize, max_frames: usize) -> Result<Self> {
        let (shape, values) = value.try_extract_tensor::<f32>()?;
        let [1, rows, frames] = &shape[..] else {
            return Err(invalid("expected a [1, channels, frames] acoustic output"));
        };
        if *rows != channels as i64
            || !(1..=max_frames as i64).contains(frames)
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid acoustic output dimensions or values"));
        }
        Ok(Self {
            channels,
            len: *frames as usize,
            values: values.to_vec(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        let len = end - start;
        let mut values = Vec::with_capacity(len * self.channels);
        for channel in 0..self.channels {
            let offset = channel * self.len;
            values.extend_from_slice(&self.values[offset + start..offset + end]);
        }
        Self {
            channels: self.channels,
            len,
            values,
        }
    }

    fn tail(&self, count: usize) -> Self {
        self.slice(self.len.saturating_sub(count), self.len)
    }

    fn append(&self, other: &Self) -> Self {
        debug_assert_eq!(self.channels, other.channels);
        let len = self.len + other.len;
        let mut values = Vec::with_capacity(self.channels * len);
        for channel in 0..self.channels {
            values.extend_from_slice(&self.values[channel * self.len..(channel + 1) * self.len]);
            values.extend_from_slice(&other.values[channel * other.len..(channel + 1) * other.len]);
        }
        Self {
            channels: self.channels,
            len,
            values,
        }
    }
}

enum Gpt {
    Onnx {
        encoder: OrtSession,
        step: OrtSession,
    },
    #[cfg(feature = "r2-trt-native")]
    TensorRt(crate::native_trt::NativeTrtGpt),
}

impl Gpt {
    fn load(dir: &Path, backend: Backend) -> Result<Self> {
        #[cfg(feature = "r2-trt-native")]
        if backend == Backend::TensorRtNative {
            return Ok(Self::TensorRt(
                crate::native_trt::NativeTrtGpt::load_graphs(
                    &dir.join("gpt_encoder.onnx"),
                    &dir.join("gpt_step_active_len.onnx"),
                )?,
            ));
        }
        let _ = backend;
        Ok(Self::Onnx {
            encoder: open_on(dir, "gpt_encoder", Backend::Cpu)?,
            step: open_on(dir, "gpt_step", Backend::Cpu)?,
        })
    }

    fn synthesize(
        &mut self,
        phones: Vec<i64>,
        bert: Vec<f32>,
        prompt: &[i64],
        cancel: &impl Fn() -> bool,
    ) -> Result<Vec<i64>> {
        let length = phones.len();
        let budget = KV_CAPACITY
            .checked_sub(length + prompt.len())
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("fragment and prompt exceed the 512-position GPT cache"))?;
        check(cancel)?;
        match self {
            Self::Onnx { encoder, step } => {
                let mut output = encoder.run(ort::inputs! {
                    "phoneme_ids" => ti(vec![1, length], phones)?,
                    "prompts" => ti(vec![1, prompt.len()], prompt.to_vec())?,
                    "bert_feature" => tf(vec![1, BERT_CHANNELS, length], bert)?,
                })?;
                let x_len = take(&mut output, "x_len")?;
                let y_len = take(&mut output, "y_len")?;
                validate_lengths(
                    x_len.try_extract_tensor::<i64>()?.1,
                    y_len.try_extract_tensor::<i64>()?.1,
                    length,
                    prompt.len(),
                )?;
                let initial = topk(&mut output)?;
                let mut key = take(&mut output, "k_cache")?;
                let mut value = take(&mut output, "v_cache")?;
                generate(initial, budget, cancel, |token, index| {
                    let mut output = step.run(ort::inputs! {
                        "samples" => ti(vec![1, 1], vec![token])?,
                        "k_cache" => &key, "v_cache" => &value,
                        "x_len" => &x_len, "y_len" => &y_len,
                        "idx" => ti(vec![1], vec![index as i64])?,
                    })?;
                    key = take(&mut output, "k_cache_new")?;
                    value = take(&mut output, "v_cache_new")?;
                    topk(&mut output)
                })
            }
            #[cfg(feature = "r2-trt-native")]
            Self::TensorRt(gpt) => {
                gpt.reset();
                let (values, indices, x, y) = gpt.run_encoder(&phones, prompt, &bert)?;
                validate_lengths(&[x], &[y], length, prompt.len())?;
                generate((values, indices), budget, cancel, |token, index| {
                    gpt.run_step(token, index)
                })
            }
        }
    }
}

fn topk(output: &mut ort::session::SessionOutputs<'_>) -> Result<(Vec<f32>, Vec<i64>)> {
    let values = take(output, "topk_values")?;
    let ids = take(output, "topk_indices")?;
    Ok((
        values.try_extract_tensor::<f32>()?.1.to_vec(),
        ids.try_extract_tensor::<i64>()?.1.to_vec(),
    ))
}

fn validate_lengths(x: &[i64], y: &[i64], phones: usize, prompt: usize) -> Result<()> {
    if x != [phones as i64] || y != [prompt as i64] {
        return Err(invalid("GPT encoder reported inconsistent prompt lengths"));
    }
    Ok(())
}

fn generate(
    mut logits: (Vec<f32>, Vec<i64>),
    budget: usize,
    cancel: &impl Fn() -> bool,
    mut step: impl FnMut(i64, usize) -> Result<(Vec<f32>, Vec<i64>)>,
) -> Result<Vec<i64>> {
    let mut seed = 1234;
    let mut tokens = Vec::with_capacity(budget);
    // The last occupied cache slot must still predict an end token.
    for index in 0..=budget {
        check(cancel)?;
        let token = pick_topk(&logits.0, &logits.1, &mut seed)?;
        if token == EOS {
            break;
        }
        if index == budget {
            return Err(invalid(
                "GPT reached the 512-position cache before finishing the fragment",
            ));
        }
        if !(0..EOS).contains(&token) {
            return Err(invalid("invalid GPT semantic token"));
        }
        tokens.push(token);
        logits = step(token, index)?;
    }
    if tokens.is_empty() {
        return Err(invalid("GPT returned no semantic tokens"));
    }
    Ok(tokens)
}

pub(super) struct Session {
    frontend: Frontend,
    gpt: Gpt,
    target: OrtSession,
    condition: OrtSession,
    step: OrtSession,
    vocoder: OrtSession,
    host: MemoryInfo,
    device: MemoryInfo,
    pack: Pack,
    schedule: Schedule,
    reference_features: Frames,
    reference_mel: Frames,
    reference_bert: Vec<f32>,
    speaker: ort::value::Tensor<f32>,
    rope: Vec<f32>,
    profile: bool,
}

#[derive(Default, Serialize)]
struct Timings {
    fragments: usize,
    chunks: usize,
    denoise_steps: usize,
    frontend_ms: f64,
    gpt_ms: f64,
    features_ms: f64,
    conditioning_ms: f64,
    denoise_ms: f64,
    vocoder_ms: f64,
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl Session {
    pub(super) fn load(
        dir: &Path,
        pack: Pack,
        backend: Backend,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        let (acoustic, device) = match backend {
            Backend::Cpu if pack.precision == Precision::Fp32 => {
                (Backend::Cpu, AllocationDevice::CPU)
            }
            #[cfg(feature = "r1-cuda-ep")]
            Backend::Cuda => (Backend::Cuda, AllocationDevice::CUDA),
            #[cfg(all(feature = "r2-trt-native", feature = "r1-cuda-ep"))]
            Backend::TensorRtNative => (Backend::Cuda, AllocationDevice::CUDA),
            _ => {
                return Err(invalid(
                    "native diffusion requires CPU with an fp32 pack, or a CUDA-enabled build",
                ))
            }
        };
        let schedule = Schedule::for_family(pack.family)?;
        let host = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let device = MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default)?;
        let reference_bert = read_f32(
            dir,
            "reference_bert.f32",
            BERT_CHANNELS * pack.reference_phones.len(),
        )?;
        let reference_features = Frames::read(
            dir,
            "reference_features.f32",
            pack.feature_channels,
            pack.reference_frames,
        )?;
        let reference_mel = Frames::read(
            dir,
            "reference_mel.f32",
            MEL_CHANNELS,
            pack.reference_frames,
        )?;
        let speaker = tf(
            vec![1, pack.speaker_channels, 1],
            read_f32(dir, "reference_speaker.f32", pack.speaker_channels)?,
        )?;
        let rope = read_f32(dir, "rope.f32", schedule.frame_budget * pack.rope_channels)?;
        check(cancel)?;
        let frontend = Frontend::load(dir, Backend::Cpu)?;
        let gpt = Gpt::load(dir, backend)?;
        check(cancel)?;
        let target = open_on(dir, "target_features", acoustic)?;
        let condition = open_on(dir, "dit_condition", acoustic)?;
        let step = open_on(dir, "dit_step", acoustic)?;
        check(cancel)?;
        let vocoder = open_on(dir, "vocoder", acoustic)?;
        check(cancel)?;
        let profile = std::env::var("ALICE_NATIVE_TTS_DIFFUSION_PROFILE").as_deref() == Ok("1");
        Ok(Self {
            frontend,
            gpt,
            target,
            condition,
            step,
            vocoder,
            host,
            device,
            pack,
            schedule,
            reference_features,
            reference_mel,
            reference_bert,
            speaker,
            rope,
            profile,
        })
    }

    pub(super) fn synthesize(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        mut emit: impl FnMut(&[f32]),
    ) -> Result<()> {
        let mut timings = Timings::default();
        let mut seed = 1234u64;
        for fragment in fragments(text) {
            check(cancel)?;
            let start = Instant::now();
            let (phones, target_bert) = self.frontend.process(fragment, cancel)?;
            if phones.is_empty() {
                continue;
            }
            let reference = self.pack.reference_phones.len();
            let target = phones.len();
            if target_bert.len() != BERT_CHANNELS * target {
                return Err(invalid("invalid frontend BERT features"));
            }
            let mut all_phones = self.pack.reference_phones.clone();
            all_phones.extend_from_slice(&phones);
            let mut bert = Vec::with_capacity(BERT_CHANNELS * (reference + target));
            for (prompt, text) in self
                .reference_bert
                .chunks_exact(reference)
                .zip(target_bert.chunks_exact(target))
            {
                bert.extend_from_slice(prompt);
                bert.extend_from_slice(text);
            }
            timings.frontend_ms += milliseconds(start);
            let start = Instant::now();
            let codes =
                self.gpt
                    .synthesize(all_phones, bert, &self.pack.reference_codes, cancel)?;
            timings.gpt_ms += milliseconds(start);
            let start = Instant::now();
            let features = {
                let mut output = self.target.run(ort::inputs! {
                    "codes" => ti(vec![1, 1, codes.len()], codes)?,
                    "phones" => ti(vec![1, target], phones)?,
                    "speaker" => &self.speaker,
                })?;
                Frames::from_output(
                    &take(&mut output, "features")?,
                    self.pack.feature_channels,
                    4 * KV_CAPACITY,
                )?
            };
            timings.features_ms += milliseconds(start);
            let mel = self.diffuse(&features, &mut seed, cancel, &mut timings)?;
            check(cancel)?;
            let start = Instant::now();
            let mut output = self.vocoder.run(ort::inputs! {
                "mel" => tf(vec![1, MEL_CHANNELS, mel.len], mel.values)?,
            })?;
            let audio = take(&mut output, "waveform")?;
            let (shape, samples) = audio.try_extract_tensor::<f32>()?;
            if shape[..] != [1, 1, (mel.len * self.schedule.samples_per_frame) as i64]
                || samples.iter().any(|sample| !sample.is_finite())
            {
                return Err(invalid("invalid vocoder waveform"));
            }
            let peak = samples
                .iter()
                .fold(1.0f32, |peak, value| peak.max(value.abs()));
            let mut normalized =
                Vec::with_capacity(samples.len() + self.pack.sample_rate as usize * 3 / 10);
            normalized.extend(samples.iter().map(|sample| sample / peak));
            normalized.resize(samples.len() + self.pack.sample_rate as usize * 3 / 10, 0.0);
            timings.vocoder_ms += milliseconds(start);
            timings.fragments += 1;
            check(cancel)?;
            emit(&normalized);
        }
        if self.profile {
            eprintln!(
                "[alice-tts] native diffusion {}",
                serde_json::to_string(&timings)?
            );
        }
        Ok(())
    }

    fn diffuse(
        &mut self,
        target: &Frames,
        seed: &mut u64,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
    ) -> Result<Frames> {
        let mut history = None::<(Frames, Frames)>;
        let reference_len = self.pack.reference_frames;
        let mut output = None::<Frames>;
        let chunk_frames = self.schedule.target_frames(reference_len);
        for start in (0..target.len).step_by(chunk_frames) {
            check(cancel)?;
            let current = target.slice(start, (start + chunk_frames).min(target.len));
            let (features, mel) = self.prompt_for_chunk(history.as_ref(), &current);
            let generated = self.denoise(features, &mel, seed, cancel, timings)?;
            history = Some((current.tail(reference_len), generated.tail(reference_len)));
            output = Some(match output {
                Some(previous) => previous.append(&generated),
                None => generated,
            });
        }
        output.ok_or_else(|| invalid("diffusion returned no mel frames"))
    }

    fn prompt_for_chunk(
        &self,
        history: Option<&(Frames, Frames)>,
        target: &Frames,
    ) -> (Frames, Frames) {
        match history {
            None => (
                self.reference_features.append(target),
                self.reference_mel.slice(0, self.reference_mel.len),
            ),
            Some((feat, mel)) if self.schedule.v5 => {
                let tail = 32
                    .min(feat.len)
                    .min(mel.len)
                    .min(self.pack.reference_frames);
                let prefix = self.pack.reference_frames - tail;
                let features = self
                    .reference_features
                    .slice(0, prefix)
                    .append(&feat.tail(tail))
                    .append(target);
                let mel = self.reference_mel.slice(0, prefix).append(&mel.tail(tail));
                (features, mel)
            }
            Some((feat, mel)) => (feat.append(target), mel.slice(0, mel.len)),
        }
    }

    fn denoise(
        &mut self,
        features: Frames,
        prompt: &Frames,
        seed: &mut u64,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
    ) -> Result<Frames> {
        let frames = features.len;
        if frames <= prompt.len || frames > self.schedule.frame_budget {
            return Err(invalid(
                "diffusion chunk is outside the exported frame budget",
            ));
        }
        let started = Instant::now();
        let mut padded = vec![0f32; MEL_CHANNELS * frames];
        for (output, input) in padded
            .chunks_exact_mut(frames)
            .zip(prompt.values.chunks_exact(prompt.len))
        {
            output[..prompt.len].copy_from_slice(input);
        }
        let mut conditions = self.condition.run(ort::inputs! {
            "features" => tf(vec![1, features.channels, frames], features.values)?,
            "prompt" => tf(vec![1, MEL_CHANNELS, frames], padded)?,
        })?;
        let condition = take(&mut conditions, "condition")?;
        let negative = if self.schedule.cfg {
            Some(take(&mut conditions, "negative")?)
        } else {
            None
        };
        timings.conditioning_ms += milliseconds(started);
        let started = Instant::now();
        let mut mask = vec![1f32; frames];
        mask[..prompt.len].fill(0.0);
        let mask = tf(vec![1, 1, frames], mask)?;
        let rope = tf(
            vec![1, frames, self.pack.rope_channels],
            self.rope[..frames * self.pack.rope_channels].to_vec(),
        )?;
        let mut state = tf(
            vec![1, MEL_CHANNELS, frames],
            noise(seed, frames, prompt.len, self.schedule.temperature),
        )?
        .into_dyn();
        let mut binding = self.step.create_binding()?;
        binding.bind_input("generation_mask", &mask)?;
        binding.bind_input("rope", &rope)?;
        binding.bind_input("condition", &condition)?;
        if let Some(negative) = &negative {
            binding.bind_input("negative", negative)?;
        }
        binding.bind_output_to_device("state_next", &self.device)?;
        for index in 0..self.schedule.steps {
            check(cancel)?;
            let time = tf(vec![1], vec![index as f32 / self.schedule.steps as f32])?;
            if index + 1 == self.schedule.steps {
                binding.clear_outputs();
                binding.bind_output_to_device("state_next", &self.host)?;
            }
            binding.bind_input("state", &state)?;
            binding.bind_input("time", &time)?;
            let mut output = self.step.run_binding(&binding)?;
            state = take(&mut output, "state_next")?;
        }
        timings.chunks += 1;
        timings.denoise_steps += self.schedule.steps;
        timings.denoise_ms += milliseconds(started);
        let result = Frames::from_output(&state, MEL_CHANNELS, frames)?;
        if result.len != frames {
            return Err(invalid("diffusion step changed the frame count"));
        }
        Ok(result.slice(prompt.len, frames))
    }
}

fn noise(seed: &mut u64, frames: usize, prompt: usize, temperature: f32) -> Vec<f32> {
    let mut uniform = || {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 11) + 1) as f64 / ((1u64 << 53) as f64 + 1.0)
    };
    let mut state = vec![0f32; MEL_CHANNELS * frames];
    for channel in state.chunks_exact_mut(frames) {
        for pair in channel[prompt..].chunks_mut(2) {
            let magnitude = (-2.0 * uniform().ln()).sqrt() * f64::from(temperature);
            let (sine, cosine) = (std::f64::consts::TAU * uniform()).sin_cos();
            pair[0] = (magnitude * cosine) as f32;
            if pair.len() == 2 {
                pair[1] = (magnitude * sine) as f32;
            }
        }
    }
    state
}

fn fragments(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    text.char_indices()
        .filter_map(move |(index, ch)| {
            let decimal = ch == '.'
                && index > 0
                && bytes[index - 1].is_ascii_digit()
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
            let delimiter = matches!(
                ch,
                '，' | '。' | '！' | '？' | '；' | '：' | ',' | '.' | '!' | '?' | ';' | ':' | '\n'
            );
            (delimiter && !decimal).then_some(index + ch.len_utf8())
        })
        .chain(std::iter::once(text.len()))
        .scan(0, move |start, end| {
            let fragment = text[*start..end].trim();
            *start = end;
            Some(fragment)
        })
        .filter(|fragment| fragment.chars().any(char::is_alphanumeric))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_gpt_cache_requires_an_end_token() {
        let score = |token| (vec![0.0], vec![token]);
        let full = generate(score(1), 1, &|| false, |_, _| Ok(score(EOS)));
        assert_eq!(full.unwrap(), [1]);
        let unfinished = generate(score(1), 1, &|| false, |_, _| Ok(score(2)));
        assert!(unfinished
            .unwrap_err()
            .to_string()
            .contains("before finishing"));
    }

    #[test]
    fn split_keeps_decimal_points_and_emits_both_languages() {
        assert_eq!(
            fragments(" 现在是3.14。 Next sentence. 最后一句 ").collect::<Vec<_>>(),
            ["现在是3.14。", "Next sentence.", "最后一句"]
        );
    }

    #[test]
    fn rolling_prompt_accepts_an_empty_reference_prefix() {
        let frames = Frames {
            channels: 2,
            len: 3,
            values: vec![1., 2., 3., 4., 5., 6.],
        };
        let prefix = frames.slice(0, 0);
        let tail = frames.tail(2);
        assert_eq!(prefix.append(&tail).values, [2., 3., 5., 6.]);
        assert_eq!(tail.append(&prefix).values, [2., 3., 5., 6.]);
    }
}
