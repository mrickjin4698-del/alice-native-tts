//! ABI-2 GPT-SoVITS diffusion graphs and native per-fragment scheduling.

#[cfg(feature = "r2-trt-native")]
mod trt;

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
const VOCODER_LOOKAHEAD_FRAMES: usize = 128;
const V5_FIRST_LOOKAHEAD_FRAMES: usize = 12;
const V5_FIRST_OVERLAP_FRAMES: usize = 4;
const V5_SECOND_PREFIX_TOKENS: usize = 72;
const NOISE_MULTIPLIER: u64 = 6364136223846793005;

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
    #[cfg(feature = "r2-trt-native")]
    gpt_checkpoint_sha256: Option<String>,
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
        Self::from_values(shape, values, channels, max_frames)
    }

    fn from_values(
        shape: &[i64],
        values: &[f32],
        channels: usize,
        max_frames: usize,
    ) -> Result<Self> {
        let [1, rows, frames] = shape else {
            return Err(invalid("expected a [1, channels, frames] acoustic output"));
        };
        if *rows != channels as i64
            || !(1..=max_frames as i64).contains(frames)
            || values.len() != channels * *frames as usize
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
    fn load(dir: &Path, backend: Backend, _pack: &Pack) -> Result<Self> {
        #[cfg(feature = "r2-trt-native")]
        if backend == Backend::TensorRtNative {
            if let Some(path) = std::env::var_os("ALICE_NATIVE_TTS_DIFFUSION_GPT_FP16_DIR") {
                let path = Path::new(&path);
                let metadata: serde_json::Value =
                    serde_json::from_slice(&fs::read(path.join("gpt.json"))?)?;
                let digest = _pack.gpt_checkpoint_sha256.as_deref();
                if digest.is_none() || metadata["gpt_checkpoint_sha256"].as_str() != digest {
                    return Err(invalid(
                        "FP16 GPT graphs target a different checkpoint than this pack",
                    ));
                }
                return Ok(Self::TensorRt(
                    crate::native_trt::NativeTrtGpt::load_fp16_graphs(path)?,
                ));
            }
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
        on_prefix: impl FnMut(&[i64]) -> Result<()>,
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
                generate(initial, budget, cancel, on_prefix, |token, index| {
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
                generate(
                    (values, indices),
                    budget,
                    cancel,
                    on_prefix,
                    |token, index| gpt.run_step(token, index),
                )
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
    mut on_prefix: impl FnMut(&[i64]) -> Result<()>,
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
        on_prefix(&tokens)?;
        logits = step(token, index)?;
    }
    if tokens.is_empty() {
        return Err(invalid("GPT returned no semantic tokens"));
    }
    Ok(tokens)
}

struct OrtAcoustics {
    target: OrtSession,
    condition: OrtSession,
    step: OrtSession,
    vocoder: OrtSession,
    host: MemoryInfo,
    device: MemoryInfo,
    speaker: ort::value::Tensor<f32>,
}

enum Acoustics {
    Ort(Box<OrtAcoustics>),
    #[cfg(feature = "r2-trt-native")]
    TensorRt(Box<trt::Acoustics>),
}

struct DenoiseWindow<'a> {
    features: Frames,
    prompt: Vec<f32>,
    mask: Vec<f32>,
    state: Vec<f32>,
    rope: &'a [f32],
    rope_channels: usize,
}

pub(super) struct Session {
    frontend: Frontend,
    gpt: Gpt,
    acoustic: AcousticSession,
    pack: Pack,
    reference_bert: Vec<f32>,
    profile: bool,
    first_pcm_tokens: Option<usize>,
}

struct AcousticSession {
    graphs: Acoustics,
    schedule: Schedule,
    reference_features: Frames,
    reference_mel: Frames,
    rope: Vec<f32>,
    rope_channels: usize,
    rolling_pcm: bool,
}

struct AcousticPrefix {
    features: Frames,
    mel: Frames,
    emitted: usize,
    overlap: Vec<f32>,
}

#[derive(Default, Serialize)]
struct Timings {
    fragments: usize,
    chunks: usize,
    denoise_steps: usize,
    prefix_windows: usize,
    leading_samples_trimmed: usize,
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

fn read_env_count(name: &str, min: usize, max: usize) -> Result<Option<usize>> {
    std::env::var_os(name)
        .map(|value| {
            value
                .to_str()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|count| (min..=max).contains(count))
                .ok_or_else(|| invalid(format!("{name} must be between {min} and {max}")))
        })
        .transpose()
}

impl Session {
    pub(super) fn load(
        dir: &Path,
        pack: Pack,
        backend: Backend,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        let schedule = Schedule::for_family(pack.family)?;
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
        let speaker = read_f32(dir, "reference_speaker.f32", pack.speaker_channels)?;
        let rope = read_f32(dir, "rope.f32", schedule.frame_budget * pack.rope_channels)?;
        check(cancel)?;
        let frontend = Frontend::load(dir, Backend::Cpu)?;
        let gpt = Gpt::load(dir, backend, &pack)?;
        check(cancel)?;
        let acoustics = Acoustics::load(dir, &pack, backend, speaker, cancel)?;
        check(cancel)?;
        let profile = std::env::var("ALICE_NATIVE_TTS_DIFFUSION_PROFILE").as_deref() == Ok("1");
        let rolling_pcm =
            std::env::var("ALICE_NATIVE_TTS_DIFFUSION_ROLLING_PCM").as_deref() == Ok("1");
        let first_pcm_tokens = if schedule.v5 {
            read_env_count("ALICE_NATIVE_TTS_V5_FIRST_PCM_TOKENS", 8, 64)?
        } else {
            None
        };
        Ok(Self {
            frontend,
            gpt,
            acoustic: AcousticSession {
                graphs: acoustics,
                schedule,
                reference_features,
                reference_mel,
                rope,
                rope_channels: pack.rope_channels,
                rolling_pcm,
            },
            pack,
            reference_bert,
            profile,
            first_pcm_tokens,
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
            let mut gpt_start = Instant::now();
            let mut prefix = None;
            let fragment_seed = seed;
            let first_pcm_tokens = self.first_pcm_tokens.filter(|_| timings.fragments == 0);
            let channels = self.pack.feature_channels;
            let codes = self.gpt.synthesize(
                all_phones,
                bert,
                &self.pack.reference_codes,
                cancel,
                |codes| {
                    if first_pcm_tokens.is_some_and(|count| {
                        codes.len() == count
                            || codes.len() == V5_SECOND_PREFIX_TOKENS.max(count + 16)
                    }) {
                        timings.gpt_ms += milliseconds(gpt_start);
                        self.acoustic.emit_prefix(
                            &mut prefix,
                            codes,
                            &phones,
                            channels,
                            &mut seed,
                            cancel,
                            &mut timings,
                            &mut emit,
                        )?;
                        check(cancel)?;
                        gpt_start = Instant::now();
                    }
                    Ok(())
                },
            )?;
            timings.gpt_ms += milliseconds(gpt_start);
            let start = Instant::now();
            let features = self.acoustic.graphs.features(codes, phones, channels)?;
            timings.features_ms += milliseconds(start);
            let has_prefix = prefix.is_some();
            let (mel, emitted, overlap) = self.acoustic.diffuse(
                &features,
                prefix,
                &mut seed,
                cancel,
                &mut timings,
                &mut emit,
            )?;
            if has_prefix {
                // Later text fragments keep the noise sequence from full-fragment synthesis.
                let window = self
                    .acoustic
                    .schedule
                    .target_frames(self.acoustic.reference_mel.len);
                seed = fragment_end_seed(fragment_seed, features.len, window);
            }
            check(cancel)?;
            let start = Instant::now();
            let samples = mel.len * self.acoustic.schedule.samples_per_frame;
            let silence = self.pack.sample_rate as usize * 3 / 10;
            let mut normalized = self.acoustic.graphs.waveform(mel, samples, silence)?;
            blend_overlap(&mut normalized[emitted..], &overlap);
            timings.vocoder_ms += milliseconds(start);
            timings.fragments += 1;
            check(cancel)?;
            emit(&normalized[emitted..]);
        }
        if self.profile {
            eprintln!(
                "[alice-tts] native diffusion {}",
                serde_json::to_string(&timings)?
            );
        }
        Ok(())
    }
}

impl AcousticSession {
    fn emit_prefix(
        &mut self,
        prefix: &mut Option<AcousticPrefix>,
        codes: &[i64],
        phones: &[i64],
        channels: usize,
        seed: &mut u64,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
        emit: &mut impl FnMut(&[f32]),
    ) -> Result<()> {
        check(cancel)?;
        let started = Instant::now();
        let features = self
            .graphs
            .features(codes.to_vec(), phones.to_vec(), channels)?;
        timings.features_ms += milliseconds(started);
        if features.len != codes.len() * 4 {
            return Err(invalid(
                "V5 target features must have four frames per token",
            ));
        }
        let start = prefix.as_ref().map_or(0, |prior| prior.mel.len);
        let current = features.slice(start, features.len);
        let history = prefix
            .as_ref()
            .map(|prior| (prior.features.tail(32), prior.mel.tail(32)));
        let (conditioning, prompt) = self.prompt_for_chunk(history.as_ref(), &current);
        let mut prefix_seed = *seed;
        let generated = self.denoise(conditioning, &prompt, &mut prefix_seed, cancel, timings)?;
        let mel = prefix.as_ref().map_or_else(
            || generated.slice(0, generated.len),
            |prior| prior.mel.append(&generated),
        );
        let frames = mel.len - V5_FIRST_LOOKAHEAD_FRAMES;
        let samples_per_frame = self.schedule.samples_per_frame;
        let next_sample = (frames - V5_FIRST_OVERLAP_FRAMES) * samples_per_frame;
        let started = Instant::now();
        let mut audio =
            self.graphs
                .waveform(mel.slice(0, mel.len), mel.len * samples_per_frame, 0)?;
        timings.vocoder_ms += milliseconds(started);
        let offset = match prefix.as_ref() {
            Some(prior) => {
                blend_overlap(&mut audio[prior.emitted..], &prior.overlap);
                prior.emitted
            }
            None => {
                let head = &audio[..next_sample];
                // Keep 10 ms of pre-roll before the first -60 dBFS frame.
                if voice_frame(head, samples_per_frame, 10f32.powf(-4.5)).is_none() {
                    return Ok(());
                }
                let quiet = voice_frame(head, samples_per_frame, 1e-6).unwrap_or(0);
                let offset = quiet.saturating_sub(1) * samples_per_frame;
                timings.leading_samples_trimmed = offset;
                offset
            }
        };
        check(cancel)?;
        *seed = prefix_seed;
        emit(&audio[offset..next_sample]);
        timings.prefix_windows += 1;
        *prefix = Some(AcousticPrefix {
            features: features.slice(0, frames),
            mel: mel.slice(0, frames),
            emitted: next_sample,
            overlap: audio[next_sample..frames * samples_per_frame].to_vec(),
        });
        Ok(())
    }

    fn diffuse(
        &mut self,
        target: &Frames,
        first: Option<AcousticPrefix>,
        seed: &mut u64,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
        emit: &mut impl FnMut(&[f32]),
    ) -> Result<(Frames, usize, Vec<f32>)> {
        let reference_len = self.reference_mel.len;
        let (start, mut history, mut output, mut emitted, overlap) = match first {
            Some(first) if first.mel.len < target.len => (
                first.mel.len,
                Some((
                    first.features.tail(reference_len),
                    first.mel.tail(reference_len),
                )),
                Some(first.mel),
                first.emitted,
                first.overlap,
            ),
            Some(_) => return Err(invalid("V5 target ended inside the first acoustic window")),
            None => (0, None, None, 0, Vec::new()),
        };
        let chunk_frames = self.schedule.target_frames(reference_len);
        for start in (start..target.len).step_by(chunk_frames) {
            check(cancel)?;
            let current = target.slice(start, (start + chunk_frames).min(target.len));
            let (features, mel) = self.prompt_for_chunk(history.as_ref(), &current);
            let generated = self.denoise(features, &mel, seed, cancel, timings)?;
            if self.rolling_pcm
                && start == 0
                && target.len > chunk_frames
                && generated.len > VOCODER_LOOKAHEAD_FRAMES
            {
                check(cancel)?;
                let started = Instant::now();
                let samples_per_frame = self.schedule.samples_per_frame;
                let waveform = self.graphs.waveform(
                    generated.slice(0, generated.len),
                    generated.len * samples_per_frame,
                    0,
                )?;
                emitted = (generated.len - VOCODER_LOOKAHEAD_FRAMES) * samples_per_frame;
                timings.vocoder_ms += milliseconds(started);
                check(cancel)?;
                emit(&waveform[..emitted]);
            }
            history = Some((current.tail(reference_len), generated.tail(reference_len)));
            output = Some(match output {
                Some(previous) => previous.append(&generated),
                None => generated,
            });
        }
        let mel = output.ok_or_else(|| invalid("diffusion returned no mel frames"))?;
        Ok((mel, emitted, overlap))
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
                let tail = 32.min(feat.len).min(mel.len).min(self.reference_mel.len);
                let prefix = self.reference_mel.len - tail;
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
        timings.conditioning_ms += milliseconds(started);
        let started = Instant::now();
        let mut mask = vec![1f32; frames];
        mask[..prompt.len].fill(0.0);
        let window = DenoiseWindow {
            features,
            prompt: padded,
            mask,
            state: noise(seed, frames, prompt.len, self.schedule.temperature),
            rope: &self.rope[..frames * self.rope_channels],
            rope_channels: self.rope_channels,
        };
        timings.denoise_ms += milliseconds(started);
        let result = self
            .graphs
            .denoise(window, self.schedule, cancel, timings)?;
        timings.chunks += 1;
        timings.denoise_steps += self.schedule.steps;
        if result.len != frames {
            return Err(invalid("diffusion step changed the frame count"));
        }
        Ok(result.slice(prompt.len, frames))
    }
}

impl Acoustics {
    fn load(
        dir: &Path,
        pack: &Pack,
        backend: Backend,
        speaker: Vec<f32>,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        #[cfg(feature = "r2-trt-native")]
        if backend == Backend::TensorRtNative {
            if let Some(root) = std::env::var_os("ALICE_NATIVE_TTS_DIFFUSION_TRT_ROOT") {
                return Ok(Self::TensorRt(Box::new(trt::Acoustics::load(
                    dir,
                    Path::new(&root),
                    pack,
                    speaker,
                    cancel,
                )?)));
            }
        }
        let (backend, device) = match backend {
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
        let target = open_on(dir, "target_features", backend)?;
        let condition = open_on(dir, "dit_condition", backend)?;
        let step = open_on(dir, "dit_step", backend)?;
        check(cancel)?;
        let vocoder = open_on(dir, "vocoder", backend)?;
        Ok(Self::Ort(Box::new(OrtAcoustics {
            target,
            condition,
            step,
            vocoder,
            host: MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
            device: MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default)?,
            speaker: tf(vec![1, speaker.len(), 1], speaker)?,
        })))
    }

    fn features(&mut self, codes: Vec<i64>, phones: Vec<i64>, channels: usize) -> Result<Frames> {
        match self {
            Self::Ort(ort) => ort_features(&mut ort.target, &ort.speaker, codes, phones, channels),
            #[cfg(feature = "r2-trt-native")]
            Self::TensorRt(native) => native.features(codes, phones, channels),
        }
    }

    fn waveform(&mut self, mel: Frames, samples: usize, silence: usize) -> Result<Vec<f32>> {
        match self {
            Self::Ort(ort) => {
                let mut output = ort.vocoder.run(ort::inputs! {
                    "mel" => tf(vec![1, MEL_CHANNELS, mel.len], mel.values)?,
                })?;
                let audio = take(&mut output, "waveform")?;
                let (shape, data) = audio.try_extract_tensor::<f32>()?;
                normalize_waveform(shape, data, samples, silence)
            }
            #[cfg(feature = "r2-trt-native")]
            Self::TensorRt(native) => native.waveform(&mel, samples, silence),
        }
    }

    fn denoise(
        &mut self,
        window: DenoiseWindow<'_>,
        schedule: Schedule,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
    ) -> Result<Frames> {
        match self {
            Self::Ort(ort) => ort.denoise(window, schedule, cancel, timings),
            #[cfg(feature = "r2-trt-native")]
            Self::TensorRt(native) => native.denoise(window, schedule, cancel, timings),
        }
    }
}

fn ort_features(
    target: &mut OrtSession,
    speaker: &ort::value::Tensor<f32>,
    codes: Vec<i64>,
    phones: Vec<i64>,
    channels: usize,
) -> Result<Frames> {
    let mut output = target.run(ort::inputs! {
        "codes" => ti(vec![1, 1, codes.len()], codes)?,
        "phones" => ti(vec![1, phones.len()], phones)?,
        "speaker" => speaker,
    })?;
    Frames::from_output(&take(&mut output, "features")?, channels, 4 * KV_CAPACITY)
}

fn normalize_waveform(
    shape: &[i64],
    data: &[f32],
    samples: usize,
    silence: usize,
) -> Result<Vec<f32>> {
    if shape != [1, 1, samples as i64]
        || data.len() != samples
        || data.iter().any(|sample| !sample.is_finite())
    {
        return Err(invalid("invalid vocoder waveform"));
    }
    let peak = data
        .iter()
        .fold(1.0f32, |peak, value| peak.max(value.abs()));
    let mut normalized = Vec::with_capacity(samples + silence);
    normalized.extend(data.iter().map(|sample| sample / peak));
    normalized.resize(samples + silence, 0.0);
    Ok(normalized)
}

fn voice_frame(audio: &[f32], samples_per_frame: usize, threshold: f32) -> Option<usize> {
    audio.chunks_exact(samples_per_frame).position(|frame| {
        let energy = frame.iter().map(|sample| sample * sample).sum::<f32>();
        energy >= frame.len() as f32 * threshold
    })
}

fn blend_overlap(audio: &mut [f32], overlap: &[f32]) {
    for (index, (&prior, output)) in overlap.iter().zip(audio).enumerate() {
        let fade = (index + 1) as f32 / (overlap.len() + 1) as f32;
        *output = prior * (1.0 - fade) + *output * fade;
    }
}

impl OrtAcoustics {
    fn denoise(
        &mut self,
        window: DenoiseWindow<'_>,
        schedule: Schedule,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
    ) -> Result<Frames> {
        let started = Instant::now();
        let frames = window.features.len;
        let mut conditions = self.condition.run(ort::inputs! {
            "features" => tf(vec![1, window.features.channels, frames], window.features.values)?,
            "prompt" => tf(vec![1, MEL_CHANNELS, frames], window.prompt)?,
        })?;
        let condition = take(&mut conditions, "condition")?;
        let negative = if schedule.cfg {
            Some(take(&mut conditions, "negative")?)
        } else {
            None
        };
        timings.conditioning_ms += milliseconds(started);
        let started = Instant::now();
        let mask = tf(vec![1, 1, frames], window.mask)?;
        let rope = tf(vec![1, frames, window.rope_channels], window.rope.to_vec())?;
        let mut state = tf(vec![1, MEL_CHANNELS, frames], window.state)?.into_dyn();
        let mut binding = self.step.create_binding()?;
        binding.bind_input("generation_mask", &mask)?;
        binding.bind_input("rope", &rope)?;
        binding.bind_input("condition", &condition)?;
        if let Some(negative) = &negative {
            binding.bind_input("negative", negative)?;
        }
        binding.bind_output_to_device("state_next", &self.device)?;
        for index in 0..schedule.steps {
            check(cancel)?;
            let time = tf(vec![1], vec![index as f32 / schedule.steps as f32])?;
            if index + 1 == schedule.steps {
                binding.clear_outputs();
                binding.bind_output_to_device("state_next", &self.host)?;
            }
            binding.bind_input("state", &state)?;
            binding.bind_input("time", &time)?;
            let mut output = self.step.run_binding(&binding)?;
            state = take(&mut output, "state_next")?;
        }
        timings.denoise_ms += milliseconds(started);
        Frames::from_output(&state, MEL_CHANNELS, frames)
    }
}

fn noise(seed: &mut u64, frames: usize, prompt: usize, temperature: f32) -> Vec<f32> {
    let mut uniform = || {
        *seed = seed.wrapping_mul(NOISE_MULTIPLIER).wrapping_add(1);
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

fn fragment_end_seed(seed: u64, frames: usize, window: usize) -> u64 {
    let draws: usize = (0..frames)
        .step_by(window)
        .map(|start| (frames - start).min(window).div_ceil(2) * 2 * MEL_CHANNELS)
        .sum();
    (0..draws).fold(seed, |seed, _| {
        seed.wrapping_mul(NOISE_MULTIPLIER).wrapping_add(1)
    })
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
        let full = generate(score(1), 1, &|| false, |_| Ok(()), |_, _| Ok(score(EOS)));
        assert_eq!(full.unwrap(), [1]);
        let unfinished = generate(score(1), 1, &|| false, |_| Ok(()), |_, _| Ok(score(2)));
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

    #[test]
    fn streaming_preserves_following_fragments_noise_seed() {
        let mut expected = 1234;
        noise(&mut expected, 517 + 483, 483, 0.875);
        noise(&mut expected, 123 + 483, 483, 0.875);
        assert_eq!(fragment_end_seed(1234, 640, 517), expected);
    }
}
