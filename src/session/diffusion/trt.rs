use super::{
    milliseconds, normalize_waveform, DenoiseWindow, Frames, ModelFamily, Pack, Schedule, Timings,
    KV_CAPACITY, MEL_CHANNELS,
};
use crate::{
    check, invalid,
    native_trt::graph::{Graph, Input},
    Result,
};
use std::{ffi::CStr, path::Path, time::Instant};

enum Target {
    TensorRt(Graph, Vec<f32>),
    #[cfg(feature = "r1-cuda-ep")]
    Cuda(Box<CudaTarget>),
}

#[cfg(feature = "r1-cuda-ep")]
struct Bucket {
    tokens: usize,
    phones: usize,
    session: ort::session::Session,
}

#[cfg(feature = "r1-cuda-ep")]
struct CudaTarget {
    dynamic: ort::session::Session,
    speaker: ort::value::Tensor<f32>,
    buckets: Vec<Bucket>,
}

#[cfg(feature = "r1-cuda-ep")]
impl CudaTarget {
    fn load(dir: &Path, family: ModelFamily, speaker: Vec<f32>) -> Result<Self> {
        use crate::Backend::Cuda;
        let mut buckets = Vec::new();
        if family == ModelFamily::V3 {
            if let Some(root) = std::env::var_os("ALICE_NATIVE_TTS_DIFFUSION_V3_BUCKETS") {
                for (tokens, phones) in [(96, 64), (256, 128)] {
                    let name = format!("target_features-{tokens}x{phones}");
                    buckets.push(Bucket {
                        tokens,
                        phones,
                        session: super::open_on(Path::new(&root), &name, Cuda)?,
                    });
                }
            }
        }
        Ok(Self {
            dynamic: super::open_on(dir, "target_features", Cuda)?,
            speaker: super::tf(vec![1, speaker.len(), 1], speaker)?,
            buckets,
        })
    }

    fn features(
        &mut self,
        mut codes: Vec<i64>,
        mut phones: Vec<i64>,
        channels: usize,
    ) -> Result<Frames> {
        let tokens = codes.len();
        let text = phones.len();
        let Some(bucket) = self
            .buckets
            .iter_mut()
            .find(|b| tokens <= b.tokens && text <= b.phones)
        else {
            return super::ort_features(&mut self.dynamic, &self.speaker, codes, phones, channels);
        };
        codes.resize(bucket.tokens, 0);
        phones.resize(bucket.phones, 0);
        let mut outputs = bucket.session.run(ort::inputs! {
            "codes" => super::ti(vec![1, 1, bucket.tokens], codes)?,
            "phones" => super::ti(vec![1, bucket.phones], phones)?,
            "speaker" => &self.speaker,
            "code_count" => super::ti(vec![1], vec![tokens as i64])?,
            "phone_count" => super::ti(vec![1], vec![text as i64])?,
        })?;
        let frames = Frames::from_output(
            &super::take(&mut outputs, "features")?,
            channels,
            bucket.tokens * 15 / 4,
        )?;
        if frames.len != bucket.tokens * 15 / 4 {
            return Err(invalid("invalid padded v3 target frame count"));
        }
        Ok(frames.slice(0, tokens * 15 / 4))
    }
}

pub(super) struct Acoustics {
    target: Target,
    condition: Graph,
    step: Graph,
    vocoder: Graph,
}

impl Acoustics {
    pub(super) fn load(
        pack_dir: &Path,
        root: &Path,
        pack: &Pack,
        speaker: Vec<f32>,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        let family = match pack.family {
            ModelFamily::V3 => "v3",
            ModelFamily::V4 => "v4",
            ModelFamily::V5Turbo => "v5turbo",
            ModelFamily::V5Dev => "v5dev",
            _ => return Err(invalid("unsupported TensorRT diffusion family")),
        };
        let dir = root.join(family);
        let load = |name| {
            check(cancel)?;
            Graph::load(&dir.join(name))
        };
        let target_selection =
            std::env::var("ALICE_NATIVE_TTS_DIFFUSION_TRT_TARGET").unwrap_or_default();
        let use_native = target_selection == "1"
            || target_selection
                .split(',')
                .any(|selected| selected.trim() == family);
        let target = if use_native {
            Target::TensorRt(load("target_features.plan")?, speaker)
        } else {
            #[cfg(feature = "r1-cuda-ep")]
            {
                Target::Cuda(Box::new(CudaTarget::load(pack_dir, pack.family, speaker)?))
            }
            #[cfg(not(feature = "r1-cuda-ep"))]
            {
                let _ = pack_dir;
                return Err(invalid(
                    "TensorRT diffusion with CUDA target features requires the CUDA build",
                ));
            }
        };
        Ok(Self {
            target,
            condition: load("dit_condition.plan")?,
            step: load("dit_step.plan")?,
            vocoder: load("vocoder.plan")?,
        })
    }

    pub(super) fn features(
        &mut self,
        codes: Vec<i64>,
        phones: Vec<i64>,
        channels: usize,
    ) -> Result<Frames> {
        let (target, speaker) = match &mut self.target {
            Target::TensorRt(session, speaker) => (session, speaker),
            #[cfg(feature = "r1-cuda-ep")]
            Target::Cuda(cuda) => return cuda.features(codes, phones, channels),
        };
        let outputs = target.run(
            &[
                Input::integers(c"codes", [1, 1, codes.len()], &codes),
                Input::integers(c"phones", [1, phones.len()], &phones),
                Input::floats(c"speaker", [1, speaker.len(), 1], speaker),
            ],
            &[c"features"],
        )?;
        let features = &outputs[0];
        Frames::from_values(&features.shape, features.values, channels, 4 * KV_CAPACITY)
    }

    pub(super) fn waveform(
        &mut self,
        mel: &Frames,
        samples: usize,
        silence: usize,
    ) -> Result<Vec<f32>> {
        let outputs = self.vocoder.run(
            &[Input::floats(
                c"mel",
                [1, MEL_CHANNELS, mel.len],
                &mel.values,
            )],
            &[c"waveform"],
        )?;
        normalize_waveform(&outputs[0].shape, outputs[0].values, samples, silence)
    }

    pub(super) fn denoise(
        &mut self,
        window: DenoiseWindow<'_>,
        schedule: Schedule,
        cancel: &impl Fn() -> bool,
        timings: &mut Timings,
    ) -> Result<Frames> {
        let started = Instant::now();
        let frames = window.features.len;
        let names: &[&CStr] = if schedule.cfg {
            &[c"condition", c"negative"]
        } else {
            &[c"condition"]
        };
        let conditions = self.condition.run(
            &[
                Input::floats(
                    c"features",
                    [1, window.features.channels, frames],
                    &window.features.values,
                ),
                Input::floats(c"prompt", [1, MEL_CHANNELS, frames], &window.prompt),
            ],
            names,
        )?;
        timings.conditioning_ms += milliseconds(started);
        let started = Instant::now();
        let mut inputs = vec![
            Input::floats(c"state", [1, MEL_CHANNELS, frames], &window.state),
            Input::floats(c"time", [1], &[0.0]),
            Input::floats(c"generation_mask", [1, 1, frames], &window.mask),
            Input::floats(c"rope", [1, frames, window.rope_channels], window.rope),
        ];
        for (name, output) in names.iter().zip(&conditions) {
            let [1, length, channels] = output.shape.as_slice() else {
                return Err(invalid("invalid TensorRT diffusion condition rank"));
            };
            if *length != frames as i64 || *channels <= 0 {
                return Err(invalid("invalid TensorRT diffusion condition shape"));
            }
            inputs.push(Input::floats(
                name,
                [1, frames, *channels as usize],
                output.values,
            ));
        }
        let outputs = self
            .step
            .run_steps(&inputs, &[c"state_next"], schedule.steps, cancel)?;
        timings.denoise_ms += milliseconds(started);
        Frames::from_values(&outputs[0].shape, outputs[0].values, MEL_CHANNELS, frames)
    }
}
