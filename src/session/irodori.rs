use super::{
    existing_directory, existing_file, invalid, python_worker, Device, ModelFamily, ModelInfo,
    Runtime, SessionOptions, SessionResult, Streaming,
};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};

const WORKER: &str = concat!(
    include_str!("../../tools/irodori-tensorrt.py"),
    "\n",
    include_str!("../../tools/irodori-rf.py"),
    "\n",
    include_str!("../../tools/irodori-worker.py"),
);
const LANGUAGES: [&str; 2] = ["ja", "auto"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    schema_version: u32,
    family: ModelFamily,
    runtime_root: PathBuf,
    checkpoint: PathBuf,
    codec_weights: Option<PathBuf>,
    reference_audio: Option<PathBuf>,
    denoiser_plan: Option<PathBuf>,
    decoder_plan: Option<PathBuf>,
    caption: Option<String>,
    seed: Option<u64>,
}

fn validated_plan(parent: &Path, path: PathBuf) -> SessionResult<PathBuf> {
    let plan = existing_file(parent.join(path), "Irodori-TTS TensorRT engine")?;
    existing_file(plan.with_extension("json"), "Irodori-TTS TensorRT metadata")?;
    Ok(plan)
}

impl Manifest {
    pub(super) fn open(parent: &Path, source: &[u8]) -> SessionResult<Self> {
        let mut manifest: Self = serde_json::from_slice(source)?;
        if manifest.schema_version != 1 || manifest.family != ModelFamily::Irodori {
            return Err(invalid("unsupported Irodori-TTS model manifest"));
        }
        manifest.runtime_root =
            existing_directory(parent.join(manifest.runtime_root), "Irodori-TTS runtime")?;
        existing_file(
            manifest
                .runtime_root
                .join("irodori_tts/inference_runtime.py"),
            "Irodori-TTS inference",
        )?;
        manifest.checkpoint =
            existing_file(parent.join(manifest.checkpoint), "Irodori-TTS checkpoint")?;
        manifest.codec_weights = manifest
            .codec_weights
            .map(|path| existing_file(parent.join(path), "Irodori-TTS codec weights"))
            .transpose()?;
        manifest.reference_audio = manifest
            .reference_audio
            .map(|path| existing_file(parent.join(path), "reference audio"))
            .transpose()?;
        manifest.denoiser_plan = manifest
            .denoiser_plan
            .map(|path| validated_plan(parent, path))
            .transpose()?;
        manifest.decoder_plan = manifest
            .decoder_plan
            .map(|path| validated_plan(parent, path))
            .transpose()?;
        if manifest.decoder_plan.is_some() && manifest.codec_weights.is_none() {
            return Err(invalid(
                "Irodori-TTS decoder_plan requires local codec_weights",
            ));
        }
        manifest.caption = manifest
            .caption
            .map(|caption| caption.trim().to_owned())
            .filter(|caption| !caption.is_empty());
        Ok(manifest)
    }

    pub(super) fn info(&self) -> ModelInfo {
        ModelInfo {
            family: self.family,
            runtime: if self.denoiser_plan.is_some() || self.decoder_plan.is_some() {
                Runtime::IrodoriTensorRt
            } else {
                Runtime::IrodoriTts
            },
            streaming: Streaming::Utterance,
            sample_rate: None,
            languages: LANGUAGES.map(str::to_owned).to_vec(),
        }
    }

    pub(super) fn load(
        self,
        options: &SessionOptions,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<python_worker::Session> {
        if !matches!(options.device, Device::Cpu | Device::Cuda) {
            return Err(invalid("Irodori-TTS uses cpu or cuda"));
        }
        if (self.denoiser_plan.is_some() || self.decoder_plan.is_some())
            && options.device != Device::Cuda
        {
            return Err(invalid("Irodori-TTS TensorRT engines require cuda"));
        }
        let bundled = self.runtime_root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        let python = python_worker::select_python(options.python.as_deref(), bundled)?;
        let config = json!({
            "device": options.device,
            "checkpoint": self.checkpoint,
            "codec_weights": self.codec_weights,
            "reference_audio": self.reference_audio,
            "denoiser_plan": self.denoiser_plan,
            "decoder_plan": self.decoder_plan,
            "caption": self.caption,
            "seed": self.seed,
            "languages": LANGUAGES,
        });
        python_worker::Session::load(
            python_worker::Launch {
                name: "Irodori-TTS",
                script: WORKER,
                runtime_root: self.runtime_root,
                python,
            },
            config,
            cancelled,
        )
    }
}
