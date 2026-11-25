//! Model discovery and local synthesis sessions for standalone applications.

mod checkpoint;
mod diffusion;

use crate::{validate_pack, validate_streaming_pack, Backend, Config, Error, NativeTts, Pack};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("synthesis cancelled")]
    Cancelled,
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Native(#[from] Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type SessionResult<T> = Result<T, SessionError>;

fn invalid(message: impl Into<String>) -> SessionError {
    SessionError::Invalid(message.into())
}

fn check(cancelled: &impl Fn() -> bool) -> SessionResult<()> {
    if cancelled() {
        Err(SessionError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ModelFamily {
    #[serde(rename = "v1")]
    V1,
    #[serde(rename = "v2")]
    V2,
    #[serde(rename = "v2Pro")]
    V2Pro,
    #[serde(rename = "v2ProPlus")]
    V2ProPlus,
    #[serde(rename = "v3")]
    V3,
    #[serde(rename = "v4")]
    V4,
    #[serde(rename = "v5turbo")]
    V5Turbo,
    #[serde(rename = "v5dev")]
    V5Dev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    NativeOnnx,
    GptSovitsCheckpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Streaming {
    Utterance,
    SemanticTokens,
    TextFragments,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub family: ModelFamily,
    pub runtime: Runtime,
    pub streaming: Streaming,
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub languages: Vec<String>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Cpu,
    Cuda,
    Directml,
    TensorRt,
    TensorRtNative,
}

#[derive(Debug, Default)]
pub struct SessionOptions {
    pub device: Device,
    pub gpt_sovits_root: Option<PathBuf>,
    pub python: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointManifest {
    schema_version: u32,
    family: ModelFamily,
    gpt: PathBuf,
    sovits: PathBuf,
    reference_audio: PathBuf,
    reference_text: String,
    reference_language: String,
    #[serde(default = "automatic_language")]
    default_language: String,
    bert: Option<PathBuf>,
    hubert: Option<PathBuf>,
}

fn automatic_language() -> String {
    "auto".into()
}

fn existing_file(path: PathBuf, name: &str) -> SessionResult<PathBuf> {
    if !path.is_file() {
        return Err(invalid(format!(
            "{name} file is missing at {}",
            path.display()
        )));
    }
    Ok(path)
}

fn existing_directory(path: PathBuf, name: &str) -> SessionResult<PathBuf> {
    if !path.is_dir() {
        return Err(invalid(format!(
            "{name} directory is missing at {}",
            path.display()
        )));
    }
    Ok(path)
}

enum ModelSource {
    Native(PathBuf),
    Diffusion(PathBuf, Box<diffusion::Pack>),
    Checkpoint(Box<CheckpointManifest>),
}

/// A validated native pack directory or a schema-versioned checkpoint manifest.
pub struct Model {
    source: ModelSource,
    info: ModelInfo,
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> SessionResult<Self> {
        let path = std::path::absolute(path)?;
        if path.is_dir() {
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("pack.json"))?)?;
            if manifest["abi"].as_u64() == Some(2) {
                let pack = diffusion::Pack::open(&path, manifest).map_err(native_error)?;
                let info = ModelInfo {
                    family: pack.family,
                    runtime: Runtime::NativeOnnx,
                    streaming: Streaming::TextFragments,
                    sample_rate: Some(pack.sample_rate),
                    languages: vec![automatic_language()],
                };
                return Ok(Self {
                    source: ModelSource::Diffusion(path, Box::new(pack)),
                    info,
                });
            }
            validate_pack(&path)?;
            let pack: Pack = serde_json::from_value(manifest)?;
            let streaming = if pack.streaming.is_some() {
                validate_streaming_pack(&path)?;
                Streaming::SemanticTokens
            } else {
                Streaming::Utterance
            };
            return Ok(Self {
                source: ModelSource::Native(path),
                info: ModelInfo {
                    family: ModelFamily::V2Pro,
                    runtime: Runtime::NativeOnnx,
                    streaming,
                    sample_rate: Some(pack.sample_rate),
                    languages: vec![automatic_language()],
                },
            });
        }

        let parent = path
            .parent()
            .ok_or_else(|| invalid("model manifest has no parent"))?;
        let mut manifest: CheckpointManifest = serde_json::from_slice(&fs::read(&path)?)?;
        if manifest.schema_version != 1 {
            return Err(invalid("unsupported checkpoint manifest schema_version"));
        }
        if manifest.reference_text.trim().is_empty()
            || manifest.reference_language.trim().is_empty()
            || manifest.default_language.trim().is_empty()
        {
            return Err(invalid(
                "checkpoint manifest requires a reference transcript and language",
            ));
        }
        manifest.gpt = existing_file(parent.join(manifest.gpt), "GPT checkpoint")?;
        manifest.sovits = existing_file(parent.join(manifest.sovits), "SoVITS checkpoint")?;
        manifest.reference_audio =
            existing_file(parent.join(manifest.reference_audio), "reference audio")?;
        manifest.bert = manifest
            .bert
            .map(|path| existing_directory(parent.join(path), "BERT"))
            .transpose()?;
        manifest.hubert = manifest
            .hubert
            .map(|path| existing_directory(parent.join(path), "HuBERT"))
            .transpose()?;
        let streaming = match manifest.family {
            ModelFamily::V3 | ModelFamily::V4 | ModelFamily::V5Turbo | ModelFamily::V5Dev => {
                Streaming::TextFragments
            }
            _ => Streaming::SemanticTokens,
        };
        Ok(Self {
            info: ModelInfo {
                family: manifest.family,
                runtime: Runtime::GptSovitsCheckpoint,
                streaming,
                sample_rate: None,
                languages: Vec::new(),
            },
            source: ModelSource::Checkpoint(Box::new(manifest)),
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

enum Engine {
    Native(Box<NativeTts>),
    Diffusion(Box<diffusion::Session>),
    Checkpoint(Box<checkpoint::Session>),
}

fn native_backend(device: Device) -> SessionResult<Backend> {
    serde_json::from_value(serde_json::to_value(device)?).map_err(|_| {
        invalid(format!(
            "native device {device:?} is unavailable in this build"
        ))
    })
}

/// A loaded model that owns its runtime and accepts one synthesis at a time.
pub struct Session {
    engine: Engine,
    info: ModelInfo,
}

impl Session {
    pub fn load(
        model: Model,
        options: &SessionOptions,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<Self> {
        check(cancelled)?;
        let mut info = model.info;
        let engine = match model.source {
            ModelSource::Native(model_dir) => {
                let config = Config {
                    model_dir,
                    reference_audio: None,
                    reference_text: None,
                    backend: native_backend(options.device)?,
                    streaming: info.streaming == Streaming::SemanticTokens,
                };
                Engine::Native(Box::new(
                    NativeTts::load(&config, cancelled).map_err(native_error)?,
                ))
            }
            ModelSource::Diffusion(dir, pack) => Engine::Diffusion(Box::new(
                diffusion::Session::load(&dir, *pack, native_backend(options.device)?, cancelled)
                    .map_err(native_error)?,
            )),
            ModelSource::Checkpoint(manifest) => {
                let session = checkpoint::Session::load(*manifest, options, cancelled)?;
                info.sample_rate = Some(session.sample_rate());
                info.languages = session.languages().to_vec();
                Engine::Checkpoint(Box::new(session))
            }
        };
        Ok(Self { engine, info })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Emit mono PCM chunks at the session's sample rate and return the sample count.
    pub fn synthesize(
        &mut self,
        text: &str,
        language: Option<&str>,
        cancelled: &impl Fn() -> bool,
        mut emit: impl FnMut(&[i16]) -> SessionResult<()>,
    ) -> SessionResult<u64> {
        check(cancelled)?;
        if text.trim().is_empty() {
            return Err(invalid("synthesis text is empty"));
        }
        if let Some(language) = language {
            if !self
                .info
                .languages
                .iter()
                .any(|supported| supported == language)
            {
                return Err(invalid(format!(
                    "this model does not support language {language}"
                )));
            }
        }
        if let Engine::Checkpoint(session) = &mut self.engine {
            return session.synthesize(text, language, cancelled, emit);
        }
        let mut samples = 0;
        let mut error = None;
        let failed = Cell::new(false);
        let mut forward = |audio: &[f32]| {
            if failed.get() || cancelled() || audio.is_empty() {
                return;
            }
            let result = if audio.iter().all(|sample| sample.is_finite()) {
                let pcm: Vec<_> = audio
                    .iter()
                    .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                    .collect();
                samples += pcm.len() as u64;
                emit(&pcm)
            } else {
                Err(invalid("native model returned non-finite PCM"))
            };
            if let Err(fail) = result {
                error = Some(fail);
                failed.set(true);
            }
        };
        let stop = || cancelled() || failed.get();
        let result = match &mut self.engine {
            Engine::Native(native) if self.info.streaming == Streaming::SemanticTokens => native
                .synthesize_streaming(text, &stop, |audio, _| forward(audio))
                .map(|_| ()),
            Engine::Native(native) => native.synthesize(text, &stop).map(|audio| forward(&audio)),
            Engine::Diffusion(session) => session.synthesize(text, &stop, forward),
            Engine::Checkpoint(_) => unreachable!("checkpoint synthesis returns above"),
        };
        if let Some(error) = error {
            return Err(error);
        }
        result.map_err(native_error)?;
        check(cancelled)?;
        if samples == 0 {
            return Err(invalid("native model returned no PCM samples"));
        }
        Ok(samples)
    }
}

fn native_error(error: Error) -> SessionError {
    match error {
        Error::Cancelled => SessionError::Cancelled,
        error => error.into(),
    }
}
