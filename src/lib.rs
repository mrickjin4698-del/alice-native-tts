//! Local inference for the versioned Alice GPT-SoVITS v2Pro ONNX pack.
//! Training/export are offline operations; runtime inference does not start Python.
mod audio;
mod frontend;
mod inference;
pub mod profiling;
mod streaming;
pub use frontend::normalize::normalize as normalize_text;
pub use inference::{ChunkInfo, NativeTts, SynthesisTimings};
#[cfg(feature = "r0-profiling")]
pub use profiling::{AllocationSnapshot, PhaseAllocation, PhaseTiming, ProfilePhase, R0Timings};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const PROVIDER_ID: &str = "gpt-sovits-onnx";
pub const STREAMING_GRAPH: &str = "sovits_stream.onnx";
pub const PACK_FILES: &[&str] = &[
    "pack.json",
    "config.json",
    "bert.onnx",
    "ssl.onnx",
    "gpt_encoder.onnx",
    "gpt_step.onnx",
    "sovits.onnx",
    "spectrogram.onnx",
    "sv_embedding.onnx",
    "vq_encoder.onnx",
    "g2pw.onnx",
    "cmudict.json",
    "g2p-en.json",
    "LICENSES.txt",
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("native TTS cancelled")]
    Cancelled,
    #[error("native TTS: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Ort(#[from] ort::Error),
    #[error(transparent)]
    Wav(#[from] hound::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub(crate) fn check(cancelled: &impl Fn() -> bool) -> Result<()> {
    if cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub model_dir: PathBuf,
    pub reference_audio: Option<PathBuf>,
    pub reference_text: Option<String>,
    pub backend: Backend,
    pub streaming: bool,
}

/// CPU remains the portable default. DirectML keeps GPT on CPU. The opt-in
/// CUDA/TensorRT R1 backends target GPT Encoder and GPT Step; frontend and
/// vocoder sessions remain explicitly on CPU. Unsupported GPT nodes fail
/// instead of silently falling back to the CPU EP.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Cpu,
    Directml,
    #[cfg(feature = "r1-cuda-ep")]
    Cuda,
    #[cfg(feature = "r1-tensorrt-ep")]
    TensorRt,
}

#[derive(Deserialize)]
struct Pack {
    abi: u32,
    architecture: String,
    sample_rate: u32,
    max_len: usize,
    reference_audio: String,
    reference_text: String,
    #[serde(default)]
    streaming: Option<StreamingPack>,
}

#[derive(Deserialize)]
struct StreamingPack {
    abi: u32,
    graph: String,
    latent_channels: usize,
    samples_per_token: usize,
}

/// The streaming extension is optional; original ABI-1 packs remain usable.
pub fn validate_streaming_pack(dir: &Path) -> Result<()> {
    validate_pack(dir)?;
    let pack: Pack = serde_json::from_slice(&std::fs::read(dir.join("pack.json"))?)?;
    let extension = pack.streaming.ok_or_else(|| invalid("model pack lacks the token-streaming extension; use clause mode or install a streaming pack"))?;
    if extension.abi != 1
        || extension.graph != STREAMING_GRAPH
        || extension.latent_channels != 192
        || extension.samples_per_token != 1280
    {
        return Err(invalid("unsupported token-streaming graph interface"));
    }
    if !dir.join(STREAMING_GRAPH).is_file() {
        return Err(invalid(
            "missing sovits_stream.onnx; use clause mode or install a streaming pack",
        ));
    }
    Ok(())
}

/// Cheap structural discovery; graph loading and reference validation occur in preflight.
pub fn validate_pack(dir: &Path) -> Result<()> {
    for file in PACK_FILES {
        if !dir.join(file).is_file() {
            return Err(invalid(format!("missing model pack file: {file}")));
        }
    }
    let pack: Pack = serde_json::from_slice(&std::fs::read(dir.join("pack.json"))?)?;
    if pack.abi != 1
        || pack.architecture != "v2Pro"
        || pack.sample_rate != 32_000
        || pack.max_len != 512
    {
        return Err(invalid(
            "unsupported model pack ABI/architecture (requires v2Pro, ABI 1, 32 kHz, KV 512)",
        ));
    }
    let reference = Path::new(&pack.reference_audio);
    if reference.is_absolute()
        || reference
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(invalid(
            "pack reference_audio must be a relative path within the pack",
        ));
    }
    if !dir.join(reference).is_file() || pack.reference_text.trim().is_empty() {
        return Err(invalid(
            "model pack requires a reference WAV and its actual transcript",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_streaming_extension_requires_its_exact_interface_and_graph() {
        let temp = tempfile::tempdir().unwrap();
        for name in PACK_FILES {
            std::fs::write(temp.path().join(name), []).unwrap();
        }
        std::fs::write(temp.path().join("reference.wav"), []).unwrap();
        let mut manifest = serde_json::json!({"abi":1,"architecture":"v2Pro","sample_rate":32000,"max_len":512,"reference_audio":"reference.wav","reference_text":"hello"});
        let save = |m: &serde_json::Value| {
            std::fs::write(temp.path().join("pack.json"), m.to_string()).unwrap()
        };
        save(&manifest);
        assert!(validate_pack(temp.path()).is_ok());
        assert!(validate_streaming_pack(temp.path()).is_err());
        manifest["streaming"] = serde_json::json!({"abi":1,"graph":"sovits_stream.onnx","latent_channels":192,"samples_per_token":1280});
        save(&manifest);
        assert!(validate_streaming_pack(temp.path()).is_err());
        std::fs::write(temp.path().join(STREAMING_GRAPH), []).unwrap();
        assert!(validate_streaming_pack(temp.path()).is_ok());
        manifest["streaming"]["graph"] = "../outside.onnx".into();
        save(&manifest);
        assert!(validate_streaming_pack(temp.path()).is_err());
        manifest["streaming"]["graph"] = STREAMING_GRAPH.into();
        manifest["streaming"]["samples_per_token"] = 2560.into();
        save(&manifest);
        assert!(validate_streaming_pack(temp.path()).is_err());
    }
    #[test]
    fn invalid_pack_is_rejected_before_loading_graphs() {
        let temp = tempfile::tempdir().unwrap();
        for name in PACK_FILES {
            std::fs::write(temp.path().join(name), []).unwrap();
        }
        let manifest = serde_json::json!({"abi":1,"architecture":"v2Pro","sample_rate":32000,"max_len":512,"reference_audio":"../other.wav","reference_text":"hello"});
        std::fs::write(temp.path().join("pack.json"), manifest.to_string()).unwrap();
        assert!(validate_pack(temp.path())
            .unwrap_err()
            .to_string()
            .contains("relative path"));
        let mut manifest = manifest;
        manifest["abi"] = 2.into();
        std::fs::write(temp.path().join("pack.json"), manifest.to_string()).unwrap();
        assert!(validate_pack(temp.path())
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
    }
}
