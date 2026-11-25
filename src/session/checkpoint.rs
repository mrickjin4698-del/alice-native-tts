use super::{
    existing_directory, existing_file, invalid, python_worker, CheckpointManifest, Device,
    ModelFamily, SessionOptions, SessionResult,
};
use serde::Serialize;
use std::path::PathBuf;

const WORKER: &str = concat!(
    include_str!("../../tools/gpt-sovits-tensorrt.py"),
    "\n",
    include_str!("../../tools/gpt-sovits-worker.py")
);

#[derive(Serialize)]
struct WorkerConfig {
    runtime_root: PathBuf,
    device: Device,
    family: ModelFamily,
    gpt: PathBuf,
    sovits: PathBuf,
    bert: PathBuf,
    hubert: PathBuf,
    reference_audio: PathBuf,
    reference_text: String,
    reference_language: String,
    default_language: String,
}

pub(super) fn load(
    manifest: CheckpointManifest,
    options: &SessionOptions,
    cancelled: &impl Fn() -> bool,
) -> SessionResult<python_worker::Session> {
    if !matches!(
        options.device,
        Device::Cpu | Device::Cuda | Device::TensorRt
    ) {
        return Err(invalid(
            "GPT-SoVITS checkpoints use cpu, cuda, or tensor_rt",
        ));
    }
    let root = options.gpt_sovits_root.as_ref().ok_or_else(|| {
        invalid("GPT-SoVITS checkpoints require an installation (--gpt-sovits-root)")
    })?;
    let runtime_root = std::path::absolute(root)?;
    existing_file(
        runtime_root.join("GPT_SoVITS/TTS_infer_pack/TTS.py"),
        "GPT-SoVITS inference",
    )?;
    if matches!(manifest.family, ModelFamily::V5Turbo | ModelFamily::V5Dev) {
        existing_file(
            runtime_root.join("GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"),
            "GPT-SoVITS V5 vocoder",
        )?;
    }
    let bundled = runtime_root.join(if cfg!(windows) {
        "runtime/python.exe"
    } else {
        "runtime/bin/python"
    });
    let python = python_worker::select_python(options.python.as_deref(), bundled)?;
    let pre = runtime_root.join("GPT_SoVITS/pretrained_models");
    let bert = existing_directory(
        manifest
            .bert
            .unwrap_or_else(|| pre.join("chinese-roberta-wwm-ext-large")),
        "BERT",
    )?;
    let hubert = existing_directory(
        manifest
            .hubert
            .unwrap_or_else(|| pre.join("chinese-hubert-base")),
        "HuBERT",
    )?;
    let config = WorkerConfig {
        runtime_root: runtime_root.clone(),
        device: options.device,
        family: manifest.family,
        gpt: manifest.gpt,
        sovits: manifest.sovits,
        bert,
        hubert,
        reference_audio: manifest.reference_audio,
        reference_text: manifest.reference_text,
        reference_language: manifest.reference_language,
        default_language: manifest.default_language,
    };
    python_worker::Session::load(
        python_worker::Launch {
            name: "GPT-SoVITS",
            script: WORKER,
            runtime_root,
            python,
        },
        config,
        cancelled,
    )
}
