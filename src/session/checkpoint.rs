use super::{
    check, existing_directory, existing_file, invalid, CheckpointManifest, Device, ModelFamily,
    SessionError, SessionOptions, SessionResult,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

const WORKER: &str = include_str!("../../tools/gpt-sovits-worker.py");

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

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Ready {
        sample_rate: u32,
        languages: Vec<String>,
    },
    Audio {
        pcm: String,
    },
    Done {
        samples: u64,
    },
    Error {
        message: String,
    },
}

struct Process {
    child: Child,
    input: BufWriter<ChildStdin>,
    events: Receiver<SessionResult<Event>>,
}

impl Process {
    fn spawn(python: &PathBuf, config: &WorkerConfig) -> SessionResult<Self> {
        let mut child = Command::new(python)
            .args(["-I", "-B", "-u", "-c", WORKER])
            .current_dir(&config.runtime_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                invalid(format!(
                    "could not start GPT-SoVITS Python at {}: {error}",
                    python.display()
                ))
            })?;
        let stdout = child.stdout.take().expect("piped child stdout");
        let stdin = child.stdin.take().expect("piped child stdin");
        let (send, events) = mpsc::sync_channel(4);
        let mut worker = Self {
            child,
            input: BufWriter::new(stdin),
            events,
        };
        std::thread::Builder::new()
            .name("gpt-sovits-output".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let event = line
                        .map_err(SessionError::from)
                        .and_then(|line| serde_json::from_str(&line).map_err(SessionError::from));
                    let invalid = event.is_err();
                    if send.send(event).is_err() || invalid {
                        break;
                    }
                }
            })?;
        worker.send(config)?;
        Ok(worker)
    }

    fn send(&mut self, message: &impl Serialize) -> SessionResult<()> {
        serde_json::to_writer(&mut self.input, message)?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        Ok(())
    }

    fn next(&self, cancelled: &impl Fn() -> bool) -> SessionResult<Event> {
        loop {
            check(cancelled)?;
            match self.events.recv_timeout(Duration::from_millis(15)) {
                Ok(Ok(Event::Error { message })) => {
                    return Err(invalid(format!("GPT-SoVITS: {message}")))
                }
                Ok(value) => return value,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(invalid("GPT-SoVITS worker closed its output"));
                }
            }
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) struct Session {
    python: PathBuf,
    config: WorkerConfig,
    process: Option<Process>,
    sample_rate: u32,
    languages: Vec<String>,
}

impl Session {
    pub(super) fn load(
        manifest: CheckpointManifest,
        options: &SessionOptions,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<Self> {
        if !matches!(options.device, Device::Cpu | Device::Cuda) {
            return Err(invalid("GPT-SoVITS checkpoints use cpu or cuda"));
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
        let python = match &options.python {
            Some(path) if path.components().count() > 1 => std::path::absolute(path)?,
            Some(path) => path.clone(),
            None if bundled.is_file() => bundled,
            None => PathBuf::from(if cfg!(windows) { "python" } else { "python3" }),
        };
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
            runtime_root,
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
        let (process, sample_rate, languages) = Self::open(&python, &config, cancelled)?;
        Ok(Self {
            python,
            config,
            process: Some(process),
            sample_rate,
            languages,
        })
    }

    fn open(
        python: &PathBuf,
        config: &WorkerConfig,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<(Process, u32, Vec<String>)> {
        check(cancelled)?;
        let process = Process::spawn(python, config)?;
        match process.next(cancelled)? {
            Event::Ready {
                sample_rate,
                languages,
            } if sample_rate > 0 && !languages.is_empty() => Ok((process, sample_rate, languages)),
            _ => Err(invalid("invalid GPT-SoVITS startup response")),
        }
    }

    pub(super) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(super) fn languages(&self) -> &[String] {
        &self.languages
    }

    pub(super) fn synthesize(
        &mut self,
        text: &str,
        language: Option<&str>,
        cancelled: &impl Fn() -> bool,
        emit: impl FnMut(&[i16]) -> SessionResult<()>,
    ) -> SessionResult<u64> {
        let mut process = if let Some(process) = self.process.take() {
            process
        } else {
            let (process, rate, languages) = Self::open(&self.python, &self.config, cancelled)?;
            if rate != self.sample_rate || languages != self.languages {
                return Err(invalid(
                    "checkpoint runtime capabilities changed during restart",
                ));
            }
            process
        };
        let result = Self::generate(&mut process, text, language, cancelled, emit);
        if result.is_ok() {
            self.process = Some(process);
        }
        result
    }

    fn generate(
        process: &mut Process,
        text: &str,
        language: Option<&str>,
        cancelled: &impl Fn() -> bool,
        mut emit: impl FnMut(&[i16]) -> SessionResult<()>,
    ) -> SessionResult<u64> {
        check(cancelled)?;
        process.send(&serde_json::json!({ "text": text, "language": language }))?;
        let mut count = 0;
        loop {
            match process.next(cancelled)? {
                Event::Audio { pcm } => {
                    let bytes = STANDARD
                        .decode(pcm)
                        .map_err(|error| invalid(format!("invalid checkpoint PCM: {error}")))?;
                    if bytes.is_empty() || bytes.len() % 2 != 0 {
                        return Err(invalid("invalid PCM length from GPT-SoVITS worker"));
                    }
                    let samples: Vec<_> = bytes
                        .chunks_exact(2)
                        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                        .collect();
                    check(cancelled)?;
                    emit(&samples)?;
                    count += samples.len() as u64;
                }
                Event::Done { samples } if samples == count && count > 0 => return Ok(count),
                _ => return Err(invalid("unexpected GPT-SoVITS synthesis response")),
            }
        }
    }
}
