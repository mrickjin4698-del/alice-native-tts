use super::{check, invalid, SessionError, SessionResult};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

pub(super) struct Launch {
    pub(super) name: &'static str,
    pub(super) script: &'static str,
    pub(super) runtime_root: PathBuf,
    pub(super) python: PathBuf,
}

pub(super) fn select_python(explicit: Option<&Path>, bundled: PathBuf) -> SessionResult<PathBuf> {
    match explicit {
        Some(path) if path.components().count() > 1 => Ok(std::path::absolute(path)?),
        Some(path) => Ok(path.to_path_buf()),
        None if bundled.is_file() => Ok(bundled),
        None => Ok(PathBuf::from(if cfg!(windows) {
            "python"
        } else {
            "python3"
        })),
    }
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
    name: &'static str,
    child: Child,
    input: BufWriter<ChildStdin>,
    events: Receiver<SessionResult<Event>>,
}

impl Process {
    fn spawn(launch: &Launch, config: &serde_json::Value) -> SessionResult<Self> {
        let mut child = Command::new(&launch.python)
            .args(["-I", "-B", "-u", "-c", launch.script])
            .current_dir(&launch.runtime_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                invalid(format!(
                    "could not start {} Python at {}: {error}",
                    launch.name,
                    launch.python.display()
                ))
            })?;
        let stdout = child.stdout.take().expect("piped child stdout");
        let stdin = child.stdin.take().expect("piped child stdin");
        let (send, events) = mpsc::sync_channel(4);
        let mut worker = Self {
            name: launch.name,
            child,
            input: BufWriter::new(stdin),
            events,
        };
        std::thread::Builder::new()
            .name(format!("{}-output", launch.name))
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
                    return Err(invalid(format!("{}: {message}", self.name)))
                }
                Ok(value) => return value,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(invalid(format!("{} worker closed its output", self.name)));
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
    launch: Launch,
    config: serde_json::Value,
    process: Option<Process>,
    sample_rate: u32,
    languages: Vec<String>,
}

impl Session {
    pub(super) fn load(
        launch: Launch,
        config: impl Serialize,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<Self> {
        let config = serde_json::to_value(config)?;
        let (process, sample_rate, languages) = Self::open(&launch, &config, cancelled)?;
        Ok(Self {
            launch,
            config,
            process: Some(process),
            sample_rate,
            languages,
        })
    }

    fn open(
        launch: &Launch,
        config: &serde_json::Value,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<(Process, u32, Vec<String>)> {
        check(cancelled)?;
        let process = Process::spawn(launch, config)?;
        match process.next(cancelled)? {
            Event::Ready {
                sample_rate,
                languages,
            } if sample_rate > 0 && !languages.is_empty() => Ok((process, sample_rate, languages)),
            _ => Err(invalid(format!("invalid {} startup response", launch.name))),
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
            let (process, rate, languages) = Self::open(&self.launch, &self.config, cancelled)?;
            if rate != self.sample_rate || languages != self.languages {
                return Err(invalid(format!(
                    "{} runtime capabilities changed during restart",
                    self.launch.name
                )));
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
                    let bytes = STANDARD.decode(pcm).map_err(|error| {
                        invalid(format!("invalid {} PCM: {error}", process.name))
                    })?;
                    if bytes.is_empty() || bytes.len() % 2 != 0 {
                        return Err(invalid(format!(
                            "invalid PCM length from {} worker",
                            process.name
                        )));
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
                _ => {
                    return Err(invalid(format!(
                        "unexpected {} synthesis response",
                        process.name
                    )))
                }
            }
        }
    }
}
