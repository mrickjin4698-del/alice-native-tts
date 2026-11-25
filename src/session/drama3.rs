use super::{
    check, existing_directory, existing_file, invalid, python_worker, Device, ModelFamily,
    ModelInfo, Runtime, SessionOptions, SessionResult, Streaming,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tempfile::NamedTempFile;

const SAMPLE_RATE: u32 = 44_100;
const WORKER: &str = include_str!("../../tools/drama3-worker.py");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    schema_version: u32,
    family: ModelFamily,
    runtime_root: PathBuf,
    checkpoint: PathBuf,
    reference_audio: Option<PathBuf>,
    reference_text: Option<String>,
    #[serde(default = "default_threads")]
    threads: u16,
    #[serde(default)]
    session_options: BTreeMap<String, String>,
    #[serde(default)]
    request_options: BTreeMap<String, String>,
}

fn default_threads() -> u16 {
    8
}

impl Manifest {
    fn executable(&self) -> PathBuf {
        self.runtime_root.join(if cfg!(windows) {
            "audiocpp_server.exe"
        } else {
            "audiocpp_server"
        })
    }

    pub(super) fn open(parent: &Path, source: &[u8]) -> SessionResult<Self> {
        let mut manifest: Self = serde_json::from_slice(source)?;
        if manifest.schema_version != 1 || manifest.family != ModelFamily::Drama3 {
            return Err(invalid("unsupported Drama 3 model manifest"));
        }
        if manifest.threads == 0 {
            return Err(invalid("Drama 3 threads must be positive"));
        }
        if manifest.reference_audio.is_some() != manifest.reference_text.is_some() {
            return Err(invalid(
                "Drama 3 reference_audio and reference_text must be supplied together",
            ));
        }
        if manifest
            .reference_text
            .as_ref()
            .is_some_and(|text| text.trim().is_empty())
        {
            return Err(invalid("Drama 3 reference_text is empty"));
        }
        manifest.runtime_root =
            existing_directory(parent.join(manifest.runtime_root), "audio.cpp runtime")?;
        existing_file(manifest.executable(), "audio.cpp server")?;
        manifest.checkpoint = existing_file(parent.join(manifest.checkpoint), "Drama 3 GGUF")?;
        let mut magic = [0; 4];
        File::open(&manifest.checkpoint)?.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(invalid("Drama 3 checkpoint must be an audio.cpp GGUF"));
        }
        manifest.reference_audio = manifest
            .reference_audio
            .map(|path| existing_file(parent.join(path), "Drama 3 reference audio"))
            .transpose()?;
        Ok(manifest)
    }

    pub(super) fn info(&self) -> ModelInfo {
        ModelInfo {
            family: self.family,
            runtime: Runtime::AudioCpp,
            streaming: Streaming::SemanticTokens,
            sample_rate: Some(SAMPLE_RATE),
            languages: vec!["auto".into()],
        }
    }

    pub(super) fn load(
        mut self,
        options: &SessionOptions,
        cancelled: &impl Fn() -> bool,
    ) -> SessionResult<Session> {
        check(cancelled)?;
        if !matches!(options.device, Device::Cpu | Device::Cuda) {
            return Err(invalid("Drama 3 uses cpu or cuda"));
        }
        self.session_options
            .entry("fish_drama3.context_length".into())
            .or_insert_with(|| "2048".into());
        self.session_options
            .entry("fish_drama3.cpu_expert_layers".into())
            .or_insert_with(|| "24".into());
        self.request_options
            .entry("seed".into())
            .or_insert_with(|| "123".into());
        self.request_options
            .entry("max_tokens".into())
            .or_insert_with(|| "512".into());

        let mut config_file = tempfile::Builder::new()
            .prefix("alice-drama3-")
            .suffix(".json")
            .tempfile()?;
        let id = config_file
            .path()
            .file_stem()
            .expect("temporary config has a file stem")
            .to_string_lossy()
            .into_owned();
        let socket = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let address = socket.local_addr()?;
        let server_config = json!({
            "host": "127.0.0.1",
            "port": address.port(),
            "backend": options.device,
            "threads": self.threads,
            "lazy_load": false,
            "ui": false,
            "models": [{
                "id": id,
                "family": "fish_drama3",
                "path": self.checkpoint,
                "task": if self.reference_audio.is_some() { "clon" } else { "tts" },
                "mode": "streaming",
                "session_options": self.session_options,
            }],
        });
        serde_json::to_writer(config_file.as_file_mut(), &server_config)?;
        config_file.as_file_mut().flush()?;
        drop(socket);

        let mut command = Command::new(self.executable());
        command
            .current_dir(&self.runtime_root)
            .arg("--config")
            .arg(config_file.path())
            .arg("--no-ui")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut server = Server {
            child: command.spawn()?,
            _config: config_file,
        };
        loop {
            check(cancelled)?;
            server.ensure_running()?;
            if TcpStream::connect(address).is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(35));
        }

        let bundled = self.runtime_root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        let python = python_worker::select_python(options.python.as_deref(), bundled)?;
        let worker = python_worker::Session::load(
            python_worker::Launch {
                name: "Drama 3",
                runtime_root: self.runtime_root,
                python,
                script: WORKER,
            },
            json!({
                "port": address.port(), "model": id, "sample_rate": SAMPLE_RATE,
                "reference_audio": self.reference_audio, "reference_text": self.reference_text,
                "options": self.request_options,
            }),
            cancelled,
        )?;
        Ok(Session { worker, server })
    }
}

struct Server {
    child: Child,
    _config: NamedTempFile,
}

impl Server {
    fn ensure_running(&mut self) -> SessionResult<()> {
        if let Some(status) = self.child.try_wait()? {
            return Err(invalid(format!("audio.cpp server exited with {status}")));
        }
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) struct Session {
    worker: python_worker::Session,
    server: Server,
}

impl Session {
    pub(super) fn synthesize(
        &mut self,
        text: &str,
        language: Option<&str>,
        cancelled: &impl Fn() -> bool,
        emit: impl FnMut(&[i16]) -> SessionResult<()>,
    ) -> SessionResult<u64> {
        self.server.ensure_running()?;
        self.worker.synthesize(text, language, cancelled, emit)
    }
}
