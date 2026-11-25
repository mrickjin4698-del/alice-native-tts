#[path = "alice-tts/worker.rs"]
mod worker;

use alice_native_tts::session::{
    Device, Model, Session, SessionError, SessionOptions, SessionResult,
};
use serde_json::json;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

const USAGE: &str = "\
alice-tts inspect MODEL
alice-tts synthesize MODEL --text TEXT --output FILE.wav [--language LANG] [RUNTIME OPTIONS]
alice-tts worker MODEL [--warmup-text TEXT] [RUNTIME OPTIONS]

MODEL is a native ONNX pack directory or a model manifest JSON file.
RUNTIME OPTIONS
  --device cpu|cuda|directml|tensor_rt|tensor_rt_native   Default cpu
  --gpt-sovits-root DIR   Local GPT-SoVITS installation for checkpoints
  --python PATH          Interpreter for the local Python backend
";

fn invalid(message: impl Into<String>) -> SessionError {
    SessionError::Invalid(message.into())
}

fn string(value: OsString, name: &str) -> SessionResult<String> {
    value
        .into_string()
        .map_err(|_| invalid(format!("{name} must be UTF-8")))
}

#[derive(Default)]
struct Flags {
    runtime: SessionOptions,
    text: Option<String>,
    output: Option<PathBuf>,
    language: Option<String>,
    warmup: Option<String>,
}

fn flags(args: impl IntoIterator<Item = OsString>) -> SessionResult<Flags> {
    let mut result = Flags::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let flag = string(flag, "flag")?;
        let value = args
            .next()
            .ok_or_else(|| invalid(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--device" => {
                let value = string(value, "device")?;
                result.runtime.device = serde_json::from_value::<Device>(json!(value))
                    .map_err(|_| invalid(format!("unknown device {value}")))?;
            }
            "--gpt-sovits-root" => result.runtime.gpt_sovits_root = Some(value.into()),
            "--python" => result.runtime.python = Some(value.into()),
            "--text" => result.text = Some(string(value, "text")?),
            "--output" => result.output = Some(value.into()),
            "--language" => result.language = Some(string(value, "language")?),
            "--warmup-text" => result.warmup = Some(string(value, "warmup text")?),
            _ => return Err(invalid(format!("unknown option {flag}"))),
        }
    }
    Ok(result)
}

fn render(
    session: &mut Session,
    text: &str,
    language: Option<&str>,
    output: &Path,
) -> SessionResult<()> {
    let rate = session
        .info()
        .sample_rate
        .expect("loaded model sample rate");
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    let result = (|| {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav = hound::WavWriter::new(BufWriter::new(file), spec)
            .map_err(alice_native_tts::Error::from)?;
        let samples = session.synthesize(text, language, &|| false, |pcm| {
            for &sample in pcm {
                wav.write_sample(sample)
                    .map_err(alice_native_tts::Error::from)?;
            }
            Ok(())
        })?;
        wav.finalize().map_err(alice_native_tts::Error::from)?;
        Ok(samples)
    })();
    match result {
        Ok(samples) => {
            println!(
                "{}",
                json!({
                    "type": "done",
                    "samples": samples,
                    "sample_rate": rate,
                    "output": output,
                })
            );
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(output);
            Err(error)
        }
    }
}

fn run() -> SessionResult<()> {
    let mut args = std::env::args_os().skip(1);
    let command = args.next().map(|arg| string(arg, "command")).transpose()?;
    let command = match command.as_deref() {
        None | Some("--help" | "-h") => {
            print!("{USAGE}");
            return Ok(());
        }
        Some("--version" | "-V") => {
            println!("alice-tts {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(command @ ("inspect" | "synthesize" | "worker")) => command,
        Some(command) => return Err(invalid(format!("unknown command {command}"))),
    };
    let path = args
        .next()
        .ok_or_else(|| invalid("a model path is required"))?;
    if command == "inspect" {
        if args.next().is_some() {
            return Err(invalid("inspect takes exactly one model path"));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(Model::open(path)?.info())?
        );
        return Ok(());
    }

    let flags = flags(args)?;
    let model = Model::open(path)?;
    if command == "synthesize" {
        let output = flags
            .output
            .ok_or_else(|| invalid("synthesize requires --output"))?;
        let text = flags
            .text
            .ok_or_else(|| invalid("synthesize requires --text"))?;
        if flags.warmup.is_some() {
            return Err(invalid("--warmup-text is available in worker mode"));
        }
        let mut session = Session::load(model, &flags.runtime, &|| false)?;
        render(&mut session, &text, flags.language.as_deref(), &output)
    } else {
        if flags.text.is_some() || flags.output.is_some() || flags.language.is_some() {
            return Err(invalid(
                "worker receives text and language in JSONL requests",
            ));
        }
        let mut session = Session::load(model, &flags.runtime, &|| false)?;
        if let Some(text) = flags.warmup {
            session.synthesize(&text, None, &|| false, |_| Ok(()))?;
        }
        worker::run(session, flags.runtime.device)
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("alice-tts: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
