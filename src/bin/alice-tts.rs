#[path = "alice-tts/worker.rs"]
mod worker;

#[cfg(feature = "pack-tools")]
#[path = "alice-tts/pack.rs"]
mod pack;
#[cfg(feature = "server")]
#[path = "alice-tts/serve.rs"]
mod serve;

use alice_native_tts::session::{
    Device, Model, Session, SessionError, SessionOptions, SessionResult,
};
use serde_json::json;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
#[cfg(feature = "server")]
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

const USAGE: &str = "\
alice-tts inspect MODEL
alice-tts synthesize MODEL --text TEXT --output FILE.wav [--language LANG] [RUNTIME OPTIONS]
alice-tts worker MODEL [--warmup-text TEXT] [RUNTIME OPTIONS]
";

fn print_usage() {
    print!("{USAGE}");
    #[cfg(feature = "server")]
    println!("alice-tts serve MODEL [--bind ADDRESS] [--warmup-text TEXT] [RUNTIME OPTIONS]");
    #[cfg(feature = "pack-tools")]
    println!("alice-tts pack --help");
    print!(
        "
MODEL is a native ONNX pack directory or a checkpoint manifest JSON file.
RUNTIME OPTIONS
  --device cpu|cuda|directml|tensor_rt|tensor_rt_native   Default cpu
  --gpt-sovits-root DIR   Local GPT-SoVITS installation for checkpoints
  --python PATH          Interpreter for that installation
"
    );
    #[cfg(feature = "server")]
    println!("  --bind ADDRESS         HTTP and WebSocket listen address. Default 127.0.0.1:8090");
}

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
    #[cfg(feature = "server")]
    bind: Option<SocketAddr>,
}

fn flags(_command: &str, args: impl IntoIterator<Item = OsString>) -> SessionResult<Flags> {
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
            #[cfg(feature = "server")]
            "--bind" if _command == "serve" => {
                let address = string(value, "bind address")?;
                result.bind = Some(address.parse().map_err(|_| {
                    invalid("--bind requires an IP address and port, for example 127.0.0.1:8090")
                })?);
            }
            _ => return Err(invalid(format!("unknown option {flag}"))),
        }
    }
    Ok(result)
}

fn wav_spec(rate: u32) -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
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
        let mut wav = hound::WavWriter::new(BufWriter::new(file), wav_spec(rate))
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
            print_usage();
            return Ok(());
        }
        Some("--version" | "-V") => {
            println!("alice-tts {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        #[cfg(feature = "pack-tools")]
        Some("pack") => return pack::run(args),
        #[cfg(not(feature = "pack-tools"))]
        Some("pack") => return Err(invalid("pack requires a build with --features pack-tools")),
        #[cfg(not(feature = "server"))]
        Some("serve") => return Err(invalid("serve requires a build with --features server")),
        #[cfg(feature = "server")]
        Some(command @ "serve") => command,
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

    let flags = flags(command, args)?;
    let model = Model::open(path)?;
    if command == "synthesize" {
        let output = flags
            .output
            .ok_or_else(|| invalid("synthesize requires --output"))?;
        let text = flags
            .text
            .ok_or_else(|| invalid("synthesize requires --text"))?;
        if flags.warmup.is_some() {
            return Err(invalid(
                "--warmup-text is available in worker and serve modes",
            ));
        }
        let mut session = Session::load(model, &flags.runtime, &|| false)?;
        render(&mut session, &text, flags.language.as_deref(), &output)
    } else {
        if flags.text.is_some() || flags.output.is_some() || flags.language.is_some() {
            return Err(invalid(format!(
                "{command} receives text and language in requests"
            )));
        }
        #[cfg(feature = "server")]
        if command == "serve" {
            return serve::run(
                model,
                flags.runtime,
                flags.warmup,
                flags.bind.unwrap_or_else(|| ([127, 0, 0, 1], 8090).into()),
            );
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
