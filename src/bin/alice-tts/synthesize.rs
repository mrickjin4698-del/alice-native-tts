use super::{invalid, wav_spec};
use alice_native_tts::session::{Session, SessionResult};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};

#[derive(Default)]
pub(super) enum Format {
    #[default]
    Wav,
    S16le,
}

impl Format {
    pub(super) fn parse(value: &str) -> SessionResult<Self> {
        match value {
            "wav" => Ok(Self::Wav),
            "s16le" => Ok(Self::S16le),
            _ => Err(invalid(format!(
                "unknown output format {value} (expected wav or s16le)"
            ))),
        }
    }
}

pub(super) fn read_text(text: Option<String>, file: Option<PathBuf>) -> SessionResult<String> {
    match (text, file) {
        (Some(text), None) => Ok(text),
        (None, Some(file)) => {
            let mut text = if file == Path::new("-") {
                let mut input = String::new();
                io::stdin().lock().read_to_string(&mut input)?;
                input
            } else {
                fs::read_to_string(file)?
            };
            if text.starts_with('\u{feff}') {
                text.remove(0);
            }
            Ok(text)
        }
        (None, None) => Err(invalid("synthesize requires --text or --text-file")),
        (Some(_), Some(_)) => Err(invalid("use only one of --text and --text-file")),
    }
}

fn write_wav(
    session: &mut Session,
    text: &str,
    language: Option<&str>,
    rate: u32,
    writer: impl Write + Seek,
) -> SessionResult<u64> {
    let mut wav =
        hound::WavWriter::new(writer, wav_spec(rate)).map_err(alice_native_tts::Error::from)?;
    let samples = session.synthesize(text, language, &|| false, |pcm| {
        for &sample in pcm {
            wav.write_sample(sample)
                .map_err(alice_native_tts::Error::from)?;
        }
        Ok(())
    })?;
    wav.finalize().map_err(alice_native_tts::Error::from)?;
    Ok(samples)
}

fn write_pcm(
    session: &mut Session,
    text: &str,
    language: Option<&str>,
    mut writer: impl Write,
) -> SessionResult<u64> {
    let mut bytes = Vec::new();
    session.synthesize(text, language, &|| false, |pcm| {
        bytes.clear();
        bytes.extend(pcm.iter().flat_map(|sample| sample.to_le_bytes()));
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(())
    })
}

pub(super) fn render(
    session: &mut Session,
    text: &str,
    language: Option<&str>,
    output: &Path,
    format: Format,
) -> SessionResult<()> {
    let rate = session
        .info()
        .sample_rate
        .expect("loaded model sample rate");

    if output == Path::new("-") {
        let mut stdout = io::stdout().lock();
        match format {
            Format::Wav => {
                // Finalize the RIFF size fields on a seekable file before copying to the pipe.
                let mut wav = tempfile::tempfile()?;
                write_wav(session, text, language, rate, BufWriter::new(&mut wav))?;
                wav.rewind()?;
                io::copy(&mut wav, &mut stdout)?;
                stdout.flush()?;
            }
            Format::S16le => {
                write_pcm(session, text, language, stdout)?;
            }
        }
        return Ok(());
    }

    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    let writer = BufWriter::new(file);
    let result = match format {
        Format::Wav => write_wav(session, text, language, rate, writer),
        Format::S16le => write_pcm(session, text, language, writer),
    };
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
