use alice_native_tts::{Backend, Config, NativeTts};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: synthesize PACK TEXT OUTPUT.wav [cpu|directml] [cancel-ms]".into());
    }
    let backend = match args.get(4).map(String::as_str).unwrap_or("cpu") {
        "cpu" => Backend::Cpu,
        "directml" => Backend::Directml,
        value => return Err(format!("unsupported backend: {value}").into()),
    };
    let config = Config {
        model_dir: PathBuf::from(&args[1]),
        reference_audio: None,
        reference_text: None,
        streaming: false,
        backend,
    };
    let loaded = Instant::now();
    let mut model = NativeTts::load(&config, &|| false)?;
    println!(
        "model_ready_ms={:.1}",
        loaded.elapsed().as_secs_f64() * 1000.0
    );
    let started = Instant::now();
    let cancel = args.get(5).map(|s| s.parse::<u64>()).transpose()?;
    let audio = model.synthesize(args.get(2).ok_or("missing text")?, &|| {
        cancel.is_some_and(|ms| started.elapsed().as_millis() >= ms as u128)
    })?;
    let mut wav = hound::WavWriter::create(
        &args[3],
        hound::WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in &audio {
        wav.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    wav.finalize()?;
    println!(
        "synthesis_ms={:.1},audio_seconds={:.3}",
        started.elapsed().as_secs_f64() * 1000.0,
        audio.len() as f64 / 32000.0
    );
    Ok(())
}
