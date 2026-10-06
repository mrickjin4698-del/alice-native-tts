//! Same model, seed and texts on CPU/DirectML, with stage timings and WAVs.
use alice_native_tts::{Backend, Config, NativeTts};
use std::{path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let pack = PathBuf::from(
        args.get(1)
            .ok_or("usage: backend_bench PACK OUTPUT_DIR cpu|directml")?,
    );
    let output = PathBuf::from(args.get(2).ok_or("missing output directory")?);
    let backend: Backend =
        serde_json::from_value(serde_json::json!(args.get(3).ok_or("missing backend")?))?;
    std::fs::create_dir_all(&output)?;
    let start = Instant::now();
    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            streaming: false,
            backend,
        },
        &|| false,
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut rows = Vec::new();
    for repeat in 0..2 {
        for (id, text) in [
            ("zh", "你好，今天我们测试本地语音合成。"),
            ("polyphonic", "重庆的银行今天有十二个人，大家一起重复测试。"),
            ("en", "Hello Alice, welcome back."),
            (
                "clause",
                "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
            ),
        ] {
            let start = Instant::now();
            let (samples, stages) = model.synthesize_profiled(text, &|| false)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let wav_name = format!("{id}-{repeat}.wav");
            let mut wav = hound::WavWriter::create(
                output.join(&wav_name),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 32000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for sample in &samples {
                wav.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
            }
            wav.finalize()?;
            let row = serde_json::json!({ "id": format!("{id}-{repeat}"), "text": text, "elapsed_ms": elapsed_ms, "audio_seconds": samples.len() as f64 / 32000.0, "stages": stages });
            println!("{row}");
            rows.push(row);
        }
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"backend":backend,"model_load_ms":load_ms,"samples":rows}),
        )?,
    )?;
    Ok(())
}
