//! Silent warmup, mode-switch reuse and unchanged waveforms with real graphs.
use alice_native_tts::{Backend, Config, Error, NativeTts};
use std::{path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let pack = PathBuf::from(
        args.get(1)
            .ok_or("usage: prepare_bench PACK OUTPUT cpu|directml")?,
    );
    let output = PathBuf::from(args.get(2).ok_or("missing output directory")?);
    let backend: Backend =
        serde_json::from_value(serde_json::json!(args.get(3).ok_or("missing backend")?))?;
    std::fs::create_dir_all(&output)?;
    let started = Instant::now();
    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            backend,
            streaming: true,
        },
        &|| false,
    )?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    assert!(!model.prepare(&|| false)?);
    let warmup_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    assert!(model.prepare(&|| false)?);
    let repeated_ms = started.elapsed().as_secs_f64() * 1000.0;
    let text = "你好，今天我们测试本地语音合成。";
    let mut original = Vec::new();
    let mut first_ms = 0.0;
    let started = Instant::now();
    model.synthesize_streaming(text, &|| false, |audio, _| {
        if original.is_empty() {
            first_ms = started.elapsed().as_secs_f64() * 1000.0;
        }
        original.extend_from_slice(audio);
    })?;
    let mut modes = Vec::new();
    for enabled in [false, true] {
        let started = Instant::now();
        model.set_streaming(enabled, &|| false)?;
        let switch_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert!(!model.prepare(&|| false)?);
        let audio = model.synthesize(text, &|| false)?;
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|x| x.is_finite()));
        if enabled {
            assert_eq!(audio.len(), original.len());
            let error: f64 = audio
                .iter()
                .zip(&original)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let energy: f64 = original.iter().map(|x| f64::from(*x).powi(2)).sum();
            assert!((error / energy.max(1e-12)).sqrt() < 1e-5);
        }
        let name = if enabled { "restored-stream" } else { "clause" };
        let mut wav = hound::WavWriter::create(
            output.join(format!("{name}.wav")),
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
        modes.push(
            serde_json::json!({"streaming":enabled,"switch_ms":switch_ms,"samples":audio.len()}),
        );
    }
    assert!(matches!(
        model.set_streaming(false, &|| true),
        Err(Error::Cancelled)
    ));
    assert!(model.prepare(&|| false)?); // Cancelled switch kept the ready stream graph.
    let restored = model.synthesize(text, &|| false)?;
    assert_eq!(restored.len(), original.len());
    let report = serde_json::json!({"backend":backend,"load_ms":load_ms,"warmup_ms":warmup_ms,"repeated_prepare_ms":repeated_ms,"first_pcm_after_prepare_ms":first_ms,"modes":modes,"restored_waveform_matches":true,"cancelled_switch_preserved_ready_decoder":true});
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
