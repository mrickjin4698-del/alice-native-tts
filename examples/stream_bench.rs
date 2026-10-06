//! Token-streaming latency, waveform, short-tail and post-audio cancellation.
use alice_native_tts::{Backend, Config, Error, NativeTts};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let dir = PathBuf::from(
        args.get(1)
            .ok_or("usage: stream_bench PACK OUTPUT_DIR cpu|directml")?,
    );
    let output = PathBuf::from(args.get(2).ok_or("missing output directory")?);
    let backend: Backend =
        serde_json::from_value(serde_json::json!(args.get(3).ok_or("missing backend")?))?;
    std::fs::create_dir_all(&output)?;
    let started = Instant::now();
    let mut model = NativeTts::load(
        &Config {
            model_dir: dir,
            reference_audio: None,
            reference_text: None,
            backend,
            streaming: true,
        },
        &|| false,
    )?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut rows = Vec::new();
    for (id, text) in [
        ("zh", "你好，今天我们测试本地语音合成。"),
        ("mixed", "你好，Alice, welcome back. 这是本地语音测试。"),
        ("en", "Hello Alice, welcome back."),
        (
            "clause",
            "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
        ),
        ("short", "你好。"),
    ] {
        let started = Instant::now();
        let mut samples = Vec::new();
        let mut chunks = Vec::new();
        let timings=model.synthesize_streaming(text,&||false,|audio,info|{
            chunks.push(serde_json::json!({"at_ms":started.elapsed().as_secs_f64()*1000.0,"sample_count":audio.len(),"semantic_tokens":info.semantic_tokens,"final":info.final_chunk}));
            samples.extend_from_slice(audio);
        })?;
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|x| x.is_finite()));
        if timings.semantic_tokens > 24 {
            assert!(timings.first_audio_tokens < timings.semantic_tokens);
            assert!(chunks.len() > 1);
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut wav = hound::WavWriter::create(
            output.join(format!("{id}.wav")),
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
        let row = serde_json::json!({"id":id,"text":text,"elapsed_ms":elapsed_ms,"audio_seconds":samples.len() as f64/32000.0,"stages":timings,"chunks":chunks});
        println!("{row}");
        rows.push(row);
    }
    let cancel = AtomicBool::new(false);
    let mut callbacks = 0;
    let mut first_ms = 0.0;
    let started = Instant::now();
    let result = model.synthesize_streaming(
        "这是一段需要在开始播放以后被打断的语音测试。",
        &|| cancel.load(Ordering::Acquire),
        |_, info| {
            assert!(!info.final_chunk);
            callbacks += 1;
            first_ms = started.elapsed().as_secs_f64() * 1000.0;
            cancel.store(true, Ordering::Release);
        },
    );
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(callbacks, 1);
    // A new utterance has fresh token/latent/SOLA state after cancellation.
    let resumed = model.synthesize("你好。", &|| false)?;
    assert!(!resumed.is_empty());
    let report = serde_json::json!({"backend":backend,"model_load_ms":load_ms,"samples":rows,"cancel_after_first_audio":true,"cancel_extra_callbacks":0,"cancel_after_audio_ms":first_ms,"next_utterance_recovered":true});
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
