use alice_native_tts::{Backend, Config, Error, NativeTts};
use std::{env, path::PathBuf, time::Instant};

const TEXT: &str = "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let pack = PathBuf::from(
        args.get(1)
            .ok_or("usage: r2_1_trt_native_cancel PACK [cancel-ms]")?,
    );
    let cancel_after_ms = args.get(2).map_or(Ok(500u64), |value| value.parse())?;
    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            backend: Backend::TensorRtNative,
            streaming: true,
        },
        &|| false,
    )?;

    let cancel_started = Instant::now();
    let mut cancelled_callbacks = 0usize;
    let cancelled = model.synthesize_streaming(
        TEXT,
        &|| cancel_started.elapsed().as_millis() >= cancel_after_ms as u128,
        |_, _| cancelled_callbacks += 1,
    );
    match cancelled {
        Err(Error::Cancelled) => {}
        Err(error) => return Err(format!("cancellation returned the wrong error: {error}").into()),
        Ok(_) => return Err("synthesis completed before the cancellation deadline".into()),
    }

    let recovered_started = Instant::now();
    let mut recovered_samples = 0usize;
    let recovered = model.synthesize_streaming(TEXT, &|| false, |audio, _| {
        recovered_samples += audio.len();
    })?;
    if recovered_samples == 0 || recovered.semantic_tokens == 0 {
        return Err("post-cancellation synthesis returned no audio or semantic tokens".into());
    }

    println!(
        "cancelled_after_ms={}, callbacks_before_cancel={}, recovery_ms={:.1}, recovery_tokens={}, recovery_samples={}",
        cancel_after_ms,
        cancelled_callbacks,
        recovered_started.elapsed().as_secs_f64() * 1000.0,
        recovered.semantic_tokens,
        recovered_samples,
    );
    Ok(())
}
