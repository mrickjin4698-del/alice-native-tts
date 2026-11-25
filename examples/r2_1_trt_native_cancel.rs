use alice_native_tts::{Backend, Config, Error, NativeTts};
use std::{cell::Cell, env, path::PathBuf, time::Instant};

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

    // A first callback proves GPT steps and the decoder completed. Warm first,
    // then cancel after that callback instead of relying only on wall-clock time.
    let after_callback = args.iter().any(|arg| arg == "--after-callback");
    let mut clean_audio = Vec::new();
    if after_callback {
        model.synthesize_streaming(TEXT, &|| false, |audio, _| {
            clean_audio.extend_from_slice(audio);
        })?;
    }
    let callback_seen = Cell::new(false);
    let cancel_started = Instant::now();
    let mut cancelled_callbacks = 0usize;
    let cancelled = model.synthesize_streaming(
        TEXT,
        &|| {
            if after_callback {
                callback_seen.get()
            } else {
                cancel_started.elapsed().as_millis() >= cancel_after_ms as u128
            }
        },
        |_, _| {
            cancelled_callbacks += 1;
            callback_seen.set(true);
        },
    );
    match cancelled {
        Err(Error::Cancelled) => {}
        Err(error) => return Err(format!("cancellation returned the wrong error: {error}").into()),
        Ok(_) => return Err("synthesis completed before the cancellation deadline".into()),
    }

    if after_callback && cancelled_callbacks == 0 {
        return Err("did not exercise GPT steps before cancellation".into());
    }
    let mut recovered_audio = Vec::new();
    let recovered_started = Instant::now();
    let mut recovered_samples = 0usize;
    let recovered = model.synthesize_streaming(TEXT, &|| false, |audio, _| {
        recovered_samples += audio.len();
        recovered_audio.extend_from_slice(audio);
    })?;
    if recovered_samples == 0 || recovered.semantic_tokens == 0 {
        return Err("post-cancellation synthesis returned no audio or semantic tokens".into());
    }

    if after_callback {
        if clean_audio.len() != recovered_audio.len() {
            return Err("post-cancellation PCM length changed".into());
        }
        let max_error = clean_audio
            .iter()
            .zip(&recovered_audio)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if !clean_audio
            .iter()
            .chain(&recovered_audio)
            .all(|x| x.is_finite())
            || max_error > 1e-5
        {
            return Err(format!("post-cancellation PCM mismatch: {max_error}").into());
        }
        println!("warmed_callback_cancel_pcm_max_error={max_error}");
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
