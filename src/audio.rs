use crate::{invalid, Result};
use std::path::Path;

pub(crate) fn reference(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    if std::fs::metadata(path)?.len() > 8 * 1024 * 1024 {
        return Err(invalid("reference WAV is too large"));
    }
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.channels > 2 || !(8_000..=96_000).contains(&spec.sample_rate) {
        return Err(invalid("reference requires mono/stereo WAV at 8–96 kHz"));
    }
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            if !matches!(spec.bits_per_sample, 8 | 16 | 24 | 32) {
                return Err(invalid("unsupported reference PCM bit depth"));
            }
            let divisor = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / divisor))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    if data.iter().any(|x| !x.is_finite()) {
        return Err(invalid("non-finite reference samples"));
    }
    let mono: Vec<f32> = data
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    let duration = mono.len() as f64 / spec.sample_rate as f64;
    if !(3.0..=10.0).contains(&duration) {
        return Err(invalid("reference must be 3–10 seconds long"));
    }
    let peak = mono.iter().copied().map(f32::abs).fold(0.0, f32::max);
    if peak < 0.001 {
        return Err(invalid("reference audio is silent"));
    }
    Ok((
        resample(&mono, spec.sample_rate, 16_000),
        resample(&mono, spec.sample_rate, 32_000),
    ))
}

/// Windowed sinc with a low-pass cutoff for downsampling. Reference preparation
/// runs once per configuration; this avoids aliasing from linear interpolation.
fn resample(audio: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return audio.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0) * 0.95;
    let radius = 32i64;
    (0..(audio.len() as f64 * ratio).round() as usize)
        .map(|i| {
            let pos = i as f64 / ratio;
            let center = pos.floor() as i64;
            let mut total = 0.0;
            let mut weight = 0.0;
            for j in center - radius..=center + radius {
                if j < 0 || j >= audio.len() as i64 {
                    continue;
                }
                let distance = pos - j as f64;
                if distance.abs() > radius as f64 {
                    continue;
                }
                let x = std::f64::consts::PI * distance * cutoff;
                let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos();
                let w = sinc * window * cutoff;
                total += audio[j as usize] as f64 * w;
                weight += w;
            }
            (total / weight) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn downsampling_suppresses_out_of_band_tones() {
        let signal: Vec<f32> = (0..32000)
            .map(|i| (2.0 * std::f32::consts::PI * 12000.0 * i as f32 / 32000.0).sin())
            .collect();
        let output = resample(&signal, 32000, 16000);
        assert_eq!(output.len(), 16000);
        let rms = (output[100..15900].iter().map(|x| x * x).sum::<f32>() / 15800.0).sqrt();
        assert!(rms < 0.001, "{rms}");
    }
}
