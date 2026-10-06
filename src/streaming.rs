//! v2Pro latent overlap and SOLA stitching, independent for each utterance.
use crate::{
    check,
    inference::{tf, ti},
    invalid, Result,
};
use ort::{
    memory::Allocator,
    session::Session,
    value::{DynValue, Tensor},
};

pub(crate) const CHUNK_TOKENS: usize = 24;
const CHANNELS: usize = 192;
const SAMPLES_PER_TOKEN: usize = 1280;
const OVERLAP_TOKENS: usize = 2;
const OVERLAP_SAMPLES: usize = OVERLAP_TOKENS * SAMPLES_PER_TOKEN;

#[derive(Default)]
pub(crate) struct StreamDecoder {
    latent: Vec<f32>,
    latent_frames: usize,
    tokens: usize,
    tail: Vec<f32>,
}

impl StreamDecoder {
    pub fn decode(
        &mut self,
        session: &mut Session,
        tokens: &[i64],
        phones: &[i64],
        spec: &DynValue,
        sv: &DynValue,
        final_chunk: bool,
        cancel: &impl Fn() -> bool,
    ) -> Result<Vec<f32>> {
        check(cancel)?;
        let added = tokens
            .len()
            .checked_sub(self.tokens)
            .ok_or_else(|| invalid("streaming token count moved backwards"))?;
        if added == 0 {
            return if final_chunk {
                Ok(std::mem::take(&mut self.tail))
            } else {
                Err(invalid("empty non-final semantic chunk"))
            };
        }
        // Original decode_streaming keeps at least ten new/context tokens
        // for the vocoder when EOS leaves a short last chunk.
        let overlap = if self.tokens == 0 {
            0
        } else {
            OVERLAP_TOKENS + 10usize.saturating_sub(added)
        };
        let frames = overlap * 2;
        if frames > self.latent_frames {
            return Err(invalid("insufficient streaming latent context"));
        }
        let mut previous = Vec::with_capacity(CHANNELS * frames);
        for channel in 0..CHANNELS {
            let end = (channel + 1) * self.latent_frames;
            previous.extend_from_slice(&self.latent[end - frames..end]);
        }
        let result = added + overlap;
        // from_array rejects zero-sized dimensions; ORT's allocator supports
        // the empty context required by the first streaming chunk.
        let previous = if frames == 0 {
            Tensor::<f32>::new(&Allocator::default(), [1_i64, CHANNELS as i64, 0])?
        } else {
            tf(vec![1, CHANNELS, frames], previous)?
        };
        let out = session.run(ort::inputs! {
            "pred_semantic" => ti(vec![1,1,tokens.len()],tokens.to_vec())?,
            "text_seq" => ti(vec![1,phones.len()],phones.to_vec())?,
            "refer_spec" => spec, "sv_emb" => sv,
            "result_length" => ti(vec![1],vec![result as i64])?,
            "overlap_latent" => previous
        })?;
        let (shape, latent) = out
            .get("latent")
            .ok_or_else(|| invalid("missing streaming latent output"))?
            .try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, CHANNELS as i64, (result * 2) as i64]
            || latent.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid streaming latent output"));
        }
        let (shape, samples) = out
            .get("audio")
            .ok_or_else(|| invalid("missing streaming audio output"))?
            .try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, 1, (result * SAMPLES_PER_TOKEN) as i64]
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid streaming audio length/data"));
        }
        check(cancel)?;
        self.latent = latent.to_vec();
        self.latent_frames = result * 2;
        self.tokens = tokens.len();
        let extra = overlap.saturating_sub(OVERLAP_TOKENS) * SAMPLES_PER_TOKEN;
        let mut audio = samples[extra..].to_vec();
        if !self.tail.is_empty() {
            audio = sola(&self.tail, &audio);
        }
        self.tail.clear();
        if !final_chunk {
            if audio.len() <= OVERLAP_SAMPLES {
                return Err(invalid("streaming chunk is shorter than its overlap"));
            }
            self.tail = audio.split_off(audio.len() - OVERLAP_SAMPLES);
        }
        Ok(audio)
    }
}

/// Align at most 10 ms and blend the withheld 80 ms with a periodic Hann
/// window, matching GPT-SoVITS' SOLA rule. Never repeat already-emitted PCM.
fn sola(tail: &[f32], audio: &[f32]) -> Vec<f32> {
    let overlap = tail.len();
    let search = 320.min(audio.len().saturating_sub(overlap));
    let mut best = f64::NEG_INFINITY;
    let mut offset = 0;
    for candidate in 0..=search {
        let segment = &audio[candidate..candidate + overlap];
        let dot: f64 = tail
            .iter()
            .zip(segment)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let energy: f64 = segment.iter().map(|x| f64::from(*x).powi(2)).sum();
        let score = dot / (energy + 1e-8).sqrt();
        if score > best {
            best = score;
            offset = candidate;
        }
    }
    let mut result = audio[offset..].to_vec();
    for (i, previous) in tail.iter().enumerate() {
        let weight = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / overlap as f64).cos();
        result[i] = (weight * f64::from(result[i]) + (1.0 - weight) * f64::from(*previous)) as f32;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_chunk_accepts_empty_context_tensor() {
        let tensor = Tensor::<f32>::new(&Allocator::default(), [1_i64, 192, 0]).unwrap();
        let (shape, data) = tensor.try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.as_ref(), [1, 192, 0]);
        assert!(data.is_empty());
    }
    #[test]
    fn sola_recovers_known_offset_without_repeating_previous_audio() {
        let tail: Vec<f32> = (0..2560)
            .map(|i| (i as f32 * 0.13).sin() * (i as f32 * 0.071).cos())
            .collect();
        let mut audio = vec![0.0; 79];
        audio.extend_from_slice(&tail);
        audio.extend(vec![0.25; 1280]);
        let joined = sola(&tail, &audio);
        assert_eq!(joined.len(), 3840);
        assert!(joined[..2560]
            .iter()
            .zip(&tail)
            .all(|(a, b)| (a - b).abs() < 1e-6));
        assert_eq!(&joined[2560..], vec![0.25; 1280]);
    }
    #[test]
    fn silent_overlap_uses_zero_offset_and_remains_finite() {
        let joined = sola(&vec![0.0; 2560], &vec![0.0; 3840]);
        assert_eq!(joined, vec![0.0; 3840]);
    }
}
