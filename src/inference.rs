use crate::{
    audio, check,
    frontend::Frontend,
    invalid,
    streaming::{StreamDecoder, CHUNK_TOKENS},
    validate_pack, validate_streaming_pack, Backend, Config, Pack, Result,
};
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, serde::Serialize)]
pub struct SynthesisTimings {
    pub frontend_ms: f64,
    pub gpt_ms: f64,
    pub vocoder_ms: f64,
    pub semantic_tokens: usize,
    pub first_audio_tokens: usize,
    pub audio_chunks: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ChunkInfo {
    pub semantic_tokens: usize,
    pub final_chunk: bool,
}

pub(crate) fn tf(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Tensor<f32>> {
    Ok(Tensor::from_array((shape.into(), data))?)
}
pub(crate) fn ti(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Result<Tensor<i64>> {
    Ok(Tensor::from_array((shape.into(), data))?)
}
pub(crate) fn open(dir: &Path, name: &str) -> Result<Session> {
    open_on(dir, name, Backend::Cpu)
}
pub(crate) fn open_on(dir: &Path, name: &str, backend: Backend) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(4)
        .map_err(|e| invalid(e.to_string()))?
        .with_inter_threads(1)
        .map_err(|e| invalid(e.to_string()))?;
    if backend == Backend::Directml {
        #[cfg(windows)]
        {
            // Required by DirectML. Explicit selection must report a failure,
            // rather than silently presenting a CPU session as GPU inference.
            builder = builder
                .with_memory_pattern(false)
                .map_err(|e| invalid(e.to_string()))?
                .with_parallel_execution(false)
                .map_err(|e| invalid(e.to_string()))?
                .with_execution_providers([ort::ep::DirectML::default()
                    .with_performance_preference(
                        ort::ep::directml::PerformancePreference::HighPerformance,
                    )
                    .build()
                    .error_on_failure()])
                .map_err(|e| invalid(e.to_string()))?;
        }
        #[cfg(not(windows))]
        return Err(invalid("DirectML TTS requires Windows; select CPU"));
    }
    Ok(builder.commit_from_file(dir.join(format!("{name}.onnx")))?)
}
fn take(out: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    out.remove(name)
        .ok_or_else(|| invalid(format!("incompatible ONNX graph: missing {name}")))
}

pub struct NativeTts {
    model_dir: PathBuf,
    backend: Backend,
    warmed: bool,
    frontend: Frontend,
    enc: Session,
    step: Session,
    sovits: Session,
    prompts: Tensor<i64>,
    spec: DynValue,
    sv: DynValue,
    reference_phones: Vec<i64>,
    reference_features: Vec<f32>,
    prompt_tokens: usize,
    streaming: bool,
}

impl NativeTts {
    pub fn load(config: &Config, cancel: &impl Fn() -> bool) -> Result<Self> {
        validate_pack(&config.model_dir)?;
        if config.streaming {
            validate_streaming_pack(&config.model_dir)?;
        }
        check(cancel)?;
        let dir = &config.model_dir;
        let pack: Pack = serde_json::from_slice(&std::fs::read(dir.join("pack.json"))?)?;
        if config.reference_audio.is_some() != config.reference_text.is_some() {
            return Err(invalid(
                "custom reference WAV and transcript must be configured together",
            ));
        }
        let reference_path = config
            .reference_audio
            .clone()
            .unwrap_or_else(|| dir.join(pack.reference_audio));
        let reference_text = config
            .reference_text
            .as_deref()
            .unwrap_or(&pack.reference_text);
        let (wav16, wav32) = audio::reference(&reference_path)?;
        let mut frontend = Frontend::load(dir, config.backend)?;
        check(cancel)?;
        let (reference_phones, reference_features) = frontend.process(reference_text, cancel)?;
        let mut ssl = open(dir, "ssl")?;
        check(cancel)?;
        let mut vq = open(dir, "vq_encoder")?;
        check(cancel)?;
        let mut spectrogram = open(dir, "spectrogram")?;
        check(cancel)?;
        let mut speaker = open(dir, "sv_embedding")?;
        check(cancel)?;
        let mut padded = wav16.clone();
        padded.resize(padded.len() + 4800, 0.0);
        let mut ssl_out = ssl.run(ort::inputs! {"audio"=>tf(vec![1,padded.len()],padded)?})?;
        let content = take(&mut ssl_out, "last_hidden_state")?;
        check(cancel)?;
        let vq_out = vq.run(ort::inputs! {"ssl_content"=>&content})?;
        let (_, codes) = vq_out
            .get("codes")
            .ok_or_else(|| invalid("incompatible VQ output"))?
            .try_extract_tensor::<i64>()?;
        let prompt_tokens = codes.len();
        let prompts = ti(vec![1, prompt_tokens], codes.to_vec())?;
        let spec = take(
            &mut spectrogram.run(ort::inputs! {"audio"=>tf(vec![1,wav32.len()],wav32)?})?,
            "spectrogram",
        )?;
        check(cancel)?;
        let sv = take(
            &mut speaker.run(ort::inputs! {"audio"=>tf(vec![1,wav16.len()],wav16)?})?,
            "sv_embedding",
        )?;
        check(cancel)?;
        // Reference-only sessions can release their weights before GPT/VITS load.
        drop(vq_out);
        drop(ssl_out);
        drop(content);
        drop((ssl, vq, spectrogram, speaker));
        let enc = open(dir, "gpt_encoder")?;
        check(cancel)?;
        let step = open(dir, "gpt_step")?;
        check(cancel)?;
        let sovits = open_on(
            dir,
            if config.streaming {
                "sovits_stream"
            } else {
                "sovits"
            },
            config.backend,
        )?;
        check(cancel)?;
        if reference_phones.len() + prompt_tokens + 32 >= 512 {
            return Err(invalid("reference exceeds KV cache budget"));
        }
        Ok(Self {
            model_dir: dir.to_owned(),
            backend: config.backend,
            warmed: false,
            frontend,
            enc,
            step,
            sovits,
            prompts,
            spec,
            sv,
            reference_phones,
            reference_features,
            prompt_tokens,
            streaming: config.streaming,
        })
    }

    /// Change only the vocoder. Keep GPT, frontend and reference tensors;
    /// failed/cancelled preparation leaves the original decoder usable.
    pub fn set_streaming(&mut self, enabled: bool, cancel: &impl Fn() -> bool) -> Result<()> {
        check(cancel)?;
        if enabled == self.streaming {
            return Ok(());
        }
        if enabled {
            validate_streaming_pack(&self.model_dir)?;
        }
        let next = open_on(
            &self.model_dir,
            if enabled { "sovits_stream" } else { "sovits" },
            self.backend,
        )?;
        check(cancel)?;
        self.sovits = next;
        self.streaming = enabled;
        self.warmed = false;
        Ok(())
    }

    /// Run representative inference without sending PCM to any consumer.
    /// Returns whether inference warmup was already completed in this mode.
    pub fn prepare(&mut self, cancel: &impl Fn() -> bool) -> Result<bool> {
        check(cancel)?;
        let already_warm = self.warmed;
        if !already_warm {
            self.synthesize_streaming("你好，欢迎回来。", cancel, |_, _| {})?;
        }
        check(cancel)?;
        Ok(already_warm)
    }

    /// Collects the selected decoder's audio. Cancellation is checked at every
    /// semantic token and after VITS, so interrupted output is never published.
    pub fn synthesize(&mut self, text: &str, cancel: &impl Fn() -> bool) -> Result<Vec<f32>> {
        self.synthesize_profiled(text, cancel)
            .map(|(audio, _)| audio)
    }

    pub fn synthesize_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
    ) -> Result<(Vec<f32>, SynthesisTimings)> {
        let mut samples = Vec::new();
        let timings =
            self.synthesize_streaming(text, cancel, |audio, _| samples.extend_from_slice(audio))?;
        Ok((samples, timings))
    }

    /// In token mode, emits complete audio fragments during GPT generation.
    /// In compatibility mode, delivers one complete sentence. Every fragment
    /// is checked for cancellation before and after the caller's FIFO write.
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        mut emit: impl FnMut(&[f32], ChunkInfo),
    ) -> Result<SynthesisTimings> {
        check(cancel)?;
        if text.chars().count() > 180 {
            return Err(invalid("native TTS sentence exceeds 180 characters"));
        }
        let started = Instant::now();
        let (phones, features) = self.frontend.process(text, cancel)?;
        let frontend_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let n = self.reference_phones.len() + phones.len();
        let budget = 512usize
            .checked_sub(n + self.prompt_tokens)
            .ok_or_else(|| {
                invalid("sentence exceeds KV cache budget; split into shorter sentences")
            })?;
        if budget < 32 {
            return Err(invalid(
                "sentence leaves insufficient semantic token budget",
            ));
        }
        let mut all_phones = self.reference_phones.clone();
        all_phones.extend_from_slice(&phones);
        let mut all_features = vec![0.0; 1024 * n];
        for row in 0..1024 {
            let r = self.reference_phones.len();
            let t = phones.len();
            all_features[row * n..row * n + r]
                .copy_from_slice(&self.reference_features[row * r..(row + 1) * r]);
            all_features[row * n + r..(row + 1) * n]
                .copy_from_slice(&features[row * t..(row + 1) * t]);
        }
        check(cancel)?;
        let mut out=self.enc.run(ort::inputs!{"phoneme_ids"=>ti(vec![1,n],all_phones)?,"prompts"=>&self.prompts,"bert_feature"=>tf(vec![1,1024,n],all_features)?})?;
        let x_len = out
            .get("x_len")
            .ok_or_else(|| invalid("missing GPT x_len"))?
            .try_extract_tensor::<i64>()?
            .1
            .to_vec();
        let y_len = out
            .get("y_len")
            .ok_or_else(|| invalid("missing GPT y_len"))?
            .try_extract_tensor::<i64>()?
            .1
            .to_vec();
        if x_len != [n as i64] || y_len != [self.prompt_tokens as i64] {
            return Err(invalid("GPT encoder has fixed fixture lengths"));
        }
        let mut values = take(&mut out, "topk_values")?;
        let mut ids = take(&mut out, "topk_indices")?;
        let mut k = take(&mut out, "k_cache")?;
        let mut v = take(&mut out, "v_cache")?;
        drop(out);
        let mut random = 1234u64;
        let mut tokens = Vec::new();
        let mut eos = false;
        let mut decoder = StreamDecoder::default();
        let mut vocoder_ms = 0.0;
        let mut delivery_ms = 0.0;
        let mut first_audio_tokens = 0;
        let mut audio_chunks = 0;
        for idx in 0..budget.min(350) {
            check(cancel)?;
            let token = pick(&values, &ids, &mut random)?;
            if token == 1024 {
                eos = true;
                break;
            }
            if !(0..1024).contains(&token) {
                return Err(invalid("invalid semantic token"));
            }
            tokens.push(token);
            let mut next=self.step.run(ort::inputs!{"samples"=>ti(vec![1,1],vec![token])?,"k_cache"=>&k,"v_cache"=>&v,
                "x_len"=>ti(vec![1],x_len.clone())?,"y_len"=>ti(vec![1],y_len.clone())?,"idx"=>ti(vec![1],vec![idx as i64])?})?;
            values = take(&mut next, "topk_values")?;
            ids = take(&mut next, "topk_indices")?;
            k = take(&mut next, "k_cache_new")?;
            v = take(&mut next, "v_cache_new")?;
            drop(next);
            if self.streaming && tokens.len() % CHUNK_TOKENS == 0 {
                let decode_started = Instant::now();
                let audio = decoder.decode(
                    &mut self.sovits,
                    &tokens,
                    &phones,
                    &self.spec,
                    &self.sv,
                    false,
                    cancel,
                )?;
                vocoder_ms += decode_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
                if first_audio_tokens == 0 {
                    first_audio_tokens = tokens.len();
                }
                audio_chunks += 1;
                let delivery_started = Instant::now();
                emit(
                    &audio,
                    ChunkInfo {
                        semantic_tokens: tokens.len(),
                        final_chunk: false,
                    },
                );
                delivery_ms += delivery_started.elapsed().as_secs_f64() * 1000.0;
                check(cancel)?;
            }
        }
        check(cancel)?;
        if !eos || tokens.is_empty() {
            return Err(invalid(
                "semantic generation did not finish within the cache budget; shorten the sentence",
            ));
        }
        let gpt_ms = (started.elapsed().as_secs_f64() * 1000.0 - vocoder_ms - delivery_ms).max(0.0);
        let semantic_tokens = tokens.len();
        let started = Instant::now();
        let samples = if self.streaming {
            decoder.decode(
                &mut self.sovits,
                &tokens,
                &phones,
                &self.spec,
                &self.sv,
                true,
                cancel,
            )?
        } else {
            let audio=self.sovits.run(ort::inputs!{"pred_semantic"=>ti(vec![1,1,tokens.len()],tokens)?,"text_seq"=>ti(vec![1,phones.len()],phones)?,
            "refer_spec"=>&self.spec,"sv_emb"=>&self.sv,"noise_scale"=>tf(vec![1],vec![0.0])?})?;
            let (_, samples) = audio
                .get("audio")
                .ok_or_else(|| invalid("incompatible VITS output"))?
                .try_extract_tensor::<f32>()?;
            samples.to_vec()
        };
        vocoder_ms += started.elapsed().as_secs_f64() * 1000.0;
        check(cancel)?;
        if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid synthesized audio"));
        }
        if first_audio_tokens == 0 {
            first_audio_tokens = semantic_tokens;
        }
        audio_chunks += 1;
        emit(
            &samples,
            ChunkInfo {
                semantic_tokens,
                final_chunk: true,
            },
        );
        check(cancel)?;
        self.warmed = true;
        Ok(SynthesisTimings {
            frontend_ms,
            gpt_ms,
            vocoder_ms,
            semantic_tokens,
            first_audio_tokens,
            audio_chunks,
        })
    }
}

fn pick(values: &DynValue, indices: &DynValue, state: &mut u64) -> Result<i64> {
    let (_, v) = values.try_extract_tensor::<f32>()?;
    let (_, ids) = indices.try_extract_tensor::<i64>()?;
    if v.is_empty() || v.len() != ids.len() || v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid semantic logits"));
    }
    let count = 15.min(v.len());
    let peak = v[..count].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let weights: Vec<f64> = v[..count]
        .iter()
        .map(|x| (*x as f64 - peak).exp())
        .collect();
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let target = (*state >> 11) as f64 / ((1u64 << 53) as f64) * weights.iter().sum::<f64>();
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return Ok(ids[i]);
        }
    }
    Ok(ids[count - 1])
}
