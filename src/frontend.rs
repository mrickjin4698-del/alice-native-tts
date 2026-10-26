pub mod normalize;
use normalize::normalize;
mod pinyin;
use crate::profiling::{ProfileCollector, ProfilePhase};
use crate::{
    check,
    inference::{open_on, tf, ti},
    invalid, Backend, Result,
};
use ort::session::Session;
use serde::Deserialize;
use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};
use tokenizers::Tokenizer;

const FRONTEND_CACHE_MAX_BYTES: usize = 8 * 1024 * 1024;
const FRONTEND_CACHE_MAX_ENTRIES: usize = 32;
const FRONTEND_CACHE_SCHEMA: u8 = 1;

struct FrontendCacheEntry {
    phones: Vec<i64>,
    features: Vec<f32>,
    bytes: usize,
}

struct G2pwBatchInputs {
    count: usize,
    len: usize,
    positions: Vec<usize>,
    input_ids: Vec<i64>,
    token_type_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    phoneme_mask: Vec<f32>,
    char_ids: Vec<i64>,
    position_ids: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FrontendLanguageRoute {
    Chinese,
    English,
    Mixed,
    Empty,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FrontendCacheConfig {
    schema: u8,
    g2pw_batch_size: usize,
    g2pw_backend: u8,
    bert_backend: u8,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FrontendCacheKey {
    normalized_text: String,
    language_route: FrontendLanguageRoute,
    config: FrontendCacheConfig,
}

fn frontend_graph_backend(name: &str, fallback: Backend) -> Result<Backend> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(fallback);
    };
    match value.as_str() {
        "cpu" => Ok(Backend::Cpu),
        "directml" => Ok(Backend::Directml),
        "cuda" => {
            #[cfg(feature = "r1-cuda-ep")]
            {
                Ok(Backend::Cuda)
            }
            #[cfg(not(feature = "r1-cuda-ep"))]
            {
                Err(invalid(format!(
                    "{name}=cuda requires the r1-cuda-ep feature"
                )))
            }
        }
        "tensor_rt" => {
            #[cfg(feature = "r1-tensorrt-ep")]
            {
                Ok(Backend::TensorRt)
            }
            #[cfg(not(feature = "r1-tensorrt-ep"))]
            {
                Err(invalid(format!(
                    "{name}=tensor_rt requires the r1-tensorrt-ep feature"
                )))
            }
        }
        _ => Err(invalid(format!(
            "{name} must be cpu, directml, cuda, or tensor_rt"
        ))),
    }
}

fn backend_cache_tag(backend: Backend) -> u8 {
    match backend {
        Backend::Cpu => 0,
        Backend::Directml => 1,
        #[cfg(feature = "r1-cuda-ep")]
        Backend::Cuda => 2,
        #[cfg(feature = "r1-tensorrt-ep")]
        Backend::TensorRt => 3,
    }
}

fn frontend_language_route(text: &str) -> FrontendLanguageRoute {
    let has_chinese = text.chars().any(is_han);
    let has_english = text.chars().any(|ch| ch.is_ascii_alphabetic());
    match (has_chinese, has_english) {
        (true, true) => FrontendLanguageRoute::Mixed,
        (true, false) => FrontendLanguageRoute::Chinese,
        (false, true) => FrontendLanguageRoute::English,
        (false, false) => FrontendLanguageRoute::Empty,
    }
}

fn frontend_cache_key(normalized_text: &str, config: FrontendCacheConfig) -> FrontendCacheKey {
    FrontendCacheKey {
        language_route: frontend_language_route(normalized_text),
        normalized_text: normalized_text.to_owned(),
        config,
    }
}

#[derive(Deserialize)]
struct Mono {
    phone: String,
}
#[derive(Deserialize)]
struct Poly {
    index: usize,
    phones: Vec<(String, usize)>,
}
#[derive(Deserialize)]
struct EnglishModel {
    phonemes: Vec<String>,
    enc_emb: Vec<f32>,
    enc_w_ih: Vec<f32>,
    enc_w_hh: Vec<f32>,
    enc_b_ih: Vec<f32>,
    enc_b_hh: Vec<f32>,
    dec_emb: Vec<f32>,
    dec_w_ih: Vec<f32>,
    dec_w_hh: Vec<f32>,
    dec_b_ih: Vec<f32>,
    dec_b_hh: Vec<f32>,
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

/// The exported BERT expects Chinese character features; English features are zero.
pub(crate) struct Frontend {
    bert: Session,
    g2pw: Session,
    tokenizer: Tokenizer,
    mono: HashMap<char, Mono>,
    poly: HashMap<char, Poly>,
    labels: Vec<String>,
    symbols: HashMap<String, i64>,
    cmu: HashMap<String, Vec<String>>,
    english: EnglishModel,
    frontend_cache_enabled: bool,
    frontend_cache_force_miss: bool,
    frontend_cache: HashMap<FrontendCacheKey, FrontendCacheEntry>,
    frontend_cache_lru: VecDeque<FrontendCacheKey>,
    frontend_cache_bytes: usize,
    g2pw_batch_size: usize,
    cache_config: FrontendCacheConfig,
    parallel_frontend: bool,
}

impl Frontend {
    pub fn load(dir: &Path, backend: Backend) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let symbols = serde_json::from_value(
            config
                .get("symbol_to_id")
                .cloned()
                .ok_or_else(|| invalid("missing symbol_to_id"))?,
        )?;
        let english: EnglishModel =
            serde_json::from_slice(&std::fs::read(dir.join("g2p-en.json"))?)?;
        for (values, len) in [
            (&english.enc_emb, 29 * 256),
            (&english.enc_w_ih, 768 * 256),
            (&english.enc_w_hh, 768 * 256),
            (&english.enc_b_ih, 768),
            (&english.enc_b_hh, 768),
            (&english.dec_emb, 74 * 256),
            (&english.dec_w_ih, 768 * 256),
            (&english.dec_w_hh, 768 * 256),
            (&english.dec_b_ih, 768),
            (&english.dec_b_hh, 768),
            (&english.fc_w, 74 * 256),
            (&english.fc_b, 74),
        ] {
            if values.len() != len || values.iter().any(|x| !x.is_finite()) {
                return Err(invalid("invalid English G2P weights"));
            }
        }
        if english.phonemes.len() != 74 {
            return Err(invalid("invalid English phoneme vocabulary"));
        }
        let g2pw_batch_size = std::env::var("ALICE_NATIVE_TTS_G2PW_BATCH_SIZE")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| invalid("ALICE_NATIVE_TTS_G2PW_BATCH_SIZE must be an integer"))
            })
            .transpose()?
            .unwrap_or(4);
        if !(1..=64).contains(&g2pw_batch_size) {
            return Err(invalid(
                "ALICE_NATIVE_TTS_G2PW_BATCH_SIZE must be between 1 and 64",
            ));
        }
        let g2pw_backend =
            frontend_graph_backend("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND", backend)?;
        let bert_backend =
            frontend_graph_backend("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND", backend)?;
        Ok(Self {
            bert: open_on(dir, "bert", bert_backend)?,
            g2pw: open_on(dir, "g2pw", g2pw_backend)?,
            tokenizer: Tokenizer::from_bytes(include_bytes!("frontend/g2pw_tokenizer.json"))
                .map_err(|e| invalid(e.to_string()))?,
            mono: serde_json::from_str(include_str!("frontend/dict_mono_chars.json"))?,
            poly: serde_json::from_str(include_str!("frontend/dict_poly_chars.json"))?,
            labels: serde_json::from_str(include_str!("frontend/dict_poly_index_list.json"))?,
            cmu: serde_json::from_slice(&std::fs::read(dir.join("cmudict.json"))?)?,
            symbols,
            english,
            frontend_cache_enabled: std::env::var("ALICE_NATIVE_TTS_FRONTEND_CACHE")
                .ok()
                .as_deref()
                != Some("0"),
            frontend_cache_force_miss: std::env::var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS")
                .ok()
                .as_deref()
                == Some("1"),
            frontend_cache: HashMap::new(),
            frontend_cache_lru: VecDeque::new(),
            frontend_cache_bytes: 0,
            g2pw_batch_size,
            cache_config: FrontendCacheConfig {
                schema: FRONTEND_CACHE_SCHEMA,
                g2pw_batch_size,
                g2pw_backend: backend_cache_tag(g2pw_backend),
                bert_backend: backend_cache_tag(bert_backend),
            },
            parallel_frontend: std::env::var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL")
                .ok()
                .as_deref()
                == Some("1"),
        })
    }

    #[cfg(feature = "r0-profiling")]
    pub(crate) fn finish_ort_profiling(&mut self) -> Result<Vec<std::path::PathBuf>> {
        Ok(vec![
            self.bert.end_profiling()?.into(),
            self.g2pw.end_profiling()?.into(),
        ])
    }

    fn phone(&self, s: &str) -> Result<i64> {
        self.symbols
            .get(s)
            .copied()
            .ok_or_else(|| invalid(format!("unsupported phoneme: {s}")))
    }

    fn chinese_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        profile: &mut ProfileCollector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let batch_size = self.g2pw_batch_size;
        self.chinese_with_batch_profiled(text, cancel, batch_size, profile)
    }

    #[cfg(test)]
    fn chinese_with_batch(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        batch_size: usize,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut profile = ProfileCollector::disabled();
        self.chinese_with_batch_profiled(text, cancel, batch_size, &mut profile)
    }

    fn chinese_with_batch_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        batch_size: usize,
        profile: &mut ProfileCollector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        if self.parallel_frontend {
            return self.chinese_parallel_profiled(text, cancel, batch_size, profile);
        }
        let (chars, ids) = profile.measure(ProfilePhase::FrontendTokenizer, || {
            let chars: Vec<char> = text.chars().collect();
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| invalid(e.to_string()))?;
            let ids: Vec<i64> = encoding.get_ids().iter().map(|x| i64::from(*x)).collect();
            Ok::<_, crate::Error>((chars, ids))
        })?;
        if ids.len() != chars.len() + 2 {
            return Err(invalid("Chinese tokenizer/character alignment mismatch"));
        }
        let len = ids.len();
        let (mut pinyins, queries) = profile.measure(ProfilePhase::FrontendG2pwPrepare, || {
            let pinyins = chars
                .iter()
                .map(|ch| {
                    self.mono
                        .get(ch)
                        .map_or_else(|| ch.to_string(), |m| m.phone.clone())
                })
                .collect::<Vec<_>>();
            // Own the small dictionary payload so profiling can isolate input
            // preparation from G2PW execution without holding a self borrow.
            let queries = chars
                .iter()
                .enumerate()
                .filter_map(|(i, ch)| self.poly.get(ch).map(|p| (i, p.index, p.phones.clone())))
                .collect::<Vec<_>>();
            (pinyins, queries)
        });
        for batch in queries.chunks(batch_size) {
            check(cancel)?;
            let count = batch.len();
            let (input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids) =
                profile.measure(ProfilePhase::FrontendG2pwPrepare, || {
                    let mut masks = vec![0.0; count * self.labels.len()];
                    let mut char_ids = Vec::with_capacity(count);
                    let mut positions = Vec::with_capacity(count);
                    for (row, (i, poly_index, poly_phones)) in batch.iter().enumerate() {
                        for (_, index) in poly_phones {
                            masks[row * self.labels.len() + *index] = 1.0;
                        }
                        char_ids.push(*poly_index as i64);
                        positions.push((i + 1) as i64);
                    }
                    Ok::<_, crate::Error>((
                        ti(vec![count, len], ids.repeat(count))?,
                        ti(vec![count, len], vec![0; count * len])?,
                        ti(vec![count, len], vec![1; count * len])?,
                        tf(vec![count, self.labels.len()], masks)?,
                        ti(vec![count], char_ids)?,
                        ti(vec![count], positions)?,
                    ))
                })?;
            let out = profile.measure(ProfilePhase::FrontendG2pwRun, || {
                self.g2pw.run(ort::inputs! {
                    "input_ids" => input_ids,
                    "token_type_ids" => token_type_ids,
                    "attention_mask" => attention_mask,
                    "phoneme_mask" => phoneme_mask,
                    "char_ids" => char_ids,
                    "position_ids" => position_ids
                })
            })?;
            let (_, probs) = out
                .get("probs")
                .ok_or_else(|| invalid("incompatible G2PW output"))?
                .try_extract_tensor::<f32>()?;
            if probs.len() != count * self.labels.len() || probs.iter().any(|v| !v.is_finite()) {
                return Err(invalid("invalid G2PW probabilities"));
            }
            for ((position, _, _), row) in batch.iter().zip(probs.chunks_exact(self.labels.len())) {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                pinyins[*position] = self.labels[index].clone();
            }
        }
        let (phones, word2ph) = profile.measure(ProfilePhase::FrontendPhonemeMap, || {
            tone_sandhi(&chars, &mut pinyins);
            let mut phones = Vec::new();
            let mut word2ph = Vec::new();
            for p in &pinyins {
                if p.len() == 1 && is_punctuation(p.chars().next().unwrap()) {
                    phones.push(self.phone(p)?);
                    word2ph.push(1usize);
                } else {
                    let (initial, final_) = pinyin::split_zh_ph(p);
                    phones.push(self.phone(initial)?);
                    if !final_.is_empty() {
                        phones.push(self.phone(final_)?);
                        word2ph.push(2);
                    } else {
                        word2ph.push(1);
                    }
                }
            }
            Ok::<_, crate::Error>((phones, word2ph))
        })?;
        check(cancel)?;
        let (input_ids, attention_mask, token_type_ids) =
            profile.measure(ProfilePhase::FrontendBertInput, || {
                Ok::<_, crate::Error>((
                    ti(vec![1, len], ids)?,
                    ti(vec![1, len], vec![1; len])?,
                    ti(vec![1, len], vec![0; len])?,
                ))
            })?;
        let out = profile.measure(ProfilePhase::FrontendBertRun, || {
            self.bert.run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids
            })
        })?;
        let (shape, data) = out
            .get("hidden_states")
            .ok_or_else(|| invalid("incompatible BERT output"))?
            .try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, len as i64, 1024] {
            return Err(invalid("incompatible BERT export"));
        }
        let features = profile.measure(ProfilePhase::FrontendFeatureExpand, || {
            let n = phones.len();
            let mut features = vec![0.0; n * 1024];
            let mut col = 0;
            for (i, count) in word2ph.iter().enumerate() {
                for _ in 0..*count {
                    for row in 0..1024 {
                        features[row * n + col] = data[(i + 1) * 1024 + row];
                    }
                    col += 1;
                }
            }
            features
        });
        Ok((phones, features))
    }

    fn chinese_parallel_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        batch_size: usize,
        profile: &mut ProfileCollector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let (chars, ids) = profile.measure(ProfilePhase::FrontendTokenizer, || {
            let chars: Vec<char> = text.chars().collect();
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| invalid(e.to_string()))?;
            let ids: Vec<i64> = encoding.get_ids().iter().map(|x| i64::from(*x)).collect();
            Ok::<_, crate::Error>((chars, ids))
        })?;
        if ids.len() != chars.len() + 2 {
            return Err(invalid("Chinese tokenizer/character alignment mismatch"));
        }
        let len = ids.len();
        let (mut pinyins, queries) = profile.measure(ProfilePhase::FrontendG2pwPrepare, || {
            let pinyins = chars
                .iter()
                .map(|ch| {
                    self.mono
                        .get(ch)
                        .map_or_else(|| ch.to_string(), |m| m.phone.clone())
                })
                .collect::<Vec<_>>();
            let queries = chars
                .iter()
                .enumerate()
                .filter_map(|(i, ch)| self.poly.get(ch).map(|p| (i, p.index, p.phones.clone())))
                .collect::<Vec<_>>();
            (pinyins, queries)
        });
        let batches = profile.measure(ProfilePhase::FrontendG2pwPrepare, || {
            queries
                .chunks(batch_size)
                .map(|batch| {
                    let count = batch.len();
                    let mut phoneme_mask = vec![0.0; count * self.labels.len()];
                    let mut char_ids = Vec::with_capacity(count);
                    let mut position_ids = Vec::with_capacity(count);
                    let mut positions = Vec::with_capacity(count);
                    for (row, (position, poly_index, poly_phones)) in batch.iter().enumerate() {
                        for (_, index) in poly_phones {
                            phoneme_mask[row * self.labels.len() + *index] = 1.0;
                        }
                        char_ids.push(*poly_index as i64);
                        position_ids.push((*position + 1) as i64);
                        positions.push(*position);
                    }
                    G2pwBatchInputs {
                        count,
                        len,
                        positions,
                        input_ids: ids.repeat(count),
                        token_type_ids: vec![0; count * len],
                        attention_mask: vec![1; count * len],
                        phoneme_mask,
                        char_ids,
                        position_ids,
                    }
                })
                .collect::<Vec<_>>()
        });
        let (bert_input_ids, bert_attention_mask, bert_token_type_ids) =
            profile.measure(ProfilePhase::FrontendBertInput, || {
                Ok::<_, crate::Error>((
                    ti(vec![1, len], ids)?,
                    ti(vec![1, len], vec![1; len])?,
                    ti(vec![1, len], vec![0; len])?,
                ))
            })?;
        check(cancel)?;

        let parallel_started = std::time::Instant::now();
        let (g2pw_started, g2pw_ended, g2pw_result, bert_started, bert_ended, bert_result) =
            std::thread::scope(|scope| {
                let bert_session = &mut self.bert;
                let bert_handle = scope.spawn(move || {
                    let started = std::time::Instant::now();
                    let result = (|| {
                        let out = bert_session.run(ort::inputs! {
                            "input_ids" => bert_input_ids,
                            "attention_mask" => bert_attention_mask,
                            "token_type_ids" => bert_token_type_ids
                        })?;
                        let (shape, data) = out
                            .get("hidden_states")
                            .ok_or_else(|| invalid("incompatible BERT output"))?
                            .try_extract_tensor::<f32>()?;
                        if shape.as_ref() != [1, len as i64, 1024] {
                            return Err(invalid("incompatible BERT export"));
                        }
                        Ok::<_, crate::Error>(data.to_vec())
                    })();
                    (started, std::time::Instant::now(), result)
                });

                let g2pw_session = &mut self.g2pw;
                let labels_len = self.labels.len();
                let g2pw_started = std::time::Instant::now();
                let mut g2pw_result = Ok(Vec::<(usize, usize)>::new());
                for batch in batches {
                    if let Err(error) = check(cancel) {
                        g2pw_result = Err(error);
                        break;
                    }
                    let result = (|| {
                        let out = g2pw_session.run(ort::inputs! {
                            "input_ids" => ti(vec![batch.count, batch.len], batch.input_ids)?,
                            "token_type_ids" => ti(vec![batch.count, batch.len], batch.token_type_ids)?,
                            "attention_mask" => ti(vec![batch.count, batch.len], batch.attention_mask)?,
                            "phoneme_mask" => tf(vec![batch.count, labels_len], batch.phoneme_mask)?,
                            "char_ids" => ti(vec![batch.count], batch.char_ids)?,
                            "position_ids" => ti(vec![batch.count], batch.position_ids)?
                        })?;
                        let (_, probs) = out
                            .get("probs")
                            .ok_or_else(|| invalid("incompatible G2PW output"))?
                            .try_extract_tensor::<f32>()?;
                        if probs.len() != batch.count * labels_len
                            || probs.iter().any(|value| !value.is_finite())
                        {
                            return Err(invalid("invalid G2PW probabilities"));
                        }
                        let choices = batch
                            .positions
                            .iter()
                            .zip(probs.chunks_exact(labels_len))
                            .map(|(position, row)| {
                                let index = row
                                    .iter()
                                    .enumerate()
                                    .max_by(|a, b| a.1.total_cmp(b.1))
                                    .map_or(0, |(index, _)| index);
                                (*position, index)
                            })
                            .collect::<Vec<_>>();
                        Ok::<_, crate::Error>(choices)
                    })();
                    match result {
                        Ok(choices) => g2pw_result.as_mut().unwrap().extend(choices),
                        Err(error) => {
                            g2pw_result = Err(error);
                            break;
                        }
                    }
                }
                let g2pw_ended = std::time::Instant::now();
                let (bert_started, bert_ended, bert_result) = bert_handle
                    .join()
                    .map_err(|_| invalid("parallel BERT worker panicked"))?;
                Ok::<_, crate::Error>((
                    g2pw_started,
                    g2pw_ended,
                    g2pw_result,
                    bert_started,
                    bert_ended,
                    bert_result,
                ))
            })?;
        let parallel_ended = std::time::Instant::now();
        profile.record_external_event(ProfilePhase::FrontendG2pwRun, g2pw_started, g2pw_ended);
        profile.record_external_event(ProfilePhase::FrontendBertRun, bert_started, bert_ended);
        profile.record_external_event(
            ProfilePhase::FrontendParallelRun,
            parallel_started,
            parallel_ended,
        );
        let g2pw_choices = g2pw_result?;
        let bert_features = bert_result?;
        check(cancel)?;
        for (position, index) in g2pw_choices {
            pinyins[position] = self.labels[index].clone();
        }
        let (phones, word2ph) = profile.measure(ProfilePhase::FrontendPhonemeMap, || {
            tone_sandhi(&chars, &mut pinyins);
            let mut phones = Vec::new();
            let mut word2ph = Vec::new();
            for pinyin in &pinyins {
                if pinyin.len() == 1 && is_punctuation(pinyin.chars().next().unwrap()) {
                    phones.push(self.phone(pinyin)?);
                    word2ph.push(1usize);
                } else {
                    let (initial, final_) = pinyin::split_zh_ph(pinyin);
                    phones.push(self.phone(initial)?);
                    if !final_.is_empty() {
                        phones.push(self.phone(final_)?);
                        word2ph.push(2);
                    } else {
                        word2ph.push(1);
                    }
                }
            }
            Ok::<_, crate::Error>((phones, word2ph))
        })?;
        let features = profile.measure(ProfilePhase::FrontendFeatureExpand, || {
            let n = phones.len();
            let mut features = vec![0.0; n * 1024];
            let mut col = 0;
            for (i, count) in word2ph.iter().enumerate() {
                for _ in 0..*count {
                    for row in 0..1024 {
                        features[row * n + col] = bert_features[(i + 1) * 1024 + row];
                    }
                    col += 1;
                }
            }
            features
        });
        Ok((phones, features))
    }

    fn english(&self, text: &str, cancel: &impl Fn() -> bool) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut phones = Vec::new();
        let mut word = String::new();
        for ch in text.chars().chain(std::iter::once(' ')) {
            if ch.is_ascii_alphabetic() || ch == '\'' {
                word.push(ch.to_ascii_lowercase());
                continue;
            }
            if !word.is_empty() {
                check(cancel)?;
                let predicted;
                let pronunciation = if let Some(p) = self.cmu.get(&word) {
                    p
                } else {
                    predicted = self.english.predict(&word);
                    &predicted
                };
                for p in pronunciation {
                    phones.push(self.phone(p)?);
                }
                word.clear();
            }
            if is_punctuation(ch) {
                phones.push(self.phone(&ch.to_string())?);
            }
        }
        let features = vec![0.0; phones.len() * 1024];
        Ok((phones, features))
    }

    pub fn process(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut profile = ProfileCollector::disabled();
        self.process_profiled(text, cancel, &mut profile)
    }

    pub(crate) fn process_profiled(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        profile: &mut ProfileCollector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let normalized = profile.measure(ProfilePhase::FrontendNormalize, || normalize(text));
        let key = frontend_cache_key(&normalized, self.cache_config);
        if self.frontend_cache_enabled {
            let cached = profile.measure(ProfilePhase::FrontendCacheLookup, || {
                if self.frontend_cache_force_miss {
                    self.frontend_cache.clear();
                    self.frontend_cache_lru.clear();
                    self.frontend_cache_bytes = 0;
                }
                self.frontend_cache_get(&key)
            });
            if let Some(output) = cached {
                check(cancel)?;
                profile.record_frontend_cache_hit();
                return Ok(output);
            }
            profile.record_frontend_cache_miss();
        }
        let output = self.process_normalized_profiled(&normalized, cancel, profile)?;
        if self.frontend_cache_enabled {
            profile.measure(ProfilePhase::FrontendCacheStore, || {
                self.frontend_cache_insert(&key, &output.0, &output.1)
            });
        }
        Ok(output)
    }

    fn frontend_cache_get(&mut self, key: &FrontendCacheKey) -> Option<(Vec<i64>, Vec<f32>)> {
        let (phones, features) = {
            let entry = self.frontend_cache.get(key)?;
            (entry.phones.clone(), entry.features.clone())
        };
        self.frontend_cache_lru.retain(|existing| existing != key);
        self.frontend_cache_lru.push_back(key.clone());
        Some((phones, features))
    }

    fn frontend_cache_insert(&mut self, key: &FrontendCacheKey, phones: &[i64], features: &[f32]) {
        let bytes = phones
            .len()
            .saturating_mul(std::mem::size_of::<i64>())
            .saturating_add(features.len().saturating_mul(std::mem::size_of::<f32>()))
            .saturating_add(key.normalized_text.len())
            .saturating_add(std::mem::size_of::<FrontendCacheConfig>());
        if bytes > FRONTEND_CACHE_MAX_BYTES {
            return;
        }
        if let Some(previous) = self.frontend_cache.remove(key) {
            self.frontend_cache_bytes = self.frontend_cache_bytes.saturating_sub(previous.bytes);
        }
        self.frontend_cache_lru.retain(|existing| existing != key);
        while self.frontend_cache_bytes.saturating_add(bytes) > FRONTEND_CACHE_MAX_BYTES
            || self.frontend_cache.len() >= FRONTEND_CACHE_MAX_ENTRIES
        {
            let Some(oldest) = self.frontend_cache_lru.pop_front() else {
                break;
            };
            if let Some(removed) = self.frontend_cache.remove(&oldest) {
                self.frontend_cache_bytes = self.frontend_cache_bytes.saturating_sub(removed.bytes);
            }
        }
        self.frontend_cache.insert(
            key.clone(),
            FrontendCacheEntry {
                phones: phones.to_vec(),
                features: features.to_vec(),
                bytes,
            },
        );
        self.frontend_cache_lru.push_back(key.clone());
        self.frontend_cache_bytes = self.frontend_cache_bytes.saturating_add(bytes);
    }

    fn process_normalized_profiled(
        &mut self,
        normalized: &str,
        cancel: &impl Fn() -> bool,
        profile: &mut ProfileCollector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let pieces = profile.measure(ProfilePhase::FrontendRouting, || {
            let mut pieces: Vec<(bool, String)> = Vec::new();
            for ch in normalized.chars() {
                let zh = is_han(ch);
                if zh || ch.is_ascii_alphabetic() {
                    if pieces.last().is_none_or(|p| p.0 != zh) {
                        pieces.push((zh, String::new()));
                    }
                } else if ch.is_whitespace() {
                    if let Some((false, s)) = pieces.last_mut() {
                        s.push(' ');
                    }
                    continue;
                } else if !is_punctuation(ch) && ch != '\'' {
                    return Err(invalid(format!("unsupported character: {ch}")));
                }
                if let Some((_, s)) = pieces.last_mut() {
                    s.push(ch);
                }
            }
            Ok::<_, crate::Error>(pieces)
        })?;
        let mut segments = Vec::new();
        for (zh, s) in pieces {
            check(cancel)?;
            let segment = if zh {
                self.chinese_profiled(&s, cancel, profile)?
            } else {
                profile.measure(ProfilePhase::FrontendEnglishG2p, || {
                    self.english(&s, cancel)
                })?
            };
            if !segment.0.is_empty() {
                segments.push(segment);
            }
        }
        profile.measure(ProfilePhase::FrontendJoin, || {
            let n: usize = segments.iter().map(|s| s.0.len()).sum();
            if n == 0 {
                return Err(invalid("text produced no phonemes"));
            }
            let mut phones = Vec::with_capacity(n);
            let mut features = vec![0.0; n * 1024];
            for (ph, bert) in segments {
                let offset = phones.len();
                for row in 0..1024 {
                    features[row * n + offset..row * n + offset + ph.len()]
                        .copy_from_slice(&bert[row * ph.len()..(row + 1) * ph.len()]);
                }
                phones.extend(ph);
            }
            Ok((phones, features))
        })
    }
}

fn gru(x: &[f32], h: &[f32], wi: &[f32], wh: &[f32], bi: &[f32], bh: &[f32]) -> Vec<f32> {
    let mut a = vec![0.0; 768];
    let mut b = vec![0.0; 768];
    for row in 0..768 {
        a[row] = bi[row]
            + wi[row * 256..(row + 1) * 256]
                .iter()
                .zip(x)
                .map(|(w, x)| w * x)
                .sum::<f32>();
        b[row] = bh[row]
            + wh[row * 256..(row + 1) * 256]
                .iter()
                .zip(h)
                .map(|(w, x)| w * x)
                .sum::<f32>();
    }
    (0..256)
        .map(|i| {
            let r = 1.0 / (1.0 + (-(a[i] + b[i])).exp());
            let z = 1.0 / (1.0 + (-(a[256 + i] + b[256 + i])).exp());
            let n = (a[512 + i] + r * b[512 + i]).tanh();
            (1.0 - z) * n + z * h[i]
        })
        .collect()
}
impl EnglishModel {
    fn predict(&self, word: &str) -> Vec<String> {
        let mut h = vec![0.0; 256];
        for index in word
            .bytes()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    usize::from(c - b'a') + 3
                } else {
                    1
                }
            })
            .chain(std::iter::once(2))
        {
            h = gru(
                &self.enc_emb[index * 256..(index + 1) * 256],
                &h,
                &self.enc_w_ih,
                &self.enc_w_hh,
                &self.enc_b_ih,
                &self.enc_b_hh,
            );
        }
        let mut token = 2;
        let mut result = Vec::new();
        for _ in 0..20 {
            h = gru(
                &self.dec_emb[token * 256..(token + 1) * 256],
                &h,
                &self.dec_w_ih,
                &self.dec_w_hh,
                &self.dec_b_ih,
                &self.dec_b_hh,
            );
            token = (0..74)
                .max_by(|&a, &b| {
                    let score = |r: usize| {
                        self.fc_b[r]
                            + self.fc_w[r * 256..(r + 1) * 256]
                                .iter()
                                .zip(&h)
                                .map(|(w, h)| w * h)
                                .sum::<f32>()
                    };
                    score(a).total_cmp(&score(b))
                })
                .unwrap();
            if token == 3 {
                break;
            }
            result.push(self.phonemes[token].clone());
        }
        result
    }
}

fn is_han(ch: char) -> bool {
    matches!(ch as u32,0x3400..=0x9fff|0x20000..=0x2fa1f)
}
fn is_punctuation(ch: char) -> bool {
    matches!(ch, ',' | '.' | '!' | '?' | '-' | '…')
}
fn tone_sandhi(chars: &[char], pinyins: &mut [String]) {
    for i in 0..pinyins.len().saturating_sub(1) {
        let next = pinyins[i + 1].chars().last();
        let replacement = if chars[i] == '不' && next == Some('4') {
            Some('2')
        } else if chars[i] == '一'
            && next.is_some_and(|x| matches!(x, '1'..='4'))
            && i.checked_sub(1).is_none_or(|j| chars[j] != '第')
        {
            Some(if next == Some('4') { '2' } else { '4' })
        } else if pinyins[i].ends_with('3') && next == Some('3') {
            Some('2')
        } else {
            None
        };
        if let Some(tone) = replacement {
            pinyins[i].pop();
            pinyins[i].push(tone);
        }
    }
}

fn chinese_integer(n: u64) -> String {
    const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    if n == 0 {
        return DIGITS[0].into();
    }
    if n >= 10_000 {
        let (unit, base) = if n >= 100_000_000 {
            ("亿", 100_000_000)
        } else {
            ("万", 10_000)
        };
        let remain = n % base;
        return format!(
            "{}{}{}{}",
            chinese_integer(n / base),
            unit,
            if remain > 0 && remain < base / 10 {
                "零"
            } else {
                ""
            },
            if remain == 0 {
                String::new()
            } else {
                chinese_integer(remain)
            }
        );
    }
    let mut out = String::new();
    let mut zero = false;
    for (base, unit) in [(1000, "千"), (100, "百"), (10, "十"), (1, "")] {
        let d = (n / base) % 10;
        if d == 0 {
            if !out.is_empty() {
                zero = true;
            }
            continue;
        }
        if zero {
            out.push('零');
            zero = false;
        }
        if !(d == 1 && base == 10 && out.is_empty()) {
            out.push_str(DIGITS[d as usize]);
        }
        out.push_str(unit);
    }
    out
}
fn english_integer(n: u64) -> String {
    const SMALL: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    if n < 20 {
        return SMALL[n as usize].into();
    }
    if n < 100 {
        return format!(
            "{}{}",
            TENS[(n / 10) as usize],
            if n % 10 == 0 {
                String::new()
            } else {
                format!(" {}", SMALL[(n % 10) as usize])
            }
        );
    }
    for (base, unit) in [
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
        (100, "hundred"),
    ] {
        if n >= base {
            return format!(
                "{} {}{}",
                english_integer(n / base),
                unit,
                if n % base == 0 {
                    String::new()
                } else {
                    format!(" {}", english_integer(n % base))
                }
            );
        }
    }
    unreachable!()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires ALICE_NATIVE_TTS_TEST_PACK; verifies batched G2PW against scalar inference"]
    fn batched_polyphones_preserve_scalar_pronunciation() {
        let pack = std::env::var("ALICE_NATIVE_TTS_TEST_PACK").unwrap();
        let mut frontend = Frontend::load(Path::new(&pack), Backend::Cpu).unwrap();
        for text in [
            "重庆的银行今天有十二个人.",
            "大家一起重复测试长大以后的音乐.",
        ] {
            let (scalar, scalar_bert) = frontend.chinese_with_batch(text, &|| false, 1).unwrap();
            let (batch, batch_bert) = frontend.chinese_with_batch(text, &|| false, 4).unwrap();
            assert_eq!(scalar, batch);
            assert_eq!(scalar_bert, batch_bert);
            let mut profile = ProfileCollector::disabled();
            let (parallel, parallel_bert) = frontend
                .chinese_parallel_profiled(text, &|| false, 4, &mut profile)
                .unwrap();
            assert_eq!(scalar, parallel);
            assert_eq!(scalar_bert, parallel_bert);
        }
    }
    #[test]
    fn frontend_cache_key_covers_route_and_execution_configuration() {
        let config = FrontendCacheConfig {
            schema: FRONTEND_CACHE_SCHEMA,
            g2pw_batch_size: 4,
            g2pw_backend: 0,
            bert_backend: 0,
        };
        assert_eq!(
            frontend_language_route("行长来了"),
            FrontendLanguageRoute::Chinese
        );
        assert_eq!(
            frontend_language_route("Hello Alice"),
            FrontendLanguageRoute::English
        );
        assert_eq!(
            frontend_language_route("你好 Alice"),
            FrontendLanguageRoute::Mixed
        );
        let base = frontend_cache_key("你好 Alice", config);
        let different_batch = frontend_cache_key(
            "你好 Alice",
            FrontendCacheConfig {
                g2pw_batch_size: 8,
                ..config
            },
        );
        let different_provider = frontend_cache_key(
            "你好 Alice",
            FrontendCacheConfig {
                g2pw_backend: 2,
                ..config
            },
        );
        let different_bert_provider = frontend_cache_key(
            "你好 Alice",
            FrontendCacheConfig {
                bert_backend: 2,
                ..config
            },
        );
        let different_schema = frontend_cache_key(
            "你好 Alice",
            FrontendCacheConfig {
                schema: config.schema.wrapping_add(1),
                ..config
            },
        );
        assert_ne!(base, different_batch);
        assert_ne!(base, different_provider);
        assert_ne!(base, different_bert_provider);
        assert_ne!(base, different_schema);
        assert_ne!(base, frontend_cache_key("Hello Alice", config));
    }

    #[cfg(feature = "r1-cuda-ep")]
    #[test]
    #[ignore = "requires ALICE_NATIVE_TTS_TEST_PACK and the configured CUDA ORT runtime"]
    fn cuda_frontend_matches_cpu_phones_and_bert_features() {
        let pack = std::env::var("ALICE_NATIVE_TTS_TEST_PACK").unwrap();
        std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "0");
        std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL", "0");
        std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU", "1");

        let load = |g2pw: &str, bert: &str| {
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND", g2pw);
            std::env::set_var("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND", bert);
            Frontend::load(Path::new(&pack), Backend::Cpu).unwrap()
        };
        let mut cpu = load("cpu", "cpu");
        let texts = ["重庆的银行今天有十二个人.", "行长在重庆重看长长的乐章."];
        let mut cpu_outputs = Vec::new();
        for text in texts {
            let mut profile = ProfileCollector::disabled();
            cpu_outputs.push(cpu.chinese_profiled(text, &|| false, &mut profile).unwrap());
        }
        drop(cpu);

        for (g2pw_backend, bert_backend) in [("cuda", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")] {
            let mut candidate = load(g2pw_backend, bert_backend);
            for (text, (expected_phones, expected_features)) in texts.iter().zip(&cpu_outputs) {
                let mut profile = ProfileCollector::disabled();
                let (phones, features) = candidate
                    .chinese_profiled(text, &|| false, &mut profile)
                    .unwrap();
                assert_eq!(
                    &phones, expected_phones,
                    "phones differ: {g2pw_backend}/{bert_backend}"
                );
                assert_eq!(features.len(), expected_features.len());
                let max_abs = features
                    .iter()
                    .zip(expected_features)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                let mse = features
                    .iter()
                    .zip(expected_features)
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f32>()
                    / features.len().max(1) as f32;
                eprintln!(
                    "frontend {g2pw_backend}/{bert_backend}: feature_max_abs={max_abs:.8}, rmse={:.8}",
                    mse.sqrt()
                );
                assert!(max_abs <= 1e-3, "BERT feature drift too high: {max_abs}");
            }
        }
    }
    #[test]
    #[ignore = "requires ALICE_NATIVE_TTS_TEST_PACK, validates against original Python g2p_en predictions"]
    fn english_gru_matches_original_oov_predictions() {
        let dir = std::env::var("ALICE_NATIVE_TTS_TEST_PACK").unwrap();
        let model: EnglishModel =
            serde_json::from_slice(&std::fs::read(Path::new(&dir).join("g2p-en.json")).unwrap())
                .unwrap();
        for (word, expected) in [
            (
                "rustification",
                vec![
                    "R", "AH2", "S", "T", "AH0", "F", "AH0", "K", "EY1", "SH", "AH0", "N",
                ],
            ),
            (
                "activationist",
                vec![
                    "AE2", "K", "T", "IH0", "V", "EY1", "SH", "AH0", "N", "IH0", "S", "T",
                ],
            ),
            (
                "neuroclone",
                vec!["N", "UH1", "R", "OW0", "K", "L", "OW2", "N"],
            ),
        ] {
            assert_eq!(model.predict(word), expected, "{word}");
        }
    }
    #[test]
    fn numbers_do_not_disappear() {
        assert_eq!(
            normalize("今天有12个人，温度3.5度。"),
            "今天有 十二 个人,温度 三点五 度."
        );
        assert_eq!(normalize("Test 42."), "Test  forty two .");
        assert_eq!(chinese_integer(10001), "一万零一");
    }
    #[test]
    fn common_tone_changes() {
        let mut p = vec!["ni3".into(), "hao3".into(), "bu4".into(), "shi4".into()];
        tone_sandhi(&['你', '好', '不', '是'], &mut p);
        assert_eq!(p, ["ni2", "hao3", "bu2", "shi4"]);
    }
}
