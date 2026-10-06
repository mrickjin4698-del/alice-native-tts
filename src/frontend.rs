pub mod normalize;
use normalize::normalize;
mod pinyin;
use crate::{
    check,
    inference::{open_on, tf, ti},
    invalid, Backend, Result,
};
use ort::session::Session;
use serde::Deserialize;
use std::{collections::HashMap, path::Path};
use tokenizers::Tokenizer;

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
        Ok(Self {
            bert: open_on(dir, "bert", backend)?,
            g2pw: open_on(dir, "g2pw", backend)?,
            tokenizer: Tokenizer::from_bytes(include_bytes!("frontend/g2pw_tokenizer.json"))
                .map_err(|e| invalid(e.to_string()))?,
            mono: serde_json::from_str(include_str!("frontend/dict_mono_chars.json"))?,
            poly: serde_json::from_str(include_str!("frontend/dict_poly_chars.json"))?,
            labels: serde_json::from_str(include_str!("frontend/dict_poly_index_list.json"))?,
            cmu: serde_json::from_slice(&std::fs::read(dir.join("cmudict.json"))?)?,
            symbols,
            english,
        })
    }

    fn phone(&self, s: &str) -> Result<i64> {
        self.symbols
            .get(s)
            .copied()
            .ok_or_else(|| invalid(format!("unsupported phoneme: {s}")))
    }

    fn chinese(&mut self, text: &str, cancel: &impl Fn() -> bool) -> Result<(Vec<i64>, Vec<f32>)> {
        self.chinese_with_batch(text, cancel, 4)
    }

    fn chinese_with_batch(
        &mut self,
        text: &str,
        cancel: &impl Fn() -> bool,
        batch_size: usize,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let chars: Vec<char> = text.chars().collect();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| invalid(e.to_string()))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|x| i64::from(*x)).collect();
        if ids.len() != chars.len() + 2 {
            return Err(invalid("Chinese tokenizer/character alignment mismatch"));
        }
        let len = ids.len();
        let mut pinyins: Vec<String> = chars
            .iter()
            .map(|ch| {
                self.mono
                    .get(ch)
                    .map_or_else(|| ch.to_string(), |m| m.phone.clone())
            })
            .collect();
        // Bounded batches amortize G2PW's BERT evaluation while keeping
        // cancellation points between runs. Preserve each query's full context.
        let queries: Vec<_> = chars
            .iter()
            .enumerate()
            .filter_map(|(i, ch)| self.poly.get(ch).map(|p| (i, p)))
            .collect();
        for batch in queries.chunks(batch_size) {
            check(cancel)?;
            let count = batch.len();
            let mut masks = vec![0.0; count * self.labels.len()];
            let mut char_ids = Vec::new();
            let mut positions = Vec::new();
            for (row, (i, poly)) in batch.iter().enumerate() {
                for (_, index) in &poly.phones {
                    masks[row * self.labels.len() + *index] = 1.0;
                }
                char_ids.push(poly.index as i64);
                positions.push((i + 1) as i64);
            }
            let out = self.g2pw.run(ort::inputs! {
                "input_ids" => ti(vec![count,len],ids.repeat(count))?,
                "token_type_ids" => ti(vec![count,len],vec![0;count * len])?,
                "attention_mask" => ti(vec![count,len],vec![1;count * len])?,
                "phoneme_mask" => tf(vec![count,self.labels.len()],masks)?,
                "char_ids" => ti(vec![count],char_ids)?,
                "position_ids" => ti(vec![count],positions)?
            })?;
            let (_, probs) = out
                .get("probs")
                .ok_or_else(|| invalid("incompatible G2PW output"))?
                .try_extract_tensor::<f32>()?;
            if probs.len() != count * self.labels.len() || probs.iter().any(|v| !v.is_finite()) {
                return Err(invalid("invalid G2PW probabilities"));
            }
            for ((position, _), row) in batch.iter().zip(probs.chunks_exact(self.labels.len())) {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                pinyins[*position] = self.labels[index].clone();
            }
        }
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
        check(cancel)?;
        let out = self.bert.run(ort::inputs! {
            "input_ids" => ti(vec![1,len],ids)?, "attention_mask" => ti(vec![1,len],vec![1;len])?,
            "token_type_ids" => ti(vec![1,len],vec![0;len])?
        })?;
        let (shape, data) = out
            .get("hidden_states")
            .ok_or_else(|| invalid("incompatible BERT output"))?
            .try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, len as i64, 1024] {
            return Err(invalid("incompatible BERT export"));
        }
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
        let normalized = normalize(text);
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
        let mut segments = Vec::new();
        for (zh, s) in pieces {
            check(cancel)?;
            let segment = if zh {
                self.chinese(&s, cancel)?
            } else {
                self.english(&s, cancel)?
            };
            if !segment.0.is_empty() {
                segments.push(segment);
            }
        }
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
