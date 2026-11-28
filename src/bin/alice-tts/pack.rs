use super::invalid;
use alice_native_tts::session::SessionResult;
use alice_native_tts::{validate_pack, validate_streaming_pack, PACK_FILES, STREAMING_GRAPH};
use clap::Parser;
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

type PackResult<T> = Result<T, Box<dyn std::error::Error>>;
type Pronunciations = BTreeMap<String, Vec<String>>;

// The checkpoint uses the phoneme order from g2p_en (see NOTICE.md).
const PHONEMES: &str = "<pad> <unk> <s> </s> AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 \
    AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH \
    IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH \
    UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH";
const G2P_TENSORS: &[(&str, &[usize])] = &[
    ("enc_emb", &[29, 256]),
    ("enc_w_ih", &[768, 256]),
    ("enc_w_hh", &[768, 256]),
    ("enc_b_ih", &[768]),
    ("enc_b_hh", &[768]),
    ("dec_emb", &[74, 256]),
    ("dec_w_ih", &[768, 256]),
    ("dec_w_hh", &[768, 256]),
    ("dec_b_ih", &[768]),
    ("dec_b_hh", &[768]),
    ("fc_w", &[74, 256]),
    ("fc_b", &[74]),
];

#[derive(Parser)]
#[command(name = "alice-tts pack", about = "Assemble an ABI-1 v2Pro ONNX pack")]
struct Args {
    /// Directory containing the v2Pro ONNX exports and config.json.
    #[arg(long)]
    exports: PathBuf,
    /// G2PW ONNX graph.
    #[arg(long)]
    g2pw: PathBuf,
    /// CMUdict text corpus or an existing cmudict.json.
    #[arg(long)]
    cmudict: PathBuf,
    /// g2p_en checkpoint20.npz or an existing g2p-en.json.
    #[arg(long)]
    g2p_en: PathBuf,
    /// Provenance and licenses for the model and frontend assets.
    #[arg(long)]
    licenses: PathBuf,
    /// Reference voice WAV.
    #[arg(long)]
    reference: PathBuf,
    /// Actual transcript of the reference audio.
    #[arg(long)]
    transcript: String,
    /// New model pack directory.
    #[arg(long)]
    output: PathBuf,
    /// Optional streaming graph with an adjacent .validation.json report.
    #[arg(long)]
    streaming_graph: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct EnglishAssets {
    phonemes: Vec<String>,
    #[serde(flatten)]
    tensors: BTreeMap<String, Vec<f32>>,
}

#[derive(Deserialize)]
struct StreamingReport {
    streaming: Value,
    graph_sha256: String,
    validation: Vec<StreamingCase>,
}

#[derive(Deserialize)]
struct StreamingCase {
    outputs: Vec<StreamingOutput>,
}

#[derive(Deserialize)]
struct StreamingOutput {
    output: String,
    original_relative_l2: f64,
}

fn is_json(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

fn write_json(path: &Path, value: &impl Serialize) -> PackResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer(&mut file, value)?;
    file.flush()?;
    Ok(())
}

fn copy(source: &Path, destination: &Path) -> PackResult<()> {
    fs::copy(source, destination).map_err(|error| {
        format!(
            "copy {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn cmu_text(reader: impl BufRead) -> PackResult<Pronunciations> {
    let mut dictionary = BTreeMap::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty()
            || [";;;", "##", "# "]
                .iter()
                .any(|&prefix| line.starts_with(prefix))
        {
            continue;
        }
        let mut fields = line.split_ascii_whitespace().peekable();
        let word = fields.next().expect("non-empty dictionary line");
        // Both NLTK's counter column and CMUdict's WORD(N) variants use the
        // first pronunciation, as did the original preparation script.
        let word = word
            .strip_suffix(')')
            .and_then(|stem| stem.rsplit_once('('))
            .filter(|(_, variant)| {
                !variant.is_empty() && variant.bytes().all(|b| b.is_ascii_digit())
            })
            .map_or(word, |(word, _)| word);
        if fields
            .peek()
            .is_some_and(|field| field.bytes().all(|b| b.is_ascii_digit()))
        {
            fields.next();
        }
        let phones: Vec<_> = fields
            .take_while(|phone| *phone != "#")
            .map(str::to_owned)
            .collect();
        if phones.is_empty() {
            return Err(format!("no pronunciation at CMUdict line {}", index + 1).into());
        }
        dictionary
            .entry(word.to_ascii_lowercase())
            .or_insert(phones);
    }
    Ok(dictionary)
}

fn prepare_cmudict(input: &Path, output: &Path) -> PackResult<()> {
    let reader = BufReader::new(File::open(input)?);
    let dictionary: Pronunciations = if is_json(input) {
        serde_json::from_reader(reader)?
    } else {
        cmu_text(reader)?
    };
    if dictionary.is_empty() {
        return Err("CMUdict has no entries".into());
    }
    write_json(output, &dictionary)
}

fn read_g2p_checkpoint(input: &Path) -> PackResult<EnglishAssets> {
    let mut arrays = NpzReader::new(File::open(input)?)?;
    let mut tensors = BTreeMap::new();
    for &(name, shape) in G2P_TENSORS {
        let values: ArrayD<f32> = arrays.by_name(name)?;
        if values.shape() != shape {
            return Err(format!("g2p_en tensor {name} must have shape {shape:?}").into());
        }
        tensors.insert(name.to_owned(), values.iter().copied().collect());
    }
    Ok(EnglishAssets {
        phonemes: PHONEMES
            .split_ascii_whitespace()
            .map(str::to_owned)
            .collect(),
        tensors,
    })
}

fn prepare_g2p(input: &Path, output: &Path) -> PackResult<()> {
    let assets: EnglishAssets = if is_json(input) {
        serde_json::from_reader(BufReader::new(File::open(input)?))?
    } else {
        read_g2p_checkpoint(input)?
    };
    if assets.phonemes.len() != 74 {
        return Err("g2p_en requires 74 phonemes".into());
    }
    for &(name, shape) in G2P_TENSORS {
        if !assets.tensors.get(name).is_some_and(|tensor| {
            tensor.len() == shape.iter().product::<usize>() && tensor.iter().all(|v| v.is_finite())
        }) {
            return Err(format!(
                "g2p_en tensor {name} must contain finite values of shape {shape:?}"
            )
            .into());
        }
    }
    write_json(output, &assets)
}

fn prepare_streaming(graph: &Path, output: &Path) -> PackResult<Value> {
    let report_path = graph.with_extension("validation.json");
    let report: StreamingReport = serde_json::from_reader(BufReader::new(
        File::open(&report_path)
            .map_err(|error| format!("read {}: {error}", report_path.display()))?,
    ))?;
    let metadata = json!({
        "abi": 1, "graph": STREAMING_GRAPH, "latent_channels": 192, "samples_per_token": 1280,
    });
    if report.streaming != metadata || report.validation.len() < 4 {
        return Err("streaming graph requires an ABI-1 report with at least four cases".into());
    }
    for case in report.validation {
        if case.outputs.len() != 2
            || !["audio", "latent"].iter().all(|name| {
                case.outputs.iter().any(|value| {
                    value.output == *name && (0.0..=2e-4).contains(&value.original_relative_l2)
                })
            })
        {
            return Err("streaming report lacks passing audio and latent comparisons".into());
        }
    }

    let destination = output.join(STREAMING_GRAPH);
    copy(graph, &destination)?;
    let mut digest = Sha256::new();
    io::copy(&mut File::open(destination)?, &mut digest)?;
    if format!("{:x}", digest.finalize()) != report.graph_sha256 {
        return Err("streaming graph SHA-256 differs from its validation report".into());
    }
    Ok(metadata)
}

fn build_pack(args: &Args) -> PackResult<PathBuf> {
    if args.transcript.trim().is_empty() {
        return Err("reference transcript must be non-empty".into());
    }
    if hound::WavReader::open(&args.reference)?.duration() == 0 {
        return Err("reference WAV has no samples".into());
    }
    let output = std::path::absolute(&args.output)?;
    match fs::symlink_metadata(&output) {
        Ok(_) => return Err(format!("output already exists: {}", output.display()).into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let parent = output
        .parent()
        .ok_or("output directory requires a parent")?;
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".alice-tts-pack-")
        .tempdir_in(parent)?;
    let dir = staging.path();
    for &name in PACK_FILES {
        let destination = dir.join(name);
        match name {
            "pack.json" => {}
            "cmudict.json" => prepare_cmudict(&args.cmudict, &destination)?,
            "g2p-en.json" => prepare_g2p(&args.g2p_en, &destination)?,
            "g2pw.onnx" => copy(&args.g2pw, &destination)?,
            "LICENSES.txt" => copy(&args.licenses, &destination)?,
            _ => copy(&args.exports.join(name), &destination)?,
        }
    }
    copy(&args.reference, &dir.join("reference.wav"))?;
    let mut manifest = json!({
        "abi": 1, "architecture": "v2Pro", "sample_rate": 32000, "max_len": 512,
        "reference_audio": "reference.wav", "reference_text": args.transcript,
    });
    if let Some(graph) = &args.streaming_graph {
        manifest["streaming"] = prepare_streaming(graph, dir)?;
    }
    write_json(&dir.join("pack.json"), &manifest)?;
    if args.streaming_graph.is_some() {
        validate_streaming_pack(dir)?;
    } else {
        validate_pack(dir)?;
    }
    fs::rename(dir, &output)?;
    Ok(output)
}

pub(super) fn run(args: impl IntoIterator<Item = OsString>) -> SessionResult<()> {
    let args = Args::parse_from(std::iter::once(OsString::from("alice-tts pack")).chain(args));
    let pack = build_pack(&args).map_err(|error| invalid(error.to_string()))?;
    println!("{}", json!({"pack": pack, "abi": 1}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmudict_keeps_the_first_pronunciation_in_both_formats() {
        let corpus = b";;; comment\nA 1 AH0\nA 2 EY1\nREAD R IY1 D # comment\nREAD(2) R EH1 D\n";
        let dictionary = cmu_text(corpus.as_slice()).unwrap();
        assert_eq!(dictionary["a"], ["AH0"]);
        assert_eq!(dictionary["read"], ["R", "IY1", "D"]);
    }

    #[test]
    fn g2p_checkpoint_phoneme_inventory_has_74_entries() {
        assert_eq!(PHONEMES.split_ascii_whitespace().count(), 74);
    }
}
