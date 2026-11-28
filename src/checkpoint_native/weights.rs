use crate::{invalid, Result};
use half::{bf16, f16};
use repugnant_pickle::{evaluate, ops::PickleOp, parse_ops, SequenceType, Value};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use zip::{result::ZipError, ZipArchive};

const MAX_PICKLE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(super) enum Component {
    Gpt,
    Sovits,
}

impl Component {
    fn name(self) -> &'static str {
        match self {
            Self::Gpt => "GPT",
            Self::Sovits => "SoVITS",
        }
    }

    fn profile(self) -> &'static str {
        match self {
            Self::Gpt => include_str!("gpt-profile.json"),
            Self::Sovits => include_str!("sovits-profile.json"),
        }
    }
}

/// GPT-SoVITS can replace the first two bytes of the standard ZIP header with
/// a family tag. Present the ZIP reader with the standard header in memory.
struct CheckpointReader {
    file: File,
    restore_zip_magic: bool,
    position: u64,
}

impl CheckpointReader {
    fn open(path: &Path, component: Component) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut header = [0; 4];
        file.read_exact(&mut header)?;
        let restore_zip_magic = match (&header, component) {
            (b"PK\x03\x04", _) => false,
            (b"05\x03\x04", Component::Sovits) => true,
            _ => {
                return Err(invalid(format!(
                    "{} checkpoint is not a supported v2Pro PyTorch ZIP: {}",
                    component.name(),
                    path.display()
                )))
            }
        };
        file.rewind()?;
        Ok(Self {
            file,
            restore_zip_magic,
            position: 0,
        })
    }
}

impl Read for CheckpointReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.file.read(buffer)?;
        if self.restore_zip_magic && self.position < 2 {
            let start = self.position as usize;
            let count = read.min(2 - start);
            buffer[..count].copy_from_slice(&b"PK"[start..start + count]);
        }
        self.position += read as u64;
        Ok(read)
    }
}

impl Seek for CheckpointReader {
    fn seek(&mut self, seek: SeekFrom) -> io::Result<u64> {
        self.position = self.file.seek(seek)?;
        Ok(self.position)
    }
}

fn global(value: &Value<'_>, module: &str, name: &str) -> bool {
    matches!(value, Value::Raw(op) if matches!(op.as_ref(), PickleOp::GLOBAL(m,n) if *m == module && *n == name))
}

fn fields<'v, 'p>(value: &'v Value<'p>) -> Option<&'v [Value<'p>]> {
    match value {
        Value::Build(_, attributes) => fields(attributes),
        Value::Global(class, arguments) if global(class, "collections", "OrderedDict") => {
            match arguments.as_slice() {
                [_, Value::Seq(SequenceType::Tuple, items)] => Some(items),
                _ => None,
            }
        }
        Value::Seq(SequenceType::Dict, items) => match items.as_slice() {
            // SETITEMS may group the dictionary entries in a single tuple.
            [Value::Seq(SequenceType::Tuple, pairs)]
                if !matches!(pairs.first(), Some(Value::String(_))) =>
            {
                Some(pairs)
            }
            _ => Some(items),
        },
        _ => None,
    }
}

fn entry<'v, 'p>(value: &'v Value<'p>) -> Option<(&'p str, &'v Value<'p>)> {
    if let Value::Seq(SequenceType::Tuple, pair) = value {
        if let [Value::String(name), item] = pair.as_slice() {
            return Some((name, item));
        }
    }
    None
}

fn field<'v, 'p>(value: &'v Value<'p>, key: &str) -> Option<&'v Value<'p>> {
    fields(value)?
        .iter()
        .filter_map(entry)
        .find_map(|(name, value)| (name == key).then_some(value))
}

fn same_setting(value: &Value<'_>, expected: &Json) -> bool {
    match (value, expected) {
        (Value::Int(n), Json::Number(want)) => Some(*n) == want.as_i64(),
        (Value::String(text), Json::String(want)) => *text == want,
        (Value::Seq(SequenceType::List | SequenceType::Tuple, values), Json::Array(want)) => {
            values.len() == want.len()
                && values
                    .iter()
                    .zip(want)
                    .all(|(value, expected)| same_setting(value, expected))
        }
        _ => false,
    }
}

fn check_profile(root: &Value<'_>, component: Component) -> Result<()> {
    let expected: HashMap<String, HashMap<String, Json>> =
        serde_json::from_str(component.profile())?;
    let config =
        field(root, "config").ok_or_else(|| invalid("checkpoint is missing its config"))?;
    for (section, settings) in expected {
        let section_value = field(config, &section);
        for (name, expected) in settings {
            let actual = section_value.and_then(|value| field(value, &name));
            if !actual.is_some_and(|value| same_setting(value, &expected)) {
                return Err(invalid(format!(
                    "{} checkpoint config {section}.{name} must be {expected} for the v2Pro architecture",
                    component.name()
                )));
            }
        }
    }
    Ok(())
}

enum FloatStorage {
    F32,
    F16,
    Bf16,
}

impl FloatStorage {
    fn size(&self) -> usize {
        match self {
            Self::F32 => 4,
            _ => 2,
        }
    }

    fn read(&self, data: &[u8]) -> Vec<f32> {
        match self {
            Self::F32 => data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect(),
            Self::F16 => data
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect(),
            Self::Bf16 => data
                .chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect(),
        }
    }
}

struct StoredTensor {
    storage: String,
    dtype: FloatStorage,
    elements: usize,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn dimensions(values: &[Value<'_>]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            if let Value::Int(n) = value {
                if let Ok(n) = usize::try_from(*n) {
                    return Ok(n);
                }
            }
            Err(invalid(
                "checkpoint has an invalid tensor dimension or stride",
            ))
        })
        .collect()
}

impl StoredTensor {
    fn parse(tensor: &Value<'_>) -> Result<Option<Self>> {
        let Value::Global(function, arguments) = tensor else {
            return Ok(None);
        };
        if !global(function, "torch._utils", "_rebuild_tensor_v2") {
            return Ok(None);
        }
        let [Value::Seq(SequenceType::Tuple, arguments)] = arguments.as_slice() else {
            return Err(invalid("malformed PyTorch tensor arguments"));
        };
        let [Value::PersId(storage), Value::Int(offset), Value::Seq(SequenceType::Tuple, shape), Value::Seq(SequenceType::Tuple, strides), ..] =
            arguments.as_slice()
        else {
            return Err(invalid("malformed PyTorch tensor layout"));
        };
        let Value::Seq(SequenceType::Tuple, descriptor) = storage.as_ref() else {
            return Err(invalid("malformed PyTorch storage reference"));
        };
        let [Value::String("storage"), dtype, Value::String(key), Value::String(_), Value::Int(size)] =
            descriptor.as_slice()
        else {
            return Err(invalid("malformed PyTorch storage record"));
        };
        let dtype = if global(dtype, "torch", "FloatStorage") {
            FloatStorage::F32
        } else if global(dtype, "torch", "HalfStorage") {
            FloatStorage::F16
        } else if global(dtype, "torch", "BFloat16Storage") {
            FloatStorage::Bf16
        } else {
            return Ok(None);
        };
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("invalid PyTorch storage key"));
        }
        Ok(Some(Self {
            storage: (*key).to_owned(),
            dtype,
            elements: usize::try_from(*size).map_err(|_| invalid("invalid storage size"))?,
            offset: usize::try_from(*offset).map_err(|_| invalid("invalid storage offset"))?,
            shape: dimensions(shape)?,
            strides: dimensions(strides)?,
        }))
    }

    fn byte_range(&self, expected: &[usize]) -> Option<(usize, usize)> {
        if self.shape != expected || self.strides.len() != self.shape.len() {
            return None;
        }
        let mut count = 1usize;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension != 1 && stride != count {
                return None;
            }
            count = count.checked_mul(dimension)?;
        }
        if self.offset.checked_add(count)? > self.elements {
            return None;
        }
        Some((
            self.offset.checked_mul(self.dtype.size())?,
            count.checked_mul(self.dtype.size())?,
        ))
    }
}

pub(super) struct Weights {
    archive: ZipArchive<CheckpointReader>,
    prefix: String,
    tensors: HashMap<String, StoredTensor>,
}

fn zip_error(error: ZipError) -> crate::Error {
    invalid(format!("PyTorch archive: {error}"))
}

impl Weights {
    pub(super) fn open(path: &Path, component: Component) -> Result<Self> {
        let reader = CheckpointReader::open(path, component)?;
        let mut archive = ZipArchive::new(reader).map_err(zip_error)?;
        let pickle_name = archive
            .file_names()
            .find(|name| name.ends_with("/data.pkl"))
            .map(str::to_owned)
            .ok_or_else(|| invalid("PyTorch archive has no data.pkl"))?;
        let (prefix, _) = pickle_name.rsplit_once('/').expect("data.pkl has a parent");
        match archive.by_name(&format!("{prefix}/byteorder")) {
            Ok(entry) => {
                let mut bytes = Vec::new();
                entry.take(16).read_to_end(&mut bytes)?;
                if bytes != b"little" {
                    return Err(invalid("checkpoint storage must be little endian"));
                }
            }
            Err(ZipError::FileNotFound) => {}
            Err(error) => return Err(zip_error(error)),
        }
        let mut bytes = Vec::new();
        archive
            .by_name(&pickle_name)
            .map_err(zip_error)?
            .take(MAX_PICKLE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_PICKLE_BYTES {
            return Err(invalid("checkpoint tensor metadata exceeds 2 MiB"));
        }
        // The evaluator builds a tree of pickle records. It cannot import or run
        // Python globals. Only the known tensor and configuration records are used.
        let (_, operations) = parse_ops::<()>(&bytes)
            .map_err(|_| invalid("could not parse PyTorch pickle metadata"))?;
        let (values, _) = evaluate(&operations, true)
            .map_err(|error| invalid(format!("could not read PyTorch tensor metadata: {error}")))?;
        let [root] = values.as_slice() else {
            return Err(invalid("checkpoint must contain a single metadata object"));
        };
        check_profile(root, component)?;
        let weights = field(root, "weight")
            .and_then(fields)
            .ok_or_else(|| invalid("checkpoint has no weight dictionary"))?;
        let mut tensors = HashMap::new();
        for (name, value) in weights.iter().filter_map(entry) {
            if let Some(tensor) = StoredTensor::parse(value)? {
                if tensors.insert(name.to_owned(), tensor).is_some() {
                    return Err(invalid(format!("duplicate checkpoint tensor {name}")));
                }
            }
        }
        Ok(Self {
            archive,
            prefix: prefix.to_owned(),
            tensors,
        })
    }

    pub(super) fn read(&mut self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("checkpoint is missing float tensor {name}")))?;
        let (offset, len) = tensor.byte_range(expected).ok_or_else(|| {
            invalid(format!(
                "checkpoint tensor {name} needs contiguous shape {expected:?}"
            ))
        })?;
        let mut entry = self
            .archive
            .by_name(&format!("{}/data/{}", self.prefix, tensor.storage))
            .map_err(zip_error)?;
        if (offset as u64)
            .checked_add(len as u64)
            .is_none_or(|end| end > entry.size())
        {
            return Err(invalid(format!(
                "storage for checkpoint tensor {name} is truncated"
            )));
        }
        let skipped = io::copy(&mut entry.by_ref().take(offset as u64), &mut io::sink())?;
        if skipped != offset as u64 {
            return Err(invalid(format!(
                "cannot reach storage for checkpoint tensor {name}"
            )));
        }
        let mut data = vec![0; len];
        entry.read_exact(&mut data)?;
        let floats = tensor.dtype.read(&data);
        if floats.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "checkpoint tensor {name} contains non-finite values"
            )));
        }
        Ok(floats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_a_v2pro_header_only_in_the_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.pth");
        let original = b"05\x03\x04data";
        std::fs::write(&path, original).unwrap();
        let mut reader = CheckpointReader::open(&path, Component::Sovits).unwrap();
        let mut first = [0];
        reader.read_exact(&mut first).unwrap();
        assert_eq!(first, *b"P");
        reader.seek(SeekFrom::Start(1)).unwrap();
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"K\x03\x04data");
        assert_eq!(std::fs::read(path).unwrap(), original);
    }

    #[test]
    fn checks_contiguous_tensor_ranges_before_reading_storage() {
        let mut tensor = StoredTensor {
            storage: "0".into(),
            dtype: FloatStorage::F16,
            elements: 7,
            offset: 3,
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        assert_eq!(tensor.byte_range(&[2, 2]), Some((6, 8)));
        assert_eq!(tensor.byte_range(&[4]), None);
        tensor.offset = 4;
        assert_eq!(tensor.byte_range(&[2, 2]), None);
        tensor.offset = 3;
        tensor.strides.swap(0, 1);
        assert_eq!(tensor.byte_range(&[2, 2]), None);
    }
}
