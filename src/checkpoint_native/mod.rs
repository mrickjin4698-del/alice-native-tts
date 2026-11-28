mod weights;

use crate::{check, invalid, session::NativePrecision, Result};
use ort::{
    session::{builder::SessionBuilder, Session},
    value::Tensor,
};
use prost::Message;
use std::path::Path;
use weights::{Component, Weights};

const ONNX_FLOAT: i32 = 1;
#[cfg(feature = "r1-cuda-ep")]
const ONNX_FLOAT16: i32 = 10;
const ONNX_EXTERNAL: i32 = 1;

const ASSETS: &[&str] = &[
    "config.json",
    "bert.onnx",
    "ssl.onnx",
    "spectrogram.onnx",
    "sv_embedding.onnx",
    "g2pw.onnx",
    "cmudict.json",
    "g2p-en.json",
    "LICENSES.txt",
];

pub(crate) fn validate_assets(dir: &Path) -> Result<()> {
    for name in ASSETS {
        if !dir.join(name).is_file() {
            return Err(invalid(format!(
                "missing shared v2Pro asset {}",
                dir.join(name).display()
            )));
        }
    }
    Ok(())
}

// Select only the ONNX ModelProto fields needed to identify external
// initializers. ONNX Runtime reads the complete operator graph.
#[derive(Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

#[derive(Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
}

#[derive(Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(int32, tag = "14")]
    data_location: i32,
}

pub(crate) struct VoiceGraphs {
    gpt: Weights,
    sovits: Weights,
    #[cfg(feature = "r1-cuda-ep")]
    precision: NativePrecision,
}

impl VoiceGraphs {
    pub(crate) fn open(
        gpt: &Path,
        sovits: &Path,
        precision: NativePrecision,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        #[cfg(not(feature = "r1-cuda-ep"))]
        if precision == NativePrecision::Fp16 {
            return Err(invalid(
                "native_precision fp16 requires the r1-cuda-ep feature",
            ));
        }
        check(cancel)?;
        let gpt = Weights::open(gpt, Component::Gpt)?;
        check(cancel)?;
        let sovits = Weights::open(sovits, Component::Sovits)?;
        Ok(Self {
            gpt,
            sovits,
            #[cfg(feature = "r1-cuda-ep")]
            precision,
        })
    }

    pub(crate) fn session(
        &mut self,
        name: &str,
        mut builder: SessionBuilder,
        cancel: &impl Fn() -> bool,
    ) -> Result<Session> {
        let (graph, weights, prefix) = match name {
            "gpt_encoder" => (
                include_bytes!("graphs/gpt_encoder.onnx").as_slice(),
                &mut self.gpt,
                "t2s_model.",
            ),
            "gpt_step" => (
                include_bytes!("graphs/gpt_step.onnx").as_slice(),
                &mut self.gpt,
                "t2s_model.",
            ),
            "vq_encoder" => (
                include_bytes!("graphs/vq_encoder.onnx").as_slice(),
                &mut self.sovits,
                "vq_model.",
            ),
            "sovits_stream" => (
                include_bytes!("graphs/sovits_stream.onnx").as_slice(),
                &mut self.sovits,
                "model.",
            ),
            _ => return Err(invalid(format!("no v2Pro checkpoint graph for {name}"))),
        };
        #[cfg(feature = "r1-cuda-ep")]
        let graph = match (self.precision, name) {
            (NativePrecision::Fp16, "gpt_encoder") => {
                include_bytes!("graphs/gpt_encoder_fp16.onnx").as_slice()
            }
            (NativePrecision::Fp16, "gpt_step") => {
                include_bytes!("graphs/gpt_step_fp16.onnx").as_slice()
            }
            (NativePrecision::Fp16, "sovits_stream") => {
                include_bytes!("graphs/sovits_stream_fp16.onnx").as_slice()
            }
            _ => graph,
        };
        let model = ModelProto::decode(graph)
            .map_err(|error| invalid(format!("invalid embedded graph {name}: {error}")))?;
        let model = model
            .graph
            .ok_or_else(|| invalid("embedded model graph is missing"))?;
        for tensor in model
            .initializer
            .into_iter()
            .filter(|tensor| tensor.data_location == ONNX_EXTERNAL)
        {
            check(cancel)?;
            let key = tensor
                .name
                .strip_prefix(prefix)
                .ok_or_else(|| invalid("embedded graph contains an unknown weight name"))?;
            let dims: Vec<_> = tensor
                .dims
                .into_iter()
                .map(usize::try_from)
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| invalid("embedded graph contains an invalid weight shape"))?;
            let weight = match tensor.data_type {
                ONNX_FLOAT => {
                    let values = weights.read(key, &dims)?;
                    Tensor::from_array((dims, values))?.into_dyn()
                }
                #[cfg(feature = "r1-cuda-ep")]
                ONNX_FLOAT16 => {
                    let values: Vec<_> = weights
                        .read(key, &dims)?
                        .into_iter()
                        .map(half::f16::from_f32)
                        .collect();
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(invalid(format!("checkpoint tensor {key} overflows fp16")));
                    }
                    Tensor::from_array((dims, values))?.into_dyn()
                }
                _ => {
                    return Err(invalid(
                        "embedded graph has an unsupported checkpoint tensor type",
                    ))
                }
            };
            builder = builder
                .with_external_initializer(&tensor.name, weight)
                .map_err(|error| invalid(format!("{name}: {error}")))?;
        }
        Ok(builder.commit_from_memory(graph)?)
    }
}
