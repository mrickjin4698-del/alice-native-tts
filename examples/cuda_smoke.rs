//! Minimal CUDA EP smoke test. The graph has a single Add node and disables
//! the CPU EP, so a successful output proves that the node ran on CUDA.

use ort::{ep::CUDA, session::Session, value::Tensor};
use std::{env, error::Error, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cuda-smoke"));
    fs::create_dir_all(&output_dir)?;

    let mut session = Session::builder()?
        .with_execution_providers([CUDA::default()
            .with_device_id(0)
            .with_tf32(false)
            .build()
            .error_on_failure()])?
        .with_disable_cpu_fallback()?
        .with_profiling(output_dir.join("cuda-add"))?
        .commit_from_memory(&add_model())?;

    let a = Tensor::from_array(([1, 4], vec![1.0f32, 2.0, 3.0, 4.0]))?;
    let b = Tensor::from_array(([1, 4], vec![10.0f32, 20.0, 30.0, 40.0]))?;
    let outputs = session.run(ort::inputs! { "a" => a, "b" => b })?;
    let (_, output) = outputs
        .get("y")
        .ok_or("CUDA smoke graph did not return y")?
        .try_extract_tensor::<f32>()?;
    let output = output.to_vec();
    let expected = [11.0, 22.0, 33.0, 44.0];
    if output != expected {
        return Err(format!("CUDA Add mismatch: got {output:?}, expected {expected:?}").into());
    }
    drop(outputs);
    let profile_path = session.end_profiling()?;
    println!("runtime_build_info={}", ort::info());
    println!("provider=CUDAExecutionProvider");
    println!("cpu_fallback=disabled");
    println!("output={output:?}");
    println!("profile={profile_path}");
    Ok(())
}

fn add_model() -> Vec<u8> {
    let node = message(&[
        bytes_field(1, b"a"),
        bytes_field(1, b"b"),
        bytes_field(2, b"y"),
        bytes_field(3, b"add"),
        bytes_field(4, b"Add"),
    ]);
    let graph = message(&[
        bytes_field(1, &node),
        bytes_field(2, b"cuda_add_smoke"),
        bytes_field(11, &value_info(b"a")),
        bytes_field(11, &value_info(b"b")),
        bytes_field(12, &value_info(b"y")),
    ]);
    let opset = message(&[varint_field(2, 13)]);
    message(&[
        varint_field(1, 8),
        bytes_field(2, b"alice-native-tts-cuda-smoke"),
        bytes_field(7, &graph),
        bytes_field(8, &opset),
    ])
}

fn value_info(name: &[u8]) -> Vec<u8> {
    let dim_one = message(&[varint_field(1, 1)]);
    let dim_four = message(&[varint_field(1, 4)]);
    let shape = message(&[bytes_field(1, &dim_one), bytes_field(1, &dim_four)]);
    let tensor_type = message(&[varint_field(1, 1), bytes_field(2, &shape)]);
    let value_type = message(&[bytes_field(1, &tensor_type)]);
    message(&[bytes_field(1, name), bytes_field(2, &value_type)])
}

fn varint_field(field: u64, value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_varint((field << 3) | 0, &mut bytes);
    write_varint(value, &mut bytes);
    bytes
}

fn bytes_field(field: u64, value: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_varint((field << 3) | 2, &mut bytes);
    write_varint(value.len() as u64, &mut bytes);
    bytes.extend_from_slice(value);
    bytes
}

fn message(fields: &[Vec<u8>]) -> Vec<u8> {
    fields
        .iter()
        .flat_map(|field| field.iter().copied())
        .collect()
}

fn write_varint(mut value: u64, bytes: &mut Vec<u8>) {
    while value >= 0x80 {
        bytes.push((value as u8) | 0x80);
        value >>= 7;
    }
    bytes.push(value as u8);
}
