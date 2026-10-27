use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/trt_gpt.cpp");
    println!("cargo:rerun-if-changed=native/trt_gpt.h");
    println!("cargo:rerun-if-env-changed=ALICE_TRT_HOME");
    println!("cargo:rerun-if-env-changed=ALICE_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ALICE_CUDA_NVCC_HOME");

    if env::var_os("CARGO_FEATURE_R2_TRT_NATIVE").is_none() {
        return;
    }

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        panic!("the experimental native TensorRT backend currently requires Windows x64");
    }

    let trt_home = required_path("ALICE_TRT_HOME");
    let cuda_home = required_path("ALICE_CUDA_HOME");
    let cuda_nvcc_home = required_path("ALICE_CUDA_NVCC_HOME");
    for required in [
        trt_home.join("include/NvInfer.h"),
        trt_home.join("include/NvOnnxParser.h"),
        trt_home.join("lib/nvinfer_11.lib"),
        trt_home.join("lib/nvonnxparser_11.lib"),
        cuda_home.join("include/cuda_runtime_api.h"),
        cuda_home.join("lib/x64/cudart.lib"),
        cuda_nvcc_home.join("include/crt/host_defines.h"),
    ] {
        if !required.is_file() {
            panic!(
                "native TensorRT prerequisite not found: {}",
                required.display()
            );
        }
    }

    cc::Build::new()
        .cpp(true)
        .file("native/trt_gpt.cpp")
        .include("native")
        .include(trt_home.join("include"))
        .include(cuda_nvcc_home.join("include"))
        .include(cuda_home.join("include"))
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("/Zc:__cplusplus")
        .warnings(true)
        .compile("alice_native_trt_gpt");

    println!(
        "cargo:rustc-link-search=native={}",
        trt_home.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib/x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=nvinfer_11");
    println!("cargo:rustc-link-lib=dylib=nvonnxparser_11");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn required_path(name: &str) -> PathBuf {
    let value = env::var_os(name)
        .unwrap_or_else(|| panic!("set {name} to the extracted native dependency directory"));
    PathBuf::from(value)
}
