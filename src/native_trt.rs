use crate::{invalid, Result};
use std::{
    ffi::{CStr, CString},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    ptr::NonNull,
};

pub(crate) mod graph;

const TOPK_CAPACITY: usize = 256;
const ERROR_CAPACITY: usize = 2048;

unsafe extern "C" {
    fn alice_trt_gpt_create(
        encoder_onnx: *const std::ffi::c_char,
        encoder_plan: *const std::ffi::c_char,
        step_onnx: *const std::ffi::c_char,
        step_plan: *const std::ffi::c_char,
        error: *mut std::ffi::c_char,
        error_capacity: usize,
    ) -> *mut std::ffi::c_void;
    fn alice_trt_gpt_run_encoder(
        handle: *mut std::ffi::c_void,
        phoneme_ids: *const i64,
        phoneme_count: usize,
        prompt_ids: *const i64,
        prompt_count: usize,
        bert_features: *const f32,
        bert_feature_count: usize,
        topk_values: *mut f32,
        topk_indices: *mut i64,
        topk_capacity: usize,
        topk_count: *mut usize,
        x_length: *mut i64,
        y_length: *mut i64,
        error: *mut std::ffi::c_char,
        error_capacity: usize,
    ) -> i32;
    fn alice_trt_gpt_run_step(
        handle: *mut std::ffi::c_void,
        sample: i64,
        token_index: i64,
        topk_values: *mut f32,
        topk_indices: *mut i64,
        topk_capacity: usize,
        topk_count: *mut usize,
        error: *mut std::ffi::c_char,
        error_capacity: usize,
    ) -> i32;
    fn alice_trt_gpt_reset(handle: *mut std::ffi::c_void);
    fn alice_trt_gpt_destroy(handle: *mut std::ffi::c_void);
}

pub(crate) struct NativeTrtGpt {
    handle: NonNull<std::ffi::c_void>,
}

impl NativeTrtGpt {
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let step = std::env::var_os("ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join("gpt_step.onnx"));
        Self::load_graphs(&model_dir.join("gpt_encoder.onnx"), &step)
    }

    pub(crate) fn load_graphs(encoder: &Path, step: &Path) -> Result<Self> {
        Self::load_graphs_with_precision(encoder, step, "fp32")
    }

    pub(crate) fn load_fp16_graphs(dir: &Path) -> Result<Self> {
        Self::load_graphs_with_precision(
            &dir.join("gpt_encoder.onnx"),
            &dir.join("gpt_step_active_len.onnx"),
            "fp16-cache",
        )
    }

    fn load_graphs_with_precision(encoder: &Path, step: &Path, precision: &str) -> Result<Self> {
        let cache_dir = engine_cache_dir()?;
        let encoder_plan = cache_dir.join(format!(
            "gpt_encoder-{}-trt112-{precision}.plan",
            model_fingerprint(encoder)?
        ));
        let step_plan = cache_dir.join(format!(
            "gpt_step-{}-trt112-{precision}.plan",
            model_fingerprint(step)?
        ));
        std::fs::create_dir_all(&cache_dir)?;
        let encoder = c_string_path(encoder)?;
        let encoder_plan = c_string_path(&encoder_plan)?;
        let step = c_string_path(step)?;
        let step_plan = c_string_path(&step_plan)?;
        let mut error = [0i8; ERROR_CAPACITY];
        let handle = unsafe {
            alice_trt_gpt_create(
                encoder.as_ptr(),
                encoder_plan.as_ptr(),
                step.as_ptr(),
                step_plan.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| ffi_error(&error))?;
        Ok(Self { handle })
    }

    pub(crate) fn run_encoder(
        &mut self,
        phoneme_ids: &[i64],
        prompt_ids: &[i64],
        bert_features: &[f32],
    ) -> Result<(Vec<f32>, Vec<i64>, i64, i64)> {
        #[cfg(all(windows, feature = "r1-cuda-ep"))]
        let _range = crate::nvtx::Range::push(b"gpt_encoder_native_trt\0");
        let mut values = vec![0.0f32; TOPK_CAPACITY];
        let mut indices = vec![0i64; TOPK_CAPACITY];
        let mut count = 0usize;
        let mut x_length = 0i64;
        let mut y_length = 0i64;
        let mut error = [0i8; ERROR_CAPACITY];
        let status = unsafe {
            alice_trt_gpt_run_encoder(
                self.handle.as_ptr(),
                phoneme_ids.as_ptr(),
                phoneme_ids.len(),
                prompt_ids.as_ptr(),
                prompt_ids.len(),
                bert_features.as_ptr(),
                bert_features.len(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                TOPK_CAPACITY,
                &mut count,
                &mut x_length,
                &mut y_length,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check_status(status, &error)?;
        validate_topk_count(count)?;
        values.truncate(count);
        indices.truncate(count);
        Ok((values, indices, x_length, y_length))
    }

    pub(crate) fn run_step(
        &mut self,
        sample: i64,
        token_index: usize,
    ) -> Result<(Vec<f32>, Vec<i64>)> {
        #[cfg(all(windows, feature = "r1-cuda-ep"))]
        let _range = crate::nvtx::Range::push(b"gpt_step_native_trt\0");
        let mut values = vec![0.0f32; TOPK_CAPACITY];
        let mut indices = vec![0i64; TOPK_CAPACITY];
        let mut count = 0usize;
        let mut error = [0i8; ERROR_CAPACITY];
        let status = unsafe {
            alice_trt_gpt_run_step(
                self.handle.as_ptr(),
                sample,
                token_index as i64,
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                TOPK_CAPACITY,
                &mut count,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check_status(status, &error)?;
        validate_topk_count(count)?;
        values.truncate(count);
        indices.truncate(count);
        Ok((values, indices))
    }

    pub(crate) fn reset(&mut self) {
        unsafe { alice_trt_gpt_reset(self.handle.as_ptr()) };
    }
}

impl Drop for NativeTrtGpt {
    fn drop(&mut self) {
        unsafe { alice_trt_gpt_destroy(self.handle.as_ptr()) };
    }
}

fn validate_topk_count(count: usize) -> Result<()> {
    if count == 0 || count > TOPK_CAPACITY {
        return Err(invalid(format!(
            "TensorRT GPT returned an invalid top-k length: {count}"
        )));
    }
    Ok(())
}

fn check_status(status: i32, error: &[i8; ERROR_CAPACITY]) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(ffi_error(error))
    }
}

fn ffi_error(error: &[i8; ERROR_CAPACITY]) -> crate::Error {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    invalid(if message.is_empty() {
        "TensorRT native GPT operation failed".to_owned()
    } else {
        message.into_owned()
    })
}

fn c_string_path(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| invalid("TensorRT path contains an embedded NUL"))
}

fn engine_cache_dir() -> Result<PathBuf> {
    let path = std::env::var_os("ALICE_NATIVE_TTS_TRT_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .map(|root| root.join("AliceNativeTts").join("Trt11GptCache"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("alice-native-tts-trt11-gpt-cache"));
    Ok(path)
}

fn model_fingerprint(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        for byte in &buffer[..bytes] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

#[cfg(test)]
mod repeatability_tests {
    use super::*;

    #[test]
    #[ignore = "requires local TRT/CUDA runtime and ABI-1 model fixture; run serially"]
    fn native_topk_reset_repeatability() {
        let pack = std::env::var_os("ALICE_NATIVE_TTS_TEST_PACK").expect("set fixture directory");
        let output =
            std::env::var_os("ALICE_NATIVE_TTS_TEST_TOPK_OUTPUT").expect("set unique output file");
        assert!(
            !Path::new(&output).exists(),
            "do not overwrite prior evidence"
        );
        let mut runner = NativeTrtGpt::load(Path::new(&pack)).unwrap();
        assert!(runner.run_step(1, 0).is_err());
        let mut requests = Vec::new();
        for (phonemes, prompts) in [(4usize, 4usize), (7, 5), (4, 4)] {
            runner.reset();
            assert!(runner.run_step(1, 0).is_err());
            let encoder = runner
                .run_encoder(
                    &vec![1; phonemes],
                    &vec![3; prompts],
                    &vec![0.0; 1024 * phonemes],
                )
                .unwrap();
            let mut steps = Vec::new();
            for idx in 0..16 {
                // Predetermined identical inputs isolate DMA from sampling.
                let (values, indices) = runner.run_step((idx * 7 + 1) as i64, idx).unwrap();
                assert!(values.iter().all(|x| x.is_finite()));
                steps.push((values, indices));
            }
            // Exercise a late valid cache boundary with identical synthetic cache
            // contents in both modes (not a natural-language quality test).
            let last_index = (511 - encoder.2 - encoder.3) as usize;
            for idx in [247, last_index] {
                let (values, indices) = runner.run_step(1, idx).unwrap();
                assert!(values.iter().all(|x| x.is_finite()));
                steps.push((values, indices));
            }
            if std::env::var_os("ALICE_NATIVE_TTS_TRT_SHAPE_STEP_ONNX").is_some() {
                assert!(runner.run_step(1, last_index + 1).is_err());
                assert_eq!(
                    runner.run_step(1, last_index).unwrap(),
                    *steps.last().unwrap(),
                    "rejected active length must not corrupt cache"
                );
            }
            assert!(runner.run_step(1, 512).is_err());
            requests.push((encoder, steps));
        }
        assert_eq!(
            requests[0], requests[2],
            "reset must restore deterministic top-k outputs"
        );
        std::fs::write(output, serde_json::to_vec_pretty(&requests).unwrap()).unwrap();
    }
}
