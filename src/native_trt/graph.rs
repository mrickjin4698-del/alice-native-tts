//! Host bindings for locally built TensorRT acoustic engines.

use super::{c_string_path, check_status, ffi_error, ERROR_CAPACITY};
use crate::{invalid, Error, Result};
use std::{
    any::Any,
    ffi::{c_char, c_void, CStr},
    marker::PhantomData,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    path::Path,
    ptr::NonNull,
};

#[repr(C)]
pub(crate) struct Input<'a> {
    name: *const c_char,
    data: *const c_void,
    bytes: usize,
    shape: [i64; 4],
    rank: usize,
    borrow: PhantomData<&'a ()>,
}

impl<'a> Input<'a> {
    pub(crate) fn floats<const N: usize>(
        name: &'static CStr,
        dims: [usize; N],
        values: &'a [f32],
    ) -> Self {
        Self::new(name, dims, values)
    }

    pub(crate) fn integers<const N: usize>(
        name: &'static CStr,
        dims: [usize; N],
        values: &'a [i64],
    ) -> Self {
        Self::new(name, dims, values)
    }

    fn new<T, const N: usize>(name: &'static CStr, dims: [usize; N], values: &'a [T]) -> Self {
        let mut shape = [0; 4];
        for (axis, dim) in shape[..N].iter_mut().zip(dims) {
            *axis = dim as i64;
        }
        Self {
            name: name.as_ptr(),
            data: values.as_ptr().cast(),
            bytes: std::mem::size_of_val(values),
            shape,
            rank: N,
            borrow: PhantomData,
        }
    }
}

#[repr(C)]
struct OutputBinding {
    name: *const c_char,
    data: *const f32,
    len: usize,
    shape: [i64; 4],
    rank: usize,
}

pub(crate) struct Output<'a> {
    pub(crate) shape: Vec<i64>,
    pub(crate) values: &'a [f32],
}

unsafe extern "C" {
    fn alice_trt_graph_load(
        path: *const c_char,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    fn alice_trt_graph_run(
        handle: *mut c_void,
        inputs: *const Input<'_>,
        input_count: usize,
        outputs: *mut OutputBinding,
        output_count: usize,
        steps: usize,
        cancel: unsafe extern "C" fn(*mut c_void) -> bool,
        caller: *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn alice_trt_graph_destroy(handle: *mut c_void);
}

struct Observer<'a, F> {
    callback: &'a F,
    cancelled: bool,
    panic: Option<Box<dyn Any + Send>>,
}

unsafe extern "C" fn observe<F: Fn() -> bool>(context: *mut c_void) -> bool {
    // The in-flight FFI invocation owns this observer on the calling thread.
    let observer = unsafe { &mut *context.cast::<Observer<'_, F>>() };
    match catch_unwind(AssertUnwindSafe(|| (observer.callback)())) {
        Ok(cancelled) => {
            observer.cancelled = cancelled;
            cancelled
        }
        Err(payload) => {
            observer.panic = Some(payload);
            true
        }
    }
}

pub(crate) struct Graph(NonNull<c_void>);

impl Graph {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let path = c_string_path(path)?;
        let mut error = [0; ERROR_CAPACITY];
        let handle =
            unsafe { alice_trt_graph_load(path.as_ptr(), error.as_mut_ptr(), error.len()) };
        NonNull::new(handle)
            .map(Self)
            .ok_or_else(|| ffi_error(&error))
    }

    pub(crate) fn run<'a>(
        &'a mut self,
        inputs: &[Input<'_>],
        names: &[&CStr],
    ) -> Result<Vec<Output<'a>>> {
        self.run_steps(inputs, names, 0, &|| false)
    }

    pub(crate) fn run_steps<'a, F: Fn() -> bool>(
        &'a mut self,
        inputs: &[Input<'_>],
        names: &[&CStr],
        steps: usize,
        cancel: &F,
    ) -> Result<Vec<Output<'a>>> {
        let mut outputs: Vec<_> = names
            .iter()
            .map(|name| OutputBinding {
                name: name.as_ptr(),
                data: std::ptr::null(),
                len: 0,
                shape: [0; 4],
                rank: 0,
            })
            .collect();
        let mut observer = Observer {
            callback: cancel,
            cancelled: false,
            panic: None,
        };
        let mut error = [0; ERROR_CAPACITY];
        let status = unsafe {
            alice_trt_graph_run(
                self.0.as_ptr(),
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_mut_ptr(),
                outputs.len(),
                steps,
                observe::<F>,
                std::ptr::from_mut(&mut observer).cast(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if let Some(payload) = observer.panic {
            resume_unwind(payload);
        }
        if observer.cancelled {
            return Err(Error::Cancelled);
        }
        check_status(status, &error)?;
        outputs
            .into_iter()
            .map(|output| {
                if output.data.is_null()
                    || output.len == 0
                    || output.len > isize::MAX as usize / std::mem::size_of::<f32>()
                    || !(1..=output.shape.len()).contains(&output.rank)
                {
                    return Err(invalid("invalid TensorRT graph output"));
                }
                // C++ owns these host buffers until the graph is mutated or dropped.
                let values = unsafe { std::slice::from_raw_parts(output.data, output.len) };
                Ok(Output {
                    shape: output.shape[..output.rank].to_vec(),
                    values,
                })
            })
            .collect()
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { alice_trt_graph_destroy(self.0.as_ptr()) };
    }
}
