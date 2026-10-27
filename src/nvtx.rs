//! Opt-in NVTX ranges used only to correlate ORT calls with Nsight timelines.

use std::{
    ffi::{c_char, c_void, OsStr},
    sync::OnceLock,
};

type PushRange = unsafe extern "C" fn(*const c_char) -> i32;
type PopRange = unsafe extern "C" fn() -> i32;

struct Api {
    _module: usize,
    push: PushRange,
    pop: PopRange,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static API: OnceLock<Option<Api>> = OnceLock::new();

pub(crate) struct Range {
    active: bool,
}

impl Range {
    pub(crate) fn push(name: &'static [u8]) -> Self {
        if !*ENABLED.get_or_init(|| {
            std::env::var("ALICE_NATIVE_TTS_NSIGHT_NVTX")
                .ok()
                .as_deref()
                == Some("1")
        }) || name.last() != Some(&0)
        {
            return Self { active: false };
        }

        let Some(api) = API.get_or_init(load_api).as_ref() else {
            return Self { active: false };
        };
        unsafe { (api.push)(name.as_ptr().cast()) };
        Self { active: true }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if self.active {
            if let Some(api) = API.get().and_then(Option::as_ref) {
                unsafe { (api.pop)() };
            }
        }
    }
}

fn load_api() -> Option<Api> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(path: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    use std::os::windows::ffi::OsStrExt;
    let dll = std::env::var_os("ALICE_NATIVE_TTS_NVTX_DLL")
        .unwrap_or_else(|| OsStr::new("nvToolsExt64_1.dll").to_os_string());
    let wide: Vec<u16> = OsStr::new(&dll).encode_wide().chain(Some(0)).collect();
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module.is_null() {
        return None;
    }
    let push = unsafe { GetProcAddress(module, b"nvtxRangePushA\0".as_ptr()) };
    let pop = unsafe { GetProcAddress(module, b"nvtxRangePop\0".as_ptr()) };
    if push.is_null() || pop.is_null() {
        return None;
    }
    Some(Api {
        _module: module as usize,
        push: unsafe { std::mem::transmute(push) },
        pop: unsafe { std::mem::transmute(pop) },
    })
}
