//! Optional trtllm-gen long-context prefill attention backend.
//!
//! grout's checked cuTile LPT kernel is the default and is at measured
//! parity with its zero-check unsafe twin; the remaining long-context gap
//! vs vLLM (-29%/layer at 16K/32K on sm_100) is trtllm-gen's CUDA C++
//! kernels, which this module dlopen's via the shim in
//! benchmarks/trtllm_shim (build instructions there). Opt-in and external
//! by design — the same category as cuBLAS: a vendored, arch-locked
//! backend outside the safety story.
//!
//! Enable with GROUT_TRTLLM_ATTN=1 and GROUT_TRTLLM_ATTN_LIB=<path to
//! libgrout_trtllm.so>. Falls back to the cuTile kernel if the library is
//! missing or a call fails.

use anyhow::{bail, Result};
use libloading::{Library, Symbol};
use std::ffi::CStr;
use std::sync::OnceLock;

type RaggedFn = unsafe extern "C" fn(
    out: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    k: *mut core::ffi::c_void,
    v: *mut core::ffi::c_void,
    q_len: i64,
    kv_len: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    k_stride_tokens: i64,
    k_stride_heads: i64,
    v_stride_tokens: i64,
    v_stride_heads: i64,
    bmm1_scale: f32,
    enable_pdl: i32,
    stream: *mut core::ffi::c_void,
) -> i32;
type LastErrFn = unsafe extern "C" fn() -> *const core::ffi::c_char;
type SyncFn = unsafe extern "C" fn() -> i32;

struct Shim {
    _lib: Library,
    ragged: libloading::os::unix::Symbol<RaggedFn>,
    last_err: libloading::os::unix::Symbol<LastErrFn>,
    sync: libloading::os::unix::Symbol<SyncFn>,
}

static SHIM: OnceLock<Option<Shim>> = OnceLock::new();

fn shim() -> Option<&'static Shim> {
    SHIM.get_or_init(|| {
        if std::env::var("GROUT_TRTLLM_ATTN").ok().as_deref() != Some("1") {
            return None;
        }
        let path = match std::env::var("GROUT_TRTLLM_ATTN_LIB") {
            Ok(p) => p,
            Err(_) => {
                eprintln!(
                    "GROUT_TRTLLM_ATTN=1 but GROUT_TRTLLM_ATTN_LIB is unset; \
                     falling back to the cuTile attention kernel"
                );
                return None;
            }
        };
        // SAFETY: loading the shim we built; symbols are checked below.
        let lib = match unsafe { Library::new(&path) } {
            Ok(l) => l,
            Err(e) => {
                eprintln!("failed to load {path}: {e}; falling back to cuTile attention");
                return None;
            }
        };
        unsafe {
            let ragged: Symbol<RaggedFn> = match lib.get(b"grout_trtllm_ragged_context_f16") {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("symbol missing in {path}: {e}");
                    return None;
                }
            };
            let last_err: Symbol<LastErrFn> = lib.get(b"grout_trtllm_last_error").ok()?;
            let sync: Symbol<SyncFn> = lib.get(b"grout_trtllm_sync").ok()?;
            let ragged = ragged.into_raw();
            let last_err = last_err.into_raw();
            let sync = sync.into_raw();
            Some(Shim {
                _lib: lib,
                ragged,
                last_err,
                sync,
            })
        }
    })
    .as_ref()
}

/// True when the env opts in and the shim library loaded.
pub fn enabled() -> bool {
    shim().is_some()
}

/// Launch the trtllm-gen ragged context attention on dense grout buffers.
/// All strides in elements. Brackets the call with device-wide syncs (v1
/// ordering contract; see module docs).
#[allow(clippy::too_many_arguments)]
pub fn ragged_context_f16(
    out: u64,
    q: u64,
    k: u64,
    v: u64,
    q_len: usize,
    kv_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_head_stride: usize,
    qk_scale: f32,
) -> Result<()> {
    let Some(s) = shim() else {
        bail!("trtllm shim not loaded");
    };
    // SAFETY: pointers come from live device tensors owned by the caller;
    // the sync bracket orders the foreign launch against grout's stream.
    unsafe {
        if (s.sync)() != 0 {
            bail!("cuda sync before trtllm launch failed");
        }
        let rc = (s.ragged)(
            out as *mut _,
            q as *mut _,
            k as *mut _,
            v as *mut _,
            q_len as i64,
            kv_len as i64,
            num_q_heads as i64,
            num_kv_heads as i64,
            head_dim as i64,
            head_dim as i64,
            kv_head_stride as i64,
            head_dim as i64,
            kv_head_stride as i64,
            qk_scale,
            0,
            core::ptr::null_mut(),
        );
        if rc != 0 {
            let msg = CStr::from_ptr((s.last_err)()).to_string_lossy().to_string();
            bail!("trtllm ragged context attention failed (rc={rc}): {msg}");
        }
        if (s.sync)() != 0 {
            bail!("cuda sync after trtllm launch failed");
        }
    }
    Ok(())
}
