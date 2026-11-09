//! A C interface to the native udlex reader. See `include/udlex.h`.

use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice, str};

use udlex_rs::Lexicon;

// C can share and close handles across threads without Rust's type checks.
const _: fn() = || {
    fn require_shared<T: Send + Sync>() {}
    require_shared::<Lexicon>();
};

/// Opaque handle whose allocation is owned by the library.
#[repr(C)]
pub struct UdlexNativeLexicon {
    _opaque: [u8; 0],
}

#[repr(i32)]
enum Status {
    Ok = 0,
    MoreSpace = 1,
    InvalidArgument = -1,
    InvalidUtf8 = -2,
    OpenFailed = -3,
    Panic = -4,
}

fn boundary(body: impl FnOnce() -> Result<(), Status>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => Status::Ok as i32,
        Ok(Err(status)) => status as i32,
        Err(_) => Status::Panic as i32,
    }
}

// The caller supplies a live, readable buffer for nonzero lengths in range.
unsafe fn text<'a>(data: *const u8, len: usize) -> Result<&'a str, Status> {
    if len == 0 {
        return Ok("");
    }
    if data.is_null() || len > isize::MAX as usize {
        return Err(Status::InvalidArgument);
    }
    str::from_utf8(unsafe { slice::from_raw_parts(data, len) }).map_err(|_| Status::InvalidUtf8)
}

/// Returns a read-only, null-terminated version string owned by the library.
#[unsafe(no_mangle)]
pub extern "C" fn udlex_native_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Opens a compiled lexicon directory and writes an owned handle to `out`.
///
/// # Safety
/// `path` must be readable for `path_len` bytes when that length is nonzero
/// and within range. `out` must be writable and aligned for one handle pointer.
/// The input and output must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn udlex_native_open(
    path: *const u8,
    path_len: usize,
    out: *mut *mut UdlexNativeLexicon,
) -> i32 {
    boundary(|| {
        if out.is_null() || !out.is_aligned() {
            return Err(Status::InvalidArgument);
        }
        unsafe { out.write(ptr::null_mut()) };
        let path = unsafe { text(path, path_len) }?;
        if path.is_empty() || path.contains('\0') {
            return Err(Status::InvalidArgument);
        }
        let lexicon = Lexicon::open(path).map_err(|_| Status::OpenFailed)?;
        unsafe { out.write(Box::into_raw(Box::new(lexicon)).cast()) };
        Ok(())
    })
}

/// Releases a handle. A null handle is accepted.
///
/// # Safety
/// A non-null handle must come from this library and still be live. Wait for
/// every reader before closing it. Do not use the handle again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn udlex_native_close(lexicon: *mut UdlexNativeLexicon) -> i32 {
    boundary(|| {
        let lexicon = lexicon.cast::<Lexicon>();
        if !lexicon.is_null() {
            if !lexicon.is_aligned() {
                return Err(Status::InvalidArgument);
            }
            drop(unsafe { Box::from_raw(lexicon) });
        }
        Ok(())
    })
}

/// Writes one UTF-8 lemma to a caller-owned buffer without a terminator.
///
/// # Safety
/// The handle must remain live until the call returns. `token` and `out` must
/// cover their declared lengths when nonzero and within range. `out_len` must
/// be writable and aligned for one `usize`. Buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn udlex_native_lemma(
    lexicon: *const UdlexNativeLexicon,
    token: *const u8,
    token_len: usize,
    sentence_initial: u8,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> i32 {
    boundary(|| {
        if out_len.is_null() || !out_len.is_aligned() {
            return Err(Status::InvalidArgument);
        }
        unsafe { out_len.write(0) };
        let lexicon = lexicon.cast::<Lexicon>();
        if lexicon.is_null()
            || !lexicon.is_aligned()
            || out_cap > isize::MAX as usize
            || (out.is_null() && out_cap != 0)
        {
            return Err(Status::InvalidArgument);
        }
        let token = unsafe { text(token, token_len) }?;
        let lemma = unsafe { &*lexicon }.lemma(token, sentence_initial != 0);
        let len = lemma.len();
        unsafe { out_len.write(len) };
        if len > out_cap {
            return Err(Status::MoreSpace);
        }
        if len != 0 {
            unsafe { ptr::copy_nonoverlapping(lemma.as_ptr(), out, len) };
        }
        Ok(())
    })
}
