//! A candidate's on-disk image: 64-byte-aligned sections behind a small
//! directory, mapped read-only. Every structure is queried from such a
//! mapping so that candidates differ in layout, not in where the bytes live.

use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;

const MAGIC: u64 = 0x3147_4D49_4241_4C45; // "ELABIMG1"
const ALIGN: usize = 64;

pub fn write_sections(path: &Path, parts: &[&[u8]]) -> io::Result<u64> {
    let mut out = io::BufWriter::new(std::fs::File::create(path)?);
    let head = (2 + 2 * parts.len()) * 8;
    let mut at = head.next_multiple_of(ALIGN);
    out.write_all(&MAGIC.to_le_bytes())?;
    out.write_all(&(parts.len() as u64).to_le_bytes())?;
    for part in parts {
        out.write_all(&(at as u64).to_le_bytes())?;
        out.write_all(&(part.len() as u64).to_le_bytes())?;
        at = (at + part.len()).next_multiple_of(ALIGN);
    }
    let mut written = head;
    for part in parts {
        let pad = written.next_multiple_of(ALIGN) - written;
        out.write_all(&[0_u8; ALIGN][..pad])?;
        out.write_all(part)?;
        written += pad + part.len();
    }
    out.flush()?;
    Ok(written as u64)
}

pub fn map_file(path: &Path) -> io::Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the lab maps files it wrote itself and never rewrites them
    // while a candidate is open.
    unsafe { Mmap::map(&file) }
}

/// `ENCODING_LAB_HEAP=1` reads images into anonymous memory instead of
/// mapping them: the control arm for "a heap-built candidate lies denser
/// than a mapped file" (and, with THP left on, on huge pages).
pub fn heap_mode() -> bool {
    std::env::var_os("ENCODING_LAB_HEAP").is_some()
}

enum Store {
    Mapped(Mmap),
    // Words, so the copy is as aligned as a mapping's sections need.
    Heap(Vec<u64>, usize),
}

impl Store {
    fn open(path: &Path) -> io::Result<Self> {
        if !heap_mode() {
            return map_file(path).map(Self::Mapped);
        }
        let bytes = std::fs::read(path)?;
        let mut words = vec![0_u64; bytes.len().div_ceil(8)];
        // SAFETY: the destination holds at least `bytes.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast(), bytes.len())
        };
        Ok(Self::Heap(words, bytes.len()))
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            // SAFETY: the vector holds `len` initialised bytes.
            Self::Heap(words, len) => unsafe {
                std::slice::from_raw_parts(words.as_ptr().cast(), *len)
            },
        }
    }
}

pub struct Img {
    map: Store,
    secs: Vec<(usize, usize)>,
}

impl Img {
    pub fn open(path: &Path) -> io::Result<Self> {
        let store = Store::open(path)?;
        let map = store.bytes();
        let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_owned());
        let word = |i: usize| -> io::Result<usize> {
            let b = map.get(i * 8..i * 8 + 8).ok_or_else(|| bad("truncated"))?;
            Ok(u64::from_le_bytes(b.try_into().unwrap()) as usize)
        };
        if word(0)? as u64 != MAGIC {
            return Err(bad("not a lab image"));
        }
        let mut secs = Vec::new();
        for i in 0..word(1)? {
            let (at, len) = (word(2 + 2 * i)?, word(3 + 2 * i)?);
            if at % ALIGN != 0 || at.checked_add(len).is_none_or(|end| end > map.len()) {
                return Err(bad("section outside the file"));
            }
            secs.push((at, len));
        }
        Ok(Self { map: store, secs })
    }

    /// Section `i` for as long as the process lives: callers store the view
    /// next to the `Img` that owns the mapping and drop both together.
    pub fn bytes(&self, i: usize) -> &'static [u8] {
        let (at, len) = self.secs[i];
        // SAFETY: the range was checked against the store in `open`; the
        // store's address is stable until `self` is dropped.
        unsafe { std::slice::from_raw_parts(self.map.bytes().as_ptr().add(at), len) }
    }

    pub fn words<T: Word>(&self, i: usize) -> &'static [T] {
        let bytes = self.bytes(i);
        // SAFETY: plain little-endian integers; sections start 64-aligned.
        let (pre, mid, post) = unsafe { bytes.align_to::<T>() };
        assert!(pre.is_empty() && post.is_empty(), "misaligned section");
        mid
    }
}

/// Integer types a section may be viewed as.
pub trait Word: Copy {}
impl Word for u16 {}
impl Word for u32 {}
impl Word for u64 {}

pub fn as_bytes<T: Word>(words: &[T]) -> &[u8] {
    // SAFETY: integers have no padding; the lab runs little-endian only.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
