//! A string arena: one blob and `u32` boundaries. Keys, lemmas and query
//! streams all live in this shape, in memory and on disk.

use std::io::{self, Write};
use std::path::Path;

#[derive(Default)]
pub struct Strs {
    offs: Vec<u32>,
    blob: Vec<u8>,
}

impl Strs {
    pub fn new() -> Self {
        Self {
            offs: vec![0],
            blob: Vec::new(),
        }
    }

    pub fn push(&mut self, s: &[u8]) {
        self.blob.extend_from_slice(s);
        let end = u32::try_from(self.blob.len()).expect("arena exceeds 4 GiB");
        self.offs.push(end);
    }

    pub fn len(&self) -> usize {
        self.offs.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn blob_len(&self) -> usize {
        self.blob.len()
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> &[u8] {
        &self.blob[self.offs[i] as usize..self.offs[i + 1] as usize]
    }

    /// The hot-loop accessor: `i` must be below `len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> &[u8] {
        // SAFETY: the caller keeps `i < len()`; boundaries ascend inside the blob.
        unsafe {
            let a = *self.offs.get_unchecked(i) as usize;
            let b = *self.offs.get_unchecked(i + 1) as usize;
            self.blob.get_unchecked(a..b)
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone {
        (0..self.len()).map(|i| self.get(i))
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut out = io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(&(self.len() as u64).to_le_bytes())?;
        for off in &self.offs {
            out.write_all(&off.to_le_bytes())?;
        }
        out.write_all(&self.blob)?;
        out.flush()
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "truncated arena");
        let n = u64::from_le_bytes(bytes.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let offs_end = 8 + (n + 1) * 4;
        let offs: Vec<u32> = bytes
            .get(8..offs_end)
            .ok_or_else(bad)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let blob = bytes[offs_end..].to_vec();
        if offs.last().copied() != Some(blob.len() as u32) || offs.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad());
        }
        Ok(Self { offs, blob })
    }
}
