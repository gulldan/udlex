//! Mirrors the core's module path so its `pub(in crate::lexicon)` codec
//! resolves here: the lab measures the production source, not a copy.

pub mod table;

use memmap2::Mmap;
use table::hybrid::{Hit, HybridTable};

pub use table::hybrid::{SPLIT_BYTES, build_hybrid};

/// The production double-array walk.
#[inline(always)]
pub fn da_walk(units: &[u8], key: &[u8]) -> Option<u32> {
    table::da_search_units(units, None, key)
}

/// What the production table answered: a string id or the lemma itself.
pub enum CoreHit<'a> {
    Id(u32),
    Text(&'a [u8]),
}

/// The production `HybridTable` over a mapped `.hybrid` file.
pub struct CoreHybrid(HybridTable);

impl CoreHybrid {
    pub fn open(map: Mmap) -> Result<Self, String> {
        HybridTable::open(table::Bytes::Mapped(map)).map(Self)
    }

    /// The same table over a heap copy of the file.
    pub fn open_owned(bytes: Vec<u8>) -> Result<Self, String> {
        HybridTable::open(table::Bytes::Owned(bytes)).map(Self)
    }

    #[inline(always)]
    pub fn get(&self, key: &[u8]) -> Option<CoreHit<'_>> {
        Some(match self.0.get(key)? {
            Hit::Id(id) => CoreHit::Id(id),
            Hit::Text(text) => CoreHit::Text(text.as_bytes()),
        })
    }
}
