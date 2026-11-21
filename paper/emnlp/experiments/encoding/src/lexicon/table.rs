//! What the core's `hybrid.rs` imports from its parent module, so that the
//! production file compiles here unchanged. Everything between the markers is
//! copied from the core's `table.rs`; `shim_is_verbatim` fails when it drifts.

// The core's `Bytes` has variants only its own callers construct.
#![allow(dead_code)]

use std::ops::Deref;

use memmap2::Mmap;

#[allow(clippy::all, dead_code)]
#[path = "../../../../../../rust/src/lexicon/table/hybrid.rs"]
pub mod hybrid;

// ---- verbatim from the core ----

/// Where a table's bytes live.
pub enum Bytes {
    /// Mapped from the bundle directory.
    Mapped(Mmap),
    /// Embedded at compile time, typically via `include_bytes!`.
    Static(&'static [u8]),
    /// Built in memory, such as a candidate phrase table.
    Owned(Vec<u8>),
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(mmap) => mmap,
            Self::Static(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }
}

/// The unit a double-array walk starts from.
const DA_ROOT: u32 = 0;

/// Reads the unit at `index` without a bounds check.
///
/// # Safety
///
/// `index` must be below the number of units in `units`. Every index this
/// module passes comes from [`da_offset`] of a non-leaf unit, which
/// [`DoubleArray::new`] validated at load time — see [`da_search`].
#[inline]
// A validated unit index times four cannot overflow; a checked multiply here
// would cost more than the read on every key byte.
#[allow(clippy::arithmetic_side_effects)]
const unsafe fn da_unit(units: &[u8], index: u32) -> u32 {
    // SAFETY: the caller guarantees the index is in bounds; a unit is four
    // bytes, so the whole read is, and `read_unaligned` drops alignment.
    let bytes = unsafe {
        units
            .as_ptr()
            .add(index as usize * yada::unit::UNIT_SIZE)
            .cast::<[u8; yada::unit::UNIT_SIZE]>()
            .read_unaligned()
    };
    u32::from_le_bytes(bytes)
}

/// The transition base of a non-leaf unit, exactly as `yada` decodes it.
#[inline]
const fn da_offset(unit: u32) -> u32 {
    (unit >> 10) << ((unit & (1 << 9)) >> 6)
}

/// Walks an optional namespace byte before the key, under the same invariant
/// as [`da_search`]; raw units so a hybrid table's short side shares it.
#[inline]
pub(super) fn da_search_units(units: &[u8], prefix: Option<u8>, key: &[u8]) -> Option<u32> {
    let mut position = DA_ROOT;
    // SAFETY: a validated array is non-empty, so the root exists.
    let mut unit = unsafe { da_unit(units, position) };
    if unit >> 31 != 0 {
        // A root with the leaf bit set means a corrupt array; miss, not UB.
        return None;
    }
    let mut advance = |byte| {
        if byte == 0 {
            // NUL is the builder's terminator and never part of a key.
            return false;
        }
        position = da_offset(unit) ^ position ^ u32::from(byte);
        // SAFETY: the root is non-leaf, and another step follows only a
        // successful label check, which includes the leaf bit. XOR with one
        // byte stays in the validated 256-unit block, including for `prefix`.
        unit = unsafe { da_unit(units, position) };
        unit & 0x8000_00FF == u32::from(byte)
    };
    if prefix.is_some_and(|byte| !advance(byte)) {
        return None;
    }
    for &byte in key {
        if !advance(byte) {
            return None;
        }
    }
    // The unit must not be a leaf (empty key: the corrupt-root case) and must
    // have a value child.
    if unit & 0x8000_0100 != 0x100 {
        return None;
    }
    let position = da_offset(unit) ^ position;
    // SAFETY: `unit` is non-leaf, so `position` is in bounds.
    let unit = unsafe { da_unit(units, position) };
    if unit >> 31 == 0 {
        return None;
    }
    Some(unit & 0x7FFF_FFFF)
}

/// Values of every leaf unit in a validated double-array trie.
fn da_leaf_values(units: &[u8]) -> impl Iterator<Item = u32> + '_ {
    units
        .as_chunks::<{ yada::unit::UNIT_SIZE }>()
        .0
        .iter()
        .filter_map(|chunk| {
            let unit = u32::from_le_bytes(*chunk);
            (unit >> 31 != 0).then_some(unit & 0x7FFF_FFFF)
        })
}

// ---- end verbatim ----

#[cfg(test)]
mod tests {
    #[test]
    fn shim_is_verbatim() {
        let core = include_str!("../../../../../../rust/src/lexicon/table.rs");
        let shim = include_str!("table.rs");
        let body = shim
            .split("// ---- verbatim from the core ----")
            .nth(1)
            .and_then(|rest| rest.split("// ---- end verbatim ----").next())
            .expect("markers");
        let items: Vec<&str> = body.split("\n\n").map(str::trim).collect();
        assert!(items.len() >= 6);
        for item in items.iter().filter(|item| !item.is_empty()) {
            assert!(core.contains(item), "drifted from the core:\n{item}");
        }
    }
}
