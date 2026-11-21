use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::MaybeUninit;
use std::sync::{Mutex, OnceLock, PoisonError};

use super::codec::{Codec, table_codec};
use super::{
    Deref, DoubleArray, Error, FileIntegrity, Meta, OFFSET_WIDTH, Path, PathBuf, Result, Rule, SEP,
    VALUE_WIDTH, expected_file, map_verified_file, open_encoded,
};
use memmap2::Mmap;

mod hybrid;
mod inline_codec;
mod map;
mod validation;

pub(super) use hybrid::Hit as HybridHit;
pub(super) use hybrid::HybridTable;
#[cfg(feature = "train")]
pub use hybrid::build_hybrid;
#[cfg(all(test, feature = "train"))]
pub(super) use inline_codec::decode_varint;
pub(super) use inline_codec::lookup as lookup_inline;
pub(super) use map::CachedMap;
pub(super) use validation::validate_tables;

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

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// One lookup table, in whichever encoding the compiler kept for it.
// Keep DA dispatch independent of payload niches.
#[repr(u8)]
pub enum Table {
    /// A double-array trie.
    Da(DoubleArray<Bytes>),
    /// A finite-state map from form to string identifier.
    Map(CachedMap),
    /// A finite-state acceptor that spells the lemma out in the key.
    Inline(fst::raw::Fst<Bytes>),
    /// Short keys in a double array, long keys under a pilot hash.
    Hybrid(HybridTable),
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

/// Walks a double array for one key: `yada`'s `exact_match_search` with its
/// per-byte asserts turned into misses (it panics on NUL or a corrupt array).
///
/// # Safety, in the type-system sense
///
/// Sound for any `units` validated by [`DoubleArray::new`]: the unit count is
/// a multiple of 256 and `da_offset(unit) ^ i` is in bounds for every non-leaf
/// unit at index `i`. XOR with a byte stays in the same 256-unit block, and
/// offsets are read only from non-leaf units (the root is checked, later
/// units passed a label check that includes the leaf bit).
#[inline]
pub(super) fn da_search(trie: &DoubleArray<Bytes>, key: &[u8]) -> Option<u32> {
    da_search_units(&trie.0, None, key)
}

/// Walks an optional namespace byte before the key, under the same invariant
/// as [`da_search`]; raw units so a hybrid table's short side shares it.
#[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
#[inline(always)]
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

/// One node of a finite-state map, part-way through being measured; decoded
/// once on push.
struct Frame<'f> {
    /// The node itself.
    node: fst::raw::Node<'f>,
    /// Which of its transitions to descend next.
    next: usize,
    /// The largest value reachable from here, so far.
    best: u64,
    /// The output on the transition that led here from the parent.
    edge: u64,
}

/// Hashes an address with one multiply; addresses are not adversarial, and
/// `SipHash` was a quarter of opening a Finnish lexicon.
#[derive(Default)]
pub struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        let mixed = (self.0 ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = mixed ^ (mixed >> 32);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

/// The largest value any key in a finite-state map carries.
///
/// Memoised per node instead of streaming every key (2.2x–32x faster on
/// shipped bundles); the graph is acyclic, so the walk terminates.
pub(super) fn max_output(fsa: &fst::raw::Fst<Bytes>) -> u64 {
    // What a key is worth if it ends here: nothing unless the node is final.
    let stops_at = |node: &fst::raw::Node<'_>| {
        if node.is_final() {
            node.final_output().value()
        } else {
            0
        }
    };

    let mut measured: HashMap<fst::raw::CompiledAddr, u64, BuildHasherDefault<AddressHasher>> =
        HashMap::default();
    let root = fsa.root();
    let mut stack = vec![Frame {
        best: stops_at(&root),
        node: root,
        next: 0,
        edge: 0,
    }];
    while let Some(frame) = stack.last_mut() {
        if frame.next < frame.node.len() {
            let transition = frame.node.transition(frame.next);
            frame.next = frame.next.saturating_add(1);
            if let Some(&known) = measured.get(&transition.addr) {
                // Saturating, so a corrupt sum fails the caller's range check.
                frame.best = frame.best.max(transition.out.value().saturating_add(known));
            } else {
                let node = fsa.node(transition.addr);
                stack.push(Frame {
                    best: stops_at(&node),
                    node,
                    next: 0,
                    edge: transition.out.value(),
                });
            }
            continue;
        }
        let Some(finished) = stack.pop() else {
            break;
        };
        let Some(parent) = stack.last_mut() else {
            return finished.best;
        };
        measured.insert(finished.node.addr(), finished.best);
        parent.best = parent.best.max(finished.edge.saturating_add(finished.best));
    }
    0
}

/// What a table lookup produced.
pub enum Found<'a> {
    /// A string identifier, to be resolved against the string table.
    Id(u64),
    /// The lemma itself, reconstructed by the inline codec.
    Lemma(String),
    /// The lemma spelled in the table's own mapping, borrowed from it.
    Text(&'a str),
}

/// Bytes reserved on the stack for a one-byte namespace and its lookup key;
/// longer keys spill to the heap.
const PREFIXED_STACK_BYTES: usize = 256;

/// Writes `prefix` then `key` into a stack buffer that was never zeroed.
///
/// `None` when the buffer is too small; the caller spills to the heap.
pub(super) fn assemble_key<'stack>(
    prefix: u8,
    key: &[u8],
    stack: &'stack mut [MaybeUninit<u8>],
) -> Option<&'stack [u8]> {
    let needed = key.len().checked_add(1)?;
    let room = stack.get_mut(..needed)?;
    let (first, rest) = room.split_first_mut()?;
    first.write(prefix);
    // SAFETY: `rest` is exactly `key.len()` bytes of writable storage, and a
    // caller's key never aliases this frame's buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(key.as_ptr(), rest.as_mut_ptr().cast::<u8>(), key.len());
    }
    // SAFETY: all `needed` bytes were just written, and `MaybeUninit<u8>` has
    // the size and alignment of `u8`.
    Some(unsafe { &*(std::ptr::from_mut::<[MaybeUninit<u8>]>(room) as *const [u8]) })
}

/// Prepends one namespace byte without allocating for ordinary lookup keys.
///
/// The result stays a `str` for the inline codec, so a non-ASCII prefix is refused.
fn prefixed_key<'stack>(
    prefix: u8,
    key: &str,
    stack: &'stack mut [MaybeUninit<u8>; PREFIXED_STACK_BYTES],
) -> Option<Cow<'stack, str>> {
    if !prefix.is_ascii() {
        return None;
    }
    if let Some(bytes) = assemble_key(prefix, key.as_bytes(), stack) {
        // SAFETY: one ASCII byte followed by the bytes of a `str`.
        return Some(Cow::Borrowed(unsafe {
            std::str::from_utf8_unchecked(bytes)
        }));
    }
    let mut spill = String::with_capacity(key.len().checked_add(1)?);
    spill.push(char::from(prefix));
    spill.push_str(key);
    Some(Cow::Owned(spill))
}

impl Table {
    /// Looks one form up; a `str` so the inline codec slices on char boundaries.
    pub(crate) fn get(&self, key: &str) -> Option<Found<'_>> {
        match self {
            Self::Da(trie) => da_search(trie, key.as_bytes()).map(|id| Found::Id(u64::from(id))),
            Self::Map(map) => map.get(key.as_bytes()).map(Found::Id),
            Self::Inline(fsa) => lookup_inline(fsa, key).map(Found::Lemma),
            Self::Hybrid(table) => table.get(key.as_bytes()).map(|hit| match hit {
                HybridHit::Id(id) => Found::Id(u64::from(id)),
                HybridHit::Text(text) => Found::Text(text),
            }),
        }
    }

    /// Looks up a key under a one-byte namespace: the entity table uses
    /// `0<exact>` and `1<locale-lower>` keys.
    pub(crate) fn get_prefixed(&self, prefix: u8, key: &str) -> Option<Found<'_>> {
        if let Self::Da(trie) = self {
            if !prefix.is_ascii() {
                return None;
            }
            return da_search_units(&trie.0, Some(prefix), key.as_bytes())
                .map(|id| Found::Id(u64::from(id)));
        }
        let mut stack = [MaybeUninit::<u8>::uninit(); PREFIXED_STACK_BYTES];
        let namespaced = prefixed_key(prefix, key, &mut stack)?;
        self.get(namespaced.as_ref())
    }

    /// Number of keys represented by this table.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Da(trie) => da_leaf_values(&trie.0).count(),
            Self::Map(map) => map.fst().len(),
            Self::Inline(fsa) => fsa.len(),
            Self::Hybrid(table) => table.len(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod prefixed_key_tests {
    use super::{
        Bytes, Cow, DoubleArray, Found, MaybeUninit, PREFIXED_STACK_BYTES, Table, da_offset,
        prefixed_key,
    };
    use yada::builder::DoubleArrayBuilder;

    fn identifier(found: Option<Found>) -> Option<u64> {
        match found {
            Some(Found::Id(id)) => Some(id),
            other => {
                assert!(other.is_none(), "a double array returns identifiers");
                None
            }
        }
    }

    #[test]
    fn ordinary_prefixed_keys_borrow_stack_and_long_ones_spill() {
        let mut stack = [MaybeUninit::<u8>::uninit(); PREFIXED_STACK_BYTES];
        let ordinary = prefixed_key(b'0', "Open AI", &mut stack).unwrap();
        assert!(matches!(ordinary, Cow::Borrowed(_)));
        assert_eq!(ordinary.as_ref(), "0Open AI");
        drop(ordinary);

        let long = "x".repeat(PREFIXED_STACK_BYTES);
        let spilled = prefixed_key(b'1', &long, &mut stack).unwrap();
        assert!(matches!(spilled, Cow::Owned(_)));
        assert_eq!(spilled.len(), long.len() + 1);
        assert_eq!(spilled.strip_prefix('1'), Some(long.as_str()));
    }

    #[test]
    fn a_non_ascii_namespace_is_refused() {
        let mut stack = [MaybeUninit::<u8>::uninit(); PREFIXED_STACK_BYTES];
        assert!(prefixed_key(0xFF, "Open AI", &mut stack).is_none());
    }

    #[test]
    fn direct_da_prefixes_agree_with_constructed_keys() {
        let long = [
            "x".repeat(254),
            "x".repeat(255),
            "x".repeat(256),
            "日本".repeat(80),
        ];
        let keys = ["", "a", "ab", "é", "e\u{301}", "ёж", "日本", "🦀"];
        let mut rows = Vec::new();
        for prefix in 1..=127_u8 {
            for key in keys.into_iter().chain(long.iter().map(String::as_str)) {
                rows.push((
                    format!("{}{key}", char::from(prefix)),
                    u32::try_from(rows.len()).unwrap(),
                ));
            }
        }
        rows.sort_unstable();
        let pairs: Vec<_> = rows.iter().map(|(key, id)| (key.as_bytes(), *id)).collect();
        let bytes = DoubleArrayBuilder::build(&pairs).unwrap();
        let table = Table::Da(DoubleArray::new(Bytes::Owned(bytes)).unwrap());
        let Table::Da(trie) = &table else {
            unreachable!()
        };
        for prefix in 0..=127_u8 {
            for key in keys
                .into_iter()
                .chain(long.iter().map(String::as_str))
                .chain(["abc", "b", "e", "ё", "日本語", "\0", "a\0b", "日本\0"])
            {
                let constructed = format!("{}{key}", char::from(prefix));
                // Yada deliberately panics on NUL; both public paths must miss.
                let expected = if constructed.as_bytes().contains(&0) {
                    None
                } else {
                    trie.exact_match_search(constructed.as_bytes())
                        .map(u64::from)
                };
                assert_eq!(
                    identifier(table.get(&constructed)),
                    expected,
                    "{constructed:?}"
                );
                assert_eq!(
                    identifier(table.get_prefixed(prefix, key)),
                    expected,
                    "{constructed:?}"
                );
            }
        }
        for prefix in 128..=u8::MAX {
            assert!(table.get_prefixed(prefix, "日本").is_none());
        }
    }

    #[test]
    fn direct_da_prefix_does_not_walk_a_leaf_as_a_node() {
        let mut root_leaf = vec![0; 256 * yada::unit::UNIT_SIZE];
        root_leaf
            .get_mut(..4)
            .unwrap()
            .copy_from_slice(&(1_u32 << 31).to_le_bytes());
        let mut prefix_leaf = DoubleArrayBuilder::build(&[(b"0a".as_slice(), 9)]).unwrap();
        let root = u32::from_le_bytes(prefix_leaf.get(..4).unwrap().try_into().unwrap());
        let at = usize::try_from(da_offset(root) ^ u32::from(b'0'))
            .unwrap()
            .checked_mul(4)
            .unwrap();
        let word = prefix_leaf.get_mut(at..at.checked_add(4).unwrap()).unwrap();
        let leaf = u32::from_le_bytes(word.try_into().unwrap()) | (1 << 31);
        word.copy_from_slice(&leaf.to_le_bytes());
        for bytes in [root_leaf, prefix_leaf] {
            // Structural validation permits leaf units; the walk must reject
            // them before interpreting their value bits as a transition base.
            let table = Table::Da(DoubleArray::new(Bytes::Owned(bytes)).unwrap());
            for key in ["", "a", "ab"] {
                assert!(table.get(&format!("0{key}")).is_none());
                assert!(table.get_prefixed(b'0', key).is_none());
            }
        }
    }
}

/// What opening the entity table will need, captured without reading it.
pub(super) struct EntityPlan {
    /// The compiled table file, extension included.
    path: PathBuf,
    /// The codec the metadata records for it.
    codec: Codec,
    /// Size and digest the manifest binds this generation to.
    expected: FileIntegrity,
    /// How many rows the metadata declares.
    declared: u64,
}

impl EntityPlan {
    /// Authenticates the file and parses it under its codec.
    fn open(&self) -> Result<Table> {
        let bytes = map_verified_file(&self.path, &self.expected)?;
        let table = open_encoded(&self.path, self.codec, bytes)?;
        let observed = table.len();
        if u64::try_from(observed).ok() != Some(self.declared) {
            return Err(Error::Invalid {
                path: self.path.clone(),
                message: format!(
                    "contains {observed} entries, metadata declares {}",
                    self.declared
                ),
            });
        }
        Ok(table)
    }
}

/// What is left to open, and the lock that keeps two threads from both doing it.
struct Pending {
    plan: EntityPlan,
    /// Orders the fallible initialisation, which `OnceLock` alone cannot
    /// express. Held only while opening; every read goes through the cell.
    gate: Mutex<()>,
}

/// The isolated entity table, authenticated when something first asks for it:
/// hashing it is the most expensive part of an open and plain lemmatisation
/// never reads it.
///
/// The cell and the plan are separate fields so the probe's accessor
/// ([`Entities::for_probe`]) stays one load: a three-way enum match there cost
/// `ru_taiga` 7.4 % of token throughput by breaking the probe's inlining.
/// Opening happens in the outer `Lexicon` methods.
#[derive(Default)]
pub(super) struct Entities {
    /// The table once opened. Failures are not memoised, so a damaged table
    /// is reported again rather than decaying into silent identity answers.
    table: OnceLock<Table>,
    /// `None` when there is nothing left to open: no table at all, or an
    /// embedded one whose bytes were already in the binary.
    pending: Option<Box<Pending>>,
    /// Whether the metadata declares any entity rows; a plain copy so
    /// [`Tables::is_empty`] needs no atomic load.
    declared: bool,
}

impl Entities {
    /// A table whose bytes are already in hand.
    pub(super) fn ready(table: Table) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(table);
        Self {
            table: cell,
            pending: None,
            declared: true,
        }
    }

    /// A table named by the manifest and read on demand.
    pub(super) fn deferred(
        path: PathBuf,
        codec: Codec,
        expected: FileIntegrity,
        declared: u64,
    ) -> Self {
        Self {
            table: OnceLock::new(),
            pending: Some(Box::new(Pending {
                plan: EntityPlan {
                    path,
                    codec,
                    expected,
                    declared,
                },
                gate: Mutex::new(()),
            })),
            declared: true,
        }
    }

    /// Whether the bundle holds entity rows at all, from the manifest.
    #[inline]
    pub(super) const fn present(&self) -> bool {
        self.declared
    }

    /// Whether a query can proceed without doing the deferred work.
    pub(super) fn is_ready(&self) -> bool {
        self.pending.is_none() || self.table.get().is_some()
    }

    /// The table for a probe: one load, no error channel, never opens.
    #[inline]
    pub(super) fn for_probe(&self) -> Option<&Table> {
        self.table.get()
    }

    /// Opens the table if it is not open yet, discarding any failure.
    ///
    /// Returns nothing on purpose: a `Result` slot in the caller's frame pushed
    /// the probe over the inliner's budget (−8 % on `ru_taiga`).
    #[cold]
    #[inline(never)]
    pub(super) fn prime(&self) {
        drop(self.get());
    }

    /// The table, authenticating it on first use.
    ///
    /// # Errors
    ///
    /// Returns the integrity, codec or count error the table carries.
    pub(super) fn get(&self) -> Result<Option<&Table>> {
        if let Some(table) = self.table.get() {
            return Ok(Some(table));
        }
        let Some(pending) = self.pending.as_deref() else {
            return Ok(None);
        };
        self.open(pending).map(Some)
    }

    /// Runs once per lexicon, on the first entity or span question.
    #[cold]
    #[inline(never)]
    fn open(&self, pending: &Pending) -> Result<&Table> {
        // The guard carries no state, so poisoning is harmless.
        let _guard = pending.gate.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(table) = self.table.get() {
            return Ok(table);
        }
        let table = pending.plan.open()?;
        Ok(self.table.get_or_init(|| table))
    }
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

/// Builds the compiled phrase table from `key -> pattern` rows, for both the
/// compiler and the candidate override.
///
/// Sorting by key sorts the encoded keys too: the separator is below every key byte.
pub(super) fn build_phrase_table(rows: &mut [(String, String)], path: &Path) -> Result<Vec<u8>> {
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    if let Some((key, _)) = rows
        .windows(2)
        .find_map(|pair| match pair {
            [left, right] if left.0 == right.0 => Some(left),
            _ => None,
        })
        .or_else(|| {
            rows.iter().find(|(key, pattern)| {
                key.is_empty()
                    || key.as_bytes().contains(&SEP)
                    || pattern.is_empty()
                    || pattern.as_bytes().contains(&SEP)
            })
        })
    {
        return Err(Error::Invalid {
            path: path.to_owned(),
            message: format!("phrase row {key:?} is duplicated, empty, or contains NUL"),
        });
    }
    let mut builder = fst::raw::Builder::memory();
    for (key, pattern) in rows.iter() {
        builder
            .insert(encode_key(key.as_bytes(), pattern.as_bytes()), 0)
            .map_err(|source| super::runtime::fst_error(path, source))?;
    }
    builder
        .into_inner()
        .map_err(|source| super::runtime::fst_error(path, source))
}

/// Encodes one (form, lemma) pair as an inline-codec key; `cut` is LEB128.
pub(super) fn encode_key(form: &[u8], lemma: &[u8]) -> Vec<u8> {
    let mut key = Vec::new();
    encode_key_into(form, lemma, &mut key);
    key
}

/// Encodes an inline key into storage a streaming compiler can reuse.
pub fn encode_key_into(form: &[u8], lemma: &[u8], key: &mut Vec<u8>) {
    let shared = form
        .iter()
        .zip(lemma)
        .take_while(|(left, right)| left == right)
        .count();
    let mut cut = form.len().saturating_sub(shared);
    key.clear();
    key.reserve(
        form.len()
            .saturating_add(2)
            .saturating_add(lemma.len().saturating_sub(shared)),
    );
    key.extend_from_slice(form);
    key.push(SEP);
    // Both casts are exact: the loop masks to seven bits and exits below 0x80.
    #[allow(clippy::cast_possible_truncation)]
    while cut >= 0x80 {
        key.push((cut & 0x7F) as u8 | 0x80);
        cut >>= 7;
    }
    #[allow(clippy::cast_possible_truncation)]
    key.push(cut as u8);
    key.extend_from_slice(lemma.get(shared..).unwrap_or_default());
}

/// Boundaries of the entries in a string-table blob.
///
/// A leading zero, one offset just past each newline, and the blob's length
/// when the last line has no terminator — so entry `i` spans
/// `offsets[i]..offsets[i + 1]` including its newline, and there is one more
/// offset than there are entries. `None` when the blob outgrows `u32`.
pub(super) fn line_offsets(blob: &[u8]) -> Option<Vec<u32>> {
    u32::try_from(blob.len()).ok()?;
    let mut offsets = vec![0_u32];
    for (position, &byte) in blob.iter().enumerate() {
        if byte == b'\n' {
            offsets.push(u32::try_from(position.saturating_add(1)).ok()?);
        }
    }
    if usize::try_from(*offsets.last()?).ok()? != blob.len() {
        offsets.push(u32::try_from(blob.len()).ok()?);
    }
    Some(offsets)
}

/// Little-endian `u32` entry boundaries; a scanned index is written into the
/// mapped `strings.idx` layout so every hit reads one representation.
struct Offsets(Bytes);

/// Serialises scanned boundaries into the layout `strings.idx` already has.
fn offset_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(OFFSET_WIDTH));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

impl Offsets {
    /// The boundaries at `index` and `index + 1` — the span of one entry —
    /// under one bounds check; the mapping need not be aligned.
    #[inline]
    fn pair(&self, index: usize) -> Option<(u32, u32)> {
        let start = index.checked_mul(OFFSET_WIDTH)?;
        if start.checked_add(2 * OFFSET_WIDTH)? > self.0.len() {
            return None;
        }
        // SAFETY: the check above leaves eight bytes from `start` inside the
        // slice, which is exactly what the two reads take.
        let bytes = unsafe {
            self.0
                .as_ptr()
                .add(start)
                .cast::<[u8; 2 * OFFSET_WIDTH]>()
                .read_unaligned()
        };
        let (first, second) = bytes.split_at(OFFSET_WIDTH);
        Some((
            u32::from_le_bytes(first.try_into().ok()?),
            u32::from_le_bytes(second.try_into().ok()?),
        ))
    }

    /// How many boundaries the index holds.
    fn words(&self) -> usize {
        self.0.len() / OFFSET_WIDTH
    }
}

/// Whether a mapped offset index truly delimits the lines of this blob; this
/// is what makes the unchecked slicing in [`Strings::entry`] sound.
pub(super) fn index_matches(index: &[u8], blob: &[u8]) -> bool {
    let words = index.len() / OFFSET_WIDTH;
    let Some(last_word) = words.checked_sub(1) else {
        return false;
    };
    if !index.len().is_multiple_of(OFFSET_WIDTH) {
        return false;
    }
    let word = |at: usize| {
        let start = at.checked_mul(OFFSET_WIDTH)?;
        index
            .get(start..start.checked_add(OFFSET_WIDTH)?)
            .and_then(|chunk| chunk.try_into().ok())
            .map(u32::from_le_bytes)
    };
    if word(0) != Some(0)
        || word(last_word).and_then(|last| usize::try_from(last).ok()) != Some(blob.len())
    {
        return false;
    }
    if u32::try_from(blob.len()).is_err() {
        return false;
    }
    let mut at = 1_usize;
    let mut last_boundary = 0_usize;
    for position in memchr::memchr_iter(b'\n', blob) {
        last_boundary = position.saturating_add(1);
        if word(at).and_then(|value| usize::try_from(value).ok()) != Some(last_boundary) {
            return false;
        }
        at = at.saturating_add(1);
    }
    if last_boundary != blob.len() {
        if word(at).and_then(|value| usize::try_from(value).ok()) != Some(blob.len()) {
            return false;
        }
        at = at.saturating_add(1);
    }
    at == words
}

/// The string table: a memory-mapped blob and one `u32` boundary per entry.
///
/// # Invariant
///
/// The blob is valid UTF-8 and every offset sits on a character boundary
/// within it, checked once at load, so [`Strings::entry`] slices unchecked.
pub struct Strings {
    /// The bytes of `strings.txt`.
    blob: Bytes,
    /// Entry boundaries; entry `i` is `blob[offsets[i]..offsets[i + 1]]`
    /// minus its trailing newline.
    offsets: Offsets,
}

impl Strings {
    /// Maps the string table of a bundle directory.
    pub(super) fn open(directory: &Path, meta: &Meta) -> Result<Self> {
        let blob_expected = expected_file(directory, meta, "strings.txt")?;
        let index_expected = expected_file(directory, meta, "strings.idx")?;
        Self::open_verified(directory, blob_expected, index_expected)
    }

    /// Maps a string table whose integrity records come from another schema.
    pub(crate) fn open_verified(
        directory: &Path,
        blob_expected: &FileIntegrity,
        index_expected: &FileIntegrity,
    ) -> Result<Self> {
        let blob_path = directory.join("strings.txt");
        let blob = map_verified_file(&blob_path, blob_expected)?;
        // One pass over the blob at load buys unchecked slicing per lookup.
        if std::str::from_utf8(&blob).is_err() {
            return Err(Error::StringsEncoding { path: blob_path });
        }
        let index_path = directory.join("strings.idx");
        let index = map_verified_file(&index_path, index_expected)?;
        let offsets = if index_matches(&index, &blob) {
            Offsets(index)
        } else {
            return Err(Error::Invalid {
                path: index_path,
                message: "offsets do not exactly delimit strings.txt lines".to_owned(),
            });
        };
        Ok(Self { blob, offsets })
    }

    /// Wraps a string table compiled into the binary.
    pub(super) fn from_static(text: &'static str) -> Result<Self> {
        let blob = Bytes::Static(text.as_bytes());
        let values = line_offsets(&blob).ok_or_else(|| Error::StringsTooLarge {
            path: PathBuf::from("<embedded>/strings.txt"),
        })?;
        let offsets = Offsets(Bytes::Owned(offset_bytes(&values)));
        Ok(Self { blob, offsets })
    }

    /// The entry at `id`, if the identifier and the table are well formed.
    #[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
    #[inline(always)]
    pub(crate) fn entry(&self, id: u64) -> Option<&str> {
        let index = usize::try_from(id).ok()?;
        let (start, end) = self.offsets.pair(index)?;
        // SAFETY: loading proved the index ascending, starting at zero and
        // ending at the blob's length — `index_matches` for a mapped one,
        // `line_offsets` by construction for a scanned one — so any pair it
        // yields is a range inside the blob.
        let bytes = unsafe { self.blob.get_unchecked(start as usize..end as usize) };
        let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
        // SAFETY: loading validated the whole blob as UTF-8 and every offset
        // as a character boundary — for the scanned index by construction,
        // for a mapped one in `index_matches` — and the stripped newline is
        // ASCII, so these bytes are a valid UTF-8 substring.
        Some(unsafe { std::str::from_utf8_unchecked(bytes) })
    }

    /// Number of strings addressable by identifiers.
    pub(crate) fn len(&self) -> usize {
        self.offsets.words().saturating_sub(1)
    }
}

/// The tables a lexicon is made of; a table with no rows is `None`.
pub(super) struct Tables {
    pub(super) exact: Option<Table>,
    pub(super) lower: Option<Table>,
    pub(super) external: Option<Table>,
    /// Opened on the first entity or span query, never at load.
    pub(super) entities: Entities,
    /// Component-wise suffix rewrites for multiword spans no entity key
    /// covered; always the inline acceptor.
    pub(super) phrases: Option<Table>,
    /// Always a double array.
    pub(super) rules: Option<DoubleArray<Bytes>>,
    /// Packed rules, indexed by the values stored in `rules`.
    pub(super) rule_values: Option<Bytes>,
}

/// Which tables are loaded and how each is encoded — not their contents.
impl std::fmt::Debug for Tables {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = |table: &Table| match table {
            Table::Da(_) => "double array",
            Table::Map(_) => "automaton",
            Table::Inline(_) => "inline automaton",
            Table::Hybrid(_) => "hybrid",
        };
        let encoding = |table: &Option<Table>| table.as_ref().map_or("absent", name);
        // Printing must not open the entity table.
        let entities = self.entities.for_probe().map_or_else(
            || {
                if self.entities.present() {
                    "deferred"
                } else {
                    "absent"
                }
            },
            name,
        );
        formatter
            .debug_struct("Tables")
            .field("exact", &encoding(&self.exact))
            .field("lower", &encoding(&self.lower))
            .field("external", &encoding(&self.external))
            .field("entities", &entities)
            .field("phrases", &encoding(&self.phrases))
            .field("rules", &self.rules.is_some())
            .field("rule_values", &self.rule_values.is_some())
            .finish()
    }
}

impl Tables {
    /// Whether no table holds anything; phrases never answer a token, so they
    /// do not count here or in [`Tables::common_is_empty`].
    pub(super) const fn is_empty(&self) -> bool {
        self.exact.is_none()
            && self.lower.is_none()
            && self.external.is_none()
            && !self.entities.present()
            && self.rules.is_none()
    }

    /// Whether no ordinary-word table or suffix rule holds anything.
    pub(super) const fn common_is_empty(&self) -> bool {
        self.exact.is_none()
            && self.lower.is_none()
            && self.external.is_none()
            && self.rules.is_none()
    }

    /// Resolves a rule index into the packed rule it stands for.
    pub(super) fn rule(&self, index: u32) -> Option<Rule> {
        let start = (index as usize).checked_mul(VALUE_WIDTH)?;
        let bytes = self
            .rule_values
            .as_deref()?
            .get(start..start.checked_add(VALUE_WIDTH)?)?;
        let packed = u64::from_le_bytes(bytes.try_into().ok()?);
        Some(Rule::unpack(packed))
    }
}
