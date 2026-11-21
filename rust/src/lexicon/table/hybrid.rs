//! The `hybrid` codec: a double array for short keys, a pilot hash for long.
//!
//! A double array costs one dependent load per key byte, a hash a fixed price,
//! so short keys walk the array and long ones hash. The long side stores the
//! lemma beside its form, saving the string-table lookup.
//!
//! The hash side is a minimal-space perfect hash in the `PtrHash` family:
//! every key hashes to a bucket, each bucket carries one 16-bit pilot, and the
//! slot is the hash mixed with that pilot. No key set is stored; a key outside
//! the set lands on some slot and is refused by the byte comparison there.
//!
//! Layout, all little-endian, every section 16-byte aligned:
//!
//! ```text
//! magic "UDLXHYB1" | split u32 | 0 u32 | n_short | n_long | seed |
//! n_buckets | n_slots | da_len | blob_len | zero to 128 bytes
//! double-array units (da_len)
//! pilots: u16 × n_buckets
//! offsets: u32 × (n_slots + 1)
//! blob: [flen u16][llen u16][form][lemma] per occupied slot, in slot order
//! ```

use std::ops::Range;

use super::Bytes;

/// Keys up to this many bytes walk the double array; longer ones are hashed.
pub const SPLIT_BYTES: usize = 8;
const MAGIC: &[u8; 8] = b"UDLXHYB1";
const HEADER: usize = 128;
const ALIGN: usize = 16;
const ENTRY_HEAD: usize = 4;
const K1: u64 = 0x9E37_79B9_7F4A_7C15;
const K2: u64 = 0xD6E8_FEB8_6659_FD93;

/// The low and high halves of a 64×64 product, folded: both are kept, so the
/// truncating casts are the operation itself.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn fold(a: u64, b: u64) -> u64 {
    let product = (a as u128).wrapping_mul(b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

/// One folded multiply per eight bytes; part of the on-disk format.
#[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
#[inline(always)]
fn key_hash(key: &[u8], seed: u64) -> u64 {
    let mut h = fold(seed ^ (key.len() as u64), K1);
    let (chunks, rest) = key.as_chunks::<8>();
    for chunk in chunks {
        h = fold(h ^ u64::from_le_bytes(*chunk), K2);
    }
    if !rest.is_empty() {
        h = fold(h ^ tail_word(rest), K1);
    }
    h
}

/// The last 1–7 key bytes as a zero-padded little-endian word, read with
/// overlapping loads: a variable-length copy compiles to a `memcpy` call.
#[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
#[inline(always)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    reason = "the tail is shorter than eight bytes, so every shift stays below 64"
)]
fn tail_word(rest: &[u8]) -> u64 {
    let n = rest.len() as u32;
    if let (Some(lo), Some(hi)) = (rest.first_chunk::<4>(), rest.last_chunk::<4>()) {
        // Overlapping bytes are equal in both reads, so OR keeps them intact.
        return u64::from(u32::from_le_bytes(*lo))
            | (u64::from(u32::from_le_bytes(*hi)) << (8 * (n - 4)));
    }
    match (rest.first(), rest.get(rest.len() / 2), rest.last()) {
        (Some(&a), Some(&b), Some(&c)) => {
            u64::from(a) | (u64::from(b) << (8 * (n / 2))) | (u64::from(c) << (8 * (n - 1)))
        }
        _ => 0,
    }
}

/// Maps a hash onto `0..n` without a division: the top 64 bits of `h × n`
/// are below `n` by construction, so the final cast cannot truncate.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::arithmetic_side_effects)]
const fn reduce(h: u64, n: usize) -> usize {
    (((h as u128) * (n as u128)) >> 64) as usize
}

#[inline]
const fn slot_of(h: u64, pilot: u16, n_slots: usize) -> usize {
    reduce(fold(h ^ (pilot as u64 + 1).wrapping_mul(K2), K1), n_slots)
}

/// Equality of two same-length keys without a `memcmp` call.
#[expect(clippy::inline_always, reason = "part of the inlined hybrid read")]
#[inline(always)]
fn same(a: &[u8], b: &[u8]) -> bool {
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        if u64::from_le_bytes(*x) != u64::from_le_bytes(*y) {
            return false;
        }
    }
    ra.iter().zip(rb).all(|(x, y)| x == y)
}

const fn aligned(at: usize) -> usize {
    at.div_ceil(ALIGN).saturating_mul(ALIGN)
}

/// Where each section of a parsed file lies; every range was checked to fit.
struct Layout {
    n_short: usize,
    n_long: usize,
    seed: u64,
    n_buckets: usize,
    n_slots: usize,
    da: Range<usize>,
    pilots_at: usize,
    offsets_at: usize,
    blob: Range<usize>,
}

fn u64_at(data: &[u8], at: usize) -> Result<u64, String> {
    data.get(at..at.checked_add(8).ok_or("overflow")?)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| "truncated header".to_owned())
}

fn count_at(data: &[u8], at: usize) -> Result<usize, String> {
    usize::try_from(u64_at(data, at)?).map_err(|_| "count exceeds this platform".to_owned())
}

fn parse_layout(data: &[u8]) -> Result<Layout, String> {
    if data.get(..8) != Some(MAGIC.as_slice()) {
        return Err("not a hybrid table".to_owned());
    }
    let split = data
        .get(8..12)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or("truncated header")?;
    if usize::try_from(split).ok() != Some(SPLIT_BYTES) {
        return Err(format!("split {split} is not the format's {SPLIT_BYTES}"));
    }
    let n_short = count_at(data, 16)?;
    let n_long = count_at(data, 24)?;
    let seed = u64_at(data, 32)?;
    let n_buckets = count_at(data, 40)?;
    let n_slots = count_at(data, 48)?;
    let da_len = count_at(data, 56)?;
    let blob_len = count_at(data, 64)?;
    let overflow = || "section offsets overflow".to_owned();
    let da_end = HEADER.checked_add(da_len).ok_or_else(overflow)?;
    let pilots_at = aligned(da_end);
    let offsets_at = aligned(
        pilots_at
            .checked_add(n_buckets.checked_mul(2).ok_or_else(overflow)?)
            .ok_or_else(overflow)?,
    );
    let blob_at = aligned(
        offsets_at
            .checked_add(
                n_slots
                    .checked_add(1)
                    .and_then(|words| words.checked_mul(4))
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?,
    );
    let blob_end = blob_at.checked_add(blob_len).ok_or_else(overflow)?;
    if blob_end != data.len() {
        return Err("section sizes do not add up to the file".to_owned());
    }
    if (n_long == 0) != (n_slots == 0) || (n_long == 0) != (n_buckets == 0) {
        return Err("an empty hash side has no buckets and no slots".to_owned());
    }
    Ok(Layout {
        n_short,
        n_long,
        seed,
        n_buckets,
        n_slots,
        da: HEADER..da_end,
        pilots_at,
        offsets_at,
        blob: blob_at..blob_end,
    })
}

/// Checks every entry once: ascending offsets, lengths that add up, keys on
/// the long side only, text on both sides. Returns the occupied slot count.
fn validate_entries(data: &[u8], layout: &Layout) -> Result<usize, String> {
    let words = layout.n_slots.checked_add(1).ok_or("overflow")?;
    let offsets_end = layout
        .offsets_at
        .checked_add(words.checked_mul(4).ok_or("overflow")?)
        .ok_or("overflow")?;
    let offsets = data
        .get(layout.offsets_at..offsets_end)
        .ok_or("offsets")?
        .as_chunks::<4>()
        .0;
    let blob = data.get(layout.blob.clone()).ok_or("blob")?;
    if offsets.first().map(|word| u32::from_le_bytes(*word)) != Some(0)
        || offsets
            .last()
            .and_then(|word| usize::try_from(u32::from_le_bytes(*word)).ok())
            != Some(blob.len())
    {
        return Err("offsets must open at zero and close the blob".to_owned());
    }
    let mut occupied = 0_usize;
    for (slot, pair) in offsets.windows(2).enumerate() {
        let (Some(start), Some(end)) = (pair.first(), pair.get(1)) else {
            break;
        };
        let start = usize::try_from(u32::from_le_bytes(*start)).map_err(|_| "offset")?;
        let end = usize::try_from(u32::from_le_bytes(*end)).map_err(|_| "offset")?;
        if end < start {
            return Err("offsets are not ascending".to_owned());
        }
        if end == start {
            continue;
        }
        let entry = blob.get(start..end).ok_or("entry outside the blob")?;
        let (head, body) = entry.split_at_checked(ENTRY_HEAD).ok_or("entry")?;
        let flen = usize::from(u16::from_le_bytes([
            *head.first().ok_or("entry")?,
            *head.get(1).ok_or("entry")?,
        ]));
        let llen = usize::from(u16::from_le_bytes([
            *head.get(2).ok_or("entry")?,
            *head.get(3).ok_or("entry")?,
        ]));
        let (form, lemma) = body.split_at_checked(flen).ok_or("entry")?;
        if lemma.len() != llen || flen <= SPLIT_BYTES {
            return Err(format!("slot {slot} entry is malformed"));
        }
        if std::str::from_utf8(form).is_err() || std::str::from_utf8(lemma).is_err() {
            return Err(format!("slot {slot} entry is not UTF-8"));
        }
        occupied = occupied.saturating_add(1);
    }
    Ok(occupied)
}

/// What a hybrid probe answered.
pub(in crate::lexicon) enum Hit<'a> {
    /// A short key's string identifier, from the double array.
    Id(u32),
    /// A long key's lemma, spelled in its own entry.
    Text(&'a str),
}

/// One `<name>.hybrid` file, validated once so lookups read unchecked.
pub struct HybridTable {
    bytes: Bytes,
    layout: Layout,
    short_entries: usize,
}

impl std::fmt::Debug for HybridTable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HybridTable")
            .field("short", &self.short_entries)
            .field("long", &self.layout.n_long)
            .finish_non_exhaustive()
    }
}

impl HybridTable {
    /// Parses and validates the whole file; every later read is unchecked.
    pub(in crate::lexicon) fn open(bytes: Bytes) -> Result<Self, String> {
        let layout = parse_layout(&bytes)?;
        let short_entries = if layout.da.is_empty() {
            0
        } else {
            let units = bytes.get(layout.da.clone()).ok_or("double array")?;
            yada::DoubleArray::new(units).map_err(|error| format!("double array: {error:?}"))?;
            super::da_leaf_values(units).count()
        };
        if short_entries != layout.n_short {
            return Err(format!(
                "double array holds {short_entries} keys, header declares {}",
                layout.n_short
            ));
        }
        let occupied = validate_entries(&bytes, &layout)?;
        if occupied != layout.n_long {
            return Err(format!(
                "{occupied} occupied slots, header declares {}",
                layout.n_long
            ));
        }
        Ok(Self {
            bytes,
            layout,
            short_entries,
        })
    }

    /// The double-array units, empty when no key was short.
    pub(in crate::lexicon) fn units(&self) -> &[u8] {
        self.bytes.get(self.layout.da.clone()).unwrap_or(&[])
    }

    pub(in crate::lexicon) const fn len(&self) -> usize {
        self.short_entries.saturating_add(self.layout.n_long)
    }

    /// Looks one form up; the key's length decides which side answers.
    // Every index below is `reduce` output, below a count whose section
    // `parse_layout` proved to lie inside the file, so the offsets cannot
    // overflow and the unchecked reads stay inside the mapping.
    // Always inline: returned from a call, `Hit` travels through memory.
    #[inline(always)]
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        clippy::inline_always
    )]
    pub(in crate::lexicon) fn get(&self, key: &[u8]) -> Option<Hit<'_>> {
        let layout = &self.layout;
        if key.len() <= SPLIT_BYTES {
            if layout.da.is_empty() {
                return None;
            }
            return super::da_search_units(self.units(), None, key).map(Hit::Id);
        }
        if layout.n_long == 0 {
            return None;
        }
        let bytes: &[u8] = &self.bytes;
        let h = key_hash(key, layout.seed);
        let bucket = reduce(h, layout.n_buckets);
        // SAFETY: `bucket < n_buckets` and the slot it yields is below
        // `n_slots`; both sections were measured against the file length.
        let (at, end) = unsafe {
            let pilot = u16::from_le_bytes(
                bytes
                    .as_ptr()
                    .add(layout.pilots_at + bucket * 2)
                    .cast::<[u8; 2]>()
                    .read_unaligned(),
            );
            let slot = slot_of(h, pilot, layout.n_slots);
            let pair = bytes
                .as_ptr()
                .add(layout.offsets_at + slot * 4)
                .cast::<[u8; 8]>()
                .read_unaligned();
            (
                u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]) as usize,
                u32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]) as usize,
            )
        };
        if at == end {
            return None;
        }
        // SAFETY: `validate_entries` proved this range is one well-formed
        // entry inside the blob: four bytes of lengths, the form, the lemma.
        let entry = unsafe { bytes.get_unchecked(layout.blob.start + at..layout.blob.start + end) };
        let flen = usize::from(u16::from_le_bytes([entry[0], entry[1]]));
        if flen != key.len() {
            return None;
        }
        let form = unsafe { entry.get_unchecked(ENTRY_HEAD..ENTRY_HEAD + flen) };
        if !same(form, key) {
            return None;
        }
        let lemma = unsafe { entry.get_unchecked(ENTRY_HEAD + flen..) };
        Some(Hit::Text(unsafe { std::str::from_utf8_unchecked(lemma) }))
    }
}

/// Seed, pilots, slot count and the key index each slot holds.
#[cfg(feature = "train")]
type Placement = (u64, Vec<u16>, usize, Vec<Option<usize>>);

/// Builds the file from sorted, unique `(form, id)` rows and their lemmas.
///
/// `lemma` resolves a string id; a row whose id has no string is an error.
#[cfg(feature = "train")]
pub fn build_hybrid(
    rows: &[(String, u64)],
    lemma: impl Fn(u64) -> Option<String>,
) -> Result<Vec<u8>, String> {
    let (short, long): (Vec<_>, Vec<_>) =
        rows.iter().partition(|(form, _)| form.len() <= SPLIT_BYTES);
    let da = if short.is_empty() {
        Vec::new()
    } else {
        let keyset = short
            .iter()
            .map(|(form, id)| {
                u32::try_from(*id)
                    .map(|id| (form.as_bytes(), id))
                    .map_err(|_| format!("value {id} exceeds the double-array u32 limit"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        yada::builder::DoubleArrayBuilder::build(&keyset)
            .map_err(|error| format!("double array: {error:?}"))?
    };
    let (seed, pilots, n_slots, order) = place(&long)?;
    let mut offsets = Vec::with_capacity(n_slots.saturating_add(1));
    let mut blob = Vec::new();
    let blob_offset = |blob: &Vec<u8>| u32::try_from(blob.len()).map_err(|_| "blob exceeds 4 GiB");
    for index in order {
        offsets.push(blob_offset(&blob)?);
        let Some((form, id)) = index.and_then(|index| long.get(index)) else {
            continue;
        };
        let text = lemma(*id).ok_or_else(|| format!("string identifier {id} is out of range"))?;
        let flen = u16::try_from(form.len()).map_err(|_| format!("form {form:?} is too long"))?;
        let llen =
            u16::try_from(text.len()).map_err(|_| format!("lemma of {form:?} is too long"))?;
        blob.extend_from_slice(&flen.to_le_bytes());
        blob.extend_from_slice(&llen.to_le_bytes());
        blob.extend_from_slice(form.as_bytes());
        blob.extend_from_slice(text.as_bytes());
    }
    offsets.push(blob_offset(&blob)?);

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&u32::try_from(SPLIT_BYTES).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    for value in [
        short.len() as u64,
        long.len() as u64,
        seed,
        pilots.len() as u64,
        n_slots as u64,
        da.len() as u64,
        blob.len() as u64,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.resize(HEADER, 0);
    out.extend_from_slice(&da);
    out.resize(aligned(out.len()), 0);
    for pilot in &pilots {
        out.extend_from_slice(&pilot.to_le_bytes());
    }
    out.resize(aligned(out.len()), 0);
    for offset in &offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.resize(aligned(out.len()), 0);
    out.extend_from_slice(&blob);
    Ok(out)
}

/// Finds a seed and pilots placing every long key on its own slot.
///
/// Ten slots per nine keys and three keys per bucket. Buckets are filled
/// largest first with the smallest pilot that keeps all of a bucket's keys on
/// free slots; a bucket that exhausts its pilots restarts under the next seed.
// Indices come from `reduce` into vectors sized by the same counts.
#[cfg(feature = "train")]
#[allow(clippy::indexing_slicing)]
fn place(long: &[&(String, u64)]) -> Result<Placement, String> {
    const SEED_ATTEMPTS: u64 = 64;
    if long.is_empty() {
        return Ok((0, Vec::new(), 0, Vec::new()));
    }
    let n = long.len();
    let n_slots = n.saturating_mul(10).div_ceil(9);
    let n_buckets = n.div_ceil(3).max(1);
    'seeds: for seed in 0..SEED_ATTEMPTS {
        let hashes: Vec<u64> = long
            .iter()
            .map(|(form, _)| key_hash(form.as_bytes(), seed))
            .collect();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_buckets];
        for (index, &h) in hashes.iter().enumerate() {
            buckets[reduce(h, n_buckets)].push(index);
        }
        let mut order: Vec<usize> = (0..n_buckets).collect();
        order.sort_by_key(|&b| std::cmp::Reverse(buckets[b].len()));
        let mut taken: Vec<Option<usize>> = vec![None; n_slots];
        let mut pilots = vec![0_u16; n_buckets];
        let mut scratch = Vec::new();
        for &b in &order {
            let keys = &buckets[b];
            if keys.is_empty() {
                continue;
            }
            let fitting = (0..=u16::MAX).find(|&pilot| {
                scratch.clear();
                keys.iter().all(|&key| {
                    let slot = slot_of(hashes[key], pilot, n_slots);
                    let free = taken[slot].is_none() && !scratch.contains(&slot);
                    scratch.push(slot);
                    free
                })
            });
            let Some(pilot) = fitting else {
                continue 'seeds;
            };
            for (&key, &slot) in keys.iter().zip(&scratch) {
                taken[slot] = Some(key);
            }
            pilots[b] = pilot;
        }
        return Ok((seed, pilots, n_slots, taken));
    }
    Err(format!(
        "no pilot placement for {n} keys in {SEED_ATTEMPTS} seeds"
    ))
}

#[cfg(all(test, feature = "train"))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;

    fn rows(keys: &[&str]) -> Vec<(String, u64)> {
        let mut rows: Vec<(String, u64)> = keys
            .iter()
            .enumerate()
            .map(|(id, key)| ((*key).to_owned(), id as u64))
            .collect();
        rows.sort();
        rows.iter_mut()
            .enumerate()
            .for_each(|(id, row)| row.1 = id as u64);
        rows
    }

    fn table(keys: &[&str]) -> (HybridTable, Vec<(String, u64)>) {
        let rows = rows(keys);
        let bytes = build_hybrid(&rows, |id| Some(format!("lemma{id}"))).unwrap();
        (HybridTable::open(Bytes::Owned(bytes)).unwrap(), rows)
    }

    #[test]
    fn short_keys_answer_ids_and_long_keys_answer_text() {
        let (table, rows) = table(&["the", "cat", "котами", "кошками", "a", "verylongword"]);
        for (form, id) in &rows {
            match table.get(form.as_bytes()).expect(form) {
                Hit::Id(found) => {
                    assert!(form.len() <= SPLIT_BYTES, "{form}");
                    assert_eq!(u64::from(found), *id);
                }
                Hit::Text(text) => {
                    assert!(form.len() > SPLIT_BYTES, "{form}");
                    assert_eq!(text, format!("lemma{id}"));
                }
            }
        }
        for miss in ["", "th", "thee", "котам", "котамиx", "verylongwor", "z"] {
            assert!(table.get(miss.as_bytes()).is_none(), "{miss}");
        }
        assert_eq!(table.len(), rows.len());
    }

    #[test]
    fn one_sided_tables_open_and_miss_the_other_side() {
        let (short_only, _) = table(&["a", "bb", "ccc"]);
        assert!(short_only.get(b"a").is_some());
        assert!(short_only.get(b"a-long-key").is_none());
        let (long_only, _) = table(&["first-long", "second-long"]);
        assert!(long_only.get(b"first-long").is_some());
        assert!(long_only.get(b"a").is_none());
        let empty = build_hybrid(&[], |_| None).unwrap();
        let empty = HybridTable::open(Bytes::Owned(empty)).unwrap();
        assert!(empty.get(b"a").is_none() && empty.get(b"something-long").is_none());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn placement_is_deterministic_and_scales() {
        let keys: Vec<String> = (0..50_000).map(|i| format!("ключ-{i}-слово")).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let rows = rows(&refs);
        let first = build_hybrid(&rows, |id| Some(id.to_string())).unwrap();
        let second = build_hybrid(&rows, |id| Some(id.to_string())).unwrap();
        assert_eq!(first, second);
        let table = HybridTable::open(Bytes::Owned(first)).unwrap();
        for (form, id) in rows.iter().step_by(997) {
            let Some(Hit::Text(text)) = table.get(form.as_bytes()) else {
                panic!("{form}");
            };
            assert_eq!(text, id.to_string());
        }
        assert!(table.get("ключ-50000-слово".as_bytes()).is_none());
    }

    #[test]
    fn tail_word_equals_the_zero_padded_copy() {
        // The hash is part of the on-disk format: the new read must give the old value.
        let bytes: Vec<u8> = (0..=255_u8).rev().chain(0..=255).collect();
        for len in 1..8 {
            for window in bytes.windows(len) {
                let mut padded = [0_u8; 8];
                padded[..len].copy_from_slice(window);
                assert_eq!(tail_word(window), u64::from_le_bytes(padded), "{window:?}");
            }
        }
    }

    #[test]
    fn damaged_files_are_refused() {
        let (_, rows) = table(&["one-long-key", "two-long-key", "ab"]);
        let good = build_hybrid(&rows, |id| Some(format!("l{id}"))).unwrap();
        let mut truncated = good.clone();
        truncated.pop();
        assert!(HybridTable::open(Bytes::Owned(truncated)).is_err());
        let mut wrong_count = good.clone();
        wrong_count[24] ^= 1;
        assert!(HybridTable::open(Bytes::Owned(wrong_count)).is_err());
        let mut bad_utf8 = good;
        let last = bad_utf8.len() - 1;
        bad_utf8[last] = 0xFF;
        assert!(HybridTable::open(Bytes::Owned(bad_utf8)).is_err());
        assert!(HybridTable::open(Bytes::Owned(b"nothing".to_vec())).is_err());
    }
}
