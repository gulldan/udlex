//! The key structures. Each builds one image from sorted `(form, payload)`
//! rows and answers `find` from the mapped image; where a crate cannot read
//! its own structure from a mapping, `HEAP` says so and the receipts carry it.

use std::cell::UnsafeCell;
use std::hash::Hasher;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use ph::BuildSeededHasher;
use ph::phast::{DefaultCompressedArray, Params, SeedOnly, bits_per_seed_to_100_bucket_size};
use ph::seeds::Bits8;
use ptr_hash::bucket_fn::Linear;
use ptr_hash::hash::KeyHasher;
use ptr_hash::{PtrHash, PtrHashParams};
use rayon::prelude::*;
use xorf::{BinaryFuse8, BinaryFuse8Ref, DmaSerializable, Filter, FilterRef};

use crate::dict::{Dict, Index, Lemma};
use crate::hashing::{K1, K2, fold, key_hash, reduce, same};
use crate::img::{Img, as_bytes, write_sections};
use crate::lexicon::da_walk;

/// Sorted unique forms with their payloads and lemmas.
pub struct Rows<'a> {
    pub keys: Vec<&'a [u8]>,
    pub payload: Vec<u32>,
    pub lemma: Vec<&'a [u8]>,
}

impl<'a> Rows<'a> {
    pub fn subset(&self, keep: impl Fn(&[u8]) -> bool) -> Rows<'a> {
        let mut out = Rows {
            keys: Vec::new(),
            payload: Vec::new(),
            lemma: Vec::new(),
        };
        for i in (0..self.keys.len()).filter(|&i| keep(self.keys[i])) {
            out.keys.push(self.keys[i]);
            out.payload.push(self.payload[i]);
            out.lemma.push(self.lemma[i]);
        }
        out
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Writes an image a candidate rebuilds at open and maps it. Under a
/// private name first: another process may be rebuilding the same image,
/// and the rename keeps every reader on a whole file.
fn publish(path: &Path, parts: &[&[u8]]) -> io::Result<Img> {
    let private = path.with_extension(format!("tmp{}", std::process::id()));
    write_sections(&private, parts)?;
    let img = Img::open(&private)?;
    std::fs::rename(&private, path)?;
    Ok(img)
}

// ---------------------------------------------------------------- sorted

/// Binary search over a sorted string array.
pub struct Sorted {
    offs: &'static [u32],
    blob: &'static [u8],
    vals: &'static [u32],
    _img: Img,
}

impl Sorted {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let mut offs = vec![0_u32];
        let mut blob = Vec::new();
        for key in &rows.keys {
            blob.extend_from_slice(key);
            offs.push(u32::try_from(blob.len()).map_err(|_| invalid("keys exceed 4 GiB"))?);
        }
        write_sections(path, &[as_bytes(&offs), &blob, as_bytes(&rows.payload)]).map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        Ok(Self {
            offs: img.words(0),
            blob: img.bytes(1),
            vals: img.words(2),
            _img: img,
        })
    }
}

impl Index for Sorted {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        let (mut lo, mut hi) = (0, self.vals.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            // SAFETY: `mid < len`, and boundaries ascend inside the blob.
            let held = unsafe {
                let a = *self.offs.get_unchecked(mid) as usize;
                let b = *self.offs.get_unchecked(mid + 1) as usize;
                self.blob.get_unchecked(a..b)
            };
            match held.cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(self.vals[mid]),
            }
        }
        None
    }
}

// ------------------------------------------------------------- hashbrown

/// The in-memory ceiling: a general-purpose hash map on the heap.
pub struct Brown(hashbrown::HashMap<Box<[u8]>, u32>);

impl Brown {
    pub fn build(rows: &Rows) -> Self {
        let mut map = hashbrown::HashMap::with_capacity(rows.keys.len());
        for (key, &payload) in rows.keys.iter().zip(&rows.payload) {
            map.insert(Box::<[u8]>::from(*key), payload);
        }
        Self(map)
    }
}

impl Index for Brown {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        self.0.get(key).copied()
    }
}

// ------------------------------------------------------------------ flat

/// Open addressing in the mapping: `tag32 | offset32` slots, linear probing,
/// load 0.75, entries `[flen u16][form][payload u32]`.
pub struct Flat {
    slots: &'static [u64],
    blob: &'static [u8],
    _img: Img,
}

impl Flat {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let cap = (rows.keys.len() * 4).div_ceil(3).max(1);
        let mut slots = vec![0_u64; cap];
        let mut blob = Vec::new();
        for (key, &payload) in rows.keys.iter().zip(&rows.payload) {
            let at = u32::try_from(blob.len() + 1).map_err(|_| invalid("entries exceed 4 GiB"))?;
            blob.extend_from_slice(&(key.len() as u16).to_le_bytes());
            blob.extend_from_slice(key);
            blob.extend_from_slice(&payload.to_le_bytes());
            let h = key_hash(key, 0);
            let mut i = reduce(h, cap);
            while slots[i] != 0 {
                i = if i + 1 == cap { 0 } else { i + 1 };
            }
            slots[i] = (h & 0xFFFF_FFFF) << 32 | u64::from(at);
        }
        write_sections(path, &[as_bytes(&slots), &blob]).map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        Ok(Self {
            slots: img.words(0),
            blob: img.bytes(1),
            _img: img,
        })
    }
}

impl Index for Flat {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        let h = key_hash(key, 0);
        let cap = self.slots.len();
        let mut i = reduce(h, cap);
        loop {
            // SAFETY: `i < cap`; a slot's offset addresses one whole entry.
            unsafe {
                let slot = *self.slots.get_unchecked(i);
                if slot == 0 {
                    return None;
                }
                if slot >> 32 == h & 0xFFFF_FFFF {
                    let at = self.blob.as_ptr().add((slot & 0xFFFF_FFFF) as usize - 1);
                    let flen = usize::from(at.cast::<u16>().read_unaligned());
                    if flen == key.len() && same(std::slice::from_raw_parts(at.add(2), flen), key) {
                        return Some(at.add(2 + flen).cast::<u32>().read_unaligned());
                    }
                }
            }
            i = if i + 1 == cap { 0 } else { i + 1 };
        }
    }
}

// ---------------------------------------------------------- double array

pub fn build_da(rows: &Rows, path: &Path) -> io::Result<()> {
    let units = if rows.keys.is_empty() {
        Vec::new()
    } else {
        let keyset: Vec<(&[u8], u32)> = rows
            .keys
            .iter()
            .copied()
            .zip(rows.payload.iter().copied())
            .collect();
        yada::builder::DoubleArrayBuilder::build(&keyset)
            .map_err(|e| invalid(format!("yada: {e:?}")))?
    };
    write_sections(path, &[&units]).map(drop)
}

/// yada's own reader over the mapped units.
pub struct Yada {
    da: yada::DoubleArray<&'static [u8]>,
    _img: Img,
}

impl Yada {
    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let da =
            yada::DoubleArray::new(img.bytes(0)).map_err(|e| invalid(format!("yada: {e:?}")))?;
        Ok(Self { da, _img: img })
    }
}

impl Index for Yada {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        self.da.exact_match_search(key)
    }
}

/// The same units under the core's walk (`da_search_units`).
pub struct DaCore {
    units: &'static [u8],
    _img: Img,
}

impl DaCore {
    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let units = img.bytes(0);
        if !units.is_empty() {
            yada::DoubleArray::new(units).map_err(|e| invalid(format!("yada: {e:?}")))?;
        }
        Ok(Self { units, _img: img })
    }
}

impl Index for DaCore {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        if self.units.is_empty() {
            return None;
        }
        da_walk(self.units, key)
    }
}

// --------------------------------------------------------------- crawdad

fn text(key: &[u8]) -> &str {
    // SAFETY: keys come from JSON strings and UTF-8-checked TSV rows, and
    // the query streams are built from those on character boundaries.
    unsafe { std::str::from_utf8_unchecked(key) }
}

macro_rules! crawdad_index {
    ($name:ident, $trie:ty) => {
        /// Character-wise double array; the crate deserialises to the heap.
        pub struct $name($trie);

        impl $name {
            pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
                let records = rows
                    .keys
                    .iter()
                    .map(|k| text(k))
                    .zip(rows.payload.iter().copied());
                let trie = <$trie>::from_records(records)
                    .map_err(|e| invalid(format!("crawdad: {e:?}")))?;
                write_sections(path, &[&trie.serialize_to_vec()]).map(drop)
            }

            pub fn open(path: &Path) -> io::Result<Self> {
                let img = Img::open(path)?;
                Ok(Self(<$trie>::deserialize_from_slice(img.bytes(0)).0))
            }
        }

        impl Index for $name {
            #[inline(always)]
            fn find(&self, key: &[u8]) -> Option<u32> {
                self.0.exact_match(text(key).chars())
            }
        }
    };
}

crawdad_index!(Crawdad, crawdad::Trie);
crawdad_index!(CrawdadMp, crawdad::MpTrie);

// ------------------------------------------------------------------- fst

pub struct Fst {
    map: fst::Map<&'static [u8]>,
    _img: Img,
}

impl Fst {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let mut builder = fst::MapBuilder::memory();
        for (key, &payload) in rows.keys.iter().zip(&rows.payload) {
            builder
                .insert(key, u64::from(payload))
                .map_err(|e| invalid(e.to_string()))?;
        }
        let bytes = builder.into_inner().map_err(|e| invalid(e.to_string()))?;
        write_sections(path, &[&bytes]).map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let map = fst::Map::new(img.bytes(0)).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { map, _img: img })
    }
}

impl Index for Fst {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        self.map.get(key).map(|v| v as u32)
    }
}

// ---------------------------------------------------------------- marisa

/// LOUDS trie (MARISA, Rust port) plus a payload array indexed by key id.
pub struct Marisa {
    trie: rsmarisa::Trie,
    agent: UnsafeCell<rsmarisa::Agent>,
    vals: &'static [u32],
    _img: Img,
    _trie_img: Img,
}

impl Marisa {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let mut keyset = rsmarisa::Keyset::new();
        for key in &rows.keys {
            keyset.push_back_bytes(key, 1.0)?;
        }
        let mut trie = rsmarisa::Trie::new();
        trie.build(&mut keyset, 0);
        let mut vals = vec![0_u32; rows.keys.len()];
        for (i, &payload) in rows.payload.iter().enumerate() {
            vals[keyset.get(i).id()] = payload;
        }
        let raw = path.with_extension("marisa");
        trie.save(raw.to_str().ok_or_else(|| invalid("path"))?)?;
        let bytes = std::fs::read(&raw)?;
        std::fs::remove_file(&raw)?;
        write_sections(path, &[as_bytes(&vals)])?;
        write_sections(&path.with_extension("trie"), &[&bytes]).map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let trie_img = Img::open(&path.with_extension("trie"))?;
        let mut trie = rsmarisa::Trie::new();
        trie.map(trie_img.bytes(0))?;
        let mut agent = rsmarisa::Agent::new();
        agent.init_state()?;
        Ok(Self {
            trie,
            agent: UnsafeCell::new(agent),
            vals: img.words(0),
            _img: img,
            _trie_img: trie_img,
        })
    }
}

impl Index for Marisa {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        // SAFETY: the lab queries from one thread; the agent is scratch
        // space of a single lookup and no reference to it escapes.
        let agent = unsafe { &mut *self.agent.get() };
        agent.set_query_bytes(key);
        if !self.trie.lookup(agent) {
            return None;
        }
        self.vals.get(agent.key().id()).copied()
    }
}

// ------------------------------------------------- hash family: entries

/// Slot-ordered entries behind a perfect hash: `[flen u16][form][value]`,
/// the value either a `u32` payload or `[llen u16][lemma]` spelled inline.
pub struct Entries {
    offs: &'static [u32],
    blob: &'static [u8],
}

impl Entries {
    fn sections(
        rows: &Rows,
        slot_of: &[usize],
        n_slots: usize,
        inline: bool,
    ) -> io::Result<(Vec<u32>, Vec<u8>)> {
        let mut holder = vec![u32::MAX; n_slots];
        for (i, &slot) in slot_of.iter().enumerate() {
            if holder[slot] != u32::MAX {
                return Err(invalid(format!("slot {slot} was assigned twice")));
            }
            holder[slot] = i as u32;
        }
        let mut offs = Vec::with_capacity(n_slots + 1);
        let mut blob = Vec::new();
        for &i in &holder {
            offs.push(u32::try_from(blob.len()).map_err(|_| invalid("entries exceed 4 GiB"))?);
            if i == u32::MAX {
                continue;
            }
            let (key, i) = (rows.keys[i as usize], i as usize);
            blob.extend_from_slice(&(key.len() as u16).to_le_bytes());
            blob.extend_from_slice(key);
            if inline {
                blob.extend_from_slice(&(rows.lemma[i].len() as u16).to_le_bytes());
                blob.extend_from_slice(rows.lemma[i]);
            } else {
                blob.extend_from_slice(&rows.payload[i].to_le_bytes());
            }
        }
        offs.push(u32::try_from(blob.len()).map_err(|_| invalid("entries exceed 4 GiB"))?);
        Ok((offs, blob))
    }

    /// The payload, or with `INLINE` the blob offset of `[llen][lemma]`.
    #[inline(always)]
    fn probe<const INLINE: bool>(&self, slot: usize, key: &[u8]) -> Option<u32> {
        // SAFETY: `slot < n_slots` by the slotter's contract, and each
        // occupied slot's range is one whole entry.
        unsafe {
            let at = *self.offs.get_unchecked(slot) as usize;
            let end = *self.offs.get_unchecked(slot + 1) as usize;
            if at == end {
                return None;
            }
            let entry = self.blob.as_ptr().add(at);
            let flen = usize::from(entry.cast::<u16>().read_unaligned());
            if flen != key.len() || !same(std::slice::from_raw_parts(entry.add(2), flen), key) {
                return None;
            }
            let value = at + 2 + flen;
            Some(if INLINE {
                value as u32
            } else {
                self.blob.as_ptr().add(value).cast::<u32>().read_unaligned()
            })
        }
    }
}

/// Maps any key, member or not, to a slot.
pub trait Slotter {
    /// Whether every key lands below the slot count without a check.
    const TOTAL: bool;
    fn slot(&self, key: &[u8]) -> usize;
}

/// A perfect hash in front of slot-ordered entries.
pub struct Hashed<S, const INLINE: bool> {
    slotter: S,
    entries: Entries,
    n_slots: usize,
    _img: Img,
}

impl<S, const INLINE: bool> Hashed<S, INLINE> {
    pub fn blob(&self) -> &'static [u8] {
        self.entries.blob
    }
}

impl<S: Slotter, const INLINE: bool> Index for Hashed<S, INLINE> {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        let slot = self.slotter.slot(key);
        if !S::TOTAL && slot >= self.n_slots {
            return None;
        }
        self.entries.probe::<INLINE>(slot, key)
    }
}

// ------------------------------------------------------ pilot (the core's)

#[inline(always)]
const fn slot_of(h: u64, pilot: u16, n_slots: usize) -> usize {
    reduce(fold(h ^ (pilot as u64 + 1).wrapping_mul(K2), K1), n_slots)
}

/// The core's hash side: 16-bit pilots, three keys per bucket, load 0.9.
pub struct Pilot {
    seed: u64,
    n_buckets: usize,
    n_slots: usize,
    pilots: &'static [u16],
}

impl Slotter for Pilot {
    const TOTAL: bool = true;

    #[inline(always)]
    fn slot(&self, key: &[u8]) -> usize {
        let h = key_hash(key, self.seed);
        // SAFETY: `reduce` stays below the bucket count.
        let pilot = unsafe { *self.pilots.get_unchecked(reduce(h, self.n_buckets)) };
        slot_of(h, pilot, self.n_slots)
    }
}

/// Seed, pilots, slot count and each key's slot: the core's `place`, restated.
pub struct Placement {
    pub seed: u64,
    pub pilots: Vec<u16>,
    pub n_slots: usize,
    pub slot_of: Vec<usize>,
}

pub fn place(keys: &[&[u8]]) -> io::Result<Placement> {
    const SEED_ATTEMPTS: u64 = 64;
    let n = keys.len();
    if n == 0 {
        // One empty slot behind one bucket keeps lookups branch-free.
        return Ok(Placement {
            seed: 0,
            pilots: vec![0],
            n_slots: 1,
            slot_of: Vec::new(),
        });
    }
    let n_slots = (n * 10).div_ceil(9);
    let n_buckets = n.div_ceil(3).max(1);
    'seeds: for seed in 0..SEED_ATTEMPTS {
        let hashes: Vec<u64> = keys.iter().map(|k| key_hash(k, seed)).collect();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_buckets];
        for (index, &h) in hashes.iter().enumerate() {
            buckets[reduce(h, n_buckets)].push(index);
        }
        let mut order: Vec<usize> = (0..n_buckets).collect();
        order.sort_by_key(|&b| std::cmp::Reverse(buckets[b].len()));
        let mut taken = vec![false; n_slots];
        let mut out = vec![0_usize; n];
        let mut pilots = vec![0_u16; n_buckets];
        let mut scratch = Vec::new();
        for &b in &order {
            let members = &buckets[b];
            if members.is_empty() {
                continue;
            }
            let fitting = (0..=u16::MAX).find(|&pilot| {
                scratch.clear();
                members.iter().all(|&key| {
                    let slot = slot_of(hashes[key], pilot, n_slots);
                    let free = !taken[slot] && !scratch.contains(&slot);
                    scratch.push(slot);
                    free
                })
            });
            let Some(pilot) = fitting else {
                continue 'seeds;
            };
            for (&key, &slot) in members.iter().zip(&scratch) {
                taken[slot] = true;
                out[key] = slot;
            }
            pilots[b] = pilot;
        }
        return Ok(Placement {
            seed,
            pilots,
            n_slots,
            slot_of: out,
        });
    }
    Err(invalid(format!("no pilot placement for {n} keys")))
}

impl<const INLINE: bool> Hashed<Pilot, INLINE> {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let placed = place(&rows.keys)?;
        let (offs, blob) = Entries::sections(rows, &placed.slot_of, placed.n_slots, INLINE)?;
        let head = [
            placed.seed,
            placed.pilots.len() as u64,
            placed.n_slots as u64,
        ];
        write_sections(
            path,
            &[
                as_bytes(&head),
                as_bytes(&placed.pilots),
                as_bytes(&offs),
                &blob,
            ],
        )
        .map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let head: &[u64] = img.words(0);
        let slotter = Pilot {
            seed: head[0],
            n_buckets: head[1] as usize,
            n_slots: head[2] as usize,
            pilots: img.words(1),
        };
        let n_slots = slotter.n_slots;
        Ok(Self {
            slotter,
            entries: Entries {
                offs: img.words(2),
                blob: img.bytes(3),
            },
            n_slots,
            _img: img,
        })
    }
}

// --------------------------------------------------------------- ptr_hash

/// The lab's key hash behind `ptr_hash`, so structures differ and hashes don't.
#[derive(Clone)]
pub struct FoldH;

impl KeyHasher<[u8]> for FoldH {
    type H = u64;

    #[inline(always)]
    fn hash(key: &[u8], seed: u64) -> u64 {
        key_hash(key, seed)
    }
}

/// `FastPtrHash`: single part, linear buckets, 8-bit pilots, no remap.
type Ptr = PtrHash<[u8], Linear, Vec<u32>, FoldH, Vec<u8>, true, false>;

pub struct PtrSlots(Option<Ptr>);

impl Slotter for PtrSlots {
    const TOTAL: bool = true;

    #[inline(always)]
    fn slot(&self, key: &[u8]) -> usize {
        match &self.0 {
            Some(mphf) => mphf.index(key),
            None => 0,
        }
    }
}

impl<const INLINE: bool> Hashed<PtrSlots, INLINE> {
    /// Builds, writes and maps in one step: without `epserde` the crate has
    /// no way to reopen its pilots, so opening rebuilds them (`HEAP`).
    pub fn build_open(rows: &Rows, path: &Path) -> io::Result<Self> {
        let (mphf, n_slots) = if rows.keys.is_empty() {
            (None, 1)
        } else {
            let keys = rows.keys.par_iter().map(|k| *k);
            let mphf = Ptr::new_from_par_iter(rows.keys.len(), keys, PtrHashParams::default());
            let n_slots = mphf.max_index();
            (Some(mphf), n_slots)
        };
        let slotter = PtrSlots(mphf);
        let slot_of: Vec<usize> = rows.keys.iter().map(|k| slotter.slot(k)).collect();
        let (offs, blob) = Entries::sections(rows, &slot_of, n_slots, INLINE)?;
        let img = publish(path, &[as_bytes(&offs), &blob])?;
        Ok(Self {
            slotter,
            entries: Entries {
                offs: img.words(0),
                blob: img.bytes(1),
            },
            n_slots,
            _img: img,
        })
    }
}

// ------------------------------------------------------------------ PHast

/// The lab's key hash as a seeded streaming hasher for `ph`.
#[derive(Clone, Copy, Default)]
pub struct FoldSeeded;

pub struct FoldStream(u64);

impl Hasher for FoldStream {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = key_hash(bytes, self.0);
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
}

impl BuildSeededHasher for FoldSeeded {
    type Hasher = FoldStream;

    #[inline(always)]
    fn build_hasher(&self, seed: u64) -> FoldStream {
        FoldStream(fold(seed ^ K2, K1))
    }
}

type Phast = ph::phast::Function<Bits8, SeedOnly, DefaultCompressedArray, FoldSeeded>;

/// How many lookups ended in the crate's `unreachable!` and were caught.
pub static PHAST_PANICS: AtomicU64 = AtomicU64::new(0);

/// `ph` answers only for member keys: some non-members reach `unreachable!`.
/// The panic is caught and counted; `quiet_phast_panics` keeps it silent.
pub struct PhastSlots(Phast);

impl Slotter for PhastSlots {
    const TOTAL: bool = false;

    #[inline(always)]
    fn slot(&self, key: &[u8]) -> usize {
        catch_unwind(AssertUnwindSafe(|| self.0.get(key))).unwrap_or_else(|_| {
            PHAST_PANICS.fetch_add(1, Relaxed);
            usize::MAX
        })
    }
}

pub fn quiet_phast_panics() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let from_ph = info.location().is_some_and(|l| l.file().contains("phast"));
        if !from_ph {
            default(info);
        }
    }));
}

impl<const INLINE: bool> Hashed<PhastSlots, INLINE> {
    pub fn build(rows: &Rows, path: &Path) -> io::Result<()> {
        let params = Params::new(Bits8, bits_per_seed_to_100_bucket_size(8));
        let f = Phast::with_slice_p_hash_sc(&rows.keys, &params, FoldSeeded, SeedOnly);
        let slot_of: Vec<usize> = rows.keys.iter().map(|k| f.get(*k)).collect();
        let n_slots = rows.keys.len().max(1);
        let (offs, blob) = Entries::sections(rows, &slot_of, n_slots, INLINE)?;
        let mut mphf = Vec::new();
        f.write(&mut mphf)?;
        write_sections(path, &[&mphf, as_bytes(&offs), &blob]).map(drop)
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let img = Img::open(path)?;
        let f = Phast::read_with_hasher_sc(&mut img.bytes(0), FoldSeeded, SeedOnly)?;
        let offs: &[u32] = img.words(1);
        Ok(Self {
            slotter: PhastSlots(f),
            entries: Entries {
                offs,
                blob: img.bytes(2),
            },
            n_slots: offs.len() - 1,
            _img: img,
        })
    }
}

// ------------------------------------------------------------------ VFunc

/// A static function (sux `VFunc`) from form to its rank: three independent
/// loads, no key set. The crate reopens it only through `epserde`, so the
/// lab rebuilds it at open (`HEAP`).
pub struct VSlots(Option<sux::func::VFunc<[u8], Box<[u32]>>>);

impl Slotter for VSlots {
    const TOTAL: bool = false;

    #[inline(always)]
    fn slot(&self, key: &[u8]) -> usize {
        self.0.as_ref().map_or(usize::MAX, |f| f.get(key) as usize)
    }
}

impl<const INLINE: bool> Hashed<VSlots, INLINE> {
    pub fn build_open(rows: &Rows, path: &Path) -> io::Result<Self> {
        let n = rows.keys.len();
        let ranks: Vec<u32> = (0..n as u32).collect();
        let func = if n == 0 {
            None
        } else {
            let built = <sux::func::VFunc<[u8], Box<[u32]>>>::try_new(
                sux::utils::FromSlice::new(&rows.keys),
                sux::utils::FromSlice::new(&ranks),
                dsi_progress_logger::no_logs!(),
            );
            Some(built.map_err(|e| invalid(format!("vfunc: {e}")))?)
        };
        let slot_of: Vec<usize> = (0..n).collect();
        let n_slots = n.max(1);
        let (offs, blob) = Entries::sections(rows, &slot_of, n_slots, INLINE)?;
        let img = publish(path, &[as_bytes(&offs), &blob])?;
        Ok(Self {
            slotter: VSlots(func),
            entries: Entries {
                offs: img.words(0),
                blob: img.bytes(1),
            },
            n_slots,
            _img: img,
        })
    }
}

// ----------------------------------------------------- MPHF + fingerprint

/// `ptr_hash` with no keys stored: a 16-bit fingerprint and the payload per
/// slot. Not exact: a non-key passes with probability 2^-16.
pub struct PtrFp {
    slotter: PtrSlots,
    recs: &'static [u8],
    _img: Img,
}

const FP_SEED: u64 = 0xF1F0;
const FP_REC: usize = 6;

impl PtrFp {
    pub fn build_open(rows: &Rows, path: &Path) -> io::Result<Self> {
        let (mphf, n_slots) = if rows.keys.is_empty() {
            (None, 1)
        } else {
            let keys = rows.keys.par_iter().map(|k| *k);
            let mphf = Ptr::new_from_par_iter(rows.keys.len(), keys, PtrHashParams::default());
            let n_slots = mphf.max_index();
            (Some(mphf), n_slots)
        };
        let slotter = PtrSlots(mphf);
        // An empty slot holds a payload no key can carry.
        let mut recs = vec![0xFF_u8; n_slots * FP_REC];
        for (key, &payload) in rows.keys.iter().zip(&rows.payload) {
            let at = slotter.slot(key) * FP_REC;
            let fp = (key_hash(key, FP_SEED) >> 48) as u16;
            recs[at..at + 2].copy_from_slice(&fp.to_le_bytes());
            recs[at + 2..at + 6].copy_from_slice(&payload.to_le_bytes());
        }
        let img = publish(path, &[&recs])?;
        Ok(Self {
            slotter,
            recs: img.bytes(0),
            _img: img,
        })
    }
}

impl Index for PtrFp {
    #[inline(always)]
    fn find(&self, key: &[u8]) -> Option<u32> {
        let at = self.slotter.slot(key) * FP_REC;
        // SAFETY: the slot is below the record count.
        let (fp, payload) = unsafe {
            let rec = self.recs.as_ptr().add(at);
            (
                rec.cast::<u16>().read_unaligned(),
                rec.add(2).cast::<u32>().read_unaligned(),
            )
        };
        (fp == (key_hash(key, FP_SEED) >> 48) as u16 && payload != u32::MAX).then_some(payload)
    }
}

// ------------------------------------------------------- negative filter

const FILTER_SEED: u64 = 0xB1F8;

/// A binary fuse filter (8-bit fingerprints) asked before the structure.
pub struct Filtered<D> {
    filter: BinaryFuse8Ref<'static>,
    inner: D,
    _img: Img,
}

pub fn build_filter(keys: &[&[u8]], path: &Path) -> io::Result<()> {
    let hashes: Vec<u64> = keys.iter().map(|k| key_hash(k, FILTER_SEED)).collect();
    let filter = BinaryFuse8::try_from(&hashes).map_err(invalid)?;
    let mut descriptor = vec![0_u8; BinaryFuse8::DESCRIPTOR_LEN];
    filter.dma_copy_descriptor_to(&mut descriptor);
    write_sections(path, &[&descriptor, filter.dma_fingerprints()]).map(drop)
}

impl<D> Filtered<D> {
    pub fn open(path: &Path, inner: D) -> io::Result<Self> {
        let img = Img::open(path)?;
        Ok(Self {
            filter: BinaryFuse8Ref::from_dma(img.bytes(0), img.bytes(1)),
            inner,
            _img: img,
        })
    }
}

impl<D: Dict> Dict for Filtered<D> {
    #[inline(always)]
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>> {
        if !self.filter.contains(&key_hash(key, FILTER_SEED)) {
            return None;
        }
        self.inner.get(key)
    }
}
