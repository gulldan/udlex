//! The core's key hash, restated: `key_hash` and `reduce` are private in
//! `hybrid.rs`. `tests::pilot_matches_core` proves the copy places keys
//! exactly as the production builder does.

pub const K1: u64 = 0x9E37_79B9_7F4A_7C15;
pub const K2: u64 = 0xD6E8_FEB8_6659_FD93;

#[inline(always)]
pub const fn fold(a: u64, b: u64) -> u64 {
    let product = (a as u128).wrapping_mul(b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

#[inline(always)]
pub fn key_hash(key: &[u8], seed: u64) -> u64 {
    let mut h = fold(seed ^ (key.len() as u64), K1);
    let (chunks, rest) = key.as_chunks::<8>();
    for chunk in chunks {
        h = fold(h ^ u64::from_le_bytes(*chunk), K2);
    }
    if !rest.is_empty() {
        let mut buf = [0_u8; 8];
        buf[..rest.len()].copy_from_slice(rest);
        h = fold(h ^ u64::from_le_bytes(buf), K1);
    }
    h
}

#[inline(always)]
pub const fn reduce(h: u64, n: usize) -> usize {
    (((h as u128) * (n as u128)) >> 64) as usize
}

/// Equality of two same-length keys without a `memcmp` call, as the core does.
#[inline(always)]
pub fn same(a: &[u8], b: &[u8]) -> bool {
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        if u64::from_le_bytes(*x) != u64::from_le_bytes(*y) {
            return false;
        }
    }
    ra.iter().zip(rb).all(|(x, y)| x == y)
}

/// Deterministic stream for workload sampling.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        reduce(self.next(), n)
    }
}
