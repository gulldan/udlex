//! Root transitions decoded once for the lifetime of a verified FST map.

use fst::raw::Output;

use super::Bytes;

/// A map whose first lookup step does not decode or search the root again.
pub struct CachedMap {
    // Keep Map pointer-sized with one allocation; DA dispatch depends on Table's layout.
    data: Box<CachedMapData>,
}

struct CachedMapData {
    fst: fst::Map<Bytes>,
    first: [Option<(usize, Output)>; 256],
}

impl CachedMap {
    // Every transition input is a byte, so it indexes the 256 cache slots.
    #[allow(clippy::indexing_slicing)]
    pub(crate) fn new(fst: fst::Map<Bytes>) -> Self {
        let mut data = Box::new(CachedMapData {
            fst,
            first: [None; 256],
        });
        let fields = &mut *data;
        for transition in fields.fst.as_fst().root().transitions() {
            fields.first[usize::from(transition.inp)] = Some((transition.addr, transition.out));
        }
        Self { data }
    }

    pub(super) fn fst(&self) -> &fst::Map<Bytes> {
        &self.data.fst
    }

    #[inline]
    pub(super) fn get(&self, key: &[u8]) -> Option<u64> {
        let data = &*self.data;
        let Some((&first, rest)) = key.split_first() else {
            return data.fst.get(key);
        };
        let &(address, mut output) = data.first.get(usize::from(first))?.as_ref()?;
        let fst = data.fst.as_fst();
        let mut node = fst.node(address);
        for &byte in rest {
            let transition = node.transition(node.find_input(byte)?);
            output = output.cat(transition.out);
            node = fst.node(transition.addr);
        }
        node.is_final()
            .then(|| output.cat(node.final_output()).value())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Bytes, CachedMap};

    fn map(rows: &[(&str, u64)]) -> CachedMap {
        let mut rows = rows.to_vec();
        rows.sort_unstable();
        let mut builder = fst::MapBuilder::memory();
        for (key, value) in rows {
            builder.insert(key, value).unwrap();
        }
        CachedMap::new(fst::Map::new(Bytes::Owned(builder.into_inner().unwrap())).unwrap())
    }

    #[test]
    fn root_cache_preserves_prefixes_unicode_and_edge_and_final_outputs() {
        let map = map(&[
            ("", 71),
            ("a", 100),
            ("ab", 150),
            ("ac", 90),
            ("z", u64::MAX),
            ("é", 0),
            ("ёж", 700),
            ("ёжик", 720),
            ("日本", 400),
        ]);
        let root = map.fst().as_fst().root();
        let edge = root.transition(root.find_input(b'a').unwrap());
        assert!(!edge.out.is_zero());
        assert!(!map.fst().as_fst().node(edge.addr).final_output().is_zero());
        assert!(root.transitions().any(|transition| transition.addr == 0));

        for key in [
            "",
            "a",
            "ab",
            "ac",
            "ad",
            "aba",
            "b",
            "z",
            "é",
            "e\u{301}",
            "ё",
            "ёж",
            "ёжик",
            "ёжики",
            "日",
            "日本",
            "日本語",
            "\0",
        ] {
            assert_eq!(map.get(key.as_bytes()), map.fst().get(key), "{key:?}");
        }
    }

    #[test]
    fn empty_and_single_edge_maps_keep_their_terminal_values() {
        for rows in [
            &[][..],
            &[("", 0)][..],
            &[("", u64::MAX)][..],
            &[("x", 73)][..],
        ] {
            let map = map(rows);
            for key in ["", "x", "xx", "y", "🦀"] {
                assert_eq!(map.get(key.as_bytes()), map.fst().get(key), "{key:?}");
            }
        }
    }

    #[test]
    fn every_possible_root_byte_has_its_own_cache_slot() {
        let mut builder = fst::MapBuilder::memory();
        for byte in 0..=u8::MAX {
            builder.insert([byte], u64::from(byte)).unwrap();
        }
        let map =
            CachedMap::new(fst::Map::new(Bytes::Owned(builder.into_inner().unwrap())).unwrap());
        assert_eq!(map.fst().as_fst().root().len(), 256);
        assert_eq!(map.get(b""), map.fst().get(b""));
        for byte in 0..=u8::MAX {
            for key in [&[byte][..], &[byte, b'x'][..]] {
                assert_eq!(map.get(key), map.fst().get(key), "{key:?}");
            }
        }
    }
}
