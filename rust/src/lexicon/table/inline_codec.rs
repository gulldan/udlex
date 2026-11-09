//! Streaming decoder for the inline `form NUL varint(cut) tail` codec.

use super::{Bytes, SEP};

/// Decodes an LEB128 value from a byte slice for format regression tests.
#[cfg(all(test, feature = "train"))]
pub(in crate::lexicon) fn decode_varint(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0_usize;
    let mut shift = 0_u32;
    for (position, &byte) in bytes.iter().enumerate() {
        if decode_group(&mut value, &mut shift, position, byte)? {
            return Some((value, position.saturating_add(1)));
        }
    }
    None
}

fn decode_group(value: &mut usize, shift: &mut u32, position: usize, byte: u8) -> Option<bool> {
    if *shift >= usize::BITS {
        return None;
    }
    let part = usize::from(byte & 0x7F);
    if part > (usize::MAX >> *shift) {
        return None;
    }
    *value |= part << *shift;
    if byte & 0x80 == 0 {
        // Only the shortest LEB128 spelling is part of the bundle format.
        return (position == 0 || part != 0).then_some(true);
    }
    *shift = shift.saturating_add(7);
    Some(false)
}

/// Room for a tail longer than the cut it replaces, so the buffer is sized once.
const TAIL_HEADROOM: usize = 8;

/// Looks up a form while allocating only the final lemma buffer.
pub(in crate::lexicon) fn lookup(fsa: &fst::raw::Fst<Bytes>, form: &str) -> Option<String> {
    let mut node = fsa.root();
    for &byte in form.as_bytes() {
        if byte == SEP {
            return None;
        }
        let index = node.find_input(byte)?;
        node = fsa.node(node.transition_addr(index));
    }
    node = fsa.node(node.transition_addr(node.find_input(SEP)?));

    // Decode the cut directly from the unique payload transition path.
    let mut cut = 0_usize;
    let mut shift = 0_u32;
    let mut position = 0_usize;
    loop {
        if node.is_final() || node.len() != 1 {
            return None;
        }
        let transition = node.transition(0);
        node = fsa.node(transition.addr);
        if decode_group(&mut cut, &mut shift, position, transition.inp)? {
            break;
        }
        position = position.saturating_add(1);
    }
    let kept = form.len().checked_sub(cut)?;
    let head = form.as_bytes().get(..kept)?;
    // The cut is a byte prefix (`яблоки`/`яблоко` share a lead byte), so the
    // head may end inside a character; validation starts at the last boundary.
    let mut proved = kept;
    while !form.is_char_boundary(proved) {
        proved = proved.checked_sub(1)?;
    }

    let mut lemma = Vec::with_capacity(form.len().saturating_add(TAIL_HEADROOM));
    lemma.extend_from_slice(head);
    loop {
        if node.is_final() {
            if !node.is_empty() {
                return None;
            }
            break;
        }
        if node.len() != 1 {
            return None;
        }
        let transition = node.transition(0);
        lemma.push(transition.inp);
        node = fsa.node(transition.addr);
    }

    // Authenticated is not trusted: prove the tail (and the torn head bytes) UTF-8.
    std::str::from_utf8(lemma.get(proved..)?).ok()?;
    // SAFETY: everything below `proved` is `form` cut on a character boundary,
    // the rest was just validated, and valid text joined at a boundary to
    // valid text is valid text.
    Some(unsafe { String::from_utf8_unchecked(lemma) })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{Bytes, lookup};
    use crate::lexicon::table::{encode_key, encode_key_into};

    fn fst(key: &[u8]) -> fst::raw::Fst<Bytes> {
        let mut builder = fst::raw::Builder::memory();
        builder
            .insert(key, 0)
            .expect("one key is sorted and unique");
        let bytes = builder.into_inner().expect("memory writer cannot fail");
        fst::raw::Fst::new(Bytes::Owned(bytes)).expect("fresh FST parses")
    }

    #[test]
    fn decodes_a_multibyte_cut_and_long_tail() {
        let form = "a".repeat(130);
        let lemma = "словарный-хвост-длиннее-прежнего-reserve";
        let table = fst(&encode_key(form.as_bytes(), lemma.as_bytes()));
        assert_eq!(lookup(&table, &form).as_deref(), Some(lemma));
    }

    #[test]
    fn a_cut_inside_a_character_is_kept_when_the_tail_completes_it() {
        // The shared byte prefix ends between the two bytes of the last
        // letter, which is what the encoder writes for this pair.
        let table = fst(&encode_key("яблоки".as_bytes(), "яблоко".as_bytes()));
        assert_eq!(lookup(&table, "яблоки").as_deref(), Some("яблоко"));
    }

    #[test]
    fn a_cut_landing_inside_a_character_is_a_miss() {
        // The head would end between the two bytes of `я`, so the lemma the
        // table spells is not text; a miss, never a panic or a torn string.
        let table = fst(b"\xd1\x8f\0\x01!");
        assert_eq!(lookup(&table, "я"), None);
    }

    #[test]
    fn a_tail_that_is_not_utf8_is_a_miss() {
        let table = fst(b"cat\0\x03\xff");
        assert_eq!(lookup(&table, "cat"), None);
    }

    #[test]
    fn reusable_encoder_is_byte_for_byte_identical() {
        let mut storage = b"old allocation".to_vec();
        for (form, lemma) in [
            ("cats", "cat"),
            ("unchanged", "unchanged"),
            ("a very long shared prefix", "a very long shared suffix"),
            ("слова", "слово"),
        ] {
            encode_key_into(form.as_bytes(), lemma.as_bytes(), &mut storage);
            assert_eq!(storage, encode_key(form.as_bytes(), lemma.as_bytes()));
        }
    }

    #[test]
    fn rejects_a_noncanonical_streamed_varint() {
        assert_eq!(lookup(&fst(b"cat\0\x81\x00tail"), "cat"), None);
    }
}
