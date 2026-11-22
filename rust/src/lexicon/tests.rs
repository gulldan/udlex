use std::borrow::Cow;
use std::collections::BTreeMap;

use super::generation::write_file;
use super::runtime::read_to_string;
use super::table::{Bytes, da_search, decode_varint, index_matches, lookup_inline, max_output};
use super::*;
use crate::probe::Options;
use yada::builder::DoubleArrayBuilder;

/// Stages learned tables, an optional phrase source and their metadata.
fn stage(
    root: &Path,
    tables: &crate::train::Tables,
    phrases: &[(String, String)],
    training_tokens: usize,
) -> GenerationWriter {
    let writer = GenerationWriter::begin(root).unwrap();
    let entries = crate::train::write_sources(
        tables,
        crate::train::Variant::Full,
        crate::train::Thresholds::default(),
        writer.directory(),
    )
    .unwrap();
    let entries: serde_json::Map<String, serde_json::Value> = entries
        .into_iter()
        .chain([(
            "phrases".to_owned(),
            crate::train::write_phrase_source(phrases, writer.directory()).unwrap(),
        )])
        .map(|(name, count)| (name, count.into()))
        .collect();
    let meta = serde_json::json!({
        "name": root.file_name().unwrap().to_str().unwrap(),
        "language": "Test",
        "iso639_3": "zxx",
        "source": "test.train",
        "training_tokens": training_tokens,
        "variant": "full",
        "selected_on": null,
        "max_suffix": crate::format::MAX_SUFFIX_LENGTH,
        "format_version": FORMAT_VERSION,
        "entity_encoding": ENTITY_ENCODING,
        "generation": writer.id(),
        "normalization": NORMALIZATION,
        "files": {},
        "entries": entries,
        "traffic": {},
        "gold_tokens": null,
        "accuracy": null,
        "baseline": null,
        "beats_baseline": null,
        "codecs": {}
    });
    write_file(&writer.directory().join("meta.json"), &render_meta(&meta)).unwrap();
    writer
}

/// Publishes a complete source generation suitable for `compile` tests.
fn publish_sources(root: &Path, pairs: &[crate::train::Pair]) {
    let tables = crate::train::build(pairs, Vec::new());
    stage(root, &tables, &[], pairs.len()).publish().unwrap();
}

/// Publishes a generation whose entity table holds one namespaced pair.
fn publish_with_entities(root: &Path, surface: &str, lemma: &str) {
    publish_with_entities_and_phrases(root, &[(surface, lemma)], &[]);
}

/// Publishes a generation carrying both curated entities and phrase patterns.
fn publish_with_entities_and_phrases(
    root: &Path,
    entities: &[(&str, &str)],
    phrases: &[(&str, &str)],
) {
    let owned = |rows: &[(&str, &str)]| -> Vec<(String, String)> {
        rows.iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    };
    let tables =
        crate::train::build_with_entities(&[pair("cats", "cat")], Vec::new(), owned(entities));
    stage(root, &tables, &owned(phrases), 1)
        .compile_and_publish()
        .unwrap();
}

/// The compiled entity table of a published generation.
fn entity_table_path(root: &Path) -> PathBuf {
    let resolved = resolve_generation(root).unwrap();
    Codec::ALL
        .into_iter()
        .map(|codec| codec.file(&resolved.directory, "entities"))
        .find(|path| path.exists())
        .expect("a published generation with entity rows has a compiled table")
}

fn pair(form: &str, lemma: &str) -> crate::train::Pair {
    crate::train::Pair {
        form: form.to_owned(),
        lemma: lemma.to_owned(),
        sentence_initial: false,
    }
}

/// Builds a double array over `keys`, numbering the values from one.
fn trie(keys: &[&str]) -> (DoubleArray<Bytes>, Vec<(String, u32)>) {
    let mut rows: Vec<(String, u32)> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| ((*key).to_owned(), u32::try_from(index).unwrap() + 1))
        .collect();
    rows.sort();
    let pairs: Vec<(&[u8], u32)> = rows
        .iter()
        .map(|(key, value)| (key.as_bytes(), *value))
        .collect();
    let bytes = DoubleArrayBuilder::build(&pairs).expect("the builder accepts sorted keys");
    // Leaked because `Bytes` only holds what a bundle holds: a mapping or
    // something static. A test that ends with the process can afford it.
    let array =
        DoubleArray::new(Bytes::Static(Vec::leak(bytes))).expect("a freshly built array validates");
    (array, rows)
}

/// The hand-rolled walk must answer exactly what `yada`'s safe one answers;
/// it reimplements the dependency's bit layout.
#[test]
fn the_unchecked_walk_agrees_with_the_checked_one() {
    let keys = ["кот", "коты", "котами", "к", "zebra", "zeb"];
    let (array, rows) = trie(&keys);
    for (key, expected) in &rows {
        assert_eq!(da_search(&array, key.as_bytes()), Some(*expected), "{key}");
        assert_eq!(
            da_search(&array, key.as_bytes()),
            array.exact_match_search(key.as_bytes()),
            "{key}"
        );
    }
    // Misses, near-misses and prefixes of real keys. No NUL here: the
    // safe function asserts against it and panics, which is the one place
    // the two deliberately differ — see the test below.
    for absent in ["кош", "котамиx", "ко", "z", "zebras"] {
        assert_eq!(
            da_search(&array, absent.as_bytes()),
            array.exact_match_search(absent.as_bytes()),
            "{absent}"
        );
    }
}

/// A key holding NUL is a miss, not a panic as in `yada::exact_match_search`.
#[test]
fn a_nul_byte_in_the_key_is_answered_like_any_other_miss() {
    let (array, _) = trie(&["кот"]);
    assert_eq!(da_search(&array, b"\0"), None);
    assert_eq!(da_search(&array, b"\xd0\xba\0\xd0\xbe"), None);
}

/// The node walk must find exactly the largest value streaming finds: shared
/// suffixes under different prefixes, a key prefixing another, and the
/// maximum on neither the first nor the last key.
#[test]
fn the_node_walk_finds_the_same_largest_value_as_streaming() {
    use fst::Streamer as _;

    let mut rows: Vec<(&str, u64)> = vec![
        ("делать", 7),
        ("сделать", 41),
        ("переделать", 99_000),
        ("кот", 3),
        ("коты", 12_500),
        ("котами", 8),
        ("к", 1),
        ("zeb", 0),
        ("zebra", 640),
    ];
    rows.sort_unstable();

    let mut builder = fst::MapBuilder::memory();
    for (key, value) in &rows {
        builder
            .insert(key, *value)
            .expect("keys are sorted and unique");
    }
    let bytes = builder
        .into_inner()
        .expect("the in-memory writer never fails");
    // Leaked for the same reason `trie` leaks: `Bytes` holds a mapping or
    // something static, and a test that ends with the process can afford it.
    let map = fst::Map::new(Bytes::Static(Vec::leak(bytes))).expect("a freshly built map parses");

    let mut streamed = 0_u64;
    let mut stream = map.stream();
    while let Some((_, value)) = stream.next() {
        streamed = streamed.max(value);
    }

    assert_eq!(streamed, 99_000);
    assert_eq!(max_output(map.as_fst()), streamed);
}

/// Malformed inline payloads must not wrap their cut length or pick one lemma
/// according to automaton transition order.
#[test]
fn inline_lookup_rejects_overflow_noncanonical_and_ambiguous_payloads() {
    assert_eq!(decode_varint(&[0x7f]), Some((127, 1)));
    assert_eq!(decode_varint(&[0x80, 0]), None);

    let groups = usize::try_from(usize::BITS.div_ceil(7)).unwrap();
    let remaining = usize::BITS - u32::try_from((groups - 1) * 7).unwrap();
    let mut overflow = vec![0xff; groups - 1];
    overflow.push(1_u8 << remaining);
    assert_eq!(decode_varint(&overflow), None);

    let build = |keys: &[&[u8]]| {
        let mut builder = fst::raw::Builder::memory();
        for key in keys {
            builder.insert(*key, 0).unwrap();
        }
        let bytes = builder.into_inner().unwrap();
        fst::raw::Fst::new(Bytes::Static(Vec::leak(bytes))).unwrap()
    };

    let ambiguous = build(&[b"cat\0\x03", b"cat\0\x03x"]);
    assert_eq!(lookup_inline(&ambiguous, "cat"), None);

    let valid = build(&[b"cats\0\x01"]);
    assert_eq!(lookup_inline(&valid, "cats"), Some("cat".to_owned()));
    assert_eq!(lookup_inline(&valid, "cats\0"), None);
}

/// An empty key cannot be stored, and asking for one is still safe.
#[test]
fn an_empty_key_cannot_be_stored_and_is_safe_to_ask_for() {
    assert!(DoubleArrayBuilder::build(&[(b"".as_slice(), 1_u32)]).is_err());
    let (array, _) = trie(&["кот"]);
    assert_eq!(da_search(&array, b""), array.exact_match_search(b""));
}

/// Canonically equivalent spellings must reach the same entry: a decomposed
/// token misses the raw probe and is found by the normalised one.
#[test]
fn a_decomposed_token_finds_the_composed_key() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_test");
    publish_sources(&root, &[pair("caf\u{e9}s", "caf\u{e9}")]);
    compile(&root).unwrap();
    let lexicon = Lexicon::open(&root).unwrap();

    // Composed: the spelling the table stores, found by the raw walk.
    assert_eq!(lexicon.lemma("caf\u{e9}s", false), "caf\u{e9}");
    // Decomposed: the same word spelled `e` + combining acute. It misses the
    // raw walk byte for byte and must still come back with the same lemma.
    assert_eq!(lexicon.lemma("cafe\u{301}s", false), "caf\u{e9}");
    // And the lemma itself is returned composed, whichever spelling went in.
    assert!(unicode_normalization::is_nfc(
        &lexicon.lemma("cafe\u{301}s", false)
    ));
}

#[test]
fn empty_sentence_batch_preserves_borrowed_surfaces() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_empty_batch");
    publish_sources(&root, &[]);
    let metadata = resolve_generation(&root)
        .unwrap()
        .directory
        .join("meta.json");
    let mut meta: serde_json::Value =
        serde_json::from_str(&read_to_string(&metadata).unwrap()).unwrap();
    meta.as_object_mut()
        .unwrap()
        .insert("variant".to_owned(), "identity".into());
    write_file(&metadata, &render_meta(&meta)).unwrap();
    compile(&root).unwrap();
    let lexicon = Lexicon::open(&root).unwrap();
    let tokens = ["cafe\u{301}", "", "UNKNOWN"];
    let batch = lexicon.lemmatize(&tokens);
    assert_eq!(batch.len(), tokens.len());
    for (index, (answer, token)) in batch.iter().zip(tokens).enumerate() {
        assert_eq!(*answer, lexicon.lemma(token, index == 0));
        assert!(matches!(answer, Cow::Borrowed(found) if std::ptr::eq(*found, token)));
    }
    assert_eq!(lexicon.lemmatize(&[]), [] as [Cow<'_, str>; 0]);
}

#[test]
fn sentence_batch_matches_scalar_probes_and_surface_casing() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_batch");
    publish_sources(
        &root,
        &[
            pair("cats", "cat"),
            pair("dogs", "dog"),
            pair("rats", "rat"),
            pair("bats", "bat"),
            pair("cafés", "café"),
            pair("FOOS", "foo"),
        ],
    );
    compile(&root).unwrap();
    let metadata = resolve_generation(&root)
        .unwrap()
        .directory
        .join("meta.json");
    let mut meta: serde_json::Value =
        serde_json::from_str(&read_to_string(&metadata).unwrap()).unwrap();
    let tokens = ["Foos", "Foos", "cats", "cafe\u{301}s", "hats", "", "§§§"];
    for surface in [false, true] {
        meta.as_object_mut().unwrap().insert(
            "token_case".to_owned(),
            if surface { "surface" } else { "keep" }.into(),
        );
        write_file(&metadata, &render_meta(&meta)).unwrap();
        let lexicon = Lexicon::open(&root).unwrap();
        let scalar: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| lexicon.lemma_with_info(token, index == 0, Options::default()))
            .collect();
        assert_eq!(
            scalar
                .iter()
                .map(|answer| answer.source)
                .collect::<Vec<_>>(),
            [
                crate::Source::Lower,
                crate::Source::Lower,
                crate::Source::Exact,
                crate::Source::Exact,
                crate::Source::Rule,
                crate::Source::Identity,
                crate::Source::Identity,
            ]
        );
        let batch = lexicon.lemmatize(&tokens);
        assert_eq!(
            batch,
            [
                "foo",
                if surface { "Foo" } else { "foo" },
                "cat",
                "café",
                "hat",
                "",
                "§§§"
            ]
        );
        assert_eq!(
            batch,
            scalar
                .into_iter()
                .map(|answer| answer.lemma)
                .collect::<Vec<_>>()
        );
        assert_eq!(lexicon.lemmatize(&[]), [] as [Cow<'_, str>; 0]);
    }
}

#[test]
fn rich_sentence_answers_preserve_order_and_provenance() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_provenance");
    publish_sources(&root, &[pair("cats", "cat"), pair("lemma", "lemma")]);
    compile(&root).unwrap();
    let lexicon = Lexicon::open(&root).unwrap();

    let answers = lexicon.lemmatize_with_info(&["lemma", "cats", "§§§"]);
    assert_eq!(answers.len(), 3);
    let mut answers_iter = answers.iter();
    let unchanged = answers_iter.next().unwrap();
    assert_eq!(unchanged.lemma, "lemma");
    assert_eq!(unchanged.source, crate::Source::Exact);
    assert!(unchanged.source.is_lemmatized());
    let changed = answers_iter.next().unwrap();
    assert_eq!(changed.lemma, "cat");
    assert_eq!(changed.source, crate::Source::Exact);
    assert!(changed.source.is_lemmatized());
    let fallback = answers_iter.next().unwrap();
    assert_eq!(fallback.lemma, "§§§");
    assert_eq!(fallback.source, crate::Source::Identity);
    assert!(!fallback.source.is_lemmatized());
    assert!(answers_iter.next().is_none());

    let plain = lexicon.lemmatize(&["lemma", "cats", "§§§"]);
    assert_eq!(
        answers
            .iter()
            .map(|answer| answer.lemma.as_ref())
            .collect::<Vec<_>>(),
        plain
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<_>>()
    );
}

/// The language code in the metadata must reach the probe's case mapping.
#[test]
fn the_language_code_in_metadata_reaches_the_probe_casing() {
    use crate::probe::Tables as _;

    for (iso, expected) in [
        ("tur", crate::normalization::Casing::Turkish),
        ("ell", crate::normalization::Casing::Greek),
        ("nld", crate::normalization::Casing::Dutch),
        ("rus", crate::normalization::Casing::Root),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("xx_test");
        publish_sources(&root, &[pair("cats", "cat")]);
        compile(&root).unwrap();
        set_test_iso(&root, iso);
        let lexicon = Lexicon::open(&root).unwrap();
        assert_eq!(lexicon.casing(), expected, "{iso}");
    }
}

#[test]
fn generations_keep_loaded_mmaps_immutable_across_rebuilds() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_test");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let old = Lexicon::open(&root).unwrap();
    assert_eq!(old.lemma("cats", false), "cat");

    publish_sources(&root, &[pair("dogs", "dog")]);
    // Publishing sources makes new opens fail closed, while the already
    // loaded mapping continues to see its complete old generation.
    assert!(matches!(
        Lexicon::open(&root),
        Err(Error::NotCompiled { .. })
    ));
    assert_eq!(old.lemma("cats", false), "cat");
    compile(&root).unwrap();
    let new = Lexicon::open(&root).unwrap();
    assert_eq!(new.lemma("dogs", false), "dog");
    assert_eq!(new.lemma("cats", false), "cats");
    assert_eq!(old.lemma("cats", false), "cat");
}

#[test]
fn catalogue_rejects_duplicate_iso_codes_instead_of_loading_one_arbitrarily() {
    let temporary = tempfile::tempdir().unwrap();
    for name in ["xx_first", "xx_second"] {
        let root = temporary.path().join(name);
        publish_sources(&root, &[pair("cats", "cat")]);
        compile(&root).unwrap();
    }

    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));
    assert!(matches!(
        load(temporary.path(), "zxx"),
        Err(Error::Invalid { .. })
    ));
}

fn write_bundle_manifest(directory: &Path, rows: &[(&str, &str)], format_version: u32) {
    let lexicons: Vec<serde_json::Value> = rows
        .iter()
        .map(|(iso639_3, config)| serde_json::json!({"iso639_3": iso639_3, "config": config}))
        .collect();
    let manifest = serde_json::json!({
        "format_version": format_version,
        "lexicons": lexicons,
    });
    write_file(&directory.join("manifest.json"), &render_meta(&manifest)).unwrap();
}

fn set_test_iso(root: &Path, iso639_3: &str) {
    let metadata = resolve_generation(root)
        .unwrap()
        .directory
        .join("meta.json");
    let raw = read_to_string(&metadata).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("iso639_3".to_owned(), iso639_3.into());
    write_file(&metadata, &render_meta(&value)).unwrap();
}

fn write_wheel_manifest(
    directory: &Path,
    filename_iso: &str,
    declared_iso: &str,
    config: &str,
    generation: Option<&str>,
) {
    let manifests = directory.join(super::catalog::WHEEL_MANIFESTS);
    std::fs::create_dir_all(&manifests).unwrap();
    let mut row = serde_json::json!({"iso639_3": declared_iso, "config": config});
    if let Some(generation) = generation {
        row.as_object_mut()
            .unwrap()
            .insert("generation".to_owned(), generation.into());
    }
    let manifest = serde_json::json!({
        "format_version": FORMAT_VERSION,
        "lexicons": [row],
    });
    write_file(
        &manifests.join(format!(
            "{}{filename_iso}.json",
            super::catalog::WHEEL_MANIFEST_PREFIX
        )),
        &render_meta(&manifest),
    )
    .unwrap();
}

#[test]
fn wheel_manifests_merge_without_scratch_fallback() {
    let temporary = tempfile::tempdir().unwrap();
    for (config, iso, form, lemma) in [
        ("aa_one", "aaa", "cats", "cat"),
        ("zz_two", "zzz", "dogs", "dog"),
    ] {
        let root = temporary.path().join(config);
        publish_sources(&root, &[pair(form, lemma)]);
        set_test_iso(&root, iso);
        compile(&root).unwrap();
        let generation = resolve_generation(&root).unwrap().generation.unwrap();
        write_wheel_manifest(temporary.path(), iso, iso, config, Some(&generation));
    }

    let listed = languages(temporary.path()).unwrap();
    assert_eq!(
        listed
            .iter()
            .map(|language| language.iso639_3.as_str())
            .collect::<Vec<_>>(),
        ["aaa", "zzz"]
    );
    assert_eq!(
        load(temporary.path(), "aaa").unwrap().lemma("cats", false),
        "cat"
    );
    assert_eq!(
        load(temporary.path(), "zz_two")
            .unwrap()
            .lemma("dogs", false),
        "dog"
    );
}

#[test]
fn wheel_manifests_reject_duplicate_iso_or_config_identity() {
    let temporary = tempfile::tempdir().unwrap();
    write_wheel_manifest(
        temporary.path(),
        "aaa",
        "aaa",
        "aa_shared",
        Some("generation-a"),
    );
    write_wheel_manifest(
        temporary.path(),
        "zzz",
        "zzz",
        "aa_shared",
        Some("generation-b"),
    );
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));

    std::fs::remove_dir_all(temporary.path().join(super::catalog::WHEEL_MANIFESTS)).unwrap();
    write_wheel_manifest(
        temporary.path(),
        "aaa",
        "aaa",
        "aa_one",
        Some("generation-a"),
    );
    write_wheel_manifest(
        temporary.path(),
        "zzz",
        "aaa",
        "zz_two",
        Some("generation-b"),
    );
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn wheel_manifest_requires_the_exact_active_generation() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_one");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();

    write_wheel_manifest(temporary.path(), "zxx", "zxx", "xx_one", None);
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));

    write_wheel_manifest(
        temporary.path(),
        "zxx",
        "zxx",
        "xx_one",
        Some("different-generation"),
    );
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn manifest_catalogue_requires_every_declared_lexicon() {
    let temporary = tempfile::tempdir().unwrap();
    write_bundle_manifest(temporary.path(), &[("zxx", "xx_missing")], FORMAT_VERSION);

    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn manifest_catalogue_never_turns_corruption_into_unsupported_language() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_broken");
    publish_sources(&root, &[pair("cats", "cat")]);
    let metadata = resolve_generation(&root)
        .unwrap()
        .directory
        .join("meta.json");
    std::fs::write(metadata, "{ broken\n").unwrap();
    write_bundle_manifest(temporary.path(), &[("zxx", "xx_broken")], FORMAT_VERSION);

    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Meta { .. })
    ));
}

#[test]
fn manifest_catalogue_authenticates_declared_identity_and_format() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_real");
    publish_sources(&root, &[pair("cats", "cat")]);

    write_bundle_manifest(temporary.path(), &[("eng", "xx_real")], FORMAT_VERSION);
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));

    write_bundle_manifest(temporary.path(), &[("zxx", "../xx_real")], FORMAT_VERSION);
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));

    write_bundle_manifest(temporary.path(), &[("zxx", "xx_real")], FORMAT_VERSION + 1);
    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Format { .. })
    ));
}

/// Rewrites the declared format of an already compiled generation.
fn set_declared_format(root: &Path, format_version: u32) {
    let metadata = resolve_generation(root)
        .unwrap()
        .directory
        .join("meta.json");
    let raw = read_to_string(&metadata).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("format_version".to_owned(), format_version.into());
    write_file(&metadata, &render_meta(&value)).unwrap();
}

#[test]
fn both_readers_accept_exactly_the_declared_range_and_name_it_when_they_do_not() {
    let range = FormatRange::READ;
    let outside = [range.min.saturating_sub(1), range.max.saturating_add(1)];
    let cases = [
        (range.min, true),
        (range.max, true),
        (outside[0], false),
        (outside[1], false),
    ];
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_range");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let generation = resolve_generation(&root).unwrap().generation.unwrap();

    for (declared, accepted) in cases {
        assert_eq!(range.accepts(declared), accepted, "{declared}");
        set_declared_format(&root, declared);
        write_bundle_manifest(temporary.path(), &[("zxx", "xx_range")], declared);
        let opened = Lexicon::open(&root);
        let listed = languages(temporary.path());
        if accepted {
            assert_eq!(opened.unwrap().lemma("cats", false), "cat");
            assert_eq!(listed.unwrap().len(), 1);
            continue;
        }
        // Both numbers and the one action that can help, for either reader.
        for refusal in [opened.unwrap_err(), listed.unwrap_err()] {
            let text = refusal.to_string();
            assert!(
                matches!(&refusal, Error::Format { found, supported, .. }
                if *found == declared && *supported == range),
                "{text}"
            );
            assert!(text.contains(&declared.to_string()), "{text}");
            assert!(text.contains(&range.to_string()), "{text}");
            let expected = if declared > range.max {
                "upgrade udlex"
            } else {
                "rebuild the bundle with udlex.build_lexicon"
            };
            assert!(text.contains(expected), "{text}");
        }
    }
    // The generation itself never moved: only the number it declares did.
    assert_eq!(
        resolve_generation(&root).unwrap().generation.as_deref(),
        Some(generation.as_str())
    );
}

#[test]
fn manifest_catalogue_lists_only_authenticated_rows() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_real");
    publish_sources(&root, &[pair("cats", "cat")]);
    write_bundle_manifest(temporary.path(), &[("zxx", "xx_real")], FORMAT_VERSION);

    let listed = languages(temporary.path()).unwrap();
    assert_eq!(listed.len(), 1);
    let language = listed.first().unwrap();
    assert_eq!(language.iso639_3, "zxx");
    assert_eq!(language.lexicon, "xx_real");
}

#[test]
fn manifest_bundle_never_loads_an_undeclared_stale_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let declared = temporary.path().join("xx_declared");
    let stale = temporary.path().join("xx_stale");
    publish_sources(&declared, &[pair("cats", "cat")]);
    publish_sources(&stale, &[pair("dogs", "dog")]);
    compile(&declared).unwrap();
    compile(&stale).unwrap();
    write_bundle_manifest(temporary.path(), &[("zxx", "xx_declared")], FORMAT_VERSION);

    assert_eq!(
        load(temporary.path(), "xx_declared")
            .unwrap()
            .lemma("cats", false),
        "cat"
    );
    assert!(matches!(
        load(temporary.path(), "xx_stale"),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn manifest_generation_never_disagrees_with_the_active_pointer() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_declared");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let manifest = serde_json::json!({
        "format_version": FORMAT_VERSION,
        "lexicons": [{
            "iso639_3": "zxx",
            "config": "xx_declared",
            "generation": "different-generation",
        }],
    });
    write_file(
        &temporary.path().join("manifest.json"),
        &render_meta(&manifest),
    )
    .unwrap();

    assert!(matches!(
        languages(temporary.path()),
        Err(Error::Invalid { .. })
    ));
    assert!(matches!(
        load(temporary.path(), "xx_declared"),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn loading_one_manifest_row_does_not_scan_unrelated_lexicon_metadata() {
    let temporary = tempfile::tempdir().unwrap();
    let declared = temporary.path().join("xx_declared");
    publish_sources(&declared, &[pair("cats", "cat")]);
    compile(&declared).unwrap();
    write_bundle_manifest(
        temporary.path(),
        &[("zxx", "xx_declared"), ("eng", "xx_missing")],
        FORMAT_VERSION,
    );

    assert_eq!(
        load(temporary.path(), "xx_declared")
            .unwrap()
            .lemma("cats", false),
        "cat"
    );
}

#[test]
fn pruning_publishes_a_runtime_only_generation_without_mutating_readers() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_prune");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let before = resolve_generation(&root).unwrap().directory;
    let old = Lexicon::open(&root).unwrap();

    assert!(prune(&root).unwrap());
    let after = resolve_generation(&root).unwrap().directory;
    assert_ne!(before, after);
    assert!(SOURCES.into_iter().all(|name| before.join(name).is_file()));
    assert!(SOURCES.into_iter().all(|name| !after.join(name).exists()));
    assert_eq!(old.lemma("cats", false), "cat");
    assert_eq!(Lexicon::open(&root).unwrap().lemma("cats", false), "cat");
    assert!(!prune(&root).unwrap());
}

#[test]
fn pruning_cuts_lemmas_only_long_hybrid_keys_reference() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_prune_tail");
    // Past the eight-byte split, so its record carries "categorical" itself.
    publish_sources(
        &root,
        &[pair("cats", "cat"), pair("categorically", "categorical")],
    );
    compile(&root).unwrap();
    let lines = |root: &Path| {
        let directory = resolve_generation(root).unwrap().directory;
        std::fs::read_to_string(directory.join("strings.txt"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let before = lines(&root);
    assert_eq!(before.last().map(String::as_str), Some("categorical"));

    assert!(prune(&root).unwrap());
    let after = lines(&root);
    assert!(!after.iter().any(|line| line == "categorical"));
    assert_eq!(after.len(), before.len() - 1);
    let lexicon = Lexicon::open(&root).unwrap();
    assert_eq!(lexicon.lemma("categorically", false), "categorical");
    assert_eq!(lexicon.lemma("cats", false), "cat");
}

#[test]
fn advisory_writer_lock_survives_stale_files_but_blocks_live_writers() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_lock");
    std::fs::create_dir(&root).unwrap();
    let lock_path = root.join(".write.lock");
    std::fs::write(&lock_path, "stale pid from a crashed process\n").unwrap();

    let first = GenerationWriter::begin(&root).unwrap();
    assert!(matches!(
        GenerationWriter::begin(&root),
        Err(Error::Io { .. })
    ));
    drop(first);

    // Closing the owning descriptor releases an advisory lock. An OS
    // process teardown (including SIGKILL) closes it in the same way.
    let recovered = GenerationWriter::begin(&root).unwrap();
    drop(recovered);
    assert!(lock_path.is_file());
}

#[test]
fn publishing_replaces_an_existing_current_pointer() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_replace");
    let first = GenerationWriter::begin(&root).unwrap();
    let first_id = first.id().to_owned();
    first.publish().unwrap();
    let second = GenerationWriter::begin(&root).unwrap();
    let second_id = second.id().to_owned();
    second.publish().unwrap();

    assert_ne!(first_id, second_id);
    assert_eq!(
        read_to_string(&root.join(CURRENT)).unwrap().trim(),
        second_id
    );
}

#[cfg(unix)]
#[test]
fn compilation_refuses_symlinked_sources_without_touching_the_target() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_symlink");
    publish_sources(&root, &[pair("cats", "cat")]);
    let source = resolve_generation(&root).unwrap();
    let outside = temporary.path().join("outside.tsv");
    std::fs::write(&outside, "sentinel\n").unwrap();
    std::fs::remove_file(source.directory.join("exact.tsv")).unwrap();
    symlink(&outside, source.directory.join("exact.tsv")).unwrap();

    assert!(matches!(compile(&root), Err(Error::Invalid { .. })));
    assert_eq!(read_to_string(&outside).unwrap(), "sentinel\n");
}

#[cfg(unix)]
#[test]
fn compiler_detects_a_symlink_swap_after_opening_its_source_descriptor() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source.tsv");
    let opened = temporary.path().join("opened.tsv");
    let outside = temporary.path().join("outside.tsv");
    let destination = temporary.path().join("copied.tsv");
    std::fs::write(&source, "safe\n").unwrap();
    std::fs::write(&outside, "sentinel\n").unwrap();

    let result = generation::copy_regular_after_open(&source, &destination, || {
        std::fs::rename(&source, &opened).unwrap();
        symlink(&outside, &source).unwrap();
    });

    assert!(matches!(result, Err(Error::Invalid { .. })));
    assert_eq!(read_to_string(&outside).unwrap(), "sentinel\n");
    assert_eq!(read_to_string(&destination).unwrap(), "safe\n");
}

#[cfg(unix)]
#[test]
fn runtime_refuses_symlinked_metadata_without_reading_the_target() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_meta_link");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let resolved = resolve_generation(&root).unwrap();
    let metadata = resolved.directory.join("meta.json");
    let outside = temporary.path().join("outside-meta.json");
    std::fs::copy(&metadata, &outside).unwrap();
    let sentinel = std::fs::read(&outside).unwrap();
    std::fs::remove_file(&metadata).unwrap();
    symlink(&outside, &metadata).unwrap();

    assert!(matches!(Lexicon::open(&root), Err(Error::Invalid { .. })));
    assert_eq!(std::fs::read(&outside).unwrap(), sentinel);
}

#[cfg(unix)]
#[test]
fn runtime_refuses_symlinked_payload_even_when_the_bytes_match() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_payload_link");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let resolved = resolve_generation(&root).unwrap();
    let payload = resolved.directory.join("strings.txt");
    let outside = temporary.path().join("outside-strings.txt");
    std::fs::copy(&payload, &outside).unwrap();
    let sentinel = std::fs::read(&outside).unwrap();
    std::fs::remove_file(&payload).unwrap();
    symlink(&outside, &payload).unwrap();

    assert!(matches!(Lexicon::open(&root), Err(Error::Invalid { .. })));
    assert_eq!(std::fs::read(&outside).unwrap(), sentinel);
}

#[test]
fn runtime_rejects_oversized_metadata_before_json_parsing() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_huge_meta");
    publish_sources(&root, &[pair("cats", "cat")]);
    let resolved = resolve_generation(&root).unwrap();
    std::fs::write(
        resolved.directory.join("meta.json"),
        vec![b' '; usize::try_from(runtime::MAX_METADATA_BYTES).unwrap() + 1],
    )
    .unwrap();

    assert!(matches!(compile(&root), Err(Error::Invalid { .. })));
}

#[test]
fn runtime_rejects_lookup_entry_totals_that_overflow_the_catalogue_type() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_entry_overflow");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let resolved = resolve_generation(&root).unwrap();
    let path = resolved.directory.join("meta.json");
    let mut meta: serde_json::Value = serde_json::from_str(&read_metadata(&path).unwrap()).unwrap();
    let entries = meta
        .get_mut("entries")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap();
    entries.insert("exact".to_owned(), u64::MAX.into());
    entries.insert("lower".to_owned(), u64::MAX.into());
    write_file(&path, &render_meta(&meta)).unwrap();

    assert!(matches!(Lexicon::open(&root), Err(Error::Invalid { .. })));
}

#[cfg(unix)]
#[test]
fn writer_lock_never_follows_a_symlink() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_lock_link");
    std::fs::create_dir(&root).unwrap();
    let outside = temporary.path().join("outside.lock");
    std::fs::write(&outside, "do not truncate\n").unwrap();
    symlink(&outside, root.join(".write.lock")).unwrap();

    assert!(GenerationWriter::begin(&root).is_err());
    assert_eq!(read_to_string(&outside).unwrap(), "do not truncate\n");
}

#[test]
fn a_tampered_generation_is_rejected_before_mapping() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_hash");
    publish_sources(&root, &[pair("cats", "cat")]);
    compile(&root).unwrap();
    let resolved = resolve_generation(&root).unwrap();
    std::fs::write(resolved.directory.join("strings.txt"), "wrong\n").unwrap();
    assert!(matches!(Lexicon::open(&root), Err(Error::Integrity { .. })));
}

#[test]
fn fst_checksum_verification_rejects_a_bit_flip() {
    let mut builder = fst::MapBuilder::memory();
    builder.insert("cats", 0).unwrap();
    let mut bytes = builder.into_inner().unwrap();
    let last = bytes.last_mut().unwrap();
    *last ^= 1;
    let error = open_encoded(
        Path::new("corrupt.map"),
        Codec::Map,
        Bytes::Static(Vec::leak(bytes)),
    )
    .err()
    .unwrap();
    assert!(matches!(error, Error::Fst { .. }));
}

#[test]
fn an_offset_index_must_match_every_real_line_boundary() {
    let blob = b"cat\ndog\n";
    let mut index = Vec::new();
    for offset in [
        0_u32,
        u32::try_from(blob.len()).unwrap(),
        u32::try_from(blob.len()).unwrap(),
    ] {
        index.extend_from_slice(&offset.to_le_bytes());
    }
    assert!(!index_matches(&index, blob));
}

#[test]
fn an_empty_string_table_has_one_zero_boundary() {
    let index = 0_u32.to_le_bytes();
    assert_eq!(line_offsets(b""), Some(vec![0]));
    assert!(index_matches(&index, b""));
}

#[test]
fn a_string_index_accepts_exact_boundaries_with_or_without_final_newline() {
    for blob in [b"cat\ndog\n".as_slice(), b"cat\ndog".as_slice()] {
        let mut index = Vec::new();
        for offset in line_offsets(blob).unwrap() {
            index.extend_from_slice(&offset.to_le_bytes());
        }
        assert!(index_matches(&index, blob));
    }
}

#[test]
fn malformed_source_rows_fail_compilation_without_publishing() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_bad_rows");
    publish_sources(&root, &[pair("cats", "cat")]);
    let source = resolve_generation(&root).unwrap();
    std::fs::write(source.directory.join("exact.tsv"), "cats\t0\nbroken\tnan\n").unwrap();
    assert!(matches!(compile(&root), Err(Error::Source { line: 2, .. })));
    assert!(matches!(
        Lexicon::open(&root),
        Err(Error::NotCompiled { .. })
    ));
}

#[test]
fn metadata_without_sources_or_artifacts_is_not_already_compiled() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_empty");
    std::fs::create_dir(&root).unwrap();
    let meta = serde_json::json!({
        "name": "xx_empty", "language": "Test", "iso639_3": "zxx",
        "source": "none", "training_tokens": 0, "variant": "identity",
        "selected_on": null, "max_suffix": 0, "format_version": FORMAT_VERSION,
        "entity_encoding": ENTITY_ENCODING,
        "generation": null, "normalization": NORMALIZATION, "files": {},
        "entries": {"exact": 0, "lower": 0, "external": 0, "entities": 0,
                    "rules": 0, "strings": 0},
        "traffic": {}, "gold_tokens": null, "accuracy": null,
        "baseline": null, "beats_baseline": null, "codecs": {}
    });
    write_file(&root.join("meta.json"), &render_meta(&meta)).unwrap();
    assert!(matches!(compile(&root), Err(Error::Invalid { .. })));
}

#[test]
fn metadata_requires_a_lowercase_ascii_iso_code_and_generation() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_meta");
    publish_sources(&root, &[pair("cats", "cat")]);
    let resolved = resolve_generation(&root).unwrap();
    let path = resolved.directory.join("meta.json");
    let raw = read_to_string(&path).unwrap();
    let original: Meta = serde_json::from_str(&raw).unwrap();

    for invalid in ["Aaa", "a1_", "aé", "abcd"] {
        let mut meta = original.clone();
        meta.iso639_3 = invalid.to_owned();
        assert!(matches!(
            validate_meta(&meta, &path, resolved.generation.as_deref()),
            Err(Error::Invalid { .. })
        ));
    }

    let mut empty_source = original.clone();
    empty_source.source.clear();
    assert!(matches!(
        validate_meta(&empty_source, &path, resolved.generation.as_deref()),
        Err(Error::Invalid { .. })
    ));

    let mut without_variant: serde_json::Value = serde_json::from_str(&raw).unwrap();
    without_variant.as_object_mut().unwrap().remove("variant");
    let defaulted: Meta = serde_json::from_value(without_variant).unwrap();
    assert_eq!(defaulted.variant, "full");
    validate_meta(&defaulted, &path, resolved.generation.as_deref()).unwrap();

    let mut missing_generation = original.clone();
    missing_generation.generation = None;
    assert!(matches!(
        validate_meta(&missing_generation, &path, None),
        Err(Error::Invalid { .. })
    ));

    for allowed in [None, Some("keep"), Some("surface")] {
        let mut meta = original.clone();
        meta.token_case = allowed.map(str::to_owned);
        validate_meta(&meta, &path, resolved.generation.as_deref()).unwrap();
    }
    let mut unknown_case = original;
    unknown_case.token_case = Some("title".to_owned());
    assert!(matches!(
        validate_meta(&unknown_case, &path, resolved.generation.as_deref()),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn entity_encoding_marker_is_required_by_compile_and_open() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_entity_encoding");
    publish_sources(&root, &[pair("cats", "cat")]);
    let source = resolve_generation(&root).unwrap();
    let source_meta = source.directory.join("meta.json");
    let mut payload: serde_json::Value =
        serde_json::from_str(&read_to_string(&source_meta).unwrap()).unwrap();
    payload.as_object_mut().unwrap().remove("entity_encoding");
    write_file(&source_meta, &render_meta(&payload)).unwrap();

    let error = compile(&root).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid { message, .. } if message.contains("entity_encoding") && message.contains("must be rebuilt")),
        "{error}"
    );

    payload
        .as_object_mut()
        .unwrap()
        .insert("entity_encoding".to_owned(), ENTITY_ENCODING.into());
    write_file(&source_meta, &render_meta(&payload)).unwrap();
    compile(&root).unwrap();

    let compiled = resolve_generation(&root).unwrap();
    let compiled_meta = compiled.directory.join("meta.json");
    let mut payload: serde_json::Value =
        serde_json::from_str(&read_to_string(&compiled_meta).unwrap()).unwrap();
    payload
        .as_object_mut()
        .unwrap()
        .insert("entity_encoding".to_owned(), "legacy-unprefixed-v0".into());
    write_file(&compiled_meta, &render_meta(&payload)).unwrap();
    assert!(matches!(
        Lexicon::open(&root),
        Err(Error::Invalid { message, .. })
            if message.contains("entity_encoding") && message.contains("must be rebuilt")
    ));
    let error = compile(&root).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid { message, .. } if message.contains("entity_encoding") && message.contains("must be rebuilt")),
        "{error}"
    );
}

/// A format-3 generation is refused by the reader and upgraded by the
/// compiler, and only when it holds no phrase source of its own: a directory
/// declaring the older format while carrying the newer table is not a bundle
/// any writer produced, so it is corruption rather than a migration.
#[test]
fn format_three_migrates_only_without_its_undefined_phrase_table() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_format_three");
    publish_sources(&root, &[pair("cats", "cat")]);
    let source = resolve_generation(&root).unwrap();
    let source_meta = source.directory.join("meta.json");
    let mut payload: serde_json::Value =
        serde_json::from_str(&read_to_string(&source_meta).unwrap()).unwrap();
    let object = payload.as_object_mut().unwrap();
    object.insert("format_version".to_owned(), 3.into());
    object
        .get_mut("entries")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap()
        .remove("phrases");
    write_file(&source_meta, &render_meta(&payload)).unwrap();

    // The reader refuses what the compiler still migrates: two ranges, one
    // rule, and the refusal names the range rather than a single number.
    let refusal = Lexicon::open(&root).unwrap_err();
    assert!(
        matches!(&refusal, Error::Format { found: 3, supported, .. } if *supported == FormatRange::READ),
        "{refusal}"
    );
    assert!(
        refusal.to_string().contains("udlex.build_lexicon"),
        "{refusal}"
    );
    let error = compile(&root).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid { message, .. } if message.contains("format-3 did not define phrase patterns")),
        "{error}"
    );

    std::fs::remove_file(source.directory.join("phrases.tsv")).unwrap();
    compile(&root).unwrap();
    let lexicon = Lexicon::open(&root).unwrap();
    assert_eq!(lexicon.lemma("cats", false), "cat");
    assert_eq!(lexicon.meta().format_version, FORMAT_VERSION);
    assert_eq!(lexicon.meta().entries.get("phrases"), Some(&0));
}

#[test]
fn compiler_rejects_unprefixed_entity_source_keys() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_unprefixed_entity");
    publish_sources(&root, &[pair("OpenAI", "OpenAI")]);
    let source = resolve_generation(&root).unwrap();
    write_file(&source.directory.join("entities.tsv"), b"OpenAI\t0\n").unwrap();

    let error = compile(&root).unwrap_err();
    assert!(
        matches!(&error, Error::Invalid { message, .. } if message.contains("0<exact>") && message.contains("1<locale-lower>")),
        "{error}"
    );
}

#[test]
fn metadata_never_accepts_test_selected_configuration() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_selection");
    publish_sources(&root, &[pair("cats", "cat")]);
    let resolved = resolve_generation(&root).unwrap();
    let path = resolved.directory.join("meta.json");
    let raw = read_to_string(&path).unwrap();
    let original: Meta = serde_json::from_str(&raw).unwrap();

    for allowed in [None, Some("dev"), Some("train-cv")] {
        let mut meta = original.clone();
        meta.selected_on = allowed.map(str::to_owned);
        validate_meta(&meta, &path, resolved.generation.as_deref()).unwrap();
    }
    for forbidden in ["test", "train", "validation"] {
        let mut meta = original.clone();
        meta.selected_on = Some(forbidden.to_owned());
        assert!(matches!(
            validate_meta(&meta, &path, resolved.generation.as_deref()),
            Err(Error::Invalid { .. })
        ));
    }
}

#[test]
fn metadata_variant_must_be_what_its_own_selection_receipt_chooses() {
    let mut meta: Meta =
        serde_json::from_str(include_str!("../../tests/data/de_tiny/meta.json")).unwrap();
    let path = Path::new("<selection-meta>");
    let counted = |hits, discordant| CandidateRecord {
        hits,
        discordant,
        ner_hits: None,
        rivals: None,
    };
    // 61 tokens won and 33 lost is 2.9 standard errors of the lead: a
    // development split ships it, a train-only holdout does not.
    let receipt = |on: &str| SelectionRecord {
        on: on.to_owned(),
        tokens: 1389,
        copied: 583,
        candidates: BTreeMap::from([
            ("identity".to_owned(), counted(583, 0)),
            ("corpus".to_owned(), counted(583, 0)),
            ("rules".to_owned(), counted(583, 0)),
            ("lookup".to_owned(), counted(583, 0)),
            ("full".to_owned(), counted(611, 94)),
        ]),
    };

    meta.variant = "full".to_owned();
    meta.selected_on = Some("dev".to_owned());
    meta.selection = Some(receipt("dev"));
    validate_meta(&meta, path, None).unwrap();

    meta.selected_on = Some("train-cv".to_owned());
    meta.selection = Some(receipt("train-cv"));
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));

    // The receipt must also name the same sample the metadata does, and a
    // candidate may not be missing from it.
    meta.selected_on = Some("dev".to_owned());
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));
    let mut short = receipt("dev");
    short.candidates.remove("lookup");
    meta.selection = Some(short);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn selection_profiles_re_derive_the_ner_aware_verdict_and_reject_half_receipts() {
    let mut meta: Meta =
        serde_json::from_str(include_str!("../../tests/data/de_tiny/meta.json")).unwrap();
    let path = Path::new("<profiled-selection-meta>");
    let names = ["identity", "corpus", "rules", "lookup", "full"];
    let hits = [500_u64, 600, 650, 648, 648];
    let ner = [500_u64, 600, 600, 640, 640];
    let rivals = [
        [0_u64, 120, 170, 160, 160],
        [120, 0, 60, 50, 50],
        [170, 60, 0, 40, 40],
        [160, 50, 40, 0, 10],
        [160, 50, 40, 10, 0],
    ];
    let receipt = SelectionRecord {
        on: "dev".to_owned(),
        tokens: 1000,
        copied: 500,
        candidates: names
            .iter()
            .zip(hits)
            .zip(ner)
            .zip(rivals)
            .map(|(((name, hits), ner_hits), row)| {
                (
                    (*name).to_owned(),
                    CandidateRecord {
                        hits,
                        // The matrix is symmetric, so a candidate's identity
                        // column is the first cell of its own row.
                        discordant: row.first().copied().unwrap_or(0),
                        ner_hits: Some(ner_hits),
                        rivals: Some(
                            names
                                .iter()
                                .zip(row)
                                .map(|(rival, count)| ((*rival).to_owned(), count))
                                .collect(),
                        ),
                    },
                )
            })
            .collect(),
    };

    // The profiled receipt resolves the rules/lookup statistical tie by the
    // NER profile, and `lookup` is what this metadata ships.
    meta.variant = "lookup".to_owned();
    meta.selected_on = Some("dev".to_owned());
    meta.selection = Some(receipt.clone());
    validate_meta(&meta, path, None).unwrap();

    // The same counts without profiles are an older receipt and are read
    // under the rule that wrote them: bare argmax, which ships `rules` and
    // therefore contradicts this metadata.
    let mut stripped = receipt.clone();
    for counted in stripped.candidates.values_mut() {
        counted.ner_hits = None;
        counted.rivals = None;
    }
    meta.selection = Some(stripped);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));

    // Half a receipt is no receipt: one candidate without its profile, or a
    // profile with only one of its two fields, or a rivals map that does not
    // spell the lattice exactly.
    let mut partial = receipt.clone();
    let corpus = partial.candidates.get_mut("corpus").unwrap();
    corpus.ner_hits = None;
    corpus.rivals = None;
    meta.selection = Some(partial);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));

    let mut widowed = receipt.clone();
    widowed.candidates.get_mut("corpus").unwrap().rivals = None;
    meta.selection = Some(widowed);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));

    let mut short = receipt;
    short
        .candidates
        .get_mut("rules")
        .unwrap()
        .rivals
        .as_mut()
        .unwrap()
        .remove("full");
    meta.selection = Some(short);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn metadata_measurement_verdict_must_match_the_declared_scores() {
    let mut meta: Meta =
        serde_json::from_str(include_str!("../../tests/data/de_tiny/meta.json")).unwrap();
    let path = Path::new("<measurement-meta>");

    meta.gold_tokens = Some(10);
    meta.accuracy = Some(0.8);
    meta.baseline = Some(0.7);
    meta.beats_baseline = Some(false);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));

    meta.beats_baseline = Some(true);
    assert!(validate_meta(&meta, path, None).is_ok());
    meta.gold_tokens = Some(0);
    assert!(matches!(
        validate_meta(&meta, path, None),
        Err(Error::Invalid { .. })
    ));
}

#[test]
fn language_names_cannot_escape_the_bundle() {
    let temporary = tempfile::tempdir().unwrap();
    assert!(matches!(
        load(temporary.path(), "../outside"),
        Err(Error::Invalid { .. })
    ));
    assert!(matches!(
        load(temporary.path(), "/tmp/outside"),
        Err(Error::Invalid { .. })
    ));
}

/// Opening must not read the entity table, and asking must; deleting the file
/// after the open proves it.
#[test]
fn opening_defers_the_entity_table_until_something_asks() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_deferred");
    publish_with_entities(&root, "Open AI", "OpenAI");

    let lexicon = Lexicon::open(&root).unwrap();
    assert!(!lexicon.entities_ready());
    // Ordinary lemmatisation is unaffected, table or no table.
    assert_eq!(lexicon.lemma("cats", false), "cat");
    assert!(!lexicon.entities_ready());

    let options = crate::probe::Options::default();
    let answer = lexicon.lemma_span_with_info("Open AI", &["Open", "AI"], None, false, options);
    assert_eq!(answer.lemma, "OpenAI");
    assert_eq!(answer.source, crate::probe::Source::Entity);
    assert!(lexicon.entities_ready());
    lexicon.ensure_entities().unwrap();
}

/// A damaged entity table still fails closed — later, and every time, never
/// decaying into silent identity answers.
#[test]
fn a_damaged_entity_table_fails_on_every_request_not_at_open() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_damaged");
    publish_with_entities(&root, "Open AI", "OpenAI");
    let table = entity_table_path(&root);

    // Same length, different bytes: only the digest can tell the difference.
    let mut bytes = std::fs::read(&table).unwrap();
    *bytes.last_mut().unwrap() ^= 0xFF;
    write_file(&table, &bytes).unwrap();

    let lexicon = Lexicon::open(&root).unwrap();
    assert_eq!(lexicon.lemma("cats", false), "cat");
    for _ in 0..3 {
        let error = lexicon.ensure_entities().unwrap_err();
        assert!(matches!(error, Error::Integrity { .. }), "{error}");
        assert!(!lexicon.entities_ready());
    }
}

/// A truncated or missing entity table is still refused at open; only the
/// digest is deferred.
#[test]
fn a_missing_or_resized_entity_table_is_still_refused_at_open() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_resized");
    publish_with_entities(&root, "Open AI", "OpenAI");
    let table = entity_table_path(&root);

    let bytes = std::fs::read(&table).unwrap();
    write_file(&table, bytes.get(..bytes.len() - 1).unwrap()).unwrap();
    let error = Lexicon::open(&root).unwrap_err();
    assert!(matches!(error, Error::Integrity { .. }), "{error}");

    std::fs::remove_file(&table).unwrap();
    let error = Lexicon::open(&root).unwrap_err();
    assert!(matches!(error, Error::Io { .. }), "{error}");
}

/// A mention handed over whole answers exactly as its token form does,
/// deferred entity table included.
#[test]
fn a_whole_string_mention_answers_like_the_token_form_it_replaces() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_text_span");
    publish_with_entities(&root, "Open AI", "OpenAI");
    let lexicon = Lexicon::open(&root).unwrap();
    let options = crate::probe::Options::default();

    let text = lexicon
        .lemma_span_text_with_info("Open AI", false, options)
        .unwrap();
    let tokens = lexicon.lemma_span_with_info("Open AI", &["Open", "AI"], None, false, options);
    assert_eq!((&text.lemma, text.source), (&tokens.lemma, tokens.source));
    assert!(lexicon.entities_ready());

    // The gap belongs to the mention: a wider one is a different key and the
    // surface comes back with both spaces intact.
    let widened = lexicon
        .lemma_span_text_with_info("Open  AI", false, options)
        .unwrap();
    assert_eq!(widened.lemma, "Open  AI");
    assert_eq!(widened.source, crate::probe::Source::Identity);

    // One word takes the ordinary token chain, as a one-token span does.
    let single = lexicon
        .lemma_span_text_with_info("cats", false, options)
        .unwrap();
    assert_eq!(single.lemma, "cat");
    assert_eq!(single.source, crate::probe::Source::Exact);

    // Degenerate surfaces are reported, never trimmed into a different query.
    for surface in ["", "   ", " Open AI", "Open AI "] {
        assert!(
            lexicon
                .lemma_span_text_with_info(surface, false, options)
                .is_err(),
            "{surface:?}"
        );
    }
}

/// Concurrent first questions open the table once and all see the same one.
#[test]
fn concurrent_first_entity_questions_open_the_table_once() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_racing");
    publish_with_entities(&root, "Open AI", "OpenAI");
    let lexicon = Lexicon::open(&root).unwrap();

    let options = crate::probe::Options::default();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                let answer =
                    lexicon.lemma_span_with_info("Open AI", &["Open", "AI"], None, false, options);
                assert_eq!(answer.lemma, "OpenAI");
                lexicon.ensure_entities().unwrap();
            });
        }
    });
    assert!(lexicon.entities_ready());
}

#[test]
fn entity_namespaces_answer_exact_lower_and_long_spill_keys() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_entity_namespaces");
    let long_surface = format!("LONG {}", "名".repeat(100));
    publish_with_entities_and_phrases(
        &root,
        &[("Open AI", "OpenAI"), (&long_surface, "LongEntity")],
        &[],
    );
    let lexicon = Lexicon::open(&root).unwrap();

    let exact_options = Options::default().entity_lookup();
    let exact = lexicon.lemma_with_info("Open AI", false, exact_options);
    assert_eq!(exact.lemma, "OpenAI");
    assert_eq!(exact.source, crate::probe::Source::Entity);

    let lower_options = Options {
        use_entities_lower: true,
        ..Options::default()
    }
    .entity_lookup();
    let lower = lexicon.lemma_with_info("OPEN AI", false, lower_options);
    assert_eq!(lower.lemma, "OpenAI");
    assert_eq!(lower.source, crate::probe::Source::EntityLower);

    // More than 256 UTF-8 bytes exercises the heap spill without changing
    // the exact or locale-lower namespaces' lookup semantics.
    assert!(long_surface.len() > 256);
    let long_exact = lexicon.lemma_with_info(&long_surface, false, exact_options);
    assert_eq!(long_exact.lemma, "LongEntity");
    assert_eq!(long_exact.source, crate::probe::Source::Entity);
    let long_surface_mixed = long_surface.replacen("LONG ", "Long ", 1);
    let long_lower = lexicon.lemma_with_info(&long_surface_mixed, false, lower_options);
    assert_eq!(long_lower.lemma, "LongEntity");
    assert_eq!(long_lower.source, crate::probe::Source::EntityLower);
}

/// The phrase key and pattern of «Нижнего Новгорода» → «Нижний Новгород»:
/// two-character lowercase tails, then `cut`/`add` per component.
const NIZHNY_KEY: &str = "го\u{1f}да";
const NIZHNY_PATTERN: &str = "его\u{1f}ий\u{1f}а\u{1f}";

#[test]
fn phrase_patterns_answer_only_uncovered_multiword_spans() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_phrases");
    publish_with_entities_and_phrases(
        &root,
        &[("Нижнего Новгорода", "Нижний Новгород")],
        &[(NIZHNY_KEY, NIZHNY_PATTERN)],
    );
    let lexicon = Lexicon::open(&root).unwrap();
    let options = Options::default().entity_lookup();

    // The gazetteer owns the span it covers: the pattern would have produced
    // the same text here, so the *stage* is the assertion, not the string.
    let covered = lexicon.lemma_span_with_info(
        "Нижнего Новгорода",
        &["Нижнего", "Новгорода"],
        None,
        false,
        options,
    );
    assert_eq!(covered.source, crate::probe::Source::Entity);

    // Same shape, different name: nothing covers it, so the pattern answers.
    let uncovered = lexicon.lemma_span_with_info(
        "Дальнего Города",
        &["Дальнего", "Города"],
        None,
        false,
        options,
    );
    assert_eq!(uncovered.lemma, "Дальний Город");
    assert_eq!(uncovered.source, crate::probe::Source::Phrase);
    assert_eq!(uncovered.source.name(), "phrase");
    assert!(!uncovered.source.is_known());

    // A single-word span is the token chain's, whatever the table holds.
    let single = lexicon.lemma_span_with_info("Города", &["Города"], None, false, options);
    assert_ne!(single.source, crate::probe::Source::Phrase);
    assert_eq!(single.lemma, "Города");

    // The gaps come back from the caller's separators, never invented here.
    let hyphenated = lexicon.lemma_span_with_info(
        "Дальнего-Города",
        &["Дальнего", "Города"],
        Some(&["-"]),
        false,
        options,
    );
    assert_eq!(hyphenated.lemma, "Дальний-Город");
}

#[test]
fn a_phrase_pattern_that_cannot_apply_leaves_the_whole_span_alone() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_phrase_guard");
    publish_with_entities_and_phrases(
        &root,
        &[("Нижнего Новгорода", "Нижний Новгород")],
        &[
            (NIZHNY_KEY, NIZHNY_PATTERN),
            // The key matches «..ым ..ой» but the cut does not: no component
            // ends in the letters it demands.
            ("ым\u{1f}ой", "zzz\u{1f}a\u{1f}zzz\u{1f}b"),
            // A cut that swallows a whole token would erase a word.
            ("ая\u{1f}ия", "Зая\u{1f}\u{1f}\u{1f}"),
            // Fewer fields than the span has components.
            ("ор\u{1f}ыр", "ор\u{1f}"),
        ],
    );
    let lexicon = Lexicon::open(&root).unwrap();
    let options = Options::default().entity_lookup();
    for (span, tokens) in [
        ("Первым Второй", ["Первым", "Второй"]),
        ("Зая Мия", ["Зая", "Мия"]),
        ("Мотор Кефыр", ["Мотор", "Кефыр"]),
        // Same key as «Нижнего Новгорода» — the tails agree — but the stem
        // does not end in the letters that pattern cuts. One component that
        // will not take the rewrite abandons the span, not just that word.
        ("Великого Города", ["Великого", "Города"]),
    ] {
        let answer = lexicon.lemma_span_with_info(span, &tokens, None, false, options);
        assert_eq!(answer.lemma, span, "{span}");
        assert_eq!(answer.source, crate::probe::Source::Identity, "{span}");
    }
}

#[test]
#[allow(deprecated)]
fn the_measurement_override_answers_exactly_as_the_shipped_table_does() {
    let temporary = tempfile::tempdir().unwrap();
    let shipped = temporary.path().join("xx_shipped");
    let bare = temporary.path().join("xx_bare");
    publish_with_entities_and_phrases(
        &shipped,
        &[("Нижнего Новгорода", "Нижний Новгород")],
        &[(NIZHNY_KEY, NIZHNY_PATTERN)],
    );
    publish_with_entities_and_phrases(&bare, &[("Нижнего Новгорода", "Нижний Новгород")], &[]);

    let compiled = Lexicon::open(&shipped).unwrap();
    let mut overridden = Lexicon::open(&bare).unwrap();
    let options = Options::default().entity_lookup();
    let span = ("Дальнего Города", ["Дальнего", "Города"]);

    // Before the override the bare lexicon declares no table and answers with
    // the span itself; the file manifest agrees that there is no file.
    assert_eq!(overridden.meta().entries.get("phrases"), Some(&0));
    assert!(!overridden.meta().files.contains_key("phrases.inline"));
    assert_eq!(
        overridden
            .lemma_span_with_info(span.0, &span.1, None, false, options)
            .source,
        crate::probe::Source::Identity
    );

    overridden
        .set_phrase_patterns(&mut [(NIZHNY_KEY.to_owned(), NIZHNY_PATTERN.to_owned())])
        .unwrap();
    let from_table = compiled.lemma_span_with_info(span.0, &span.1, None, false, options);
    let from_override = overridden.lemma_span_with_info(span.0, &span.1, None, false, options);
    assert_eq!(from_override.lemma, from_table.lemma);
    assert_eq!(from_override.source, from_table.source);

    // An empty slice is the reference profile: the table goes away again.
    overridden.set_phrase_patterns(&mut []).unwrap();
    assert_eq!(
        overridden
            .lemma_span_with_info(span.0, &span.1, None, false, options)
            .source,
        crate::probe::Source::Identity
    );
}

#[test]
fn a_shipped_phrase_table_is_manifested_and_counted_like_every_other() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("xx_phrase_manifest");
    publish_with_entities_and_phrases(
        &root,
        &[("Нижнего Новгорода", "Нижний Новгород")],
        &[(NIZHNY_KEY, NIZHNY_PATTERN)],
    );
    let lexicon = Lexicon::open(&root).unwrap();
    assert_eq!(lexicon.meta().entries.get("phrases"), Some(&1));
    assert!(lexicon.meta().files.contains_key("phrases.inline"));
    // One encoding, fixed by the format: nothing is recorded under `codecs`.
    assert!(!lexicon.meta().codecs.contains_key("phrases"));
    // Windows refuses to truncate a file that is still mapped.
    drop(lexicon);

    // Truncating the table breaks the manifest digest, like any other file.
    let directory = resolve_generation(&root).unwrap().directory;
    write_file(&directory.join("phrases.inline"), b"").unwrap();
    assert!(matches!(Lexicon::open(&root), Err(Error::Integrity { .. })));
}

#[test]
fn a_phrase_source_row_must_be_a_unique_non_empty_key_and_pattern() {
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("rows");
    std::fs::create_dir_all(&destination).unwrap();
    let repeated = [
        ("aa\u{1f}bb".to_owned(), "x\u{1f}y\u{1f}\u{1f}".to_owned()),
        ("aa\u{1f}bb".to_owned(), "p\u{1f}q\u{1f}\u{1f}".to_owned()),
    ];
    assert!(crate::train::write_phrase_source(&repeated, &destination).is_err());
    let tabbed = [("aa\tbb".to_owned(), "x\u{1f}y".to_owned())];
    assert!(crate::train::write_phrase_source(&tabbed, &destination).is_err());
}

/// A readable generation of an older format is migrated by `compile`: only the
/// hot tables are rebuilt, every other file is carried over byte for byte, and
/// the answers do not change.
#[test]
fn an_older_compiled_generation_migrates_without_touching_its_other_tables() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("zxx_fixture");
    publish_sources(
        &root,
        &[
            pair("cats", "cat"),
            pair("котами", "кот"),
            pair("verylongwords", "verylongword"),
        ],
    );
    compile(&root).unwrap();
    let current = resolve_generation(&root).unwrap();

    // The same compiled files under a different generation name, declaring
    // the previous format: what a bundle written by the previous core is.
    let legacy_root = dir.path().join("legacy").join("zxx_fixture");
    let legacy_id = format!("g{}", "f".repeat(64));
    let legacy_dir = legacy_root.join(GENERATIONS).join(&legacy_id);
    std::fs::create_dir_all(&legacy_dir).unwrap();
    for entry in std::fs::read_dir(&current.directory).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), legacy_dir.join(entry.file_name())).unwrap();
    }
    let meta_path = legacy_dir.join("meta.json");
    let mut meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    let object = meta.as_object_mut().unwrap();
    object.insert("format_version".to_owned(), (FORMAT_VERSION - 1).into());
    object.insert("generation".to_owned(), legacy_id.clone().into());
    write_file(&meta_path, &render_meta(&meta)).unwrap();
    write_file(&legacy_root.join(CURRENT), legacy_id.as_bytes()).unwrap();

    let before = Lexicon::open(&legacy_root).unwrap();
    assert_eq!(before.meta().format_version, FORMAT_VERSION - 1);
    let compiled = compile(&legacy_root).unwrap();
    assert!(compiled.from_sources);
    assert!(compiled.tables >= 1);

    let migrated = resolve_generation(&legacy_root).unwrap();
    assert_ne!(migrated.generation.as_deref(), Some(legacy_id.as_str()));
    let after = Lexicon::open(&legacy_root).unwrap();
    assert_eq!(after.meta().format_version, FORMAT_VERSION);
    assert_eq!(
        after.meta().codecs.get("exact").map(String::as_str),
        Some("hybrid")
    );
    for token in ["cats", "котами", "verylongwords", "unknown", "Cats"] {
        assert_eq!(
            after.lemma(token, false),
            before.lemma(token, false),
            "{token}"
        );
    }
    // Carried over, not rebuilt: every other runtime file is the same bytes.
    for name in runtime_files(before.meta(), &legacy_dir).unwrap() {
        if name.starts_with("exact.") || name.starts_with("lower.") {
            continue;
        }
        assert_eq!(
            std::fs::read(legacy_dir.join(&name)).unwrap(),
            std::fs::read(migrated.directory.join(&name)).unwrap(),
            "{name}"
        );
    }
    // The migrated generation remains current: compiling again is a no-op.
    assert!(!compile(&legacy_root).unwrap().from_sources);
}
