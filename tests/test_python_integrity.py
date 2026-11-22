"""Fail-closed readers, native builder inputs and legacy journal recovery."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import TYPE_CHECKING, cast

import pytest

import udlex
from udlex import _udlex, bundle
from udlex._io import (
    INSTALL_JOURNAL,
    atomic_write_text,
    bundle_lock,
    read_bounded_regular,
    recover_bundle_transactions,
    strict_json_loads,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

_normalized_rows = cast(
    "Callable[[str], int]", vars(_udlex)["_validate_normalized_rows"]
)
_source_table_batch = cast("Callable[..., object]", vars(_udlex)["_source_table_batch"])

_script_compatibility = cast(
    "Callable[[str, list[tuple[str, str]]], bytes]",
    vars(_udlex)["_script_compatibility"],
)


_locale_lower = cast(
    "Callable[[str, list[str]], list[str]]",
    vars(_udlex)["_locale_lower"],
)


_ENGLISH_FORMS = (
    ("cars", "car"),
    ("walks", "walk"),
    ("talks", "talk"),
    ("parks", "park"),
    ("marks", "mark"),
    ("Known", "Known"),
)


def _entity_lexicon(
    destination: Path,
    forms: tuple[tuple[str, str], ...] = _ENGLISH_FORMS,
    entities: tuple[tuple[str, str], ...] = (
        ("OpenAI", "OpenAI"),
        ("New York", "New York"),
    ),
    **meta: object,
) -> udlex.Lexicon:
    """Build a source fixture with entities and seal it through the compiler."""
    tables = udlex.build(
        [(form, lemma, False) for form, lemma in forms],
        [("outside", "external")],
        list(entities),
    )
    entries = dict(tables.write_sources(str(destination), "full", (0.0, 0.0)))
    (destination / "meta.json").write_text(
        json.dumps(
            {
                "name": destination.name,
                "language": "English",
                "iso639_3": "eng",
                "source": "test",
                "training_tokens": 1,
                "variant": "full",
                "selected_on": None,
                "max_suffix": 6,
                "format_version": udlex.FORMAT_VERSION,
                "entity_encoding": udlex.ENTITY_ENCODING,
                "entries": entries,
                "traffic": {},
                "gold_tokens": None,
                "accuracy": None,
                "baseline": None,
                "beats_baseline": None,
                **meta,
            }
        ),
        encoding="utf-8",
    )
    udlex.compile_bundle(destination)
    return udlex.Lexicon(str(destination))


def test_source_is_one_exhaustive_enum() -> None:
    expected = [
        (udlex.Source.ENTITY, "entity", True, True),
        (udlex.Source.ENTITY_LOWER, "entity_lower", True, True),
        (udlex.Source.EXACT, "exact", True, True),
        (udlex.Source.LOWER, "lower", True, True),
        (udlex.Source.EXACT_LOWER, "exact_lower", True, True),
        (udlex.Source.EXTERNAL, "external", True, True),
        (udlex.Source.EXTERNAL_LOWER, "external_lower", True, True),
        (udlex.Source.RULE, "rule", False, True),
        (udlex.Source.IDENTITY, "identity", False, False),
        (udlex.Source.PROTECTED, "protected", False, False),
        (udlex.Source.COMPOSED, "composed", False, True),
        (udlex.Source.PHRASE, "phrase", False, True),
        (udlex.Source.USER, "user", True, True),
        (udlex.Source.USER_LOWER, "user_lower", True, True),
        (udlex.Source.CONTEXT, "context", True, True),
    ]

    assert issubclass(udlex.Source, Enum)
    assert list(udlex.Source) == [source for source, *_ in expected]
    for source, value, known, lemmatized in expected:
        assert source.value == value
        assert str(source) == value
        assert udlex.Source(value) is source
        assert source != value
        assert source.is_known() is known
        assert source.is_lemmatized() is lemmatized
    with pytest.raises(ValueError, match="unknown"):
        udlex.Source("unknown")


def test_rich_batch_distinguishes_every_token_fallback_class(tmp_path: Path) -> None:
    _entity_lexicon(tmp_path / "en_provenance")
    lexicon = udlex.Lexicon(str(tmp_path / "en_provenance"), use_entities_lower=True)
    tokens = [
        "cars",
        "CARS",
        "Known",
        "KNOWN",
        "outside",
        "OUTSIDE",
        "zooms",
        "§§§",
        "cars",
        "OpenAI",
        "OPENAI",
    ]
    protected = [False] * len(tokens)
    protected[8] = True
    entities = [False] * len(tokens)
    entities[9:] = [True, True]

    infos = lexicon.lemmatize(tokens, protected=protected, entities=entities, info=True)
    assert [(info.lemma, info.source) for info in infos] == [
        ("car", udlex.Source.EXACT),
        ("car", udlex.Source.EXACT_LOWER),
        ("Known", udlex.Source.EXACT),
        ("Known", udlex.Source.LOWER),
        ("external", udlex.Source.EXTERNAL),
        ("external", udlex.Source.EXTERNAL_LOWER),
        ("zoom", udlex.Source.RULE),
        ("§§§", udlex.Source.IDENTITY),
        ("cars", udlex.Source.PROTECTED),
        ("OpenAI", udlex.Source.ENTITY),
        ("OpenAI", udlex.Source.ENTITY_LOWER),
    ]
    assert [info.lemma for info in infos] == lexicon.lemmatize(
        tokens, protected=protected, entities=entities
    )
    assert repr(infos[0]) == 'LemmaInfo(lemma="car", source=Source.EXACT)'
    assert all(
        not hasattr(infos[0], field) for field in ("lemmatized", "known", "changed")
    )
    assert lexicon.lemmatize([], info=True) == []

    for name in ("sentence_initial", "protected", "entities"):
        masks: dict[str, Any] = {name: [False]}
        with pytest.raises(ValueError, match=rf"{name} has 1 items, expected 2"):
            lexicon.lemmatize(["cars", "outside"], **masks, info=True)


def test_entity_protection_provenance_and_batch_masks(tmp_path: Path) -> None:
    lexicon = _entity_lexicon(tmp_path / "en_test")
    lower_entities = udlex.Lexicon(str(tmp_path / "en_test"), use_entities_lower=True)

    assert lexicon.lemma("OpenAI", info=True).source is udlex.Source.IDENTITY
    entity = lexicon.lemma("OpenAI", entity=True, info=True)
    assert (entity.lemma, entity.source) == ("OpenAI", udlex.Source.ENTITY)
    assert lexicon.lemma("OPENAI", entity=True) == "OPENAI"
    ordinary_lower = lower_entities.lemma("OPENAI", info=True)
    assert (ordinary_lower.lemma, ordinary_lower.source) == (
        "OPENAI",
        udlex.Source.IDENTITY,
    )
    assert lower_entities.lemma("OPENAI", entity=True) == "OpenAI"
    assert lower_entities.knows("OPENAI", entity=True)
    assert lexicon.lemma("cars", protected=True) == "cars"
    assert (
        lexicon.lemma("cars", protected=True, info=True).source
        is udlex.Source.PROTECTED
    )
    assert lower_entities.lemmatize(
        ["cars", "OPENAI"], entities=[False, True], protected=[False, False]
    ) == ["car", "OpenAI"]
    assert lower_entities.lemmatize(["OPENAI", "OPENAI"], entities=[False, True]) == [
        "OPENAI",
        "OpenAI",
    ]
    with pytest.raises(ValueError, match="entities has 1 items, expected 2"):
        lexicon.lemmatize(["cars", "OpenAI"], entities=[True])
    with pytest.raises(AttributeError):
        cast("Any", entity).lemma = "changed"


def test_span_lookup_probes_whole_span_then_token_chain_then_identity(
    tmp_path: Path,
) -> None:
    lexicon = _entity_lexicon(tmp_path / "en_span")

    span = lexicon.lemma_span(["New", "York"], info=True)
    assert (span.lemma, span.source) == ("New York", udlex.Source.ENTITY)
    assert lexicon.lemma_span(["went", "to", "New", "York"], 2, 4) == "New York"

    # The lowercased key needs the same opt-in as `entity=True`.
    lower_entities = udlex.Lexicon(str(tmp_path / "en_span"), use_entities_lower=True)
    assert lower_entities.lemma_span(["NEW", "YORK"]) == "New York"
    assert lexicon.lemma_span(["NEW", "YORK"]) == "NEW YORK"

    # One word falls to the token chain; an unknown multiword span is verbatim.
    single = lexicon.lemma_span(["cars"], info=True)
    assert (single.lemma, single.source) == ("car", udlex.Source.EXACT)
    unknown = lexicon.lemma_span(["Instituto", "Cervantes"], info=True)
    assert (unknown.lemma, unknown.source) == (
        "Instituto Cervantes",
        udlex.Source.IDENTITY,
    )
    assert (
        lexicon.lemma_span(["Rimskij", "Korsakov"], separators=["-"])
        == "Rimskij-Korsakov"
    )
    # A separator changes the probed key.
    assert lexicon.lemma_span(["New", "York"], separators=["  "]) == "New  York"

    guarded = lexicon.lemma_span(["New", "York"], protected=True, info=True)
    assert (guarded.lemma, guarded.source) == ("New York", udlex.Source.PROTECTED)

    with pytest.raises(ValueError, match="empty or out of bounds"):
        lexicon.lemma_span(["New", "York"], 1, 1)
    with pytest.raises(ValueError, match="empty or out of bounds"):
        lexicon.lemma_span(["New", "York"], 0, 3)
    with pytest.raises(ValueError, match="separators has 2 items, expected 1"):
        lexicon.lemma_span(["New", "York"], separators=[" ", " "])
    with pytest.raises(ValueError, match="span surface is empty"):
        lexicon.lemma_span([""])


def test_span_conventions_come_from_metadata_and_overrides_are_gated(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "tr_conventions"
    lexicon = _entity_lexicon(
        destination,
        forms=(("cars", "car"),),
        entities=(("CNN'de", "cnn"), ("Ankara'da", "ankara")),
        language="Turkish",
        iso639_3="tur",
        span_case="verbatim",
        span_multiword="compose",
    )

    # Stamped conventions: verbatim case, and a missed span composed per token.
    verbatim = lexicon.lemma_span(["CNN'de"], info=True)
    assert (verbatim.lemma, verbatim.source) == ("cnn", udlex.Source.ENTITY)
    composed = lexicon.lemma_span(["CNN'de", "Ankara'da"], info=True)
    assert (composed.lemma, composed.source) == ("cnn ankara", udlex.Source.COMPOSED)

    defaults = udlex.Lexicon(
        str(destination), span_case="transfer", span_multiword="identity"
    )
    assert defaults.lemma_span(["CNN'de"]) == "Cnn"
    assert defaults.lemma_span(["CNN'de", "Ankara'da"]) == "CNN'de Ankara'da"

    with pytest.raises(ValueError, match="span_case"):
        udlex.Lexicon(str(destination), span_case="bogus")
    with pytest.raises(ValueError, match="span_multiword"):
        udlex.Lexicon(str(destination), span_multiword="bogus")


def test_strict_json_rejects_duplicates_and_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_json_loads('{"outer":{"x":1,"x":2}}', "control.json")
    # `1e400` is well-formed JSON that silently decodes to infinity.
    for payload in ('{"value":NaN}', '{"p": 1e400}', '{"p": -1e400}'):
        with pytest.raises(ValueError, match="non-finite JSON number"):
            strict_json_loads(payload, "control.json")
    assert strict_json_loads('{"p": 1e-400}', "underflow") == {"p": 0.0}


def test_bounded_control_file_reader_rejects_links_and_oversize(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(OSError, match="unsafe"):
        read_bounded_regular(linked, 1024, description="test manifest")
    with pytest.raises(OSError, match="oversized"):
        read_bounded_regular(target, 1, description="test manifest")


def test_bundle_lock_rejects_symlink_targets(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    target = tmp_path / "victim"
    target.write_text("do not touch", encoding="utf-8")
    (directory / ".udlex.lock").symlink_to(target)
    with pytest.raises(OSError, match="unsafe bundle lock"), bundle_lock(directory):
        pass
    assert target.read_text(encoding="utf-8") == "do not touch"

    linked = tmp_path / "linked"
    linked.symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError, match="symlink bundle directory"), bundle_lock(linked):
        pass


def test_generation_resolver_rejects_whitespace_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "lexicon"
    generation = root / ".generations" / "good"
    generation.mkdir(parents=True)
    (root / "current").write_text(" good\n", encoding="utf-8")
    with pytest.raises(bundle.BundleMetadataError, match="invalid generation"):
        bundle.active_directory(root)

    (root / "current").unlink()
    (root / "current").symlink_to(tmp_path / "pointer")
    with pytest.raises(bundle.BundleMetadataError, match="regular text file"):
        bundle.active_directory(root)

    (root / "current").unlink()
    (root / "current").write_text("linked\n", encoding="utf-8")
    (root / ".generations" / "linked").symlink_to(generation, target_is_directory=True)
    with pytest.raises(bundle.BundleMetadataError, match="real directory"):
        bundle.active_directory(root)


_MEASURED = {
    "gold_tokens": 10,
    "accuracy": 0.8,
    "baseline": 0.7,
    "beats_baseline": True,
}


@pytest.fixture
def active_meta(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> tuple[Path, Path, dict[str, Any]]:
    """A compiled lexicon, its active `meta.json` and that file's payload."""
    lexicon = tmp_path / "xx_meta"
    write_lexicon(
        lexicon,
        name="xx_meta",
        iso639_3="xxa",
        language="Metaish",
        forms={"cats": "cat"},
    )
    path = bundle.active_directory(lexicon) / "meta.json"
    return lexicon, path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutate", "field", "expected"),
    [
        (lambda meta: meta.update(selected_on="train-cv"), "selected_on", "train-cv"),
        (lambda meta: meta.pop("variant"), "variant", "full"),
        (lambda meta: meta.update(_MEASURED), "beats_baseline", True),
    ],
    ids=["train-cv-selection", "rust-default-variant", "measured"],
)
def test_active_metadata_accepts(
    active_meta: tuple[Path, Path, dict[str, Any]],
    mutate: Callable[[dict[str, Any]], object],
    field: str,
    expected: object,
) -> None:
    lexicon, path, payload = active_meta
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert getattr(bundle.read_meta(lexicon), field) == expected


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda meta: meta.update(selected_on="test"), "train-cv"),
        (lambda meta: meta.update(source=""), "non-empty string"),
        (
            lambda meta: meta["entries"].update(exact=2**64 - 1, lower=1),
            r"aggregate.*u64",
        ),
        (lambda meta: meta.update(_MEASURED, gold_tokens=0), "positive u64"),
        (lambda meta: meta.update(_MEASURED, beats_baseline=None), "present exactly"),
        (lambda meta: meta.update(beats_baseline=False), "present exactly"),
        (lambda meta: meta.update(_MEASURED, beats_baseline=False), "contradicts"),
        (lambda meta: meta.pop("entity_encoding"), r"entity_encoding.*must be rebuilt"),
        (
            lambda meta: meta.update(entity_encoding="legacy-unprefixed-v0"),
            r"entity_encoding.*must be rebuilt",
        ),
    ],
    ids=[
        "test-selection",
        "empty-source",
        "u64-aggregate",
        "zero-gold-tokens",
        "half-measured",
        "unmeasured-verdict",
        "contradicted-verdict",
        "no-entity-encoding",
        "legacy-entity-encoding",
    ],
)
def test_active_metadata_rejects(
    active_meta: tuple[Path, Path, dict[str, Any]],
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    lexicon, path, payload = active_meta
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(bundle.BundleMetadataError, match=message):
        bundle.read_meta(lexicon)


def _corpus(path: Path, tokens: list[str], lemmas: list[str], upos: list[str]) -> Path:
    path.write_text(
        json.dumps({"tokens": tokens, "lemmas": lemmas, "upos": upos}) + "\n",
        encoding="utf-8",
    )
    return path


def test_no_dev_build_uses_form_disjoint_train_cv(tmp_path: Path) -> None:
    training = _corpus(
        tmp_path / "train.jsonl",
        [f"stem{index:03d}ies" for index in range(300)],
        [f"stem{index:03d}y" for index in range(300)],
        ["NOUN"] * 300,
    )
    destination = tmp_path / "xx_cv"
    outcome = _udlex.build_lexicon(
        str(destination),
        train=str(training),
        dev=str(tmp_path / "absent.dev.jsonl"),
        test=str(tmp_path / "absent.test.jsonl"),
        paradigms=[],
        name="xx_cv",
        language="Crossvalidationish",
        iso639_3="xxa",
    )

    assert outcome is not None
    assert outcome.selected_on == "train-cv"
    assert not outcome.tuned_on_dev
    assert outcome.observed_inputs == [
        (
            "corpus",
            "xx_cv.train",
            training.stat().st_size,
            hashlib.sha256(training.read_bytes()).hexdigest(),
        )
    ]
    assert bundle.read_meta(destination).selected_on == "train-cv"
    assert udlex.Lexicon(str(destination)).lemma("stem042ies") == "stem042y"


def test_outcome_selection_mirrors_the_receipt_the_generation_stores(
    tmp_path: Path,
) -> None:
    """`Outcome.selection` equals the stored receipt: hits, NER hits, rivalry."""
    training = _corpus(
        tmp_path / "train.jsonl",
        [".", "walked", "talked", "jumped", "looked"],
        [".", "walk", "talk", "jump", "look"],
        ["PUNCT", "VERB", "VERB", "VERB", "VERB"],
    )
    dev = _corpus(
        tmp_path / "dev.jsonl",
        [".", *(["helped"] * 60)],
        [".", *(["help"] * 60)],
        ["X"] * 61,
    )
    destination = tmp_path / "xx_receipt"
    outcome = _udlex.build_lexicon(
        str(destination),
        train=str(training),
        dev=str(dev),
        test=None,
        paradigms=[],
        name="xx_receipt",
        language="Receiptish",
        iso639_3="xxa",
    )

    assert outcome.variant == "rules"
    receipt = cast("dict[str, Any]", outcome.selection)
    assert receipt is not None
    generation = (destination / "current").read_text(encoding="utf-8").strip()
    meta = json.loads(
        (destination / ".generations" / generation / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt == meta["selection"]
    rules_row = receipt["candidates"]["rules"]
    assert rules_row["hits"] == 61
    assert rules_row["discordant"] == 60
    assert rules_row["ner_hits"] == 1
    assert rules_row["rivals"] == {
        "identity": 60,
        "corpus": 60,
        "rules": 0,
        "lookup": 60,
        "full": 0,
    }
    # The extended block does not break opening the bundle.
    assert bundle.read_meta(destination).variant == "rules"
    assert udlex.Lexicon(str(destination)).lemma("helped") == "help"


def test_native_builder_checks_manifest_hash_against_consumed_descriptor(
    tmp_path: Path,
) -> None:
    table = tmp_path / "xxx.tsv"
    table.write_text("cats\tcat\n", encoding="utf-8")
    destination = tmp_path / "xx_integrity"

    with pytest.raises(ValueError, match="observed input"):
        _udlex.build_lexicon(
            str(destination),
            train=None,
            dev=None,
            test=None,
            paradigms=[("authenticated", str(table), False, True)],
            name="xx_integrity",
            language="Integrityish",
            iso639_3="xxa",
            expected_inputs=[(str(table), table.stat().st_size, "0" * 64)],
        )

    assert not (destination / "current").exists()


def test_imported_tables_allow_zwnj_but_reject_controls(tmp_path: Path) -> None:
    table = tmp_path / "fas.tsv"
    table.write_text("می\u200cروم\tرفتن\n", encoding="utf-8")
    assert _normalized_rows(str(table)) == 1

    table.write_bytes(b"bad\x00form\tlemma\n")
    with pytest.raises(ValueError, match="control character"):
        _normalized_rows(str(table))


def test_imported_table_refusals_keep_their_wording_and_line_numbers(
    tmp_path: Path,
) -> None:
    """The refusal wording and row numbers are a contract; a lone CR ends a row."""
    table = tmp_path / "xxa.tsv"
    cases = [
        (b"cats\tcat", 1, "final row must end with a newline"),
        (b"ants\tant\ncats\tcat", 2, "final row must end with a newline"),
        (b"ants\tant\rcats\tcat\n", 1, "final row must end with a newline"),
        (b"cats\n", 1, "expected non-empty form<TAB>lemma"),
        (b"cats\t\n", 1, "expected non-empty form<TAB>lemma"),
        (b"cats\tcat\tNOUN\n", 1, "expected non-empty form<TAB>lemma"),
        ("cafés\tcafé\n".encode(), 1, "table text must be NFC-normalized"),
        (b"cats\tca\x7ft\n", 1, "table contains a control character"),
        (b"ants\tant\nants\tant\n", 2, "forms must be strictly sorted and unique"),
        (b"cats\tcat\nants\tant\n", 2, "forms must be strictly sorted and unique"),
        (b"cats\tca\xfft\n", 1, "table is not valid UTF-8"),
    ]
    for payload, line, message in cases:
        table.write_bytes(payload)
        expected = re.escape(f"{table}:{line}: {message}")
        with pytest.raises(ValueError, match=expected):
            _normalized_rows(str(table))

    # CRLF terminates a row, ordering is by form only, an empty table is valid.
    table.write_bytes(b"ants\tzebra\r\ncats\tcat\n")
    assert _normalized_rows(str(table)) == 2
    table.write_bytes(b"")
    assert _normalized_rows(str(table)) == 0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (b"ca\xfft\tcat\n", "invalid UTF-8"),
        (b"ca\x01t\tcat\n", "TSV row contains a control character"),
        (b"cats\rdogs\tdog\n", "TSV row contains a control character"),
        (b"lonely\n", "expected at least two tab-separated columns"),
        (b"#note\n", "expected at least two tab-separated columns"),
        (b"# note\x01\n", "TSV row contains a control character"),
        (b" \tcat\n", "the first two columns must not be whitespace-only"),
        ("cats\t \n".encode(), "the first two columns must not be whitespace-only"),
    ],
)
def test_native_source_table_rejects_malformed_bytes(row: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"fixture.tsv:2: {message}")):
        _source_table_batch(
            b"cats\tcat\n" + row,
            0,
            "fixture.tsv",
            1,
            "eng",
            None,
            lemma_first=False,
            normalized_unique=False,
            require_newline=False,
            at_eof=True,
            max_line_bytes=64,
            max_batch_rows=16,
            max_batch_bytes=128,
        )


def test_native_script_compatibility_batch_matches_builder_policy() -> None:
    assert _script_compatibility(
        "ara",
        [
            ("cats", "cat"),
            ("كَتَبُوا!", "كَتَب"),
            ("għoxorَيْ", "عشر"),
            ("Latinعربي", "Latinعربي"),
            ("123", "456"),
            ("\u0378a", "\u0378b"),
            ("", "x"),
        ],
    ) == bytes([1, 1, 0, 1, 0, 0, 0])
    assert _script_compatibility(
        "jpn",
        [("食べた", "食べる"), ("カタカナ", "かたかな"), ("Latin", "食")],
    ) == bytes([1, 1, 0])
    assert _script_compatibility("eng", [("カタカナ", "かたかな")]) == b"\x00"


def test_native_script_compatibility_batch_is_bounded() -> None:
    with pytest.raises(ValueError, match="three lowercase ASCII"):
        _script_compatibility("AR", [])
    with pytest.raises(ValueError, match="maximum is 4096"):
        _script_compatibility("ara", cast("Any", [object()] * 4097))
    with pytest.raises(ValueError, match="16777216 UTF-8 bytes"):
        _script_compatibility(
            "ara",
            cast("Any", [("a" * (16 * 1024 * 1024 + 1), ""), object()]),
        )


def test_native_locale_lower_matches_runtime_casing_and_is_bounded() -> None:
    assert _locale_lower("tur", ["İSTANBUL", "IRMAK"]) == ["istanbul", "ırmak"]
    assert _locale_lower("lit", ["I\u0301"])[0] == "i\u0307\u0301"
    with pytest.raises(ValueError, match="three lowercase ASCII"):
        _locale_lower("TR", [])
    with pytest.raises(ValueError, match="maximum is 4096"):
        _locale_lower("tur", cast("Any", ["a"] * 4097))
    with pytest.raises(ValueError, match="16777216 UTF-8 bytes"):
        _locale_lower("tur", ["a" * (16 * 1024 * 1024 + 1)])


_TRANSACTION = "f" * 32


def _install_journal(
    destination: Path, target: str, *, backup_of: str | None = None, existed: bool
) -> None:
    """Leave a `prepared` install journal for one target, as a crash would."""
    atomic_write_text(
        destination / INSTALL_JOURNAL,
        json.dumps(
            {
                "schema": "udlex-install/v1",
                "phase": "prepared",
                "manifest_existed": existed,
                "manifest_backup": f".udlex-manifest-backup-{_TRANSACTION}",
                "targets": [
                    {
                        "name": target,
                        "backup": f".udlex-config-backup-{_TRANSACTION}-"
                        f"{backup_of or target}",
                        "existed": existed,
                    }
                ],
            }
        ),
    )


def test_prepared_crash_journal_recovers_before_catalogue_read(
    tmp_path: Path, write_lexicon: Callable[..., None]
) -> None:
    destination = tmp_path / "bundle"
    original = destination / "xx_one"
    write_lexicon(
        original,
        name="xx_one",
        iso639_3="xxa",
        language="Old",
        forms={"cats": "cat"},
    )
    generation = (original / "current").read_text(encoding="utf-8").strip()
    old_manifest = (
        json.dumps(
            {
                "format_version": udlex.FORMAT_VERSION,
                "lexicons": [
                    {"iso639_3": "xxa", "config": "xx_one", "generation": generation}
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    atomic_write_text(destination / "manifest.json", old_manifest)
    atomic_write_text(
        destination / f".udlex-manifest-backup-{_TRANSACTION}", old_manifest
    )
    original.replace(destination / f".udlex-config-backup-{_TRANSACTION}-xx_one")
    atomic_write_text(destination / "manifest.json", "{}\n")
    _install_journal(destination, "xx_one", existed=True)

    recover_bundle_transactions(destination)

    assert udlex.load(destination, "xxa").lemma("cats") == "cat"
    assert (destination / "manifest.json").read_text() == old_manifest
    assert not (destination / INSTALL_JOURNAL).exists()


@pytest.mark.parametrize(
    ("target", "backup_of", "message"),
    [
        ("manifest.json", None, "target"),
        (".udlex.lock", None, "target"),
        ("../escape", None, "target"),
        ("xx_one", "yy_other", "backup"),
    ],
)
def test_install_recovery_rejects_unsafe_or_unbound_targets(
    tmp_path: Path, target: str, backup_of: str | None, message: str
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    _install_journal(destination, target, backup_of=backup_of, existed=False)
    with pytest.raises(OSError, match=message):
        recover_bundle_transactions(destination)


@pytest.mark.parametrize("with_backup", [False, True])
def test_install_recovery_rejects_linked_manifest_parent_before_any_mutation(
    tmp_path: Path, with_backup: bool
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "udlex_dict_aaa.json"
    victim.write_text("outside data")
    (destination / ".manifests").symlink_to(outside, target_is_directory=True)
    earlier = destination / "aa_one"
    earlier.mkdir()
    (earlier / "keep").write_text("earlier target must survive")
    backup_name = f".udlex-wheel-backup-{_TRANSACTION}-udlex_dict_aaa.json"
    backup = destination / backup_name
    if with_backup:
        backup.write_text("replacement")
    journal = destination / INSTALL_JOURNAL
    atomic_write_text(
        journal,
        json.dumps(
            {
                "schema": "udlex-install/v2",
                "phase": "prepared",
                "manifest_existed": False,
                "manifest_backup": f".udlex-manifest-backup-{_TRANSACTION}",
                "targets": [
                    {
                        "name": "aa_one",
                        "backup": f".udlex-config-backup-{_TRANSACTION}-aa_one",
                        "existed": False,
                    },
                    {
                        "name": ".manifests/udlex_dict_aaa.json",
                        "backup": backup_name,
                        "existed": with_backup,
                    },
                ],
            }
        ),
    )

    with pytest.raises(OSError, match="unsafe install transaction parent"):
        udlex.supported_languages(destination)

    assert victim.read_text() == "outside data"
    assert (earlier / "keep").read_text() == "earlier target must survive"
    assert journal.is_file()
    if with_backup:
        assert backup.read_text() == "replacement"
