use indexmap::{IndexMap, IndexSet};

use crate::rule::{CaseAction, Rule};

use super::{Tables, Thresholds, Variant};

/// Assigns a stable identifier to every distinct string.
///
/// Identifiers follow first use, and first use follows the order the tables
/// are walked — which is why [`write_sources`] walks them in a fixed order and
/// the maps preserve insertion. Change either and every string identifier in
/// every shipped bundle moves.
#[derive(Default)]
struct StringTable {
    items: IndexSet<String>,
}

impl StringTable {
    fn intern(&mut self, value: &str) -> u64 {
        if let Some(found) = self.items.get_index_of(value) {
            return found as u64;
        }
        let id = self.items.len() as u64;
        self.items.insert(value.to_owned());
        id
    }
}

/// Packs a rule into the unsigned 64-bit value the rule table stores.
///
/// | bits  | meaning                                     |
/// |-------|---------------------------------------------|
/// | 56–63 | characters to cut from the lowercased form |
/// | 48–55 | case action, or 255 for the stop marker     |
/// | 0–47  | identifier of the string to append          |
pub(super) fn pack(cut: usize, case: CaseAction, add: u64) -> std::io::Result<u64> {
    Rule::pack(cut, case, add).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "rule cannot be represented: cut {cut} must fit in 8 bits and string identifier {add} in 48 bits"
            ),
        )
    })
}

/// Refuses bytes that delimit the source formats or cannot be trie keys.
fn validate_key(key: &str) -> std::io::Result<()> {
    if key.is_empty() || key.contains(['\0', '\t', '\n', '\r']) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "invalid table key {key:?}: keys must be non-empty and contain no NUL/TAB/LF/CR"
            ),
        ));
    }
    Ok(())
}

/// Refuses bytes that would split the newline-delimited string table.
fn validate_string(value: &str) -> std::io::Result<()> {
    if value.contains(['\0', '\n', '\r']) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid string-table value {value:?}: values may contain no NUL/LF/CR"),
        ));
    }
    Ok(())
}

/// Writes `key <TAB> value` rows sorted by key, as a trie builder needs.
///
/// Sorted as Rust orders `String`, which for valid UTF-8 is byte order — and
/// byte order over UTF-8 is code-point order, the same sequence Python's
/// `sorted` produced.
fn write_sorted(path: &std::path::Path, mut rows: Vec<(&str, u64)>) -> std::io::Result<usize> {
    use std::io::Write as _;
    rows.sort_unstable();
    for (key, _) in &rows {
        validate_key(key)?;
    }
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "destination has no parent",
        )
    })?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    let mut out = std::io::BufWriter::new(temporary);
    for (key, value) in &rows {
        writeln!(out, "{key}\t{value}")?;
    }
    out.flush()?;
    let temporary = out
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(rows.len())
}

/// Writes the two entity namespaces without allocating a prefixed key for
/// every row. `0` sorts before `1`, so sorting each namespace separately and
/// writing them in that order is identical to sorting the materialised keys.
fn write_entity_sources(
    path: &std::path::Path,
    mut exact: Vec<(&str, u64)>,
    mut lower: Vec<(&str, u64)>,
) -> std::io::Result<usize> {
    use std::io::Write as _;

    exact.sort_unstable();
    lower.sort_unstable();
    for (key, _) in exact.iter().chain(&lower) {
        validate_key(key)?;
    }

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "destination has no parent",
        )
    })?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    let mut out = std::io::BufWriter::new(temporary);
    for (prefix, rows) in [('0', &exact), ('1', &lower)] {
        for (key, value) in rows {
            writeln!(out, "{prefix}{key}\t{value}")?;
        }
    }
    out.flush()?;
    let temporary = out
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(exact.len().saturating_add(lower.len()))
}

/// Writes the four tables and the string table a compiler reads.
///
/// The confidence gate is applied here rather than at read time: a rule that
/// does not clear its threshold is written as a stop marker, so the shipped
/// bundle needs no floating-point comparison at all.
///
/// # Errors
///
/// Returns an error when the destination cannot be created or written.
pub fn write_sources(
    tables: &Tables,
    variant: Variant,
    thresholds: Thresholds,
    destination: &std::path::Path,
) -> std::io::Result<Vec<(String, usize)>> {
    if !thresholds.is_valid() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "thresholds must be finite numbers in the inclusive range 0..=2",
        ));
    }
    std::fs::create_dir_all(destination)?;
    ensure_unpublished_destination(destination)?;
    let mut strings = StringTable::default();
    let mut entries = write_lookup_sources(tables, variant, destination, &mut strings)?;
    entries.push((
        "rules".to_owned(),
        write_rule_source(tables, variant, thresholds, destination, &mut strings)?,
    ));
    entries.push((
        "strings".to_owned(),
        write_string_source(destination, &strings)?,
    ));
    Ok(entries)
}

fn ensure_unpublished_destination(destination: &std::path::Path) -> std::io::Result<()> {
    let compiled_names = [
        "current",
        "strings.idx",
        "exact.da",
        "exact.map",
        "exact.inline",
        "exact.hybrid",
        "lower.da",
        "lower.map",
        "lower.inline",
        "lower.hybrid",
        "external.da",
        "external.map",
        "external.inline",
        "external.hybrid",
        "entities.da",
        "entities.map",
        "entities.inline",
        "entities.hybrid",
        "phrases.inline",
        "rules.da",
        "rules.values",
    ];
    if compiled_names
        .iter()
        .any(|name| destination.join(name).exists())
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "refusing to overwrite a compiled or published lexicon; write a fresh generation",
        ));
    }
    Ok(())
}

/// Writes the phrase-pattern source a build was handed, sorted by key; the
/// patterns come compiled from offline and never touch the string table.
///
/// # Errors
///
/// Returns an error when a key or pattern carries a delimiter, or the file
/// cannot be written.
pub fn write_phrase_source(
    rows: &[(String, String)],
    destination: &std::path::Path,
) -> std::io::Result<usize> {
    use std::io::Write as _;

    let mut sorted: Vec<(&str, &str)> = rows
        .iter()
        .map(|(key, pattern)| (key.as_str(), pattern.as_str()))
        .collect();
    sorted.sort_unstable();
    for (key, pattern) in &sorted {
        validate_key(key)?;
        validate_key(pattern)?;
    }
    if sorted.windows(2).any(|pair| match pair {
        [left, right] => left.0 == right.0,
        _ => false,
    }) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "phrase source repeats a span key",
        ));
    }
    let path = destination.join("phrases.tsv");
    let temporary = tempfile::NamedTempFile::new_in(destination)?;
    let mut out = std::io::BufWriter::new(temporary);
    for (key, pattern) in &sorted {
        writeln!(out, "{key}\t{pattern}")?;
    }
    out.flush()?;
    out.into_inner()
        .map_err(std::io::IntoInnerError::into_error)?
        .persist(&path)
        .map_err(|error| error.error)?;
    Ok(sorted.len())
}

fn write_lookup_sources(
    tables: &Tables,
    variant: Variant,
    destination: &std::path::Path,
    strings: &mut StringTable,
) -> std::io::Result<Vec<(String, usize)>> {
    let mut entries = Vec::with_capacity(5);

    let empty = IndexMap::new();
    let corpus = variant.keeps_corpus();
    for (name, table) in [
        ("exact", if corpus { &tables.exact } else { &empty }),
        ("lower", if corpus { &tables.lower } else { &empty }),
        (
            "external",
            if variant.keeps_external() {
                &tables.external
            } else {
                &empty
            },
        ),
    ] {
        let mut rows: Vec<(&str, u64)> = Vec::with_capacity(table.len());
        for (form, lemma) in table {
            validate_key(form)?;
            validate_string(lemma)?;
            rows.push((form, strings.intern(lemma)));
        }
        entries.push((
            name.to_owned(),
            write_sorted(&destination.join(format!("{name}.tsv")), rows)?,
        ));
    }
    let mut entity_rows = Vec::with_capacity(tables.entities.len());
    for (form, lemma) in &tables.entities {
        validate_string(lemma)?;
        entity_rows.push((form.as_str(), strings.intern(lemma)));
    }
    let mut entity_lower_rows = Vec::with_capacity(tables.entities_lower.len());
    for (form, lemma) in &tables.entities_lower {
        validate_string(lemma)?;
        entity_lower_rows.push((form.as_str(), strings.intern(lemma)));
    }
    let entity_count = write_entity_sources(
        &destination.join("entities.tsv"),
        entity_rows,
        entity_lower_rows,
    )?;
    entries.push(("entities".to_owned(), entity_count));
    Ok(entries)
}

fn write_rule_source(
    tables: &Tables,
    variant: Variant,
    thresholds: Thresholds,
    destination: &std::path::Path,
    strings: &mut StringTable,
) -> std::io::Result<usize> {
    let no_rules = IndexMap::new();
    let rules = if variant.keeps_rules() {
        &tables.rules
    } else {
        &no_rules
    };
    let mut rows: Vec<(&str, u64)> = Vec::with_capacity(rules.len());
    for (key, rule) in rules {
        validate_key(key)?;
        validate_string(&rule.add)?;
        if !rule.confidence.is_finite() || !(0.0..=1.0).contains(&rule.confidence) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("rule {key:?} has invalid confidence {}", rule.confidence),
            ));
        }
        let bucket = key.as_bytes().first().copied().unwrap_or(b'0');
        let packed = if rule.confidence >= thresholds.for_bucket(bucket) {
            pack(rule.cut, rule.case, strings.intern(&rule.add))?
        } else {
            pack(0, CaseAction::Stop, strings.intern(""))?
        };
        rows.push((key, packed));
    }
    write_sorted(&destination.join("rules.tsv"), rows)
}

fn write_string_source(
    destination: &std::path::Path,
    strings: &StringTable,
) -> std::io::Result<usize> {
    let string_path = destination.join("strings.txt");
    let mut temporary = tempfile::NamedTempFile::new_in(destination)?;
    {
        use std::io::Write as _;
        for item in &strings.items {
            validate_string(item)?;
            writeln!(temporary, "{item}")?;
        }
        temporary.as_file_mut().sync_all()?;
    }
    temporary
        .persist(&string_path)
        .map_err(|error| error.error)?;
    Ok(strings.items.len())
}
