//! Cross-file count and identifier validation for parsed tables.

use super::{
    Entities, Error, Meta, Path, Result, Rule, Strings, Table, Tables, VALUE_WIDTH, da_leaf_values,
    max_output, table_codec,
};

#[derive(Clone, Copy, Default)]
struct TableStats {
    entries: usize,
    invalid_id: Option<u64>,
}

fn table_stats(table: Option<&Table>, strings: usize) -> TableStats {
    let Some(table) = table else {
        return TableStats::default();
    };
    let valid = |value: u64| usize::try_from(value).is_ok_and(|value| value < strings);
    match table {
        Table::Da(trie) => {
            let mut stats = TableStats::default();
            for value in da_leaf_values(&trie.0) {
                stats.entries = stats.entries.saturating_add(1);
                if stats.invalid_id.is_none() && !valid(u64::from(value)) {
                    stats.invalid_id = Some(u64::from(value));
                }
            }
            stats
        }
        Table::Map(map) => {
            let largest = max_output(map.fst().as_fst());
            TableStats {
                entries: map.fst().len(),
                invalid_id: (!valid(largest)).then_some(largest),
            }
        }
        Table::Inline(fsa) => TableStats {
            entries: fsa.len(),
            invalid_id: None,
        },
        Table::Hybrid(table) => TableStats {
            entries: table.len(),
            invalid_id: da_leaf_values(table.units())
                .map(u64::from)
                .find(|&value| !valid(value)),
        },
    }
}

fn validate_lookup_tables(
    directory: &Path,
    tables: &Tables,
    strings_len: usize,
    meta: &Meta,
) -> Result<()> {
    for (name, table) in [
        ("exact", &tables.exact),
        ("lower", &tables.lower),
        ("external", &tables.external),
    ] {
        let declared = meta.entries.get(name).copied().unwrap_or(0);
        let stats = table_stats(table.as_ref(), strings_len);
        if u64::try_from(stats.entries).ok() != Some(declared) {
            return Err(Error::Invalid {
                path: directory.join(name),
                message: format!(
                    "contains {} entries, metadata declares {declared}",
                    stats.entries
                ),
            });
        }
        if table.is_some() {
            let codec = table_codec(&meta.codecs, directory, name)?;
            if let Some(value) = stats.invalid_id {
                return Err(Error::Invalid {
                    path: codec.file(directory, name),
                    message: format!("string identifier {value} is out of range 0..{strings_len}"),
                });
            }
        }
    }
    Ok(())
}

fn validate_entity_count(directory: &Path, entities: &Entities, meta: &Meta) -> Result<()> {
    if !entities.is_ready() {
        return Ok(());
    }
    let declared = meta.entries.get("entities").copied().unwrap_or(0);
    let observed = entities.for_probe().map_or(0, Table::len);
    if u64::try_from(observed).ok() != Some(declared) {
        return Err(Error::Invalid {
            path: directory.join("entities"),
            message: format!("contains {observed} entries, metadata declares {declared}"),
        });
    }
    Ok(())
}

fn validate_phrase_count(directory: &Path, phrases: Option<&Table>, meta: &Meta) -> Result<()> {
    let declared = meta.entries.get("phrases").copied().unwrap_or(0);
    let observed = phrases.map_or(0, Table::len);
    if u64::try_from(observed).ok() != Some(declared) {
        return Err(Error::Invalid {
            path: directory.join("phrases"),
            message: format!("contains {observed} entries, metadata declares {declared}"),
        });
    }
    Ok(())
}

fn validate_rules(
    directory: &Path,
    tables: &Tables,
    strings_len: usize,
    meta: &Meta,
) -> Result<()> {
    let declared = meta.entries.get("rules").copied().unwrap_or(0);
    let mut observed = 0_usize;
    let mut invalid_index = None;
    if let Some(rules) = &tables.rules {
        for index in da_leaf_values(&rules.0) {
            observed = observed.saturating_add(1);
            if invalid_index.is_none() && u64::from(index) >= declared {
                invalid_index = Some(index);
            }
        }
    }
    if u64::try_from(observed).ok() != Some(declared) {
        return Err(Error::Invalid {
            path: directory.join("rules.da"),
            message: format!("contains {observed} entries, metadata declares {declared}"),
        });
    }
    let (Some(_), Some(values)) = (&tables.rules, &tables.rule_values) else {
        if tables.rules.is_none() && tables.rule_values.is_none() && declared == 0 {
            return Ok(());
        }
        return Err(Error::Invalid {
            path: directory.to_owned(),
            message: "rules.da and rules.values must either both be absent or both be present"
                .to_owned(),
        });
    };
    if values.len() % VALUE_WIDTH != 0 || values.len() / VALUE_WIDTH != observed {
        return Err(Error::Invalid {
            path: directory.join("rules.values"),
            message: format!(
                "has {} bytes for {observed} rules; expected {}",
                values.len(),
                observed.saturating_mul(VALUE_WIDTH)
            ),
        });
    }
    if let Some(index) = invalid_index {
        return Err(Error::Invalid {
            path: directory.join("rules.da"),
            message: format!("rule index {index} is out of range 0..{observed}"),
        });
    }
    for (index, chunk) in values.as_chunks::<VALUE_WIDTH>().0.iter().enumerate() {
        let packed = u64::from_le_bytes(*chunk);
        let Some(rule) = Rule::unpack_checked(packed) else {
            let case = ((packed >> 48) & 0xFF) as u8;
            return Err(Error::Invalid {
                path: directory.join("rules.values"),
                message: format!("rule {index} has unknown case action {case}"),
            });
        };
        if rule.add >= u64::try_from(strings_len).unwrap_or(u64::MAX) {
            return Err(Error::Invalid {
                path: directory.join("rules.values"),
                message: format!(
                    "rule {index} string identifier {} is out of range 0..{strings_len}",
                    rule.add
                ),
            });
        }
    }
    Ok(())
}

/// Checks cross-file counts and identifiers after every file parsed safely.
pub(in crate::lexicon) fn validate_tables(
    directory: &Path,
    tables: &Tables,
    strings: &Strings,
    meta: &Meta,
) -> Result<()> {
    let strings_len = strings.len();
    if meta.entries.get("strings").copied() != u64::try_from(strings_len).ok() {
        return Err(Error::Invalid {
            path: directory.join("strings.txt"),
            message: format!(
                "contains {strings_len} strings, metadata declares {:?}",
                meta.entries.get("strings")
            ),
        });
    }
    validate_lookup_tables(directory, tables, strings_len, meta)?;
    validate_entity_count(directory, &tables.entities, meta)?;
    validate_phrase_count(directory, tables.phrases.as_ref(), meta)?;
    validate_rules(directory, tables, strings_len, meta)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{Bytes, DoubleArray, Table};
    use super::table_stats;
    use yada::builder::DoubleArrayBuilder;

    #[test]
    fn double_array_stats_count_and_check_ids_together() {
        let rows = [
            (b"a".as_slice(), 0_u32),
            (b"b".as_slice(), 2_u32),
            (b"c".as_slice(), 5_u32),
        ];
        let bytes = DoubleArrayBuilder::build(&rows).expect("valid sorted rows");
        let table = Table::Da(DoubleArray::new(Bytes::Owned(bytes)).expect("valid double array"));
        let stats = table_stats(Some(&table), 3);
        assert_eq!((stats.entries, stats.invalid_id), (rows.len(), Some(5)));
    }
}
