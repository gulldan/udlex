//! Build the small JNI test dictionaries with the same format writer as the release.
use std::{error::Error, fs, path::Path};
use udlex_rs::{Paradigm, Spec, build_lexicon_required};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [cases, output] = args.as_slice() else {
        return Err("expected a case file and output directory".into());
    };
    let bundle = Path::new(output);
    fs::create_dir_all(bundle)?;
    for row in fs::read_to_string(cases)?.lines().skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        let &[language, _, surface, lemma] = fields.as_slice() else {
            return Err("expected four TSV columns".into());
        };
        let mut pairs = format!("{surface}\t{lemma}\n");
        if language == "rus" {
            pairs.push_str("людей\tчеловек\nКошки\tкошка\nМоскве\tМосква\n");
        }
        let path = bundle.join(format!("{language}.tsv"));
        fs::write(&path, pairs)?;
        build_lexicon_required(&Spec {
            destination: &bundle.join(language),
            train: None,
            dev: None,
            test: None,
            paradigms: &[Paradigm {
                label: "fixture".into(),
                path,
                lemma_first: false,
                normalized_unique: false,
                expected: None,
            }],
            entities: &[],
            name: language,
            language,
            iso639_3: language,
            span_case: None,
            span_multiword: None,
            entity_lower: false,
            compound_marker: None,
            token_case: None,
            lemma_adapter: false,
            context_keys: false,
            phrases: &[],
            proper_noun_threshold: None,
        })?;
    }
    Ok(())
}
