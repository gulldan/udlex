//! Read a lossless lexical input projection or fit a new dictionary with the pinned 1.0.0 engine.
use fst::Streamer;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use udlex_rs::{Pair, Thresholds, Variant, build_for_language, write_sources};
use yada::unit::Unit;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn walk(node: usize, units: &[Unit], key: &mut Vec<u8>, strings: &[&str], output: &mut impl Write) -> Result<()> {
    let unit = &units[node];
    let base = node ^ unit.offset() as usize;
    if unit.has_leaf() {
        let leaf = &units[base];
        if !leaf.is_leaf() { return Err("Invalid terminal in lexical projection".into()); }
        writeln!(output, "{}\t{}", std::str::from_utf8(key)?, strings[leaf.value() as usize])?;
    }
    for label in 1..=u8::MAX {
        let child = base ^ usize::from(label);
        if units.get(child).is_some_and(|v| v.label() == u32::from(label)) {
            key.push(label);
            walk(child, units, key, strings, output)?;
            key.pop();
        }
    }
    Ok(())
}

// This mode reads only an external input table. Its caller verifies the exact original
// source-projection SHA-256. No predictions, training tables or rules can be read here.
fn unpack(folder: &Path, out: &Path) -> Result<()> {
    let text = fs::read_to_string(folder.join("strings.txt"))?;
    let strings: Vec<_> = text.lines().collect();
    let mut output = BufWriter::new(File::create(out)?);
    if folder.join("external.map").exists() {
        let map = fst::Map::new(fs::read(folder.join("external.map"))?)?;
        let mut entries = map.stream();
        while let Some((key, id)) = entries.next() {
            writeln!(output, "{}\t{}", std::str::from_utf8(key)?, strings[id as usize])?;
        }
    } else {
        let raw = fs::read(folder.join("external.da"))?;
        let units: Vec<_> = raw.chunks_exact(4).map(|b| Unit::from_u32(u32::from_le_bytes(b.try_into().unwrap()))).collect();
        walk(0, &units, &mut Vec::new(), &strings, &mut output)?;
    }
    output.flush()?;
    Ok(())
}

fn fit(iso: &str, input: &Path, out: &Path) -> Result<()> {
    if out.join("current").exists() || out.join("meta.json").exists() {
        return Err("Fit requires an empty destination".into());
    }
    let mut tokens = Vec::new();
    let mut external = Vec::new();
    for item in BufReader::new(File::open(input)?).lines() {
        let line = item?;
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            [x, y] if iso == "ssp" => external.push(((*x).into(), (*y).into())),
            [x, y] => tokens.push(Pair { form: (*x).into(), lemma: (*y).into(), sentence_initial: false }),
            [x, y, initial] => tokens.push(Pair { form: (*x).into(), lemma: (*y).into(), sentence_initial: *initial == "1" }),
            _ => return Err("Expected FORM, LEMMA and optional sentence position".into()),
        }
    }
    let tables = build_for_language(&tokens, external, iso);
    let entries = write_sources(&tables, Variant::Full, Thresholds::default(), out)?;
    println!("{}", serde_json::json!({"entries":entries,"training_tokens":tokens.len()}));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, folder, out] if mode == "unpack" => unpack(Path::new(folder), Path::new(out)),
        [mode, iso, input, out] if mode == "fit" => fit(iso, Path::new(input), Path::new(out)),
        _ => Err("Usage: dictionary_io unpack GENERATION TSV | fit ISO TSV DEST".into()),
    }
}

