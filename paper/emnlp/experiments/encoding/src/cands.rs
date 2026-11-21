//! Candidate names, their image files, and how each is built and opened.
//!
//! A name is `[bf8>]<index>+<layer>`:
//! * index — `sorted`, `hashbrown`, `flat`, `yada`, `dacore`, `crawdad`,
//!   `crawdadmp`, `fst`, `marisa`, `pilot`, `ptrhash`, `phast`, `vfunc`, `ptrfp`,
//!   `core`, or `split<K>:<pilot|ptrhash>` (double array up to `K` bytes);
//! * layer — `ids`, `off`, `offid`, `rule`, or `inline` for the hash family.
//!   A split keeps the layer on its short side and spells lemmas inline on
//!   the long side, except under `rule`, which both sides share.
//! * `bf8>` asks a binary fuse filter first; `call>` puts the lookup behind
//!   a real call instead of inlining it into the loop.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::alloc;
use crate::data::KeySet;
use crate::dict::{
    Dict, Direct, Flagged, Ids, Index, Lemma, Outlined, Payloads, Values, With, open_direct,
    open_ids, open_rules,
};
use crate::img::{heap_mode, map_file};
use crate::index::{
    Brown, Crawdad, CrawdadMp, DaCore, Filtered, Flat, Fst, Hashed, Marisa, PhastSlots, Pilot,
    PtrFp, PtrSlots, Rows, Sorted, VSlots, Yada, build_da, build_filter,
};
use crate::lexicon::{CoreHit, CoreHybrid, SPLIT_BYTES, build_hybrid};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Clone)]
pub struct Spec {
    pub name: String,
    outlined: bool,
    filter: bool,
    index: String,
    split: Option<usize>,
    layer: String,
}

impl Spec {
    pub fn parse(name: &str) -> io::Result<Self> {
        let (outlined, rest) = match name.strip_prefix("call>") {
            Some(rest) => (true, rest),
            None => (false, name),
        };
        let (filter, rest) = match rest.strip_prefix("bf8>") {
            Some(rest) if !outlined => (true, rest),
            _ => (false, rest),
        };
        let (index, layer) = rest
            .split_once('+')
            .ok_or_else(|| invalid(format!("{name}: no +layer")))?;
        let (split, index) = match index.strip_prefix("split").and_then(|s| s.split_once(':')) {
            Some((k, long)) => (
                Some(
                    k.parse()
                        .map_err(|_| invalid(format!("{name}: bad split")))?,
                ),
                long,
            ),
            None => (None, index),
        };
        let hashes = ["pilot", "ptrhash", "phast", "vfunc"];
        let known = [
            "sorted",
            "hashbrown",
            "flat",
            "yada",
            "dacore",
            "crawdad",
            "crawdadmp",
            "fst",
            "marisa",
            "ptrfp",
            "core",
        ];
        let ok = match (split, layer) {
            (Some(_), "ids" | "off" | "offid" | "rule") => ["pilot", "ptrhash"].contains(&index),
            (None, "inline") => hashes.contains(&index),
            (None, "ids") => hashes.contains(&index) || known.contains(&index),
            (None, "off" | "offid" | "rule") => {
                index != "core" && (hashes.contains(&index) || known.contains(&index))
            }
            _ => false,
        };
        if !ok {
            return Err(invalid(format!("{name}: unknown candidate")));
        }
        Ok(Self {
            name: name.to_owned(),
            outlined,
            filter,
            index: index.to_owned(),
            split,
            layer: layer.to_owned(),
        })
    }

    /// Structures some crate keeps on its own heap rather than in the image.
    pub fn heap(&self) -> &'static str {
        match self.index.as_str() {
            "hashbrown" => "all",
            "crawdad" | "crawdadmp" => "all (deserialised)",
            "ptrhash" | "ptrfp" => "pilots (rebuilt at open)",
            "vfunc" => "function (rebuilt at open)",
            "phast" => "seeds (deserialised)",
            _ => "",
        }
    }

    /// Whether a non-key can be answered: only the fingerprint variant.
    pub fn exact(&self) -> bool {
        self.index != "ptrfp"
    }

    /// Whether the image stays mapped while querying.
    pub fn keeps_image(&self) -> bool {
        !matches!(self.index.as_str(), "crawdad" | "crawdadmp")
    }

    /// The value layer whose image this candidate reads besides its own.
    pub fn value_layer(&self) -> Option<&'static str> {
        match self.layer.as_str() {
            "ids" => Some("ids"),
            "off" | "offid" => Some("off"),
            "rule" => Some("rule"),
            _ => None,
        }
    }

    fn long_layer(&self) -> &'static str {
        if self.layer == "rule" {
            "rule"
        } else {
            "inline"
        }
    }

    fn main_file(&self, dir: &Path) -> PathBuf {
        let stem = match self.index.as_str() {
            "yada" | "dacore" => "da",
            other => other,
        };
        match self.split {
            Some(k) => dir.join(format!("{stem}_gt{k}.{}.img", self.long_layer())),
            None if stem == "core" => dir.join("core.hybrid"),
            None => dir.join(format!("{stem}.{}.img", self.layer)),
        }
    }

    fn short_file(&self, dir: &Path) -> Option<PathBuf> {
        self.split
            .map(|k| dir.join(format!("da_le{k}.{}.img", self.layer)))
    }

    /// What `Costs` files a build under: the images, or for the heap-only
    /// map a name that never exists on disk.
    pub fn cost_keys(&self, dir: &Path) -> Vec<PathBuf> {
        let mut out = self.files(dir);
        if self.index == "hashbrown" {
            out.push(dir.join(format!("hashbrown.{}.heap", self.layer)));
        }
        out
    }

    /// Every file the candidate reads, value layer aside.
    pub fn files(&self, dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if self.index != "hashbrown" {
            out.push(self.main_file(dir));
        }
        if self.index == "marisa" {
            out.push(self.main_file(dir).with_extension("trie"));
        }
        out.extend(self.short_file(dir));
        if self.filter {
            out.push(dir.join("bf8.img"));
        }
        out
    }
}

/// The matrix: key structures under the core's value layer, the split
/// sweep, the value-layer study, and the filter study.
pub fn default_names() -> Vec<String> {
    let mut out: Vec<String> = [
        "sorted+ids",
        "hashbrown+ids",
        "flat+ids",
        "yada+ids",
        "dacore+ids",
        "crawdad+ids",
        "crawdadmp+ids",
        "fst+ids",
        "marisa+ids",
        "pilot+ids",
        "pilot+inline",
        "ptrhash+ids",
        "ptrhash+inline",
        "phast+ids",
        "phast+inline",
        "vfunc+ids",
        "vfunc+inline",
        "ptrfp+ids",
        "core+ids",
    ]
    .map(str::to_owned)
    .to_vec();
    for long in ["pilot", "ptrhash"] {
        for k in [4, 6, 8, 10, 12, 16] {
            out.push(format!("split{k}:{long}+ids"));
        }
    }
    out.extend(
        [
            "dacore+off",
            "dacore+offid",
            "dacore+rule",
            "fst+rule",
            "pilot+off",
            "pilot+rule",
            "ptrhash+rule",
            "ptrfp+rule",
            "split8:pilot+off",
            "split8:pilot+offid",
            "split8:pilot+rule",
            "split8:ptrhash+off",
            "split8:ptrhash+rule",
            "bf8>dacore+ids",
            "bf8>fst+ids",
            "bf8>marisa+ids",
            "bf8>core+ids",
            "bf8>ptrhash+inline",
            "call>dacore+ids",
            "call>split8:pilot+ids",
            "call>ptrhash+inline",
        ]
        .map(str::to_owned),
    );
    out
}

/// What a candidate is built from and where its images live.
pub struct Cx<'a> {
    pub dir: PathBuf,
    pub data: Option<(&'a KeySet, &'a Payloads)>,
}

impl<'a> Cx<'a> {
    fn rows(&self, layer: &str) -> io::Result<Rows<'a>> {
        let (set, pay) = self
            .data
            .ok_or_else(|| invalid("this candidate needs the key set"))?;
        let payload = match layer {
            "off" => &pay.off,
            "offid" => &pay.offid,
            "rule" => &pay.rule,
            _ => &pay.ids,
        };
        Ok(Rows {
            keys: set.keys.iter().collect(),
            payload: payload.clone(),
            lemma: set
                .lemma
                .iter()
                .map(|&id| set.lemmas.get(id as usize))
                .collect(),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct Cost {
    pub ns: u128,
    pub peak_heap: usize,
}

/// Build cost per image file, so candidates sharing a file share its cost.
#[derive(Default)]
pub struct Costs(pub HashMap<PathBuf, Cost>);

impl Costs {
    fn run(&mut self, file: &Path, build: impl FnOnce() -> io::Result<()>) -> io::Result<()> {
        if self.0.contains_key(file) {
            return Ok(());
        }
        let baseline = alloc::mark();
        let start = Instant::now();
        build()?;
        let cost = Cost {
            ns: start.elapsed().as_nanos(),
            peak_heap: alloc::peak_since(baseline),
        };
        self.0.insert(file.to_owned(), cost);
        Ok(())
    }

    pub fn of(&self, files: &[PathBuf]) -> Cost {
        files
            .iter()
            .filter_map(|f| self.0.get(f))
            .fold(Cost::default(), |a, c| Cost {
                ns: a.ns + c.ns,
                peak_heap: a.peak_heap.max(c.peak_heap),
            })
    }
}

/// Builds whatever images the candidate lacks. Rows are gathered outside
/// the timed region; the builder runs on one thread (`single`).
pub fn build(
    spec: &Spec,
    cx: &Cx,
    costs: &mut Costs,
    single: &rayon::ThreadPool,
) -> io::Result<()> {
    let dir = &cx.dir;
    if spec.filter {
        let file = dir.join("bf8.img");
        if !costs.0.contains_key(&file) {
            let rows = cx.rows("ids")?;
            costs.run(&file, || build_filter(&rows.keys, &file))?;
        }
    }
    let main = spec.main_file(dir);
    if let (Some(k), Some(short)) = (spec.split, spec.short_file(dir))
        && !costs.0.contains_key(&short)
    {
        let rows = cx.rows(&spec.layer)?.subset(|key| key.len() <= k);
        costs.run(&short, || build_da(&rows, &short))?;
    }
    let key = if spec.index == "hashbrown" {
        dir.join(format!("hashbrown.{}.heap", spec.layer))
    } else {
        main.clone()
    };
    if costs.0.contains_key(&key) {
        return Ok(());
    }
    let layer = if spec.split.is_some() {
        spec.long_layer()
    } else {
        spec.layer.as_str()
    };
    let mut rows = cx.rows(if layer == "inline" { "ids" } else { layer })?;
    if let Some(k) = spec.split {
        rows = rows.subset(|key| key.len() > k);
    }
    let inline = layer == "inline";
    let (rows, main) = (&rows, &main);
    costs.run(&key, || {
        single.install(|| match (spec.index.as_str(), inline) {
            ("sorted", _) => Sorted::build(rows, main),
            ("hashbrown", _) => {
                drop(Brown::build(rows));
                Ok(())
            }
            ("flat", _) => Flat::build(rows, main),
            ("yada" | "dacore", _) => build_da(rows, main),
            ("crawdad", _) => Crawdad::build(rows, main),
            ("crawdadmp", _) => CrawdadMp::build(rows, main),
            ("fst", _) => Fst::build(rows, main),
            ("marisa", _) => Marisa::build(rows, main),
            ("pilot", true) => Hashed::<Pilot, true>::build(rows, main),
            ("pilot", false) => Hashed::<Pilot, false>::build(rows, main),
            ("ptrhash", true) => Hashed::<PtrSlots, true>::build_open(rows, main).map(drop),
            ("ptrhash", false) => Hashed::<PtrSlots, false>::build_open(rows, main).map(drop),
            ("phast", true) => Hashed::<PhastSlots, true>::build(rows, main),
            ("phast", false) => Hashed::<PhastSlots, false>::build(rows, main),
            ("vfunc", true) => Hashed::<VSlots, true>::build_open(rows, main).map(drop),
            ("vfunc", false) => Hashed::<VSlots, false>::build_open(rows, main).map(drop),
            ("ptrfp", _) => PtrFp::build_open(rows, main).map(drop),
            ("core", _) => build_core(cx, main),
            (other, _) => Err(invalid(format!("no builder for {other}"))),
        })
    })
}

/// The production builder on the lab's rows: string ids are the `ids` layer's.
fn build_core(cx: &Cx, path: &Path) -> io::Result<()> {
    let (set, _) = cx
        .data
        .ok_or_else(|| invalid("the core builder needs the key set"))?;
    let text = |bytes: &[u8]| String::from_utf8(bytes.to_vec()).map_err(|e| invalid(e.to_string()));
    let rows = set
        .keys
        .iter()
        .zip(&set.lemma)
        .map(|(key, &id)| Ok((text(key)?, u64::from(id))))
        .collect::<io::Result<Vec<_>>>()?;
    let lemma = |id: u64| text(set.lemmas.get(usize::try_from(id).ok()?)).ok();
    std::fs::write(path, build_hybrid(&rows, lemma).map_err(invalid)?)
}

/// The production table, its short side resolved through the `ids` layer
/// exactly as the lexicon resolves `Found::Id` through `strings.*`.
pub struct CoreDict {
    core: CoreHybrid,
    ids: Ids,
}

impl Dict for CoreDict {
    #[inline(always)]
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>> {
        Some(match self.core.get(key)? {
            CoreHit::Id(id) => self.ids.lemma(id, key),
            CoreHit::Text(head) => Lemma { head, tail: &[] },
        })
    }
}

/// The lab's hybrid with a movable threshold: the core's double-array walk
/// for keys up to `k` bytes, a perfect hash above.
pub struct Split<L, VS, VL> {
    k: usize,
    short: DaCore,
    long: L,
    vs: VS,
    vl: VL,
}

impl<L: Index, VS: Values, VL: Values> Dict for Split<L, VS, VL> {
    #[inline(always)]
    fn get<'a>(&'a self, key: &'a [u8]) -> Option<Lemma<'a>> {
        if key.len() <= self.k {
            let payload = self.short.find(key)?;
            Some(self.vs.lemma(payload, key))
        } else {
            let payload = self.long.find(key)?;
            Some(self.vl.lemma(payload, key))
        }
    }
}

/// Receives the opened candidate under its concrete type, so the loops that
/// query it are compiled for that type alone.
pub trait Visitor {
    fn visit<D: Dict>(&mut self, dict: &D);
}

fn finish<D: Dict>(dict: D, spec: &Spec, cx: &Cx, v: &mut impl Visitor) -> io::Result<()> {
    match (spec.filter, spec.outlined) {
        (true, _) => v.visit(&Filtered::open(&cx.dir.join("bf8.img"), dict)?),
        (false, true) => v.visit(&Outlined(dict)),
        (false, false) => v.visit(&dict),
    }
    Ok(())
}

fn with_values<I: Index>(index: I, spec: &Spec, cx: &Cx, v: &mut impl Visitor) -> io::Result<()> {
    match spec.layer.as_str() {
        "ids" => finish(With(index, open_ids(&cx.dir)?), spec, cx, v),
        "off" => finish(With(index, open_direct(&cx.dir)?), spec, cx, v),
        "offid" => finish(With(index, Flagged(open_direct(&cx.dir)?)), spec, cx, v),
        "rule" => finish(With(index, open_rules(&cx.dir)?), spec, cx, v),
        other => Err(invalid(format!("no value layer {other}"))),
    }
}

fn with_split<L: Index>(
    long: L,
    blob: &'static [u8],
    spec: &Spec,
    cx: &Cx,
    v: &mut impl Visitor,
) -> io::Result<()> {
    let k = spec.split.unwrap_or(SPLIT_BYTES);
    let short = DaCore::open(
        &spec
            .short_file(&cx.dir)
            .ok_or_else(|| invalid("not a split"))?,
    )?;
    let dir = &cx.dir;
    match spec.layer.as_str() {
        "ids" => finish(
            Split {
                k,
                short,
                long,
                vs: open_ids(dir)?,
                vl: Direct::over(blob),
            },
            spec,
            cx,
            v,
        ),
        "off" => finish(
            Split {
                k,
                short,
                long,
                vs: open_direct(dir)?,
                vl: Direct::over(blob),
            },
            spec,
            cx,
            v,
        ),
        "offid" => finish(
            Split {
                k,
                short,
                long,
                vs: Flagged(open_direct(dir)?),
                vl: Direct::over(blob),
            },
            spec,
            cx,
            v,
        ),
        "rule" => finish(
            Split {
                k,
                short,
                long,
                vs: open_rules(dir)?,
                vl: open_rules(dir)?,
            },
            spec,
            cx,
            v,
        ),
        other => Err(invalid(format!("no value layer {other}"))),
    }
}

/// Opens the candidate and hands it to the visitor.
pub fn open(spec: &Spec, cx: &Cx, v: &mut impl Visitor) -> io::Result<()> {
    let path = &spec.main_file(&cx.dir);
    // The hash family: inline entries feed `Direct` over their own blob.
    macro_rules! hashed {
        ($open_inline:expr, $open_payload:expr) => {{
            let inline = spec.layer == "inline" || (spec.split.is_some() && spec.layer != "rule");
            match (inline, spec.split.is_some()) {
                (true, split) => {
                    let table = $open_inline?;
                    let blob = table.blob();
                    if split {
                        with_split(table, blob, spec, cx, v)
                    } else {
                        finish(With(table, Direct::over(blob)), spec, cx, v)
                    }
                }
                (false, true) => {
                    let table = $open_payload?;
                    let blob = table.blob();
                    with_split(table, blob, spec, cx, v)
                }
                (false, false) => {
                    // Bound first: the rows it was built from die here,
                    // before the visitor measures what the table holds.
                    let table = $open_payload?;
                    with_values(table, spec, cx, v)
                }
            }
        }};
    }
    let long_rows = |layer: &str| -> io::Result<Rows> {
        let rows = cx.rows(layer)?;
        Ok(match spec.split {
            Some(k) => rows.subset(|key| key.len() > k),
            None => rows,
        })
    };
    match spec.index.as_str() {
        "sorted" => with_values(Sorted::open(path)?, spec, cx, v),
        "hashbrown" => {
            let map = Brown::build(&cx.rows(&spec.layer)?);
            with_values(map, spec, cx, v)
        }
        "flat" => with_values(Flat::open(path)?, spec, cx, v),
        "yada" => with_values(Yada::open(path)?, spec, cx, v),
        "dacore" => with_values(DaCore::open(path)?, spec, cx, v),
        "crawdad" => with_values(Crawdad::open(path)?, spec, cx, v),
        "crawdadmp" => with_values(CrawdadMp::open(path)?, spec, cx, v),
        "fst" => with_values(Fst::open(path)?, spec, cx, v),
        "marisa" => with_values(Marisa::open(path)?, spec, cx, v),
        "pilot" => hashed!(
            Hashed::<Pilot, true>::open(path),
            Hashed::<Pilot, false>::open(path)
        ),
        "phast" => hashed!(
            Hashed::<PhastSlots, true>::open(path),
            Hashed::<PhastSlots, false>::open(path)
        ),
        "ptrhash" => hashed!(
            Hashed::<PtrSlots, true>::build_open(&long_rows("ids")?, path),
            Hashed::<PtrSlots, false>::build_open(&long_rows(&spec.layer)?, path)
        ),
        "vfunc" => hashed!(
            Hashed::<VSlots, true>::build_open(&long_rows("ids")?, path),
            Hashed::<VSlots, false>::build_open(&long_rows(&spec.layer)?, path)
        ),
        "ptrfp" => {
            let table = PtrFp::build_open(&cx.rows(&spec.layer)?, path)?;
            with_values(table, spec, cx, v)
        }
        "core" => {
            let core = if heap_mode() {
                CoreHybrid::open_owned(std::fs::read(path)?)
            } else {
                CoreHybrid::open(map_file(path)?)
            }
            .map_err(invalid)?;
            finish(
                CoreDict {
                    core,
                    ids: open_ids(&cx.dir)?,
                },
                spec,
                cx,
                v,
            )
        }
        other => Err(invalid(format!("no reader for {other}"))),
    }
}

impl Spec {
    /// Whether opening rebuilds from the key set instead of reading an image.
    pub fn needs_rows(&self) -> bool {
        matches!(
            self.index.as_str(),
            "hashbrown" | "ptrhash" | "ptrfp" | "vfunc"
        )
    }
}
