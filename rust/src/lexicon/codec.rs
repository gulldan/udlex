//! On-disk lookup-table codec names and file selection.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{Error, Result};

/// How a compiled lookup table is encoded on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// Double-array trie; the value is the string identifier.
    Da,
    /// `fst::Map` from form to string identifier.
    Map,
    /// Acceptor over keys that carry their result inline.
    Inline,
    /// Double array for keys up to eight bytes, pilot hash with inline lemmas
    /// for the rest; see `lexicon::table::hybrid`.
    Hybrid,
}

impl Codec {
    /// Every codec there is.
    pub(crate) const ALL: [Self; 4] = [Self::Da, Self::Map, Self::Inline, Self::Hybrid];

    /// The metadata name, which is also the file extension.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Da => "da",
            Self::Map => "map",
            Self::Inline => "inline",
            Self::Hybrid => "hybrid",
        }
    }

    /// Parses one metadata codec name.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|codec| codec.name() == text)
    }

    /// The file a table with this codec lives in.
    pub(crate) fn file(self, directory: &Path, table: &str) -> PathBuf {
        directory.join(format!("{table}.{}", self.name()))
    }
}

/// Resolves the codec metadata records for one table.
pub(super) fn table_codec(
    codecs: &BTreeMap<String, String>,
    directory: &Path,
    name: &str,
) -> Result<Codec> {
    codecs.get(name).map_or(Ok(Codec::Da), |text| {
        Codec::parse(text).ok_or_else(|| Error::Codec {
            path: directory.join(name),
            codec: text.clone(),
        })
    })
}
