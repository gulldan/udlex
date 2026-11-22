//! Immutable generation construction, compilation, and publication.

mod compiler;
mod identity;
mod lookup;
mod source;
mod writer;

pub use compiler::{Compiled, compile, prune};
pub use identity::render_meta;
pub use writer::GenerationWriter;

#[cfg(test)]
pub(super) use compiler::copy_regular_after_open;
#[cfg(test)]
pub(super) use writer::write_file;
