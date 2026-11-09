//! Splitting one already-delimited mention into tokens and the exact gaps
//! between them, the inverse of how the span probe rebuilds a surface.

/// Why one mention's surface cannot be split into tokens and gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    /// The surface is the empty string.
    #[error("span surface is empty")]
    Empty,
    /// The surface holds whitespace and nothing else.
    #[error("span surface has no tokens: whitespace only")]
    Blank,
    /// The surface opens or closes with whitespace; trimming is the caller's
    /// decision, since an unanswered span must come back as passed.
    #[error(
        "span surface has leading or trailing whitespace; strip the mention \
         before the call so the answer stays comparable with what was passed"
    )]
    Untrimmed,
}

/// Splits one mention's surface into its tokens and the exact text between them.
///
/// Both lists are empty for a single-token surface (the span probe's one-word
/// shape, with no allocation); otherwise the gaps are the verbatim whitespace
/// runs between tokens.
///
/// # Errors
///
/// Returns [`SurfaceError`] for an empty surface, a whitespace-only one, and
/// one that opens or closes with whitespace.
#[must_use = "the tokens and gaps are what the span probe is given"]
pub fn split_surface(surface: &str) -> Result<(Vec<&str>, Vec<&str>), SurfaceError> {
    if surface.is_empty() {
        return Err(SurfaceError::Empty);
    }
    // One iterator for both ends, so a one-character surface is checked once.
    let mut ends = surface.chars();
    if ends.next().is_some_and(char::is_whitespace)
        || ends.next_back().is_some_and(char::is_whitespace)
    {
        return Err(if surface.trim().is_empty() {
            SurfaceError::Blank
        } else {
            SurfaceError::Untrimmed
        });
    }
    let Some(first_gap) = surface.find(char::is_whitespace) else {
        return Ok((Vec::new(), Vec::new()));
    };

    let mut tokens = Vec::new();
    let mut gaps = Vec::new();
    let mut rest = surface;
    let mut next_gap = Some(first_gap);
    while let Some(at) = next_gap {
        let (token, tail) = rest.split_at(at);
        tokens.push(token);
        let width = tail
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(tail.len());
        let (gap, next) = tail.split_at(width);
        gaps.push(gap);
        rest = next;
        next_gap = rest.find(char::is_whitespace);
    }
    tokens.push(rest);
    Ok((tokens, gaps))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    /// Rebuilds the surface the way the span probe assumes it was built.
    fn rejoin(tokens: &[&str], gaps: &[&str]) -> String {
        let mut surface = String::new();
        for (index, token) in tokens.iter().enumerate() {
            if index > 0 {
                surface.push_str(gaps.get(index - 1).copied().unwrap_or(" "));
            }
            surface.push_str(token);
        }
        surface
    }

    #[test]
    fn a_two_word_mention_splits_into_its_tokens_and_one_gap() {
        let (tokens, gaps) = split_surface("Ленинградские области").unwrap();
        assert_eq!(tokens, ["Ленинградские", "области"]);
        assert_eq!(gaps, [" "]);
    }

    #[test]
    fn a_single_token_surface_needs_neither_list() {
        // The empty lists are the probe's one-word shape, not a lost token.
        let (tokens, gaps) = split_surface("Ленинград").unwrap();
        assert!(tokens.is_empty() && gaps.is_empty());
    }

    #[test]
    fn unusual_gaps_survive_and_the_surface_rebuilds_byte_for_byte() {
        // Collapsing any of these would probe a mention the caller never sent.
        for surface in [
            "Стара  Загора",
            "Совет\tЕвропы",
            "«Rady — Europy»",
            "New\u{a0}York",
            "a b  c   d",
        ] {
            let (tokens, gaps) = split_surface(surface).unwrap();
            assert_eq!(rejoin(&tokens, &gaps), surface, "{surface:?}");
        }
    }

    #[test]
    fn degenerate_surfaces_are_refused_rather_than_trimmed() {
        assert_eq!(split_surface(""), Err(SurfaceError::Empty));
        assert_eq!(split_surface(" "), Err(SurfaceError::Blank));
        assert_eq!(split_surface(" \t\n"), Err(SurfaceError::Blank));
        assert_eq!(split_surface(" Ленинград"), Err(SurfaceError::Untrimmed));
        assert_eq!(split_surface("Ленинград "), Err(SurfaceError::Untrimmed));
        assert_eq!(
            split_surface(" Ленинградские области "),
            Err(SurfaceError::Untrimmed)
        );
    }
}
