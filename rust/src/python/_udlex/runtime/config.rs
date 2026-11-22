use pyo3::prelude::*;

use crate::format::{SpanCase, SpanMultiword, TokenCase};
use crate::lexicon::{self, RuntimeLexicon};
use crate::probe::Options;

use super::{Lexicon, to_py};

/// One atomic configuration for both Python loading entry points.
pub(super) struct RuntimeConfig<'a> {
    options: Options,
    /// An explicit choice applies to both paths; `None` keeps the token path
    /// closed and lets the span path follow the generation's measured policy.
    use_entities_lower: Option<bool>,
    span_case: Option<&'a str>,
    span_multiword: Option<&'a str>,
    phrase_patterns: Option<Vec<(String, String)>>,
    /// Token case override for the acceptance machinery; `None` follows the
    /// generation's measured convention.
    token_case: Option<&'a str>,
}

impl<'a> RuntimeConfig<'a> {
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub(super) const fn new(
        use_external: bool,
        use_rules: bool,
        use_exact: bool,
        use_lower: bool,
        use_exact_lower: bool,
        use_external_lower: bool,
        use_entities: bool,
        use_entities_lower: Option<bool>,
        use_user: bool,
        protected: bool,
        span_case: Option<&'a str>,
        span_multiword: Option<&'a str>,
        phrase_patterns: Option<Vec<(String, String)>>,
        token_case: Option<&'a str>,
    ) -> Self {
        Self {
            options: Options {
                use_entities,
                use_entities_lower: false,
                use_user,
                use_exact,
                use_lower,
                use_exact_lower,
                use_external,
                use_external_lower,
                use_rules,
                protected,
            },
            use_entities_lower,
            span_case,
            span_multiword,
            phrase_patterns,
            token_case,
        }
    }

    pub(super) fn configure(
        self,
        py: Python<'_>,
        mut base: lexicon::Lexicon,
        user_bundle: Option<&str>,
    ) -> PyResult<Lexicon> {
        let user = user_bundle
            .map(|root| {
                py.detach(|| crate::UserDictionary::open_from_bundle(root, &base.meta().iso639_3))
                    .map_err(|error| to_py(&error))
            })
            .transpose()?
            .flatten();
        let mut options = self.options;
        options.use_entities_lower = self.use_entities_lower.unwrap_or(false);
        if let Some(value) = self.token_case {
            let policy = TokenCase::parse(value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "token_case must be \"keep\" or \"surface\", got {value:?}"
                ))
            })?;
            base.set_surface_case(policy == TokenCase::Surface);
        }
        let span_lower = self
            .use_entities_lower
            .unwrap_or_else(|| base.span_options().use_entities_lower);
        let mut conventions = base.span_conventions();
        if let Some(value) = self.span_case {
            let policy = SpanCase::parse(value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "span_case must be \"transfer\" or \"verbatim\", got {value:?}"
                ))
            })?;
            conventions.case_transfer = policy.transfers();
        }
        if let Some(value) = self.span_multiword {
            let policy = SpanMultiword::parse(value).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "span_multiword must be \"identity\", \"compose\" or \"head\", got {value:?}"
                ))
            })?;
            conventions.multiword = policy;
        }
        base.set_span_conventions(conventions);
        if let Some(mut rows) = self.phrase_patterns {
            #[allow(deprecated)]
            base.set_phrase_patterns(&mut rows)
                .map_err(|error| to_py(&error))?;
        }
        let inner = RuntimeLexicon::new(base, user).map_err(|error| to_py(&error))?;
        Ok(Lexicon {
            inner,
            options,
            span_lower,
            lemma_strings: std::sync::Mutex::default(),
        })
    }
}
