"""Fast lemmatisation from a compiled lexicon bundle.

A lexicon answers one question — what is the lemma of this token — with a
short sequence of memory-mapped table probes and, only for an unknown form, a
suffix rule. There is no model, no context and no tagger; a loaded lexicon is
immutable and safe to share across threads.

This is `udlex._udlex`; the package re-exports everything here, so a caller
writes ``udlex.load`` and never imports this module by name.

    >>> import udlex
    >>> [language.iso639_3 for language in udlex.languages("bundle")[:3]]
    ['abk', 'afr', 'ajp']
    >>> lexicon = udlex.load("bundle", "rus")
    >>> lexicon.lemmatize(["Мама", "мыла", "раму"])
    ['мама', 'мыть', 'рама']

``udlex.build_lexicon`` atomically publishes a compiled immutable generation.
:func:`compile_bundle` remains for migrating legacy source-only bundles and
for explicitly recompiling retained sources.
"""

from collections.abc import Sequence
from typing import Literal, final, overload

from typing_extensions import deprecated

from udlex._provenance import Source as _Source

def _training_annotation_mask(
    config: str | None, rows: list[tuple[str, str | None]]
) -> bytes:
    """Return bounded 0/1 flags using Rust train/dev annotation conventions."""

__version__: str
"""The release number, taken from the crate this was built from."""

FORMAT_VERSION: int
"""On-disk bundle layout this build *writes*; independent of release versions."""

MIN_FORMAT_VERSION: int
"""Oldest on-disk bundle layout this build still reads."""

FORMAT_RANGE: str
"""The readable range as every refusal spells it: ``4``, or ``4-6``."""

MIN_CORE_RELEASE: str
"""Earliest udlex release that reads :data:`FORMAT_VERSION`; wheels pin it."""

def format_supported(format_version: int) -> bool:
    """Whether this build reads a bundle declaring ``format_version``.

    Args:
        format_version: What a document declares, in ``[0, 2**32)``.

    Returns:
        Whether it falls inside the readable range.
    """

def format_remedy(format_version: int) -> str:
    """What a caller must do about a format this build does not read.

    Args:
        format_version: What a document declares, in ``[0, 2**32)``.

    Returns:
        Upgrading udlex when the document is ahead of this build, rebuilding
        or reinstalling the dictionary when it is behind.
    """

ENTITY_ENCODING: str
"""Encoding contract for the shared exact/lower entity table."""

NORMALIZATION: str
"""Canonical Unicode normalization and locale-casing semantics for keys."""

USER_DICTIONARY_SCHEMA: str
"""Machine contract of independent user-dictionary roots and generations."""

PHRASE_CODEC: str
"""Fixed encoding of the compiled phrase-pattern table (not recorded in ``codecs``)."""

PHRASE_SEPARATOR: str
"""Field separator inside a phrase key and a phrase pattern."""

PHRASE_TAIL: int
"""Final characters of each token a phrase key records."""

SOURCES: tuple[str, ...]
"""The source files a build writes, ``compile`` reads and ``--prune`` deletes.

Always all of them (an empty table is an empty file). ``strings.txt`` is not
one: readers memory-map it, so it survives pruning.
"""

@final
class Language:
    """One language a bundle can lemmatise: a read-only view of its metadata.

    Instances come from :func:`languages`; there is no public constructor.
    """

    @property
    def iso639_3(self) -> str:
        """ISO 639-3 code, the name to ask for this language by."""

    @property
    def name(self) -> str:
        """Human-readable language name, such as ``"Russian"``."""

    @property
    def lexicon(self) -> str:
        """The lexicon backing it, normally a Universal Dependencies treebank.

        An implementation detail of where the data came from, accepted by
        :func:`load` alongside the code but not the name to prefer.
        """

    @property
    def variant(self) -> str:
        """Which parts of the learned lexicon the bundle ships.

        ``"full"``, ``"lookup"`` (no suffix rules), ``"rules"`` (no external
        vocabulary), ``"corpus"`` (neither) or ``"identity"`` (no common-word
        stages; an entity table may remain for ``entity=True``).
        """

    @property
    def training_tokens(self) -> int:
        """How many annotated tokens it was learned from."""

    @property
    def entries(self) -> int:
        """How many form-to-lemma rows its lookup tables hold."""

    @property
    def accuracy(self) -> float | None:
        """Exact-match accuracy on the held-out test split, in ``[0, 1]``.

        ``None`` when the language had no gold data to measure against.
        """

    @property
    def baseline(self) -> float | None:
        """What copying the surface form scores on the same tokens.

        Read :attr:`accuracy` against this: it ranges from 0.13 (Arabic) to 1.0.
        """

    @property
    def is_useful(self) -> bool | None:
        """Whether independent judgement matched or exceeded copying.

        Normally from the selection receipt; ``None`` when nothing was measured.
        """

    @property
    def selected_on(self) -> str | None:
        """The sample that chose the shipped variant: ``dev`` or ``train-cv``."""

    @property
    def selection_accuracy(self) -> float | None:
        """Accuracy of the shipped variant on its selection sample.

        Shipped dictionaries carry no test score; read this against
        :attr:`selection_baseline`.
        """

    @property
    def selection_baseline(self) -> float | None:
        """What copying the surface form scored on the selection sample."""

    @property
    def gain(self) -> float | None:
        """How much accuracy the lexicon adds over copying the surface form.

        The fair way to compare languages; ``None`` when either side is unmeasured.
        """

@final
class Compiled:
    """What compiling one lexicon directory did.

    Instances come from :func:`compile_bundle`; there is no public constructor.
    """

    @property
    def tables(self) -> int:
        """How many lookup tables were written."""

    @property
    def from_sources(self) -> bool:
        """Whether the builder's sources were there; ``False`` compiled nothing."""

@final
class LemmaInfo:
    """A lemma and one typed final stage, with no duplicate booleans."""

    @property
    def lemma(self) -> str:
        """Produced lemma text."""

    @property
    def source(self) -> _Source:
        """The one final stage that produced the answer, not a miss trace."""

@final
class Lexicon:
    """A compiled lexicon, loaded once and queried many times.

    Prefer :func:`load`, which finds a language by its ISO 639-3 code. Use the
    constructor when you know the directory, for instance when you ship a single
    language on its own.
    """

    def __init__(
        self,
        path: str,
        *,
        use_external: bool = True,
        use_rules: bool = True,
        use_exact: bool = True,
        use_lower: bool = True,
        use_exact_lower: bool = True,
        use_external_lower: bool = True,
        use_entities: bool = False,
        use_entities_lower: bool | None = None,
        user_bundle: str | None = None,
        use_user: bool = True,
        protected: bool = False,
        span_case: str | None = None,
        span_multiword: str | None = None,
        phrase_patterns: list[tuple[str, str]] | None = None,
        token_case: str | None = None,
    ) -> None:
        """Open a compiled lexicon directory.

        Args:
            path: Directory holding the compiled tables.
            use_external: Consult the merged offline dictionary/analyser table
                for forms the corpus tables do not cover.
            use_rules: Guess from suffixes for unknown words. Pass ``False``
                when guesses are unacceptable. For marked entities prefer the
                stricter ``entity=True`` profile, which cannot fall through to
                common vocabulary or suffix rules.
            use_exact: Consult exact-case corpus forms.
            use_lower: Consult the corpus lowercase table.
            use_exact_lower: Retry exact corpus vocabulary with a lowercase key.
            use_external_lower: Retry external vocabulary with a lowercase key.
            use_entities: Consult isolated exact-case entity vocabulary.
            use_entities_lower: Retry isolated entities with a lowercase key;
                ``None`` keeps the generation's measured policy for spans only.
            user_bundle: Independent manual-override root. ``None`` keeps the
                object base-only.
            use_user: Consult the attached user layer. ``False`` still
                authenticates an explicitly supplied root but disables lookup.
            protected: Preserve every token without probing when true.
            span_case: Override the measured span case convention
                (``"transfer"`` or ``"verbatim"``); for the acceptance
                machinery, ``None`` in production.
            span_multiword: Override the measured multiword convention
                (``"identity"``, ``"compose"`` or ``"head"``); same
                discipline.
            phrase_patterns: Replace the generation's compiled phrase table
                with these ``(span key, pattern)`` rows; ``None`` keeps the
                shipped one and ``[]`` removes it, which is the reference
                profile the acceptance scores against.
            token_case: Override the measured token case convention
                (``"keep"`` or ``"surface"``); same discipline.

        Raises:
            OSError: If the directory is missing, unreadable, or was written by
                an incompatible builder.
            ValueError: If a convention override names an unknown value.
        """

    @overload
    def lemma(
        self,
        token: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
        entity: bool = False,
        info: Literal[False] = False,
    ) -> str: ...
    @overload
    def lemma(
        self,
        token: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
        entity: bool = False,
        info: Literal[True],
    ) -> LemmaInfo: ...
    def lemma(
        self,
        token: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
        entity: bool = False,
        info: bool = False,
    ) -> str | LemmaInfo:
        """Return the lemma of one token.

        Args:
            token: The surface form.
            sentence_initial: Whether this token opens its sentence. The flag
                only matters for capitalised tokens, where it separates a
                capitalised ordinary word from a proper noun.
            protected: Preserve this token without probing.
            entity: Use only isolated entity vocabulary, then identity.
            info: Return a :class:`LemmaInfo` — the lemma and the stage that
                produced it — instead of the bare string. Same single probe.

        Returns:
            The lemma, or the token itself when nothing is known about it.
            A lemma spelled like its token is that same string object.

        Raises:
            OSError: If ``entity=True`` and the bundle's entity table is
                damaged; that table is verified lazily on first use.
        """

    @deprecated("use lemma(..., info=True)")
    def lemma_with_info(
        self,
        token: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
        entity: bool = False,
    ) -> LemmaInfo:
        """Deprecated spelling of ``lemma(..., info=True)``."""

    def knows(
        self,
        token: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
        entity: bool = False,
    ) -> bool:
        """Whether a lookup table answers this form, suffix rules aside.

        Comparing lemmas cannot tell this: a known word may lemmatise to itself.

        Args:
            token: The surface form.
            sentence_initial: Whether the token opens its sentence.
            protected: Preserve the token without probing.
            entity: Ask only the isolated entity table.

        Returns:
            Whether the vocabulary covers it.

        Raises:
            OSError: If ``entity=True`` and the bundle's entity table is
                damaged; see :meth:`Lexicon.lemma`.
        """

    @overload
    def lemma_span(
        self,
        mention: str | Sequence[str],
        start: int = 0,
        end: int | None = None,
        *,
        separators: Sequence[str] | None = None,
        sentence_initial: bool = False,
        protected: bool = False,
        info: Literal[False] = False,
    ) -> str: ...
    @overload
    def lemma_span(
        self,
        mention: str | Sequence[str],
        start: int = 0,
        end: int | None = None,
        *,
        separators: Sequence[str] | None = None,
        sentence_initial: bool = False,
        protected: bool = False,
        info: Literal[True],
    ) -> LemmaInfo: ...
    def lemma_span(
        self,
        mention: str | Sequence[str],
        start: int = 0,
        end: int | None = None,
        *,
        separators: Sequence[str] | None = None,
        sentence_initial: bool = False,
        protected: bool = False,
        info: bool = False,
    ) -> str | LemmaInfo:
        """Return the canonical text of one entity mention.

        Where a mention starts and ends belongs to the caller's NER, never to
        this library. Pass it in whichever form you hold:

        * ``"Ленинградской области"`` — the mention's exact surface. It is
          split on its whitespace runs and the runs are kept as separators,
          so nothing is re-spaced: what no table answers comes back
          byte-for-byte, as the very object passed in.
        * ``tokens`` — the mention's own tokens, or a whole sentence narrowed
          by the half-open ``[start, end)``. ``separators`` holds the original
          text between consecutive span tokens, exactly ``end - start - 1``
          items; without it every gap becomes a single space.

        The probe order is fixed: the isolated entity table with the whole span
        — exact case, then the locale-lowercased key when the lexicon enables
        it — then the ordinary token chain for a single-word span, and the
        lexicon's measured multiword convention for a longer one.

        Prefer this over ``lemma(text, entity=True)`` for a NER mention: that
        call is the narrow entity profile — exact entity table, then the text
        itself — while this one is the whole span path.

        Args:
            mention: The mention as one string, or as tokens.
            start: Index of the first span token; token lists only.
            end: Index one past the last span token; ``None`` means the end of
                the list. Token lists only.
            separators: The original text between span tokens; token lists
                only.
            sentence_initial: Whether the span opens its sentence; consulted
                only by the single-word fall-through.
            protected: Return the surface without any lookup.
            info: Return a :class:`LemmaInfo` instead of the bare string.

        Returns:
            The canonical span text, or the surface when no table answers —
            quotes, brackets and hyphens preserved.

        Raises:
            TypeError: If ``start``, ``end`` or ``separators`` accompany a
                string mention.
            ValueError: If the interval is empty or out of bounds, the
                separator list has the wrong length, or the surface is empty,
                whitespace only, or opens or closes with whitespace (it is
                never trimmed silently).
            OSError: If the bundle's entity table is damaged; see
                :meth:`Lexicon.lemma`.
        """

    @deprecated("use lemma_span(tokens, ..., info=True)")
    def lemma_span_with_info(
        self,
        tokens: Sequence[str],
        start: int = 0,
        end: int | None = None,
        *,
        separators: Sequence[str] | None = None,
        sentence_initial: bool = False,
        protected: bool = False,
    ) -> LemmaInfo:
        """Deprecated spelling of ``lemma_span(tokens, ..., info=True)``."""

    @deprecated("use lemma_span(text)")
    def lemma_span_text(
        self,
        text: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
    ) -> str:
        """Deprecated spelling of ``lemma_span(text)``."""

    @deprecated("use lemma_span(text, info=True)")
    def lemma_span_text_with_info(
        self,
        text: str,
        *,
        sentence_initial: bool = False,
        protected: bool = False,
    ) -> LemmaInfo:
        """Deprecated spelling of ``lemma_span(text, info=True)``."""

    @overload
    def lemmatize(
        self,
        tokens: Sequence[str],
        *,
        sentence_initial: Sequence[bool] | None = None,
        protected: Sequence[bool] | None = None,
        entities: Sequence[bool] | None = None,
        info: Literal[False] = False,
        upos: Sequence[str] | None = None,
        context: bool = False,
    ) -> list[str]: ...
    @overload
    def lemmatize(
        self,
        tokens: Sequence[str],
        *,
        sentence_initial: Sequence[bool] | None = None,
        protected: Sequence[bool] | None = None,
        entities: Sequence[bool] | None = None,
        info: Literal[True],
        upos: Sequence[str] | None = None,
        context: bool = False,
    ) -> list[LemmaInfo]: ...
    def lemmatize(
        self,
        tokens: Sequence[str],
        *,
        sentence_initial: Sequence[bool] | None = None,
        protected: Sequence[bool] | None = None,
        entities: Sequence[bool] | None = None,
        info: bool = False,
        upos: Sequence[str] | None = None,
        context: bool = False,
    ) -> list[str] | list[LemmaInfo]:
        """Lemmatise a whole sentence.

        Args:
            tokens: The sentence, already tokenised. A ``str`` is refused with
                a ``TypeError`` rather than read as a sequence of characters.
            sentence_initial: Optional exact-length sentence-boundary mask.
                Several sentences can share one call: mark each one's first
                token, and the per-call overhead is paid once.
            protected: Optional exact-length preservation mask.
            entities: Optional exact-length entity-safe lookup mask.
            info: Return one :class:`LemmaInfo` per token instead of strings;
                the probes are identical and each token is probed once.
            upos: Optional exact-length UPOS tags, one per token (``""``
                skips its token). An exact answer of an ambiguous form gives
                way to the lemma the lexicon learned for that tag, with
                ``Source.CONTEXT``.
            context: Ask the same for the previous and the next word; a
                ``sentence_initial`` boundary is where the neighbours stop.
                Neither costs anything unless given.

        Returns:
            One lemma per input token, in the same order. Without a
            ``sentence_initial`` mask the first token is treated as
            sentence-initial. A lemma spelled like its token is that same
            string object.
        """

    @overload
    def lemmatize_many(
        self, sentences: Sequence[Sequence[str]], *, info: Literal[False] = False
    ) -> list[list[str]]: ...
    @overload
    def lemmatize_many(
        self, sentences: Sequence[Sequence[str]], *, info: Literal[True]
    ) -> list[list[LemmaInfo]]: ...
    def lemmatize_many(
        self, sentences: Sequence[Sequence[str]], *, info: bool = False
    ) -> list[list[str]] | list[list[LemmaInfo]]:
        """Lemmatise many sentences in one call.

        Each sentence's first token is sentence-initial; the answers equal one
        :meth:`lemmatize` call per sentence, but the GIL is released once for
        the whole batch, so threads scale.

        Args:
            sentences: Token lists, one per sentence.
            info: Return :class:`LemmaInfo` rows instead of strings.

        Returns:
            One list of lemmas per sentence, in the same order.
        """

    @deprecated("use lemmatize(..., info=True)")
    def lemmatize_with_info(
        self,
        tokens: Sequence[str],
        *,
        sentence_initial: Sequence[bool] | None = None,
        protected: Sequence[bool] | None = None,
        entities: Sequence[bool] | None = None,
    ) -> list[LemmaInfo]:
        """Deprecated spelling of ``lemmatize(..., info=True)``."""

    @property
    def name(self) -> str:
        """The lexicon identifier, normally a Universal Dependencies treebank."""

    @property
    def language(self) -> str:
        """Human-readable language name."""

    @property
    def iso639_3(self) -> str:
        """ISO 639-3 code of the language."""

    @property
    def source(self) -> str:
        """What the lexicon was learned from."""

    @property
    def training_tokens(self) -> int:
        """How many annotated tokens went into it."""

    @property
    def generation(self) -> str | None:
        """Immutable base generation resolved at open, when present."""

    @property
    def format_version(self) -> int:
        """On-disk layout version declared by the base generation."""

    @property
    def variant(self) -> str:
        """Which parts of the learned lexicon this bundle ships.

        ``"full"``, ``"lookup"`` (no suffix rules), ``"rules"`` (no external
        vocabulary), ``"corpus"`` (neither) or ``"identity"`` (no common-word
        stages; an entity table may remain for ``entity=True``).
        """

    @property
    def selected_on(self) -> str | None:
        """Selection split: ``dev``, form-disjoint ``train-cv``, or ``None``."""

    @property
    def accuracy(self) -> float | None:
        """Exact-match accuracy on the held-out test split, in ``[0, 1]``."""

    @property
    def baseline(self) -> float | None:
        """What copying the surface form scores on the same tokens.

        Read :attr:`accuracy` against this: it ranges from 0.13 (Arabic) to 1.0.
        """

    @property
    def is_useful(self) -> bool | None:
        """Whether independent judgement matched or exceeded copying.

        Normally from the selection receipt; ``None`` when nothing was measured.
        """

    @property
    def has_user_dictionary(self) -> bool:
        """Whether this ISO has at least one active manual override."""

    @property
    def user_entries(self) -> int:
        """Active manual override count."""

    @property
    def user_generation(self) -> str | None:
        """Resolved immutable user generation, if present."""

    @property
    def base_accuracy(self) -> float | None:
        """Held-out accuracy of the unchanged base generation."""

    @property
    def base_baseline(self) -> float | None:
        """Copying baseline of the unchanged base generation."""

    @property
    def base_is_useful(self) -> bool | None:
        """Usefulness verdict of the unchanged base generation."""

def languages(bundle: str) -> list[Language]:
    """List the languages a bundle can lemmatise, ordered by ISO 639-3 code.

    Reads one metadata file per language and opens no lookup table.

        >>> [language.iso639_3 for language in udlex.languages("bundle")]
        ['deu', 'eng', 'rus']

    Args:
        bundle: A bundle directory.

    Returns:
        One entry per language, ordered by code. A manifest-backed V1 bundle
        fails closed if any declared row is missing or corrupt; only an
        unmanifested local scratch directory may omit an unreadable lexicon.

    Raises:
        OSError: If the bundle directory cannot be listed.
    """

def load(
    bundle: str,
    language: str,
    *,
    use_external: bool = True,
    use_rules: bool = True,
    use_exact: bool = True,
    use_lower: bool = True,
    use_exact_lower: bool = True,
    use_external_lower: bool = True,
    use_entities: bool = False,
    use_entities_lower: bool | None = None,
    user_bundle: str | None = None,
    use_user: bool = True,
    protected: bool = False,
    span_case: str | None = None,
    span_multiword: str | None = None,
    phrase_patterns: list[tuple[str, str]] | None = None,
    token_case: str | None = None,
) -> Lexicon:
    """Open the lexicon a bundle holds for one language.

    Args:
        bundle: A bundle directory.
        language: An ISO 639-3 code such as ``"rus"``, or a treebank name such
            as ``"ru_taiga"``.
        use_external: Consult the merged offline dictionary/analyser table.
        use_rules: Guess from suffixes for unknown words.
        use_exact: Consult exact-case corpus forms.
        use_lower: Consult the corpus lowercase table.
        use_exact_lower: Retry exact corpus vocabulary lowercased.
        use_external_lower: Retry external vocabulary lowercased.
        use_entities: Consult isolated exact-case entity vocabulary.
        use_entities_lower: Retry isolated entity vocabulary lowercased;
            ``None`` keeps the generation's measured policy for spans only.
        user_bundle: Independent manual-override root. ``None`` keeps the
            object base-only.
        use_user: Consult the attached user layer. ``False`` still
            authenticates an explicitly supplied root but disables lookup.
        protected: Preserve all tokens without probing.
        span_case: Override the measured span case convention (``"transfer"``
            or ``"verbatim"``); for the acceptance machinery, ``None`` in
            production.
        span_multiword: Override the measured multiword convention
            (``"identity"``, ``"compose"`` or ``"head"``); same discipline.
        phrase_patterns: Replace the generation's compiled phrase table with
            these ``(span key, pattern)`` rows; ``None`` keeps the shipped one
            and ``[]`` removes it.
        token_case: Override the measured token case convention (``"keep"``
            or ``"surface"``); same discipline.

    Returns:
        The loaded lexicon.

    Raises:
        OSError: If the bundle holds no such language, or it cannot be opened.
        ValueError: If a convention override names an unknown value.
    """

def compile_bundle(path: str) -> Compiled:
    """Compile the sorted TSV tables in ``path`` into lookup tables.

    Each table gets its smallest encoding, recorded in the metadata. A
    directory without sources (pruned or installed) is left untouched.

    Args:
        path: A lexicon directory.

    Returns:
        What compiling did.

    Raises:
        OSError: If a source is unreadable, a destination cannot be written,
            only part of the sources is there, or the lexicon declares a format
            this build cannot compile (one version older than readable is
            migrated to :data:`FORMAT_VERSION`).
    """

def prune_bundle(path: str) -> bool:
    """Atomically publish a runtime-only generation, returning whether pruned."""

@final
class Tables:
    """The four lookup tables a build learned, held on the Rust side.

    Opaque so the lattice scores its five configurations without copying them.
    """

    def exact(self) -> list[tuple[str, str]]:
        """Surface form to lemma, case sensitive."""

    def lower(self) -> list[tuple[str, str]]:
        """Lowercased surface form to lemma."""

    def external(self) -> list[tuple[str, str]]:
        """Form to lemma harvested from a dictionary rather than the corpus."""

    def entities(self) -> list[tuple[str, str]]:
        """Curated named-entity surface forms kept out of common vocabulary."""

    def rules(self) -> list[tuple[str, tuple[int, str, int, float]]]:
        """Case bucket and suffix, to ``(cut, add, case, confidence)``."""

    def traffic(self, forms: list[str]) -> tuple[float, float] | None:
        """Share of tokens reaching the ``lower`` and ``external`` probes.

        Measured against tables built from the training split alone, over forms
        those tables never saw. ``None`` when there is nothing to measure.
        """

    def write_sources(
        self, destination: str, variant: str, thresholds: tuple[float, float]
    ) -> list[tuple[str, int]]:
        """Write the tables and the string table a compiler reads.

        Args:
            destination: Directory to create and fill.
            variant: Which parts of the lexicon to ship.
            thresholds: Confidence gates, baked into the rule table here so a
                reader needs no floating-point comparison at all.

        Returns:
            How many rows each table holds, in writing order.

        Raises:
            OSError: If the destination cannot be created or written.
        """

    def __len__(self) -> int:
        """How many entries the four tables hold together."""

def build(
    training: list[tuple[str, str, bool]],
    external: list[tuple[str, str]],
    entities: list[tuple[str, str]] = ...,
) -> Tables:
    """Learn the four lookup tables from annotated tokens.

    Args:
        training: ``(form, lemma, sentence_initial)`` per scorable token.
        external: ``(form, lemma)`` from a dictionary, used only for forms the
            corpus does not cover.
        entities: Curated named-entity ``(surface, lemma)`` pairs. Conflicting
            lemmas for one exact surface fail closed to identity; entity-link
            aliases must not be supplied as morphological lemmas.

    Returns:
        The learned tables.
    """

def tune(
    tables: Tables, development: list[tuple[str, str | None, bool]]
) -> tuple[float, float]:
    """Fit both confidence thresholds on a development split.

    The proper-noun gate is fitted on every capitalised mid-sentence form, not
    on the ``PROPN`` tag no runtime caller has.

    Args:
        tables: What :func:`build` learned from the train split.
        development: ``(form, lemma, sentence_initial)``; a lemma of ``None``
            marks a position the corpus left unannotated.

    Returns:
        The general and proper-noun thresholds, both zero when the split holds
        nothing scorable.
    """

def choose(
    tables: Tables,
    thresholds: tuple[float, float],
    gold: list[tuple[str, str | None, bool]],
    selected_on: str,
) -> tuple[str, int, float, float] | None:
    """Score the five candidate configurations and return the one that ships.

    Simplest first; a later candidate must be strictly better, and every
    winner must beat copying by more than the sample's binomial error.

    Args:
        tables: What :func:`build` learned.
        thresholds: The general and proper-noun confidence gates.
        gold: ``(form, lemma, sentence_initial)`` per token.
        selected_on: Which sample ``gold`` is, ``"dev"`` or ``"train-cv"``.
            A train-only holdout is asked for more evidence than a
            development split.

    Returns:
        ``(variant, tokens, accuracy, baseline)``, or ``None`` when the gold
        data holds nothing scorable and there is nothing to decide on.
    """

def score(
    tables: Tables,
    thresholds: tuple[float, float],
    variant: str,
    gold: list[tuple[str, str | None, bool]],
) -> tuple[int, float, float] | None:
    """Measure one configuration against gold tokens.

    Returns:
        ``(tokens, accuracy, baseline)``, or ``None`` when nothing is scorable.

    Raises:
        ValueError: If ``variant`` is not one the lattice defines.
    """

@final
class Outcome:
    """What building one lexicon produced.

    A read-only record the pipeline turns into its manifest row. Instances
    come from :func:`build_lexicon`; there is no public constructor.
    """

    @property
    def generation(self) -> str:
        """Exact immutable generation produced by this build."""

    @property
    def source(self) -> str:
        """Where the training signal came from.

        ``<config>.train`` for a treebank split, ``<label>/<iso639_3>`` for an
        external source.
        """

    @property
    def training_tokens(self) -> int:
        """How many observations were learned from, after the refit."""

    @property
    def thresholds(self) -> tuple[float, float]:
        """The fitted general and proper-noun confidence gates."""

    @property
    def tuned_on_dev(self) -> bool:
        """Whether a development split, rather than train-only CV, fitted it."""

    @property
    def variant(self) -> str:
        """Which parts of the learned lexicon were shipped."""

    @property
    def selected_on(self) -> str | None:
        """``dev``, form-disjoint ``train-cv``, or ``None``."""

    @property
    def entries(self) -> list[tuple[str, int]]:
        """Row counts of the written tables, in writing order."""

    @property
    def traffic(self) -> list[tuple[str, float]]:
        """Measured probe shares for ``lower`` and ``external``.

        Empty when no development split existed to measure them on.
        """

    @property
    def gold_tokens(self) -> int | None:
        """How many gold tokens the lexicon was scored on."""

    @property
    def accuracy(self) -> float | None:
        """Exact-match accuracy on those tokens."""

    @property
    def baseline(self) -> float | None:
        """Accuracy of copying the surface form on the same tokens."""

    @property
    def beats_baseline(self) -> bool | None:
        """Whether the lexicon is at least as good as copying."""

    @property
    def selection(self) -> dict[str, object] | None:
        """The lattice receipt behind ``variant`` (``meta.json`` ``selection``)."""

    @property
    def observed_inputs(self) -> list[tuple[str, str, int, str]]:
        """Descriptor bytes consumed as ``(role, label, size, sha256)``."""

def build_lexicon(
    destination: str,
    *,
    train: str | None,
    dev: str | None,
    test: str | None,
    paradigms: list[tuple[str, str, bool, bool]],
    name: str,
    language: str,
    iso639_3: str,
    entities: list[tuple[str, str, bool]] = ...,
    expected_inputs: list[tuple[str, int, str]] = ...,
    publish: bool = ...,
    span_case: str | None = ...,
    span_multiword: str | None = ...,
    phrases: list[tuple[str, str]] = ...,
    proper_noun_threshold: float | None = ...,
    entity_lower: bool = ...,
    compound_marker: str | None = ...,
    token_case: str | None = ...,
    lemma_adapter: bool = ...,
    context_keys: bool = ...,
) -> Outcome:
    """Build one language's lexicon end to end, from paths rather than data.

    Learns, fits on dev or a form-disjoint train holdout, refits, and seals a
    compiled immutable generation in ``destination``, with the GIL released.

    Args:
        destination: Lexicon directory to create and fill.
        train: Path to the training split, or ``None`` when the catalogue says
            the treebank has none.
        dev: Path to the development split; may not exist.
        test: Path to the held-out test split; may not exist.
        paradigms: External sources as
            ``(label, path, lemma_first, normalized_unique)``, best first,
            already minus the excluded ones. The final flag is true only for
            a manifest-verified NFC-sorted form-unique table.
        name: The lexicon identifier, normally a treebank code.
        language: Human-readable language name.
        iso639_3: ISO 639-3 code.
        entities: Isolated entity sources as ``(label, path, lemma_first)``;
            each file uses mandatory ``surface<TAB>lemma<TAB>semantics`` rows.
        expected_inputs: Manifest-backed ``(path, size, sha256)`` declarations
            checked against the exact descriptors parsed by the builder.
        publish: Whether to switch ``current`` immediately. The bundle pipeline
            seals first and commits it together with the Python manifest.
        span_case: Accepted span case convention to stamp (``"transfer"``
            or ``"verbatim"``); ``None`` keeps the default.
        span_multiword: Accepted multiword convention (``"identity"``,
            ``"compose"`` or ``"head"``); ``None`` keeps the default.
        phrases: Accepted ``(span key, pattern)`` rows; empty writes no table.
        proper_noun_threshold: Accepted capital-bucket threshold; ``None``
            keeps the tuned value.
        entity_lower: Whether the entity policy opened the lowercase retry.
        compound_marker: Accepted compound-boundary marker for external lemmas.
        token_case: Accepted token case convention (``"keep"`` or
            ``"surface"``); ``None`` keeps the default.
        lemma_adapter: Whether to respell external lemmas like the train split.
        context_keys: Whether to file context keys for ``upos=``/``context=``.

    Returns:
        What was produced. A language without usable common-word training data
        receives a sealed identity generation, optionally retaining its entity
        tables.

    Raises:
        OSError: If a split or table cannot be read, or the destination cannot
            be written.
        ValueError: If a corpus line is not valid JSON.
    """

# Native transaction helpers used by ``udlex.user_dictionary``.
def _user_dictionary_init(
    root: str,
    iso639_3: str,
    actor: str,
    reason: str,
) -> tuple[bool, str | None, int]: ...
def _user_dictionary_set(
    root: str,
    iso639_3: str,
    form: str,
    lemma: str,
    match_mode: str,
    replace: bool,
    actor: str,
    reason: str,
) -> tuple[bool, str | None, int]: ...
def _user_dictionary_remove(
    root: str,
    iso639_3: str,
    form: str,
    match_mode: str,
    actor: str,
    reason: str,
) -> tuple[bool, str | None, int]: ...
def _user_dictionary_import(
    root: str,
    iso639_3: str,
    path: str,
    replace: bool,
    actor: str,
    reason: str,
) -> tuple[bool, str | None, int]: ...
def _user_dictionary_rollback(
    root: str,
    iso639_3: str,
    generation: str,
    actor: str,
    reason: str,
) -> tuple[bool, str | None, int]: ...
def _user_dictionary_entries(
    root: str, iso639_3: str
) -> list[tuple[str, str, Literal["exact", "lower"]]]: ...
def _user_dictionary_history(root: str, iso639_3: str) -> list[dict[str, object]]: ...
def _user_dictionary_verify(root: str) -> tuple[int, int, int]: ...
