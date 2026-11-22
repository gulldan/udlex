"""Typed provenance shared by the Python package and native extension."""

from enum import Enum, unique


@unique
class Source(Enum):
    """The final stage that produced a lemma.

    The lowercase values are stable for logs and serialization.  Compare
    members as enum values (``info.source is Source.EXACT``), not as strings.
    """

    ENTITY = "entity"
    ENTITY_LOWER = "entity_lower"
    EXACT = "exact"
    LOWER = "lower"
    EXACT_LOWER = "exact_lower"
    EXTERNAL = "external"
    EXTERNAL_LOWER = "external_lower"
    RULE = "rule"
    IDENTITY = "identity"
    PROTECTED = "protected"
    COMPOSED = "composed"
    PHRASE = "phrase"
    USER = "user"
    USER_LOWER = "user_lower"
    CONTEXT = "context"

    def is_known(self) -> bool:
        """Whether the final answer came from a stored vocabulary entry."""
        return self in {
            Source.ENTITY,
            Source.ENTITY_LOWER,
            Source.EXACT,
            Source.LOWER,
            Source.EXACT_LOWER,
            Source.EXTERNAL,
            Source.EXTERNAL_LOWER,
            Source.USER,
            Source.USER_LOWER,
            Source.CONTEXT,
        }

    def is_lemmatized(self) -> bool:
        """Whether a lemmatization stage produced the final answer."""
        return self not in {Source.IDENTITY, Source.PROTECTED}

    def __str__(self) -> str:
        """Return the stable lowercase value used in logs."""
        return self.value


__all__ = ["Source"]
