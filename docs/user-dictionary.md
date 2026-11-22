# User dictionary

A user dictionary holds your own `form → lemma` overrides. It is checked
before the shipped tables, so it fixes a wrong lemma or adds a missing word
without rebuilding a package.

It lives in its own directory, the *user bundle*, never inside a dictionary
package, so reinstalling or upgrading a package does not touch it. Every change
writes a new immutable generation with an audit receipt (who, why, what
changed) and then switches to it atomically; a failed write leaves the previous
generation active. One user bundle holds any number of languages.

## Match modes

- `exact` (default): the token must match the form exactly.
- `lower`: the form is stored lowercased and matches any casing of the token.

An exact override answers with source `user`, a lowercase one with
`user_lower`.

## From Python

```py
import udlex

overrides = udlex.UserDictionary("overrides", "eng")
overrides.init(reason="product names")
overrides.set("udlex", "udlex", reason="keep the name")
overrides.set("Geese", "goose", match="lower", reason="irregular plural")
overrides.set("mice", "mouse", replace=True, reason="fix")

lexicon = udlex.load("eng", user_bundle="overrides")
print(lexicon.lemmatize(["GEESE", "mice"], info=True))
```

Every change takes a non-empty `reason`; `actor` defaults to the local user
name. Adding a form that already has an override fails unless you pass
`replace=True`. `remove(form, match=..., reason=...)` deletes one; removing a
missing form is a no-op.

`import_file(path, reason=...)` merges a strict headerless TSV file with one
`form<TAB>lemma<TAB>mode` row per line, where `mode` is `exact` or `lower`;
conflicts need `replace=True`. `entries()` lists the active overrides,
`history()` the receipts, oldest first, and `rollback(generation, reason=...)`
copies an old generation into a new one. `udlex.verify_user_bundle(path)`
authenticates every language and generation in a user bundle.

A loaded `Lexicon` keeps the generation it was opened with. Load it again to
see later changes.

## From the shell

Pass the root with `--user-bundle` or set `UDLEX_USER_BUNDLE`.

```sh
export UDLEX_USER_BUNDLE=overrides
udlex user-dict init eng --reason "product names"
udlex user-dict set eng Geese goose --match lower --reason "irregular plural"
udlex user-dict import eng fixes.tsv --reason "review batch 12"
udlex user-dict list eng
udlex user-dict status eng          # what each override changes in the base dictionary
udlex user-dict history eng
udlex user-dict rollback eng <generation> --reason "undo batch 12"
udlex user-dict verify
udlex lemma eng geese               # uses the overrides
udlex --no-user lemma eng geese     # validates them but answers from the base only
```

`status` compares every override with the base dictionary: `new_coverage`
(no base table knows the form), `redundant` (the base already gives the same
lemma), `identity_guard` (the override keeps the form as its own lemma where
the base changes it) and `changed_answer` (any other different lemma). Most
commands take `--json` for machine-readable output.

Restrict write access to the user bundle to the accounts that manage it: its
entries take precedence over everything shipped.
