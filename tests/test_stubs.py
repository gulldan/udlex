"""The type stub must describe the compiled extension, name by name."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from udlex import _udlex

STUB = Path(__file__).resolve().parent.parent / "src" / "udlex" / "_udlex.pyi"


def _stub() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _public(owner: object) -> set[str]:
    """Public names an owner exposes, ignoring nested (maturin packaging) modules."""
    return {
        name
        for name in dir(owner)
        if not name.startswith("_")
        and not isinstance(getattr(owner, name, None), ModuleType)
    }


def test_the_stub_declares_every_name_the_module_exports() -> None:
    declared = {
        node.name
        for node in _stub().body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    } | {
        target.id
        for node in _stub().body
        if isinstance(node, ast.AnnAssign)
        and isinstance(target := node.target, ast.Name)
    }
    assert _public(_udlex) <= declared


def test_the_module_exports_every_name_the_stub_declares() -> None:
    declared = {
        node.name
        for node in _stub().body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
        and not node.name.startswith("_")
    }
    assert declared <= _public(_udlex)


def test_the_stub_describes_every_member_of_lexicon() -> None:
    stubbed = {
        node.name
        for klass in _stub().body
        if isinstance(klass, ast.ClassDef) and klass.name == "Lexicon"
        for node in klass.body
        if isinstance(node, ast.FunctionDef)
    }
    actual = _public(_udlex.Lexicon)
    assert actual <= stubbed, f"missing from the stub: {sorted(actual - stubbed)}"


def test_the_shipped_package_is_marked_as_typed() -> None:
    """`py.typed` and the stub travel with the extension."""
    package = Path(_udlex.__file__).parent
    assert (package / "py.typed").exists()
    assert (package / "_udlex.pyi").exists()
    assert (package / "__init__.py").exists()
