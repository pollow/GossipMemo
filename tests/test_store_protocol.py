"""Guard the `WorldStore` Protocol against drifting from its consumers."""

from __future__ import annotations

import ast
from pathlib import Path

from gossipmemo.store import WorldStore

REASONERS_DIR = Path(__file__).resolve().parents[1] / "src" / "gossipmemo" / "reasoners"


def _referenced_store_attrs() -> set[str]:
    """Collect every `store.<attr>` / `self.store.<attr>` access in the reasoners."""

    def is_store(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "store"
        return isinstance(node, ast.Attribute) and node.attr == "store"

    attrs: set[str] = set()
    for path in sorted(REASONERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and is_store(node.value):
                attrs.add(node.attr)
    return attrs


def _protocol_methods() -> set[str]:
    return {
        name
        for name, value in vars(WorldStore).items()
        if not name.startswith("_") and callable(value)
    }


def test_protocol_matches_reasoner_usage() -> None:
    assert _protocol_methods() == _referenced_store_attrs()
