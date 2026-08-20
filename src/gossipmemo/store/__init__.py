"""World store: SQLite persistence plus the pure domain policy it applies.

`SqliteWorldStore` is one SQL surface split across `_base` and the domain
mixins that build on it. The mixins are file organization, not an extension
seam: nothing registers or discovers them, they are composed in a fixed
chain, and only the concrete class in `sqlite` is ever instantiated.
"""

from __future__ import annotations

from ._errors import AmbiguousPersonError, PersonMergeError
from ._messages import DEFAULT_EXTRACTION_COMPARISON_LIMIT, PendingExtraction
from .sqlite import SqliteWorldStore, WorldStore

__all__ = [
    "DEFAULT_EXTRACTION_COMPARISON_LIMIT",
    "AmbiguousPersonError",
    "PendingExtraction",
    "PersonMergeError",
    "SqliteWorldStore",
    "WorldStore",
]
