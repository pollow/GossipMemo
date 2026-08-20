"""Errors raised by the world store."""

from __future__ import annotations


class AmbiguousPersonError(ValueError):
    def __init__(self, reference: str) -> None:
        super().__init__(f"person reference is ambiguous: {reference}")
        self.reference = reference


class PersonMergeError(ValueError):
    """Raised when an explicit person merge cannot be applied safely."""

    def __init__(self, message: str, *, conflict: bool = False) -> None:
        super().__init__(message)
        self.conflict = conflict
