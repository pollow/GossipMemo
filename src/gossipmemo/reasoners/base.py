"""The `Reasoner` seam shared by every induction pass.

A reasoner owns exactly one attempt at one unit of work: load context from
the store, call the model through the single FIFO queue, commit the result
with an optimistic watermark check, and report whether more work remains.
The driver (`SocialMemoryWorld`) owns only the loop and the `_stopping`
flag; it never reaches into a reasoner's internals::

    while not stopping and await reasoner.attempt(space_id):
        pass
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ..queue import ReasonerCallQueue


class Reasoner(Protocol):
    name: str

    async def attempt(self, space_id: str) -> bool:
        """Do one bounded unit of work. Return True to be called again."""
        ...


class DescriptorReasoner:
    """Generic single-context reasoner: load -> call model -> apply -> decide.

    `load_context` also decides whether there is anything to do: it returns
    a falsy value when the reasoner should stop. `call` turns that context
    into the queue submission -- a label, the bound model method, and its
    positional args -- and is the ONLY step `attempt` wraps in
    `queue.submit`, so the store read (`load_context`) and the store write
    (`apply`) it brackets stay outside the queue. `continue_when` defaults to
    "retry only on a watermark conflict."

    A reasoner whose continue-logic does not fit this shape (for example one
    that must enumerate several stale targets per attempt) implements the
    `Reasoner` protocol directly instead of using this descriptor.
    """

    def __init__(
        self,
        name: str,
        queue: ReasonerCallQueue,
        load_context: Callable[[str], Any],
        call: Callable[[str, Any], tuple[str, Callable[..., Awaitable[Any]], tuple[Any, ...]]],
        apply: Callable[[str, Any, Any], bool],
        continue_when: Callable[[Any, Any, bool], bool] | None = None,
    ) -> None:
        self.name = name
        self.queue = queue
        self._load_context = load_context
        self._call = call
        self._apply = apply
        self._continue_when = continue_when or (lambda context, result, applied: not applied)

    async def attempt(self, space_id: str) -> bool:
        context = self._load_context(space_id)
        if not context:
            return False
        label, method, args = self._call(space_id, context)
        result = await self.queue.submit(label, method, *args)
        applied = self._apply(space_id, context, result)
        return self._continue_when(context, result, applied)

    # Seams reserved for a later unification pass across reasoners. Neither
    # is implemented or called today: each reasoner still reads its own
    # bounded context and applies its own single result exactly as before.
    def read_evidence_page(self, context: Any) -> Any:
        raise NotImplementedError

    def aggregate_partial_results(self, results: list[Any]) -> Any:
        raise NotImplementedError


__all__ = ["DescriptorReasoner", "Reasoner"]
