"""The `Reasoner` seam shared by every induction pass.

A reasoner runs itself to catch-up in one space::

    await reasoner.run_until_caught_up(space_id, should_continue)

That is the whole protocol. One bounded unit of work -- load context from
the store, call the model, commit with an optimistic watermark check, and
report whether more remains -- is `_attempt`, an implementation detail
behind that method: callers have no reason to drive a single unit, and the
"call me again?" bit is meaningless without the loop that honors it.
`should_continue` is re-checked before each unit, so a caller that stops
(`SocialMemoryWorld._stopping`) lands between units, never inside one.

Provider-side serialization and priority live in `llm.ProviderGate`, not
here: a reasoner's model call sets the active tier via `llm_call_tier` and
lets the adapter serialize at the single HTTP request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ..priority import TIER_BACKGROUND, llm_call_tier


def _always() -> bool:
    return True


class Reasoner(Protocol):
    name: str

    async def run_until_caught_up(
        self, space_id: str, should_continue: Callable[[], bool] = _always
    ) -> None:
        """Do work in this space until none remains, or we are told to stop."""
        ...


class AttemptLoop:
    """The one catch-up loop, shared by every reasoner implementation."""

    name: str

    async def _attempt(self, space_id: str) -> bool:
        """Do one bounded unit of work. Return True to be called again."""
        raise NotImplementedError

    async def run_until_caught_up(
        self, space_id: str, should_continue: Callable[[], bool] = _always
    ) -> None:
        while should_continue() and await self._attempt(space_id):
            pass


class DescriptorReasoner(AttemptLoop):
    """Generic single-context reasoner: load -> call model -> apply -> decide.

    `load_context` also decides whether there is anything to do: it returns
    a falsy value when the reasoner should stop. `call` turns that context
    into the model invocation -- a label, the bound model method, and its
    positional args -- and is the ONLY step `attempt` awaits directly, under
    `tier`. `continue_when` defaults to "retry only on a watermark
    conflict."

    `call` may also return `None` to skip both the model call and `apply` --
    for example when a reasoner enumerating several candidate targets finds
    that its chosen target vanished or is no longer stale but other targets
    remain. `_attempt` then calls `continue_when(context, None, False)`
    directly, so `continue_when` alone decides whether another attempt is
    warranted.
    """

    def __init__(
        self,
        name: str,
        load_context: Callable[[str], Any],
        call: Callable[
            [str, Any],
            tuple[str, Callable[..., Awaitable[Any]], tuple[Any, ...]] | None,
        ],
        apply: Callable[[str, Any, Any], bool],
        continue_when: Callable[[Any, Any, bool], bool] | None = None,
        tier: int = TIER_BACKGROUND,
    ) -> None:
        self.name = name
        self._load_context = load_context
        self._call = call
        self._apply = apply
        self._continue_when = continue_when or (
            lambda context, result, applied: not applied)
        self._tier = tier

    async def _attempt(self, space_id: str) -> bool:
        context = self._load_context(space_id)
        if not context:
            return False
        call = self._call(space_id, context)
        if call is None:
            return self._continue_when(context, None, False)
        label, method, args = call
        with llm_call_tier(self._tier, label):
            result = await method(*args)
        applied = self._apply(space_id, context, result)
        return self._continue_when(context, result, applied)

    # Seams reserved for a later unification pass across reasoners. Neither
    # is implemented or called today: each reasoner still reads its own
    # bounded context and applies its own single result exactly as before.
    def read_evidence_page(self, context: Any) -> Any:
        raise NotImplementedError

    def aggregate_partial_results(self, results: list[Any]) -> Any:
        raise NotImplementedError


__all__ = ["AttemptLoop", "DescriptorReasoner", "Reasoner"]
