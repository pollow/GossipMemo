"""Query-synthesis prompt and its transport-driven call.

Query synthesis stays a single read-only model call with no watermark to
commit, so it has no `Reasoner`; this module holds its user-prompt builder and
`synthesize`, the one place that drives it over `LlmTransport`. Having no
`ReasoningSettings`, it takes a `PromptLibrary` directly, so its system prompt
is overridable exactly like a reasoner's.
"""

from __future__ import annotations

from .models import QueryContext
from .priority import TIER_FOREGROUND, llm_call_tier
from .prompts import PromptLibrary
from .prompts.render import _json
from .transport import ChatMessage, LlmTransport


def query_synthesis_prompt(question: str, context: QueryContext) -> str:
    """Build the user prompt for read-only query synthesis."""

    return (
        "Question:\n"
        + question
        + "\n\nContext (people, relationships, and memories):\n"
        + _json(context)
    )


async def synthesize(
    transport: LlmTransport, question: str, context: QueryContext, prompts: PromptLibrary,
) -> str:
    """Answer `question` in one foreground, unstructured call over `transport`."""

    if not question.strip():
        raise ValueError("query question must not be empty")
    with llm_call_tier(TIER_FOREGROUND, "query"):
        request = transport.prepare(
            [
                ChatMessage(role="system", content=prompts.query_synthesis_system),
                ChatMessage(role="user", content=query_synthesis_prompt(question, context)),
            ],
            structured=False,
        )
        content = await transport.complete(request)
    return content.strip()


__all__ = ["query_synthesis_prompt", "synthesize"]
