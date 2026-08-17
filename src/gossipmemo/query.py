"""Query-synthesis prompt and its transport-driven call.

Query synthesis stays a single read-only model call with no watermark to
commit, so it has no `Reasoner`; this module co-locates its prompt with
`synthesize`, the one place that drives it over `LlmTransport`.
"""

from __future__ import annotations

from .models import QueryContext
from .priority import TIER_FOREGROUND, llm_call_tier
from .prompts import _json
from .transport import ChatMessage, LlmTransport

QUERY_SYNTHESIS_SYSTEM_PROMPT = """Answer the read-only question using the supplied
social-memory context. Return concise plain text only (no JSON wrapper or code
fence). Use facts and supported inferences in the context to give a direct,
useful answer; distinguish uncertainty and current conditions from historical
events. Separate Person records are not evidence that they represent different
real people; when identities may overlap, state the ambiguity instead of asserting
a distinction. Do not invent facts or claim that anything was saved. Answer in
the language of the question.
"""


def query_synthesis_prompt(question: str, context: QueryContext) -> str:
    """Build the user prompt for read-only query synthesis."""

    return (
        "Question:\n"
        + question
        + "\n\nContext (people, relationships, and memories):\n"
        + _json(context)
    )


async def synthesize(transport: LlmTransport, question: str, context: QueryContext) -> str:
    """Answer `question` in one foreground, unstructured call over `transport`."""

    if not question.strip():
        raise ValueError("query question must not be empty")
    with llm_call_tier(TIER_FOREGROUND, "query"):
        request = transport.prepare(
            [
                ChatMessage(role="system",
                            content=QUERY_SYNTHESIS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=query_synthesis_prompt(
                    question, context)),
            ],
            structured=False,
        )
        content = await transport.complete(request)
    return content.strip()


__all__ = ["QUERY_SYNTHESIS_SYSTEM_PROMPT",
           "query_synthesis_prompt", "synthesize"]
