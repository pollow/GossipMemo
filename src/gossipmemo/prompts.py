"""Prompt builders used by the structured LLM adapter.

Keeping prompts in a separate module gives the application one place to
version and test the model contract.  The builders return complete user
messages; the adapter supplies the corresponding system instruction and
serializes the result schema into it.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .models import (
    ExtractionResult,
    MemoryView,
    ModelMessage,
    PersonReasoningResult,
    PersonView,
    UserModelView,
    QueryContext,
    RelationshipReasoningResult,
    RelationshipView,
    ContinuityReasoningResult,
    ContinuityView,
    HypothesisView,
    CoverageMapView,
    LearningGoalView,
    LearningGoalCandidate,
)


EXTRACTION_SYSTEM_PROMPT = """Extract useful, provenance-aware memories from the messages.
Return only the supplied JSON schema. Keep the original meaning, speaker, and
uncertainty. Extract explicit facts, events, preferences, plans, and situations;
do not make broad personality inferences from one conversation. Leave recurring
patterns to reasoning passes. Use stated, observed, or reported basis; never use
an inferred basis for extraction. Resolve relative dates using each message's
occurred_at. A reported claim must remain attributed. The user is not a Person;
two people appearing
together do not by themselves establish a relationship. Use the language that
best matches the dominant language of current user evidence for every generated
natural-language field, including new display names; keep IDs and enum values unchanged.
"""

PERSON_REASONING_SYSTEM_PROMPT = """Reason carefully about one named Person from the
supplied owner context. Linked memories indicate
relevance to the target, not that the target is the semantic subject of every
memory. Do not transfer the current user's or any co-occurring person's traits,
preferences, intentions, or actions onto the target. Recurring patterns require
multiple distinct source memories; a narrow impression from one highly diagnostic
event is allowed when calibrated to that evidence. Identify supported patterns in behavior, preferences,
communication, decision-making, sensitivities, and helpful ways to interact.
Make reasonable social inferences when supported, with uncertainty proportional
to evidence. Do not use projections, inferred memories, or hypotheses as evidence.
Distinguish current conditions from historical events. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""


RELATIONSHIP_REASONING_SYSTEM_PROMPT = """Reason carefully about the relationship between
the two endpoint People in the supplied owner context. Linked memories indicate relevance to the
endpoints, not relationship evidence by themselves. Do not transfer the current
user's or either endpoint's standalone traits, preferences, intentions, or actions
into a relationship claim. Mere co-occurrence is not relationship evidence. Look
for recurring interaction patterns, cooperation, friction, trust, initiative, and
meaningful changes in closeness, tone, or status. Recurring patterns require
multiple distinct source memories; a narrow inference from one highly diagnostic
interaction is allowed when calibrated to that evidence. Do not use projections,
inferred memories, or hypotheses as evidence. Distinguish current from historical
conditions. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""

CONTINUITY_SYSTEM_PROMPT = """Rebuild compact cross-session continuity.
Return only the supplied JSON schema. Keep ongoing threads, recent decisions,
pending actions, and context useful for the next conversation. Do not make
long-term personality inferences or copy person/user profiles; the current user
is not a Person. Use the language that best matches supplied messages and prior
continuity; keep IDs and enum values unchanged.
"""

USER_MODEL_REASONING_SYSTEM_PROMPT = """Reason carefully about the fixed current user from
active memories marked about_user. Capture preferences, communication preferences, goals, current
situations, and practical interaction guidance. Generalize recurring patterns
when supported, but do not turn a one-off event into a stable trait or include
another person's identity. Use valid_from and valid_to to separate current
conditions from historical events. Do not use projections, inferred memories, or
hypotheses as evidence. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""

COVERAGE_AUDIT_SYSTEM_PROMPT = """Audit long-term autobiographical and persona coverage.
Return only the supplied JSON schema. Coverage is a summary of supported evidence,
not a profile and not an invitation to disclose. A hypothesis may identify an edge,
blind spot, or conflict, but never raises a coverage level. Preserve uncertainty:
unknown, private, or deferred material is valid and must not be treated as a gap to
press. Do not diagnose pathology. Use the supplied memory IDs exactly; do not invent
facts, evidence, or private details. Keep natural-language summaries concise and in
the language of the evidence."""

GOAL_PLANNING_SYSTEM_PROMPT = """Plan a very small number of optional, user-owned
learning invitations from an updated coverage map. Return only the supplied JSON
schema. Every new or changed goal must cite supplied criterion and boundary IDs.
Unknown, intimate, traumatic, stigmatized, or otherwise private areas are never
automatic targets: respect explicit readiness, consent, defer, and do-not-pursue
signals. Prefer gentle, specific questions with an easy opt-out; never pressure,
diagnose, or imply that disclosure is owed. Omission is no-op: only explicitly
transition an existing supplied goal when its lifecycle changes."""

# Stable parent IDs deliberately have richer prompt-only facets. The stored map is
# compact, while this readable rubric lets audit boundaries name meaningful blind
# spots without turning sensitive life material into a normalized schema.
COVERAGE_RUBRIC = """M1 life_chapters — Which eras, beginnings, moves, endings, and chapters are legible? Rich when chronology and transitions have texture; blind spots: childhood, family origin, education, work, migration, future chapters.
M2 everyday_life — What does ordinary life, routine, home, work, care, money, and security feel like? Rich when habits and constraints have context; blind spots: class, housing, debt, caregiving, disability access.
M3 turning_points — Which choices, accidents, losses, recoveries, and reversals changed the story? Rich when consequences and alternatives are known; blind spots: regret, repair, harm done, survival.
M4 people_and_relationship_arcs — Which attachments, ruptures, loyalties, intimacy, and family/friend arcs matter? Rich when change and boundaries are clear; blind spots: sexuality, consent, estrangement, reconciliation.
M5 places_and_context — Which places, communities, cultures, institutions, and historical contexts shape meaning? Rich when belonging and constraint are visible; blind spots: religion, politics, class, diaspora, taboo contexts.
M6 lived_scenes — What concrete scenes, sensory memories, conversations, and small moments carry the story? Rich when scenes ground abstractions; blind spots: body, health, illness, substance use, private rituals.
M7 inner_experience — How are feelings, needs, shame, guilt, secrets, fear, desire, grief, and self-protection described? Rich when the user's own meaning is present; blind spots: unspoken or explicitly private inner life.
M8 themes_and_change — What recurring themes, tensions, growth, and contradictions span time? Rich when evidence supports change and exceptions; blind spots: envy, resentment, moral injury, forgiveness, repair.
M9 unresolved_threads — What questions, conflicts, decisions, losses, or hopes remain open? Rich when uncertainty and next steps are named; blind spots: mortality, legacy, unfinished goodbyes.
P1 identity_and_self_story — How does the user name self, belonging, and self-understanding? Rich when self-authored and contextual; blind spots: protected identities and labels not offered.
P2 values_and_tradeoffs — What matters and what costs are accepted? Rich when choices reveal tensions; blind spots: morality, faith, politics, loyalty, money.
P3 worldview_and_beliefs — What assumptions, beliefs, and sources of meaning guide interpretation? Rich when complexity and change are represented; blind spots: religion, ideology, taboos.
P4 goals_motives_fears — What pulls the user forward or holds them back? Rich when motives and constraints are situated; blind spots: safety, status, intimacy, mortality.
P5 reasoning_and_decisions — How does the user decide under uncertainty, conflict, or pressure? Rich when strategies and exceptions are supported; blind spots: avoidance, risk, regret.
P6 emotional_patterns — What non-clinical emotional rhythms and coping patterns recur? Rich when grounded across contexts; blind spots: grief, shame, anger, substance-related coping. Never diagnose.
P7 social_style_and_boundaries — How does the user connect, communicate, protect space, and repair? Rich when context and consent are clear; blind spots: conflict, attachment, intimacy.
P8 preferences_and_routines — What tastes, practices, environments, and practical rhythms recur? Rich when stable versus situational preferences are separated; blind spots: health, money, accessibility constraints.
P9 voice_and_expression — How does the user tell stories, joke, ask, withhold, create, or communicate? Rich when examples span settings; blind spots: silence and code-switching.
P10 context_and_exceptions — Which roles, settings, identities, pressures, and exceptions change the pattern? Rich when it prevents overgeneralization; blind spots: home/work, safety, power, culture.
P11 skills_and_knowledge — What has the user learned, practiced, taught, or become capable of? Rich when confidence and limits are clear; blind spots: informal knowledge and blocked opportunities."""

# Method cues are deliberately non-clinical: they guide respectful inquiry and
# audit interpretation, not treatment or a demand for disclosure.
COVERAGE_METHOD = """Scan with memoir and oral-history facets: origins; childhood and
adolescence; education; work; moves; partnership/parenthood; high and low scenes;
failure, pride, kindness, loneliness; choice, consequence, and meaning; caregivers,
siblings, mentors, rivals, children, absent or lost people; homes, neighborhood,
language, institutions and history; scene details (where, when, who, dialogue,
senses, objects, action, emotion); desire, sexuality, intimacy, body, trauma and
coping; agency and communion; multiple futures, apologies, repair, mortality and
legacy. For persona also scan roles, belonging, gender/sexuality, ethnicity/class,
masks, ideal/feared self; autonomy/security, loyalty/truth, care/fairness,
achievement/rest, fidelity/taboo/harm/forgiveness; trust, justice, spirituality,
meaning/death and epistemology; attention, evidence, intuition, planning, risk and
changing one's mind; emotional triggers/regulation; closeness, power, help and
repair without attachment labels; sleep, sensory life, technology, money, rituals;
register, humor, profanity, metaphor, silence and persuasion; work/home/intimate/
public/crisis exceptions; tacit heuristics, teaching, creative and practical skill.
Use partnership, evocation, acceptance, safety, trust, collaboration, voice and
choice. Evidence-backed non-clinical patterns are allowed; direct pathology diagnosis
is not."""

QUERY_SYNTHESIS_SYSTEM_PROMPT = """Answer the read-only question using the supplied
social-memory context. Return concise plain text only (no JSON wrapper or code
fence). Use facts and supported inferences in the context to give a direct,
useful answer; distinguish uncertainty and current conditions from historical
events. Separate Person records are not evidence that they represent different
real people; when identities may overlap, state the ambiguity instead of asserting
a distinction. Do not invent facts or claim that anything was saved. Answer in
the language of the question.
"""


def schema_instruction(result_type: type[BaseModel]) -> str:
    """Return a compact instruction containing a Pydantic JSON schema."""

    schema = result_type.model_json_schema()
    return "Output schema (JSON Schema):\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )


def _json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def extraction_prompt(
    messages: list[ModelMessage],
    policy: str = "balanced",
    context: list[ModelMessage] | tuple[ModelMessage, ...] = (),
    known_people: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    user_name: str = "CurrentUser",
    comparison_memories: list[MemoryView] | tuple[MemoryView, ...] = (),
) -> str:
    """Build the user prompt for :class:`~models.ExtractionResult`."""

    policy_text = {
        "conservative": (
            "Keep only explicit, durable facts and meaningful events; skip "
            "isolated transient details such as 'today I had coffee'."
        ),
        "balanced": (
            "Keep explicit durable information and a transient detail only when "
            "it affects an ongoing situation or helps reveal a recurring pattern."
        ),
        "comprehensive": (
            "Keep explicit information plus relevant short-lived events as dated "
            "evidence, including isolated details that may reveal a later pattern."
        ),
    }
    if policy not in policy_text:
        raise ValueError(f"unknown extraction policy: {policy}")
    evidence_messages = [message for message in messages if message.author == "user"]
    assistant_messages = [
        message for message in messages if message.author == "assistant"
    ]
    return (
        f"Use the server's {policy} extraction policy for the whole batch.\n"
        f"{policy_text[policy]}\n"
        f"The fixed current user is named {user_name!r}. Use that name when referring "
        "to the current user; the current user is not a Person. User-authored "
        "messages in the current batch are the only evidence allowed to create "
        "memories. Assistant-authored messages in the batch and recent context are "
        "context only: use them to resolve references and conversational meaning, "
        "but never save their restatements, summaries, analyses, or advice as new "
        "memories. Assistant content may supply a proposition only when a current "
        "user evidence message explicitly confirms, adopts, or corrects it.\n"
        "The current user/assistant author role is context and never a Person. List every "
        "Person referenced by a memory in its `people` refs; express who said or "
        "did what in the memory content itself. Preserve reported claims as "
        "`reported`, not facts. Set `about_user` for a claim or event about "
        f"{user_name}; {user_name} must never appear in "
        "`people`. "
        "Record valid_from/valid_to when the message gives a time bound.\n\n"
        "Recent context (context only):\n"
        + _json(list(context))
        + "\n\nKnown people (identity hints only):\n"
        + _json(list(known_people))
        + "\nIn natural-language memory content and generated display fields, use "
        "a known person's canonical display_name when the messages refer to them. "
        "In every ExtractedMemory.people and ExtractedRelationship.person_a_ref/"
        "person_b_ref, use the supplied stable Person `id` (never a display_name "
        "or alias). "
        "If the messages explicitly introduce a new short name, return "
        "it in that person's `aliases` field. Omit a known person unless a new "
        "memory references them or the messages explicitly add an alias. Do not "
        "echo the known-people list.\n"
        + "\n\nCurrent batch evidence (user-authored; the only messages allowed to produce memories):\n"
        + _json(evidence_messages)
        + "\n\nCurrent batch context (assistant-authored; context only):\n"
        + _json(assistant_messages)
        + "\n\nComparison memories (deduplication/update reference only; never new evidence):\n"
        + _json(list(comparison_memories))
        + "\nFor comparison memories only: omit a memory when the current user batch merely repeats it. "
        "When current user evidence explicitly corrects, updates, or refines one, emit the new memory "
        "and set `supersedes_memory_id` to that supplied comparison memory ID. Never copy details from a "
        "comparison memory unless those details also appear in current user evidence. Do not use an inferred "
        "comparison memory as evidence."
    )


def person_reasoning_prompt(
    person: PersonView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
    user_name: str = "CurrentUser",
) -> str:
    """Build the user prompt for a person projection refresh."""

    return (
        f"The fixed current user is named {user_name!r}; the current user is not "
        "the target Person. Rebuild the profile card for the target Person from "
        "the active memories. Linked memories indicate relevance, not semantic "
        "subject. Attribute traits, preferences, and actions to the target only "
        "when the memory supports that attribution. Use only the supplied context; "
        "prefer concise traits, preferences, current_state, and interaction_notes "
        f"that help {user_name} socialize.\n\nTarget Person:\n"
        + _json(person)
        + "\n\nActive memories:\n"
        + _json(list(memories))
    )


def relationship_reasoning_prompt(
    relationship: RelationshipView,
    memories: list[MemoryView] | tuple[MemoryView, ...],
    user_name: str = "CurrentUser",
) -> str:
    """Build the user prompt for a relationship projection refresh."""

    return (
        f"The fixed current user is named {user_name!r}; the current user is not "
        "an endpoint Person. Rebuild the projection for the relationship between "
        "the two endpoint People in the input. Linked memories indicate relevance, "
        "not relationship evidence; summarize only interactions between the endpoints. "
        "Use only the supplied context; do not transfer standalone traits, preferences, "
        "or actions into a relationship claim.\n\nTarget relationship:\n"
        + _json(relationship)
        + "\n\nRelevant memories:\n"
        + _json(list(memories))
    )


def _evidence_lines(memories: list[MemoryView] | tuple[MemoryView, ...] | list[Any] | tuple[Any, ...]) -> str:
    """Compact, injection-resistant enough-for-reading evidence representation."""
    lines = []
    for m in memories:
        if hasattr(m, "source_memory_ids"):
            lines.append("- digest=" + json.dumps(m.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + " (compressed evidence; IDs refer to original Memories)")
        else:
            lines.append(f"- id={m.id!r} kind={m.kind!r} basis={m.basis!r} derivation_sources={'unavailable' if m.basis == 'inferred' else 'n/a'} text={json.dumps(m.content, ensure_ascii=False)}")
    return "\n".join(lines) or "- (none)"


def _hypothesis_lines(hypotheses: list[HypothesisView] | tuple[HypothesisView, ...]) -> str:
    return "\n".join(
        f"- id={h.id!r} confidence={h.confidence!r} evidence={[e.memory_id for e in h.evidence]!r} text={json.dumps(h.content, ensure_ascii=False)}"
        for h in hypotheses
    ) or "- (none)"


def owner_reasoning_prefix(
    target: PersonView | RelationshipView | UserModelView,
    evidence_memories: list[MemoryView] | tuple[MemoryView, ...],
    inferred_memories: list[MemoryView] | tuple[MemoryView, ...],
    hypotheses: list[HypothesisView] | tuple[HypothesisView, ...],
    *, user_name: str = "CurrentUser",
) -> str:
    """Shared immutable prefix for both stages of an owner reasoning pair."""
    return (
        f"<owner-reasoning user={json.dumps(user_name)}>\n<target>\n"
        + _json(target) + "\n</target>\n<evidence-memories>\n"
        + _evidence_lines(evidence_memories) + "\n</evidence-memories>\n"
        + "<current-inferred-memories comparison-only=\"true\">\n"
        + _evidence_lines(inferred_memories) + "\n</current-inferred-memories>\n"
        + "<open-hypotheses comparison-only=\"true\">\n" + _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\nOnly evidence-memories are evidence. Current inferred memories and open hypotheses may be reviewed for duplication or explicit lifecycle actions, never used as support."
    )

def owner_evidence_digest_prompt(memories: list[Any], user_name: str = "CurrentUser") -> str:
    return ("Compress supplied raw evidence only. Preserve chronology, basis, uncertainty, "
            "contradictions, semantic subject, and exact source_memory_ids. Do not infer people, "
            "traits, or actions; never invent IDs.\n" + _json([
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in memories
            ]))


def projection_stage_prompt() -> str:
    return "<stage>Return only the requested projection/card. Do not output inferred-memory or hypothesis actions.</stage>"


def actions_stage_prompt() -> str:
    return "<stage>Review the projection above. Return only explicit inferred-memory and hypothesis actions. Omission is always no-op. IDs must be from supplied context.</stage>"


def user_model_reasoning_prompt(
    memories: list[MemoryView], user_name: str = "CurrentUser"
) -> str:
    return (
        f"Rebuild the compact profile card for {user_name} from these active "
        "memories only. Keep it bounded and useful for interaction.\n\nActive "
        "about-user memories:\n" + _json(memories)
    )


def coverage_audit_prompt(
    coverage: CoverageMapView, memories: list[MemoryView], hypotheses: list[HypothesisView]
) -> str:
    """Immutable audit prefix plus one bounded evidence chunk."""
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<current-coverage-map>\n" + _json(coverage) + "\n</current-coverage-map>\n"
        "<new-evidence>\n" + _evidence_lines(memories) + "\n</new-evidence>\n"
        "<open-hypotheses comparison-only=\"true\">\n" + _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\nApply a patch only for this chunk. Preserve prior coverage unless new evidence changes it. "
        "Hypotheses may add a boundary or conflict with hypothesis_id, never evidence; a hypothesis never raises a coverage level. "
        "Each criterion patch needs only its stable parent criterion_id. Keep inventories compact and additive."
    )


def goal_planning_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
) -> str:
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<updated-coverage-map>\n" + _json(coverage) + "\n</updated-coverage-map>\n"
        "<open-hypotheses>\n"
        + _json([item.model_dump(mode="json") for item in hypotheses])
        + "\n</open-hypotheses>\n<open-goals>\n"
        + _json([item.model_dump(mode="json") for item in open_goals])
        + "\n</open-goals>\n<recent-closed-goals>\n"
        + _json([item.model_dump(mode="json") for item in recent_closed_goals])
        + "\n</recent-closed-goals>\n"
        "Plan only after the map is caught up. Use trauma-informed partnership, choice, and an easy decline; "
        "do not equate a blind spot with a question to ask now. Goals can focus a person or relationship but remain user-owned."
    )


def goal_candidate_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
) -> str:
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<updated-coverage-map>\n" + _json(coverage) + "\n</updated-coverage-map>\n"
        "<open-hypotheses>\n" + _json(hypotheses) + "\n</open-hypotheses>\n"
        "<open-goals>\n" + _json(open_goals) + "\n</open-goals>\n"
        "<recent-closed-goals>\n" + _json(recent_closed_goals) + "\n</recent-closed-goals>\n"
        "Propose optional candidate invitations only. Candidates are non-mutating: do not "
        "transition, retire, defer, update, or otherwise change any goal lifecycle. Respect "
        "consent, privacy, and easy decline; cite only supplied criterion and boundary IDs."
    )


def goal_reconciliation_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
    candidates: list[LearningGoalCandidate],
) -> str:
    return (
        goal_planning_prompt(coverage, hypotheses, open_goals, recent_closed_goals)
        + "\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>\n"
        "Select or reconcile these candidates into the final result. This is the only lifecycle-mutating planning pass."
    )


def goal_candidate_reduction_prompt(candidates: list[LearningGoalCandidate]) -> str:
    return (
        "Deduplicate and compress these non-mutating learning-goal candidates. "
        "Return candidates only; never transition any lifecycle.\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>"
    )


def continuity_prompt(
    continuity: ContinuityView | None, messages: list[ModelMessage]
) -> str:
    return (
        "Rebuild continuity from the prior summary and newer raw messages. "
        "Choose the last supplied message as through_message_id.\n\nPrior continuity:\n"
        + _json(continuity)
        + "\n\nNew messages:\n"
        + _json(messages)
    )


def query_synthesis_prompt(question: str, context: QueryContext) -> str:
    """Build the user prompt for read-only query synthesis."""

    return (
        "Question:\n"
        + question
        + "\n\nContext (people, relationships, and memories):\n"
        + _json(context)
    )


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "extraction_prompt",
    "person_reasoning_prompt",
    "query_synthesis_prompt",
    "relationship_reasoning_prompt",
    "owner_reasoning_prefix",
    "projection_stage_prompt",
    "actions_stage_prompt",
    "user_model_reasoning_prompt",
    "continuity_prompt",
    "coverage_audit_prompt",
    "goal_planning_prompt",
    "goal_candidate_prompt",
    "goal_reconciliation_prompt",
    "goal_candidate_reduction_prompt",
    "schema_instruction",
]
