"""Prompt scaffolding shared across reasoners.

Each reasoner now owns its own system prompt(s) and user-prompt builder(s)
(see `gossipmemo/reasoners/*.py`). What stays here is genuinely cross-
reasoner: the JSON-schema instruction helper, compact evidence/hypothesis
rendering, the owner-reasoning family shared by person/relationship/
user_model, and the coverage rubric/method shared by coverage and all three
goal-planning prompts.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .models import (
    HypothesisView,
    MemoryView,
    PersonView,
    RelationshipView,
    UserModelView,
)


def schema_instruction(result_type: type[BaseModel]) -> str:
    """Return a compact instruction containing a Pydantic JSON schema."""

    schema = result_type.model_json_schema()
    return "Output schema (JSON Schema):\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )


def _plain(value: Any) -> Any:
    """Turn nested Pydantic models into JSON-serializable data.

    A bare model was already handled, but a *list* of models was not: it
    fell through to `json.dumps(default=str)`, which stringified each one
    with `repr`. Prompts that pass a list -- recent context messages,
    evidence memories -- were embedding `id='m1' space_id='s' ...` as a
    single opaque string per item rather than a JSON object.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, indent=2, default=str)


def _evidence_lines(memories: list[MemoryView] | tuple[MemoryView, ...] | list[Any] | tuple[Any, ...]) -> str:
    """Compact, injection-resistant enough-for-reading evidence representation."""
    lines = []
    for m in memories:
        if hasattr(m, "source_memory_ids"):
            lines.append("- digest=" + json.dumps(m.model_dump(mode="json"), ensure_ascii=False,
                         separators=(",", ":")) + " (compressed evidence; IDs refer to original Memories)")
        else:
            lines.append(
                f"- id={m.id!r} kind={m.kind!r} basis={m.basis!r} derivation_sources={'unavailable' if m.basis == 'inferred' else 'n/a'} text={json.dumps(m.content, ensure_ascii=False)}")
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
        + _evidence_lines(inferred_memories) +
        "\n</current-inferred-memories>\n"
        + "<open-hypotheses comparison-only=\"true\">\n" +
        _hypothesis_lines(hypotheses)
        + "\n</open-hypotheses>\nOnly evidence-memories are evidence. Current inferred memories and open hypotheses may be reviewed for duplication or explicit lifecycle actions, never used as support."
    )


def owner_evidence_digest_prompt(memories: list[Any], user_name: str = "CurrentUser") -> str:
    return ("Compress supplied raw evidence only. Preserve chronology, basis, uncertainty, "
            "contradictions, semantic subject, and exact source_memory_ids. Do not infer people, "
            "traits, or actions; never invent IDs. Return exactly one digest item covering every "
            "supplied source ID.\n" + _json([
                item.model_dump(mode="json") if isinstance(
                    item, BaseModel) else item
                for item in memories
            ]))


def projection_stage_prompt() -> str:
    return "<stage>Return only the requested projection/card. Do not output inferred-memory or hypothesis actions.</stage>"


def actions_stage_prompt() -> str:
    return "<stage>Review the projection above. Return only explicit inferred-memory and hypothesis actions. Omission is always no-op. IDs must be from supplied context.</stage>"


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


__all__ = [
    "COVERAGE_METHOD",
    "COVERAGE_RUBRIC",
    "actions_stage_prompt",
    "owner_evidence_digest_prompt",
    "owner_reasoning_prefix",
    "projection_stage_prompt",
    "schema_instruction",
]
