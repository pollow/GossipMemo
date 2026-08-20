"""Default text for every static prompt fragment.

This module is data: named string constants and two per-root tables, with no
logic and no imports from `reasoners/` or `query.py`. `PromptLibrary` defaults
every field to a constant defined here, and an operator override file replaces
individual fields by name, so this file stays the single place the shipped
wording lives.
"""

from __future__ import annotations

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
event is allowed when calibrated to that evidence. Identify supported patterns
in behavior, preferences,
communication, decision-making, sensitivities, and helpful ways to interact.
Make reasonable social inferences when supported, with uncertainty proportional
to evidence. Do not use projections, inferred memories, or hypotheses as evidence.
Distinguish current conditions from historical events. Use the language that
best matches supplied memories; keep IDs and enum values unchanged.
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
conditions. Use the language that best matches supplied memories; keep IDs and
enum values unchanged.
"""

USER_MODEL_REASONING_SYSTEM_PROMPT = """Reason carefully about the fixed current user from
active memories marked about_user. Capture preferences, communication preferences, goals, current
situations, and practical interaction guidance. Generalize recurring patterns
when supported, but do not turn a one-off event into a stable trait or include
another person's identity. Use valid_from and valid_to to separate current
conditions from historical events. Do not use projections, inferred memories, or
hypotheses as evidence. Use the language that best matches supplied memories;
keep IDs and enum values unchanged.
"""

CONTINUITY_SYSTEM_PROMPT = """Rebuild compact cross-session continuity.
Return only the supplied JSON schema. Keep ongoing threads, recent decisions,
pending actions, and context useful for the next conversation. Do not make
long-term personality inferences or copy person/user profiles; the current user
is not a Person. Use the language that best matches supplied messages and prior
continuity; keep IDs and enum values unchanged.
"""

COVERAGE_AUDIT_SYSTEM_PROMPT = """Summarize what is known about one area of a person's
life and persona. Return only the supplied JSON schema. An entry is a
summary over many memories -- roughly dozens of memories into a short paragraph -- not a
retelling of them and not memoir prose. Write only what is known; never write what is
missing, unclear, or worth asking about. Do not invent facts, evidence, or private
details, and do not diagnose. Keep entries concise and in the language of the evidence."""

GOAL_PLANNING_SYSTEM_PROMPT = """Plan optional directions in which this user's memoir
and persona could be understood better, reading summaries of what is already
understood. Return only the supplied JSON schema. A direction is natural language: what
it is, why it is worth understanding, and one suggested wording. Private, intimate,
painful, and stigmatized areas are not off limits -- an unlit part of a life is still
part of it, and recording a direction is not asking about it. Whether to raise one now,
how to word it, and how to keep that exchange safe is the consuming agent's decision in
the moment, not this planner's. A direction may be about a friend, but it belongs to the
user's own life, relationships, or memoir: never a standalone information-gathering task
about a third party, and never a suggestion that the user test, probe, or secretly
verify anyone. Do not diagnose, and do not assume that reconciliation, disclosure, or
repair is the right ending of any thread. Omission is no-op: only explicitly transition
an existing supplied goal when its lifecycle changes."""

QUERY_SYNTHESIS_SYSTEM_PROMPT = """Answer the read-only question using the supplied
social-memory context. Return concise plain text only (no JSON wrapper or code
fence). Use facts and supported inferences in the context to give a direct,
useful answer; distinguish uncertainty and current conditions from historical
events. Separate Person records are not evidence that they represent different
real people; when identities may overlap, state the ambiguity instead of asserting
a distinction. Do not invent facts or claim that anything was saved. Answer in
the language of the question.
"""

# The rubric is split per root and per job because both reasoners fan out over
# one root at a time: the auditor gets only the viewpoint (it summarizes what the
# evidence supports), the planner gets the viewpoint plus the blind-spot cues
# (naming what is missing is its work).

# One short viewpoint line per coverage root, shared by the per-root audit and
# planning requests.
COVERAGE_ROOT_VIEWPOINTS: dict[str, str] = {
    "M1": "Which eras, beginnings, moves, endings, and chapters are legible?",
    "M2": "What do ordinary life, routine, home, work, care, money, and security look like?",
    "M3": "Which choices, accidents, losses, recoveries, and reversals changed the story?",
    "M4": "Which attachments, ruptures, loyalties, intimacies, and family or friend arcs matter?",
    "M5": "Which places, communities, cultures, institutions, and historical contexts "
          "shape meaning?",
    "M6": "Which concrete scenes, sensory memories, conversations, and small moments "
          "carry the story?",
    "M7": "How are feelings, needs, fear, desire, grief, and self-protection described?",
    "M8": "Which recurring themes, tensions, growth, and contradictions span time?",
    "M9": "Which questions, conflicts, decisions, losses, or hopes are still running?",
    "P1": "How does the user name self, belonging, and self-understanding?",
    "P2": "What matters to the user and which costs are accepted?",
    "P3": "Which assumptions, beliefs, and sources of meaning guide interpretation?",
    "P4": "What pulls the user forward or holds them back?",
    "P5": "How does the user decide under uncertainty, conflict, or pressure?",
    "P6": "Which non-clinical emotional rhythms and coping patterns recur?",
    "P7": "How does the user connect, communicate, protect space, and repair?",
    "P8": "Which tastes, practices, environments, and practical rhythms recur?",
    "P9": "How does the user tell stories, joke, ask, withhold, create, or communicate?",
    "P10": "Which roles, settings, identities, pressures, and exceptions change the pattern?",
    "P11": "What has the user learned, practiced, taught, or become capable of?",
}

# Areas that stay unsaid under each root unless something invites them. These
# are planning cues, not a ban list and not a questionnaire: an area named here
# is a direction worth holding open, and whether, when, and how to raise any of
# it is the consuming agent's call, not the planner's.
COVERAGE_ROOT_BLIND_SPOTS: dict[str, str] = {
    "M1": "childhood, family origin, education, work, migration, chapters still ahead",
    "M2": "class, housing, debt, caregiving, accessibility",
    "M3": "regret, repair, harm done, survival",
    "M4": "sexuality, consent, estrangement, reconciliation",
    "M5": "religion, politics, class, diaspora, contexts treated as taboo",
    "M6": "body, health, illness, substance use, private rituals",
    "M7": "shame, guilt, secrets, inner life left unspoken",
    "M8": "envy, resentment, moral injury, forgiveness, repair",
    "M9": "mortality, legacy, unfinished goodbyes",
    "P1": "identities and labels not yet offered",
    "P2": "morality, faith, politics, loyalty, money",
    "P3": "religion, ideology, taboos",
    "P4": "safety, status, intimacy, mortality",
    "P5": "avoidance, risk, regret",
    "P6": "grief, shame, anger, substance-related coping",
    "P7": "conflict, attachment, intimacy",
    "P8": "health, money, accessibility constraints",
    "P9": "silence and code-switching",
    "P10": "home versus work, safety, power, culture",
    "P11": "informal knowledge and opportunities that were blocked",
}

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


# The two short stage fragments of the owner-reasoning pair (`reasoners/owner.py`).
PROJECTION_STAGE_PROMPT = ("<stage>Return only the requested projection/card. "
                           "Do not output inferred-memory or hypothesis actions.</stage>")

ACTIONS_STAGE_PROMPT = ("<stage>Review the projection above. Return only explicit "
                        "inferred-memory and hypothesis actions. Omission is always no-op. "
                        "IDs must be from supplied context.</stage>")


__all__ = [
    "ACTIONS_STAGE_PROMPT",
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "COVERAGE_METHOD",
    "COVERAGE_ROOT_BLIND_SPOTS",
    "COVERAGE_ROOT_VIEWPOINTS",
    "EXTRACTION_SYSTEM_PROMPT",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "PROJECTION_STAGE_PROMPT",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
]
