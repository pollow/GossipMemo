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


# The instruction paragraphs the prompt builders assemble around rendered data.
# Each is prose only: which messages are evidence, how a memory is serialized,
# and the XML-ish scaffolding stay in the builders, so an override can retune
# wording without reshaping a request. A `$name` placeholder is filled by
# `prompts.render.fill` at use time; `library.PLACEHOLDERS` lists the names each
# fragment may use, and an override that invents or drops one fails at load.

# --- reasoners/extraction.py ---

EXTRACTION_RETENTION_RULE = (
    "Keep explicit durable information and a transient detail only when "
    "it affects an ongoing situation or helps reveal a recurring pattern."
)

EXTRACTION_USER_EVIDENCE_RULE = (
    "The fixed current user is named $quoted_user_name. Use that name when referring "
    "to the current user; the current user is not a Person. User-authored "
    "messages in the current batch are the only evidence allowed to create "
    "memories. Assistant-authored messages in the batch and recent context are "
    "context only: use them to resolve references and conversational meaning, "
    "but never save their restatements, summaries, analyses, or advice as new "
    "memories. Assistant content may supply a proposition only when a current "
    "user evidence message explicitly confirms, adopts, or corrects it."
)

EXTRACTION_PERSON_IDENTITY_RULE = (
    "The current user/assistant author role is context and never a Person. List every "
    "specific, external human individual referenced by a memory in its `people` "
    "refs; express who said or did what in the memory content itself. A Person "
    "must denote one concrete human whose identity is distinguishable from "
    "other people in the evidence and could remain recognizable across "
    "conversations. It needs an evidence-supported durable identity anchor: a "
    "proper name, an explicit alias, or a role whose single holder is uniquely "
    "and temporally determined by the evidence. Do not treat a grammatical "
    "anaphor, an unbounded group or category, a non-human entity, or a merely "
    "situational description as an identity. For an unnamed but sufficiently "
    "anchored individual, choose the most stable, specific, neutral canonical "
    "label supported by the evidence, in the evidence language, and keep each "
    "observed surface wording as an alias. Do not create separate identities "
    "merely because synonymous wording was used. Never guess that differently "
    "named people are the same person. If identity is vague, preserve any "
    "otherwise useful durable "
    "memory with the original reference in its content and leave its `people` "
    "refs empty; identity uncertainty alone is not a reason to discard that "
    "memory. Preserve reported claims as "
    "`reported`, not facts. Set `about_user` for a claim or event about "
    "$user_name; $user_name must never appear in "
    "`people`."
)

EXTRACTION_TIME_BOUND_RULE = (
    "Record valid_from/valid_to when the message gives a time bound."
)

EXTRACTION_KNOWN_PEOPLE_RULE = (
    "In natural-language memory content and generated display fields, use "
    "a known person's canonical display_name when the messages refer to them. "
    "In every ExtractedMemory.people and ExtractedRelationship.person_a_ref/"
    "person_b_ref, use the supplied stable Person `id` (never a display_name "
    "or alias). "
    "If the messages explicitly introduce a new short name, return "
    "it in that person's `aliases` field. Omit a known person unless a new "
    "memory references them or the messages explicitly add an alias. Do not "
    "echo the known-people list."
)

EXTRACTION_COMPARISON_RULE = (
    "For comparison memories only: omit a memory when the current user batch "
    "merely repeats it. When current user evidence explicitly corrects, updates, or "
    "refines one, emit the new memory and set `supersedes_memory_id` to that supplied "
    "comparison memory ID. Never copy details from a comparison memory unless those "
    "details also appear in current user evidence. Do not use an inferred comparison "
    "memory as evidence."
)

EXTRACTION_CLARIFICATION_RULE = (
    "Additionally, list any question you would have to ask the user before "
    "current user evidence can be interpreted correctly. Raise one only when "
    "the ambiguity actually blocks a durable, useful memory -- typically an "
    "unresolved reference to a concrete individual. Do not raise one for every "
    "pronoun, group, category, or incidental third party, and never ask for "
    "private detail about another person beyond what identifying them in the "
    "user's own memory requires. Say in `reason` what the ambiguity blocks, "
    "and put a short description of the kind of ambiguity in `blocked_by`. "
    "Cite the message IDs that triggered it in `evidence_message_ids`. "
    "Asking nothing is the normal outcome; return an empty list. "
    "Clarifications never change what you emit in `people` or `memories`: "
    "retain exactly the memories you would have retained without them."
)


# --- prompts/render.py (the owner-reasoning family) ---

OWNER_EVIDENCE_SCOPE_RULE = (
    "Only evidence-memories are evidence. Current inferred "
    "memories and open hypotheses may be reviewed for duplication or explicit "
    "lifecycle actions, never used as support."
)

OWNER_EVIDENCE_DIGEST_RULE = (
    "Compress supplied raw evidence only. Preserve chronology, basis, uncertainty, "
    "contradictions, semantic subject, and exact source_memory_ids. Do not infer people, "
    "traits, or actions; never invent IDs. Return exactly one digest item covering every "
    "supplied source ID."
)

# --- reasoners/continuity.py ---

CONTINUITY_REBUILD_RULE = (
    "Rebuild continuity from the prior summary and newer raw messages. "
    "Choose the last supplied message as through_message_id."
)

# --- reasoners/coverage.py ---

COVERAGE_AUDIT_FOLDING_RULE = (
    "Fold this evidence into the entries for this root. Add an entry for a topic that "
    "the entries do not cover yet, and modify an entry whose summary this evidence "
    "changes or extends; leaving an entry out changes nothing. An entry that only one "
    "memory supports is almost always wrong -- that is a single event, not an "
    "understanding of a topic."
)

COVERAGE_AUDIT_ENTRY_SHAPE_RULE = (
    "Keep exactly one entry with an empty path: the overview "
    "of this root, naming which areas exist under it and what each covers. Paths are "
    "free text; reuse a stored path when you mean the same area, and do "
    "not renumber or normalize the others. Keep content under about two hundred words: "
    "when an entry outgrows that, split it by narrowing that entry and adding the "
    "areas it no longer covers. To merge two entries, rewrite one to absorb the other "
    "and modify the other with status \"superseded\"."
)

# --- reasoners/learning_goals.py ---

GOAL_CANDIDATE_EXPANSION_RULE = (
    "These entries are everything that is understood about this root; the entry with "
    "an empty path is its overview. Propose optional candidate directions only, and "
    "none at all when this root has nothing worth opening. Expand in "
    "four ways: deeper into one entry, at something it states but never unfolds; "
    "sideways to a neighbouring or missing sibling, a stage or place or period the "
    "entries step over; forward in time along a thread the entries already contain, "
    "at what became of it -- a concrete hook like that usually yields the best "
    "direction; and along a person an entry names, where the direction is that "
    "person's part in the user's own life. Write each direction in the "
    "language of the entries. Cite in `entry_ids` the entries a direction grew out of when you "
    "can, leave it empty rather than guessing, and never withhold a direction for "
    "having nothing to cite. Do not repeat a direction an open goal already covers."
)

GOAL_CANDIDATE_CLOSURE_RULE = (
    "Candidates are non-mutating: do not transition, retire, defer, update, or "
    "otherwise change any goal lifecycle. Separately, look over the open goals above "
    "against what these entries now show: when one now reads as answered, overtaken, "
    "or no longer worth holding open, add a closure recommendation citing its "
    "`goal_id` and a short reason. This is a vote for a later pass to weigh, not a "
    "transition -- do not remove or alter the goal here, and skip any goal this "
    "root's entries say nothing new about."
)

GOAL_RECONCILIATION_MERGE_RULE = (
    "These candidates come from separate per-root passes, so near-duplicates across "
    "roots are expected: merge them, and keep the ones that read as an invitation "
    "into this user's own life rather than a survey question. Keep breadth -- several "
    "directions on one subject are worth less than the same number spread across "
    "different parts of the life."
)

GOAL_RECONCILIATION_LIFECYCLE_RULE = (
    "Reuse an existing `goal_id` to rewrite that goal, "
    "and omit it to create a new one. This is the only pass that may transition a "
    "goal's lifecycle: transition one when the evidence shows it is answered, "
    "overtaken, or no longer worth holding open. The closure recommendations are each "
    "one root's vote grounded in what its entries actually show, not an instruction: "
    "weigh a recommendation as evidence, and a goal recommended closed by one root can "
    "still be worth holding open if the rest of its scope is unanswered."
)

GOAL_CANDIDATE_REDUCTION_RULE = (
    "Deduplicate and compress these non-mutating learning-goal candidates, keeping "
    "their breadth across different parts of the life. Return candidates only; never "
    "transition any lifecycle."
)

# Query-side embedding instruction prefixes (Qwen3 asymmetric encoding:
# "Instruct: {task}\nQuery: {text}"). Each hybrid-retrieval call site has
# its own task wording; storage-side embedding never uses one of these.
EMBEDDING_TURN_RECALL_INSTRUCTION = (
    "Given the user's latest message, find memories about the user that are "
    "relevant to it."
)
EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about the user, find memories that are relevant to answering it."
)
EMBEDDING_EXTRACTION_COMPARISON_INSTRUCTION = (
    "Given a newly stated fact, find existing memories that may already state "
    "the same fact, so it can be recognized as a duplicate or update rather than "
    "new information."
)
EMBEDDING_HYPOTHESIS_DEDUP_INSTRUCTION = (
    "Given evidence currently being reasoned about, find existing hypotheses that "
    "already express the same tentative claim or direction, not merely a related topic."
)
EMBEDDING_LEARNING_GOAL_DEDUP_INSTRUCTION = (
    "Given what is currently understood about a coverage root, find existing "
    "learning goals that are already pursuing the same direction, not merely a "
    "related topic."
)
EMBEDDING_COVERAGE_ENTRY_DEDUP_INSTRUCTION = (
    "Given a piece of new evidence, find existing coverage entries that already "
    "summarize this same area, not merely a related one."
)


__all__ = [
    "ACTIONS_STAGE_PROMPT",
    "CONTINUITY_REBUILD_RULE",
    "CONTINUITY_SYSTEM_PROMPT",
    "COVERAGE_AUDIT_ENTRY_SHAPE_RULE",
    "COVERAGE_AUDIT_FOLDING_RULE",
    "COVERAGE_AUDIT_SYSTEM_PROMPT",
    "COVERAGE_METHOD",
    "COVERAGE_ROOT_BLIND_SPOTS",
    "COVERAGE_ROOT_VIEWPOINTS",
    "EMBEDDING_COVERAGE_ENTRY_DEDUP_INSTRUCTION",
    "EMBEDDING_EXTRACTION_COMPARISON_INSTRUCTION",
    "EMBEDDING_HYPOTHESIS_DEDUP_INSTRUCTION",
    "EMBEDDING_LEARNING_GOAL_DEDUP_INSTRUCTION",
    "EMBEDDING_QUERY_INSTRUCTION",
    "EMBEDDING_TURN_RECALL_INSTRUCTION",
    "EXTRACTION_CLARIFICATION_RULE",
    "EXTRACTION_COMPARISON_RULE",
    "EXTRACTION_KNOWN_PEOPLE_RULE",
    "EXTRACTION_PERSON_IDENTITY_RULE",
    "EXTRACTION_RETENTION_RULE",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_TIME_BOUND_RULE",
    "EXTRACTION_USER_EVIDENCE_RULE",
    "GOAL_CANDIDATE_CLOSURE_RULE",
    "GOAL_CANDIDATE_EXPANSION_RULE",
    "GOAL_CANDIDATE_REDUCTION_RULE",
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "GOAL_RECONCILIATION_LIFECYCLE_RULE",
    "GOAL_RECONCILIATION_MERGE_RULE",
    "OWNER_EVIDENCE_DIGEST_RULE",
    "OWNER_EVIDENCE_SCOPE_RULE",
    "PERSON_REASONING_SYSTEM_PROMPT",
    "PROJECTION_STAGE_PROMPT",
    "QUERY_SYNTHESIS_SYSTEM_PROMPT",
    "RELATIONSHIP_REASONING_SYSTEM_PROMPT",
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
]
