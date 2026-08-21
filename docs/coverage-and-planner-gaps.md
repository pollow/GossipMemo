# Coverage depth and goal ranking: two specified-but-unbuilt pieces

## Where this came from

The Hermes profile `beatrice` carried a skill, `beatrice/personal-modeling`,
that described an evidence-ledger-and-planner architecture in prompt prose,
because at the time no server implemented one. GossipMemo now does, and the
mapping is close to one-to-one:

| personal-modeling section 3 | GossipMemo |
| --- | --- |
| evidence ledger | `memories` (`kind`, `basis`, `status` active/retracted/superseded, `source_batch_id`) + `messages` |
| facet schema | `coverage_roots` |
| coverage view | `coverage_entries` |
| question log | `learning_goals.status` + `status_reason` |
| planner | the learning-goal reasoner, fanned out per coverage root |

That skill was retired on 2026-08-21. Two of its specifications are **not**
implemented here, and are recorded below so they are not lost with it. This
is not a decision record -- nothing has been decided. It is an imported
specification plus an assessment of what today's schema cannot express.

## Gap 1: coverage has no depth and no negative states

`coverage_entries` today is a root, a path, free-text `content`, and
`status CHECK(status IN ('active', 'superseded'))`. The imported spec asks
for two things that binary status cannot carry.

**A depth vector per facet**, so "understood" is not one number:
`explicitness`, `concreteness`, `temporal_breadth`, `meaning`,
`relational_links`, `confidence`, `recency`. The spec is explicit that a
single "percent understood" score is the wrong shape because it creates
false precision.

**Distinct negative states**, which it insists must never collapse into each
other:

- `unknown` -- no evidence was ever submitted.
- `not_found` -- a retrieval did not surface evidence. A search note, *not* a
  coverage state, and must not be promoted to `unknown`.
- `no_recall` -- the user tried to remember and could not.
- `declined` -- the user asked not to pursue or record this. A boundary, not
  a missing task.
- `conflict` -- active evidence is inconsistent.
- `hypothesis` -- a lead, not coverage. (GossipMemo *does* model this one, as
  a first-class `hypotheses` row.)

### Why this one is not cosmetic

`declined` has no representation anywhere in the current schema. A topic the
user explicitly asked not to pursue is therefore indistinguishable from a
topic that was never raised. Both look like absence to the goal reasoner,
which fans out per coverage root looking for thin areas -- so a declined
subject is a *prime* candidate for regeneration as an open learning goal,
and gets asked again. `learning_goals.status` has a `deferred` value, but it
describes one goal's fate, not a standing boundary on the subject, and
nothing stops a later reasoning pass from minting a fresh goal over the same
ground.

The user-visible failure is an agent that re-raises something it was asked to
drop, which reads as not listening rather than as a schema limitation.

### This argues against a decision already taken, deliberately

`design.md`'s "Epistemic guidance 与 user learning" section records that the
current shape is a considered simplification, not an omission: the 20 stable
memoir/persona perspectives were downgraded to roots plus prompt hints,
explicitly dropping both an `unknown/fragmentary/grounded/rich` scale **and**
separate boundary records. An entry records only "what is known"; finding
gaps is left to goal planning as creative work.

So Gap 1 must be read as a challenge to that decision, not as a missing
feature. The strongest form of the challenge is narrow, and does not require
reinstating the scale:

- The depth vector is the part the existing decision most directly rejected,
  and rejected for a good reason -- an LLM scoring seven dimensions per facet
  invents precision it does not have. Treat it as recorded, not as pending.
- `declined` is different. It is not a measurement of depth, it is a fact the
  user stated. Dropping "independent boundary records" removed the only place
  to put it, and the audit loop that replaced it writes only what is known,
  which structurally cannot represent "the user asked me not to know this."
  That is a gap in the substitution, not a rejected refinement.

If only one thing here is ever built, it should be a durable representation
of a user-declared boundary that goal planning reads and excludes on.

### The shape the spec proposed

```sql
CREATE TABLE coverage (
  facet_id TEXT PRIMARY KEY REFERENCES facets(id),
  state TEXT NOT NULL DEFAULT 'unknown',
  explicitness INTEGER NOT NULL DEFAULT 0,
  concreteness INTEGER NOT NULL DEFAULT 0,
  temporal_breadth INTEGER NOT NULL DEFAULT 0,
  meaning INTEGER NOT NULL DEFAULT 0,
  relational_links INTEGER NOT NULL DEFAULT 0,
  confidence TEXT NOT NULL DEFAULT 'unknown',
  recency TEXT,
  notes TEXT
);
```

Recorded as the source's shape, not as a recommendation to adopt it
verbatim -- `coverage_entries` is already root+path rather than a `facets`
table, so any real implementation would differ. The part worth preserving is
the distinction set, not the DDL.

## Gap 2: goals are generated by evidence but selected at random

Goal *generation* is evidence-driven: the reasoner fans out per coverage root
and writes `learning_goals` with a `rationale` and `entry_ids`. Goal
*selection* -- which few of the open pool reach the prompt -- is
`sample_learning_goals`, a seeded random draw.

The imported planner spec ranks instead:

1. identify facets with `unknown`, `mentioned`, low concreteness, narrow
   temporal breadth, weak relational links, or unresolved conflict;
2. exclude `declined`, recently asked, high-sensitivity, or currently
   irrelevant facets unless the user reopens them;
3. find a nearby known anchor -- a person, object, place, preference, work,
   route, or sensory cue;
4. combine one weak facet with one or two adjacent dimensions
   (`aesthetic preference x origin`, `place x body`, `person x choice`,
   `work x values`);
5. generate one concrete, low-pressure question asking for a case, scene, or
   comparison rather than a category summary.

Note that step 2 depends on Gap 1: there is nothing to exclude on until
`declined` and an asked-recently signal exist.

Here too the current behavior is documented as intentional, not accidental:
`design.md` states that Hermes consumes at most one hypothesis and a random
3-5 learning goals per turn, and is explicitly told to ignore these
directions by default, so as not to add LLM latency on the request path.
Ranking must not reintroduce that latency.

`_guidance` deliberately does not rank goals today, and its docstring
explains why: a learning goal is a direction rather than a claim, and the
store has only a query string to judge relevance with. That reasoning still
holds for the store layer. It is an argument for moving the judgment to a
caller with more context, or to an embedding match against rolling
continuity -- not an argument that random is correct.

Commit 33c1c64 added `goals` and `goal_seed` to the context and turn read
paths, which lets a caller size the sample and pin it against version churn.
That is the seam a ranked or continuity-matched selection would replace; it
is not itself the fix.
