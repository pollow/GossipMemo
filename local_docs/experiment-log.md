# Experiment log

Running log of small experiments run against real (or previously-captured
real) data to answer a specific question about GossipMemo's reasoning
passes. Each entry is a question, a method, and an honest result --
including null and ambiguous ones.

## Recipe: A/B one reasoner against a database copy

Use this when the question is "does X in the stored data change what
reasoner Y proposes", without wanting to touch the live database or run the
rest of the pipeline.

1. **Copy the database** twice under `data/experiments/` (gitignored along
   with the rest of `data/`). Never mutate `data/verify-coverage.db` or
   `data/gossipmemo.db` directly -- they are shared fixtures/live state.
2. **Mutate one side only.** Write the mutation as a small, deterministic,
   inspectable transform (a script with an explicit find/replace table or
   equivalent) so the diff between control and treatment is exactly the
   variable under test and nothing else. Print a before/after diff for
   every row it touches so the transform can be audited without re-running
   it.
3. **Clear the downstream table in both copies.** If you don't, whatever
   already accumulated there contaminates the comparison and you're
   diffing "26 old rows + N new" against "26 old rows + M new" instead of
   N against M.
4. **Restrict the pipeline to the one reasoner under test.** Do not run
   `ReasoningPipeline` end to end -- call `build_<name>_reasoner(store,
   model)` directly and drive it with `reasoner.run_until_caught_up(space_id)`,
   or use `scripts/replan_learning_goals.py` as a template for other
   reasoners. Running the full pipeline reintroduces every other reasoner's
   randomness and burns calls on stages the question doesn't touch.
5. **Turn on the LLM trace** (`GOSSIPMEMO_LLM_TRACE_PATH`, one file per
   side) before running either side. It is off by default and costs
   nothing when on; every request and completion lands in it verbatim, so
   it is the only reliable source for "how many candidates did each root
   propose" style questions that the final committed rows alone can't
   answer (a reconciliation pass can drop candidates without a trace of
   why).
6. **Run each side as a background process**, logging to a file, and poll
   for completion rather than blocking on one long foreground call --
   these runs are usually 15-25 LLM calls at 15-20s each.
7. **Compare from the trace**, not just the final table: parse each
   record's `label`, request, and completion. For a per-root fan-out
   reasoner, each candidate request names its root and each completion
   lists what it proposed; the final table only shows what survived
   reconciliation, which conflates "never proposed" with "proposed and cut
   downstream."
8. Do not commit the copied databases or trace files -- `data/` is
   gitignored in full.

## 2026-08-17: does confident coverage-overview wording suppress learning-goal candidates?

**Question.** A real run of `data/verify-coverage.db` (space `personal`)
left coverage root `P11` (`skills_and_knowledge`) with 8 entries and zero
learning goals, while other roots produced up to 5. That database's
overview entries still carry the old audit prompt's optimistic wording
("已较充分了解..." -- "already fairly well understood...", "已较清楚了解..."
-- "already fairly clearly understood..."), which the audit prompt no
longer writes as of commit `13a11ab` (it no longer rates how well anything
is understood). Hypothesis: that confident wording made the learning-goals
planner conclude P11 had nothing worth opening, i.e. the wording itself
suppressed candidate generation.

**Method.** Followed the recipe above.

- Copied `data/verify-coverage.db` to `data/experiments/control.db` and
  `data/experiments/treatment.db`.
- Wrote a one-off deterministic transform (an ordered literal find/replace
  table, ~30 entries, run once with `--apply` against `treatment.db` only)
  that strips or neutralizes phrases rating *how well* something is
  understood ("已较充分了解", "已较清楚了解", "理解较深", "较完整", "较清晰"
  used as an understanding-rating adjective, and a handful of
  sentence-specific variants of the same pattern), while leaving the
  substantive content -- which areas exist, what each one covers -- intact.
  18 of the 20 active overview entries (`path = ''`) changed; the two that
  didn't (P2, P8's near neighbors) simply didn't contain any of the listed
  phrasing.
- Exact diffs for four roots (full diff for all 18 is reproducible by
  re-running the transform script in dry-run mode against
  `data/verify-coverage.db`):

  ```
  P11 (skills_and_knowledge)
  - 已较充分了解的领域包括：模型系统与自研加速器的正确性、调试、评测和 regression
    investigation；...
  + 涉及的领域包括：模型系统与自研加速器的正确性、调试、评测和 regression
    investigation；...

  M3 (turning_points)
  - 已较清楚了解 CurrentUser 的主要人生转折：职业上经历 AAI 评价受挫、回到 MTIA
    后重新获得认可...
  + CurrentUser 的主要人生转折：职业上经历 AAI 评价受挫、回到 MTIA 后重新获得
    认可...

  M1 (life_chapters)
  - 目前较清晰的生活章节包括：浙江大学求学阶段...职业主线理解最充分，能看到从
    AAI 回到 MTIA...
  + 目前的生活章节包括：浙江大学求学阶段...职业主线能看到从 AAI 回到 MTIA...

  P7 (social_style_and_boundaries)
  - 对用户的社交风格与边界已有较完整但不均衡的了解：联结与关系维护、日常沟通及
    情绪回应方面理解较深；...
  + 用户的社交风格与边界，覆盖不均衡：联结与关系维护、日常沟通及情绪回应方面有
    记录；...
  ```

- Deleted all rows from `learning_goals` in both copies (26 pre-existing
  rows each -> 0), so both sides planned from zero.
- Ran only `build_learning_goals_reasoner` against each copy via
  `scripts/replan_learning_goals.py data/experiments/{control,treatment}.db
  --space personal`, each as a background process, each with its own
  `GOSSIPMEMO_LLM_TRACE_PATH` (`data/experiments/{control,treatment}.trace.jsonl`).
  Config (base URL, key, model) came from the repo's `.env`.
- No coverage audit, extraction, or owner induction ran on either side.

**Raw numbers.**

- Goals committed: control 15, treatment 11. Both sides' results were
  `upserts` only -- 0 `transitions` on either side.
- Trace records: 21 per side (20 per-root candidate calls + 1
  reconciliation call), verified by `wc -l` on both trace files and by
  the `label` field on every record reading `plan-learning-goals`.
- Candidate stage (parsed from the traces): **3 candidates from every one
  of the 20 roots on both sides, 60 candidates total each, zero roots
  returned an empty candidate list on either side** -- including P11,
  which produced 3 candidates in both control and treatment.
- 0 closure recommendations on both sides (see limitation 1 below).
- Surviving goals by cited root after reconciliation:
  - control: M1 2, M2 3, M3 4, M5 2, M6 3, M7 3, M9 1, P1 2, P2 1, P3 1,
    P4 2, P5 1, P6 2, P7 1, P8 1, P9 1, P10 2. (M4, M8, P11 absent.)
  - treatment: M1 2, M2 1, M5 2, M6 1, M7 3, M8 3, M9 1, P1 1, P2 2, P3 1,
    P4 1, P6 2, P7 1, P9 1, P10 2. (M3, M4, P5, P8, P11 absent.)
  - No surviving goal on either side had an empty citation list.

**Conclusion.**

The hypothesis is **rejected** at the stage it named. The per-root
candidate pass saturates its "at most three" ceiling on every root
regardless of overview wording -- confident or neutral, P11 (and every
other root) produced exactly 3 candidates on both sides. P11's zero
learning goals in the original real run were never a candidate-stage
refusal caused by confident wording; the filtering happens entirely
downstream, in reconciliation: 60 candidates collapse to 15 (control) / 11
(treatment), and P11 was dropped by reconciliation on *both* sides, wording
notwithstanding.

Treatment produced fewer surviving goals than control (11 vs 15) -- the
opposite direction from the hypothesis -- but this is a single run per
side through a stochastic merge step (reconciliation dedupes and picks
which near-duplicate candidates to keep), so it should be read as noise,
not as evidence that stripping the confident wording *hurts* planning. A
real effect claim would need several runs per side.

**Limitations to record explicitly.**

1. `learning_goals` was cleared on both sides before planning, so there
   were no open goals for the new per-root closure recommendations to vote
   against -- that mechanism (a root's vote that an existing goal now reads
   as answered or overtaken) was not exercised at all by this experiment.
   The 0/0 closure-recommendation count is a consequence of the setup, not
   a finding about whether wording affects closure judgment.
2. Both runs used the candidate prompt as it read at commit `13a11ab`,
   which still contained a breadth tie-breaker sentence ("When an entry
   already has real specificity, or an open goal already digs into a
   topic, prefer expanding a thin entry or an uncovered sibling over
   deepening that entry further"). That sentence was removed afterwards in
   commit `75f27e2` because deepening where it matters is legitimate memoir
   work. Both sides ran against the same prompt, so the *comparison* is
   still valid, but the absolute numbers (3-per-root saturation, 15 vs 11
   survivors) describe a prompt version that is no longer live.

**Implications.** If breadth across coverage roots is the goal, the lever
is the per-root candidate cap and the reconciliation merge step, not the
audit's overview wording -- the wording never reached the stage that
matters. Separately, the audit's removal of confidence verdicts
(`13a11ab`) is still justified on its own terms: a stored entry that keeps
asserting how well something is understood is a truthfulness problem in
the record regardless of whether it turns out to move planning.
