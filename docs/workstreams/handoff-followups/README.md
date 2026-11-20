# HANDOFF open follow-ups: designs and progress

This folder holds the work on the request "Fix the open follow-ups in HANDOFF" (the list in
`docs/HANDOFF.md`, "Open follow-ups", as of `c79a582`). The list was split into 13 units by
crate. Every unit has a design that went through independent adversarial review, and units are
implemented one at a time: implement, adversarial code review with fix-up rounds, commit, push.

The designs and reports were written in a session scratchpad. Paths in them that start with
`/tmp/claude-0/...` point at experiments that no longer exist; the documents themselves are
complete.

## Status

| Unit | Crates | Design | Implementation |
|---|---|---|---|
| I3 integer arithmetic | core, integers, rationals (+ callers) | approved (review 2) | `80130e7`, approved at review 1 |
| I1 primality | integers (+ callers) | approved (review 2) | `7c4d057`, approved at review 1 |
| I2 integer factorization | integers | approved (review 3) | `0ceaeaa`, approved at review 1 |
| P1 polynomials | polynomials, groups, rings, galois (+ callers) | approved (review 2) | `166ac34`, approved at review 1 |
| F1 finitefields | finitefields (+ core, polynomials) | approved (review 2) | `4b7f034`, approved at review 1 |
| R1 rings, p-adics, category | rings, padics, category, core, powerseries | approved (review 2) | `fd081fa`, approved at review 2 |
| M1a bounded rational form | matrix | approved (review 1) | `d3a133b`, approved at review 1 |
| M1b JordanForm, PrimaryRationalForm, IsSimilar, IsConsistent | matrix (+ polynomials, finitefields) | approved (review 1) | `dc367a2`, approved at review 1 |
| S1 symbolic (taylor, parser) | symbolic (+ jupyter) | never approved: reviews 4 to 9 had 1 or 2 blockers each (run 4 added four red-team rounds); frozen at revision 8 by the owner (2026-11-15) | implemented test-first from the frozen design (run 5): `4763f31` (work in progress, merged through PR #584) and the run's commits on `claude/gracious-gates-e6b6l7`; approved under the narrowed bar after the owner's decision (2026-11-16), two finish rounds and requests C-1 and C-4 |
| MF1 manifolds (d and wedge growth, checked powers) | manifolds | approved (review 4) | `f29c492`, approved at review 1 |
| C1 combinatorics | combinatorics | approved (review 4) | `2e000c9`, approved at review 2 |
| J1a REPL parser and small fixes | jupyter (+ rationals, finitefields) | approved (review 3) | `bd03278`, approved at review 1 |
| J1b REPL shared values, interrupt, limits | jupyter | approved (review 2) | `36c2170`, approved at review 1 |

Status on 2026-11-14 (run 2 of `implement-workflow.js`, 31 agents): seven units committed. S1's design
failed reviews 4 and 5 with two new budget or recursion holes each (review 4: coefficient copies and kernel
keys; review 5: left-deep monomial products, which overflow a 2 MB stack, and uncharged monomial map
searches, 13 s and 1.1 GB under the REPL budget). MF1 was skipped because the script ordered it after S1;
its design does not need S1's code. Run 3 (`implement-workflow-run3.js` in the scratchpad, summarized in
this folder at the end) gives S1 two more rounds with hard structural caps and an explicit review bar, gives
MF1 two rounds made independent of S1, and then implements each approved unit.

Run 3 (2026-11-14 and 15): MF1's design was approved at review 4 and the unit committed as `f29c492`. S1's
revisions 5 and 6 added structural caps, an "Invariants" section and a fixed review bar, but reviews 6 and 7
again found two holes each over the bar (review 6: the sparse accumulator of the series product; review 7:
MPFR's erf costs O(x^2) terms and was charged as cheap, 32 s at 12 KB; owned `Expr` coefficients copy a
kernel's top node uncharged, 2.35 GB at 100 KB). Run 4 adds a red team with separate lenses before each
formal review.

Run 4 (2026-11-15, `implement-workflow-run4.js`, 24 agents): four red-team rounds with four lenses each
(numeric, memory, series, structure; `design/S1-symbolic.redteam<round>-<lens>.md`), revisions "7, red-team
round 1" to "round 4" and revision 8, and formal reviews 8 and 9. Neither approved the design. Review 8 had one
blocker (a hint substituted into a subtree that is not regular). Review 9 has two, of one class: the engine
raises a decided `DivisionByZero` where the input divides by no exact zero. B1: the atan family builds its
integrand for an argument whose series is an exact constant, and the integrand is singular at arcsin(+-1),
arccos(+-1), arccosh(+-1) and arcsinh(+-I), although the value is a constant. B2: revision 7rt4's pole rule for
arctanh(+-1) fires on the constant term of a non-constant argument, so a logarithmic branch point that revision
8 reported as `NotLaurent` became `DivisionByZero` (a regression of that revision). Everything else review 9
checked holds, including 2000 random inputs against PARI/GP at 300 digits. The red team's last round still
listed three holes over the bar (empty truncated arguments of negative precision; `0^r` decided from syntax;
results whose trees are exponential in their DAG). The design is now 5259 lines. The prototypes (2.6 GB) stay
in the session scratchpad and are not kept. S1 was then parked until the owner decided how to continue (run 5,
below).

Run 5 (2026-11-15 to 2026-11-20; S1 only, after the owner's first decision, `impl/S1-symbolic.owner_decision.md`). The
design was frozen (no revision 9) and the tests came first: design review 9's two blockers, the red team's three holes,
a singular-point matrix of 609 rows with gp values and 479 rows of review 9's random harness, checked twice by an
independent reviewer (`impl/S1-symbolic.tests_first.md`, `impl/S1-symbolic.testcheck.md`). The port of the prototype
passed them with the owner's two rules: an exact-constant argument is evaluated, and a non-constant argument at a branch
point is `NotLaurent`. Six rounds of implementation review followed, each with a semantics and a resources reviewer
(`impl/S1-symbolic.review<r>-<lens>.md`), and a fix-up round after each, tests first. The semantics reviewer approved
every round but round 5 (a variable named pi or I could name a constant), and in round 6 found no wrong value on
13,200 rows against a gp contour oracle. The resources reviewer found one
pattern again and again. A series result is a DAG that names each kernel's source at every term, and a consumer that
walks it as a tree costs about the number of terms times its cost on that source. Rounds 2 to 5 closed each instance
with a cap on the result (its text, the folds of `simplify`, the memo payload) or with a change to `simplify` (a work
list, a DAG memo, a guard on folded powers). Round 6 found the same in `expand`, `differentiate` and the REPL's `diff`
(2.5 GB and 21.8 s; 2.1 GB). The owner accepted the recommendation (`impl/S1-symbolic.owner_decision2.md`, 2026-11-16):
narrow the bar to S1's own entry points and the printing, dropping and re-parsing of their results; document the
measured scaling of the consumers and remove the doc comments that claimed them bounded; and make a follow-up unit, S2
("DAG-aware consumers", `design/S2-dag-consumers.md`), to share repeated subexpressions. The finish under that bar had
two review rounds (`impl/S1-symbolic.fin1-*.md`: not approved, six distinct blockers, among them a parser that still
recursed and a re-parse of a result that was not bounded; `impl/S1-symbolic.fin2-*.md`: approved) and a check of the
minor fixes (`impl/S1-symbolic.finminorscheck2.md`, approved). Then requests C-1 (branch cuts) and C-4 (the dependent
crates) of `docs/workstreams/m23_golay/REQUESTS.md` were done and reviewed (`impl/S1-symbolic.c1-1-*.md`, both
approved; their minor items were fixed, and no reviewer re-checked those fixes). The report with all its rounds is
`impl/S1-symbolic.md`. S1's work-in-progress commit `4763f31` reached `main` through PR #584; the rest is in the S1 pull
request. Open: S2 and the S1 handoffs (`docs/HANDOFF.md`, "Open follow-ups").

For the four committed units the trackers are updated (2026-11-14: `PROJECT_TRACKER.md` Change Log
"Open follow-ups round, part 1", `docs/magma/status_overrides.csv`, `REVIEW_2026-11-06.md` and the
`HANDOFF.md` open list, checked by an independent verifier), and a full `cargo test --workspace` at
`922024b` passed (14,386 passed, 0 failed, 333 ignored, 194 targets; `docs/TEST_BASELINE.md`).

## Files

- `design/<unit>.md`: the design (the "Revision" sections at the top apply; the body is
  authoritative). `design/<unit>.review<r>.md`: the reviews; the last one approved the design,
  except for S1 and MF1.
- `impl/<unit>.md`: implementation reports of the committed units, with "Handoffs" sections for
  later units and the trackers. `impl/<unit>.review<r>.md`: the code reviews; the last one approved
  the unit, and its minor notes are open.
- S1 has more records, all in `impl/`: the report `S1-symbolic.md` (every round, with its handoffs); the reviews
  `S1-symbolic.review<r>-semantics.md` and `-resources.md` (rounds 1 to 6), `S1-symbolic.fin1-*.md` and
  `S1-symbolic.fin2-*.md` (the finish), `S1-symbolic.c1-1-semantics.md` and `-regressions.md` (request C-1); the check
  of the minor fixes `S1-symbolic.finminorscheck2.md`; the tests-first report `S1-symbolic.tests_first.md` and its
  check `S1-symbolic.testcheck.md`; and the owner's decisions `S1-symbolic.owner_decision.md` and
  `S1-symbolic.owner_decision2.md`. In these files, and in the comments of the S1 code and tests, a path
  `fu2/s1impl/S1.<x>.md` is `impl/S1-symbolic.<x>.md`, `S1_impl.md` is `S1-symbolic.md`, `owner_decision.md` and
  `owner_decision2.md` are `S1-symbolic.owner_decision.md` and `S1-symbolic.owner_decision2.md`, `tests_report.md` is
  `S1-symbolic.tests_first.md`, `testcheck.md` is `S1-symbolic.testcheck.md`, and `S2_spec.md` is
  `design/S2-dag-consumers.md`; their other `fu2/` paths are session scratch files that are not kept.
- `design/S2-dag-consumers.md`: the specification of the follow-up unit S2 (not designed, not implemented).
- `implement-workflow.js`: the workflow script that ran the implementation. It skips the four
  committed units by default, runs two more design rounds for S1, MF1 and C1 just before their
  turn, and parks a unit that is not approved after four reviews as a patch (later units that
  depend on it are skipped).

## Resume

All 13 units are done. The next unit of this kind is S2 (`design/S2-dag-consumers.md`): write its design and have it
reviewed like the others before any code. The steps below are those of the earlier runs.

1. Copy the designs and reports into a scratch directory, for example
   `mkdir -p $SCRATCH && cp -r docs/workstreams/handoff-followups/design docs/workstreams/handoff-followups/impl $SCRATCH/`.
2. Run `implement-workflow.js` with `args = { scratch: "$SCRATCH" }`. The order is F1, R1, M1a,
   M1b, S1, MF1, C1, J1a, J1b. With 4 cores only one unit runs at a time; a unit takes one to
   three hours.
3. After the last unit: a full `cargo test --workspace --no-fail-fast` (about an hour, detached),
   then the tracker update with an independent verifier, as in the earlier rounds.
