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
| S1 symbolic (taylor, parser) | symbolic | NOT approved: reviews 4 and 5 have 2 blockers each; revisions 5 and 6 running | not started |
| MF1 manifolds (d and wedge growth, checked powers) | manifolds | NOT approved: review 3 has 1 blocker; revisions 3 and 4 running | not started (skipped in run 2 because it was ordered after S1) |
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
- `implement-workflow.js`: the workflow script that ran the implementation. It skips the four
  committed units by default, runs two more design rounds for S1, MF1 and C1 just before their
  turn, and parks a unit that is not approved after four reviews as a patch (later units that
  depend on it are skipped).

## Resume

1. Copy the designs and reports into a scratch directory, for example
   `mkdir -p $SCRATCH && cp -r docs/workstreams/handoff-followups/design docs/workstreams/handoff-followups/impl $SCRATCH/`.
2. Run `implement-workflow.js` with `args = { scratch: "$SCRATCH" }`. The order is F1, R1, M1a,
   M1b, S1, MF1, C1, J1a, J1b. With 4 cores only one unit runs at a time; a unit takes one to
   three hours.
3. After the last unit: a full `cargo test --workspace --no-fail-fast` (about an hour, detached),
   then the tracker update with an independent verifier, as in the earlier rounds.
