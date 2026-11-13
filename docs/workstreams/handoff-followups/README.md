# HANDOFF open follow-ups: designs and progress (paused 2026-11-13)

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
| F1 finitefields | finitefields | approved (review 2) | not started |
| R1 rings, p-adics, category | rings, padics, category, core, powerseries | approved (review 2) | not started |
| M1a bounded rational form | matrix | approved (review 1) | not started |
| M1b JordanForm, PrimaryRationalForm, IsSimilar, IsConsistent | matrix | approved (review 1) | not started |
| S1 symbolic (taylor, parser) | symbolic | NOT approved: review 3 has 2 blockers | not started |
| MF1 manifolds (d and wedge growth, checked powers) | manifolds | NOT approved: review 3 has 1 blocker | not started |
| C1 combinatorics | combinatorics | NOT approved: review 3 has 1 blocker | not started |
| J1a REPL parser and small fixes | jupyter | approved (review 3) | not started |
| J1b REPL shared values, interrupt, limits | jupyter | approved (review 2) | not started |

The three open design blockers, in short (details in `design/<unit>.review3.md`):

- S1: the cost model in §2.7 does not count the full Euclid gcd behind every `Rational`
  operation, and does not charge the `nth_root` calls of radical extraction. Measure again with
  the Lehmer gcd from I3 (`80130e7`), which changes these costs.
- MF1: the derivative guard admits every two-argument `root_sum`, and `differentiate.rs:218-222`
  returns 0 whenever the first argument contains the variable, so a non-closed form can be
  reported closed.
- C1: `Subset::next_binary` and `prev_binary` return `None` for every ground set of 65 or more
  elements, because `to_bitvector` does; the design fixes only the n = 64 mask.

Not done yet for the four committed units: the tracker update (`PROJECT_TRACKER.md`,
`docs/magma/status_overrides.csv`, `REVIEW_2026-11-06.md`, `HANDOFF.md` open list) and a full
`cargo test --workspace`. Each unit ran its design's acceptance commands (its own crate, the
downstream crates it can affect, and `cargo check --workspace --tests`) and passed them. The
"Handoffs" section of each report in `impl/` lists the tracker rows to change.

## Files

- `design/<unit>.md`: the design (the "Revision" sections at the top apply; the body is
  authoritative). `design/<unit>.review<r>.md`: the reviews; the last one approved the design,
  except for S1, MF1 and C1.
- `impl/<unit>.md`: implementation reports of the committed units, with "Handoffs" sections for
  later units and the trackers. `impl/<unit>.review1.md`: the approving code reviews and their
  minor notes.
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
