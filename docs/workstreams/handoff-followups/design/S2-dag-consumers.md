# S2: DAG-aware consumers (follow-up spec of S1)

Status: specification for a follow-up unit. Not designed, not implemented. Written at the finish
of S1 (2026-11-16) on the owner's second decision (`fu2/s1impl/owner_decision2.md`): the S1 bar
covers the series entry points and the printing, dropping and re-parsing of their results; the
cost of a consumer applied to a result is documented, not bounded, and a separate unit makes those
functions share repeated subexpressions. The docs step copies this file into the repository.

Crate: `rustmath-symbolic` (and the REPL arms of `rustmath-jupyter` that call it). MAGMA Handbook
ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`) is the
interface the S1 results come from; the consumers themselves are crate functions outside S1.

## 1. The problem

A series result of S1 is a DAG. A kernel made from an input node renders as that node, and every
term that names the kernel shares the same `Arc`. As a tree, the result repeats that subtree at
every term. Its DAG is small: the engine charges it, and the caps of S1 (`MAX_RESULT_TREE`,
`MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`, `MAX_RESULT_MEMO_WORDS` in `series_const.rs`) bound its
tree, its printed text and the folds of `simplify` on it.

The consumers below walk the result as a tree. They rebuild every occurrence of a shared subtree,
so their cost on a result is about (number of terms) x (their cost on the kernel's source). The
source is part of the input, so a short input can make a large result for them even when the
result itself is small.

| consumer | file | how it walks | what it builds |
|---|---|---|---|
| `Expr::simplify` | `simplify.rs:423` | explicit stack; a node with two or more parents in the walked DAG is simplified once (S1 rounds 3 to 5, memo of at most 32 MiB of literals) | already DAG-aware on its input; its cost blows up only on inputs that other consumers have unshared |
| `Expr::expand` | `expand.rs:26` | recursive; distributes every product of sums, also inside a call's argument | a fresh tree at every occurrence |
| `Expr::differentiate` | `differentiate.rs:24` | recursive; product, quotient and power rules | 290 to 650 bytes per tree node, at every occurrence, also for subtrees free of the variable |
| `Expr::substitute` | `substitute.rs:17` | recursive | a fresh node at every occurrence, also where nothing changes |
| `PartialEq for Expr` | `expression.rs:50` (derived) | recursive over the tree | nothing; time linear in the tree, stack as deep as the tree |
| `Expr::integrate` | `integrate.rs:61` | tier 1 (rational Risch) on the whole input, then tier 3 recursing over sums | measured cubic in the number of terms (section 3) |
| REPL `diff`, `expand`, `subs`, `integrate`, `simplify` | `rustmath-jupyter/src/repl.rs` | the functions above; `diff` is `differentiate` then `simplify` (`repl.rs:2024`) | as above |

`symbols()` and the other tree walks of the crate (`walker.rs`) have the same shape; S2 lists them
in its design and decides which to convert.

## 2. Goal

For every result R that an S1 entry point accepts (`taylor_with_budget` with the REPL's budget
1e8, `series_expansion` with `to_expr`), each consumer C of the table runs in time and memory
linear in the DAG of R times C's own cost per distinct node, and its output shares what R shares.
Concretely, on the acceptance inputs of section 4:

- time within 20 s and peak memory within 512 MB (release, 8 MB thread), and the same in debug
  for the inputs marked "both";
- the scaling with the order is linear: for one input at orders N and 10 N, the ratio of the times
  and of the peak memory growths is at most about 15 (linear with slack);
- outputs unchanged in value: every consumer's output equals HEAD's output as a tree (structural
  `==`), except where section 5 allows a change, which the callers' tests must then accept.

Non-goals: changing the series engine or its caps (an optional phase 2 below may relax them);
canonical forms or hash-consing of equal subtrees that are not shared (`x + 1` built twice);
making `simplify` stronger.

## 3. Measured cases (rounds 4 to 6 of the S1 implementation)

Release, 8 MB thread, budget 1e8 unless named. Input files are in
`fu2/s1impl/rev6-resources/cons/` and `repl/` (round 6), `fu2/s1impl/rev5-resources/cons/`
(round 5) and the round-4 records.

| # | input (at 0) | order | result | consumer | measured | HEAD on the input itself |
|---|---|---|---|---|---|---|
| E1 | `prodsum12.txt`: `sin((y0+z0)*(y1+z1)*...*(y11+z11))/(1-x)` (110 bytes) | 100 | Ok, 0.006 s | `expand`; `simplify(expand)` | 0.29 s, 257 MB; 1.87 s, 622 MB | `expand` 0.004 s, 7 MB; `simplify` 0.152 s, 23 MB (debug) |
| E2 | the same | 400 | Ok, 0.007 s | the same | 1.74 s, 1 007 MB; 8.47 s, 2 433 MB | |
| E3 | the same | 1000 | Ok, 0.007 s | the same | 4.01 s, 2 507 MB; 21.8 s, 5 608 MB (98 407 302 tree nodes) | |
| E4 | the same | 3000 | Ok | `expand` | abort under a 4 GB limit | |
| E5 | `prodsum8.txt` (74 bytes, 8 factors) | 20000 | Ok, 0.020 s | `expand` | 3.40 s, 3 132 MB | |
| E6 | REPL `replE_prodsum12_1000.txt` (debug, 64 MiB thread) | 1000 | `t` Ok, 0.049 s | `e = expand(t)` | 5.69 s, 2 512 MB | |
| D1 | `prodchain1000.txt`: `sin(y0*y1*...*y999)/(1-x)` (4 901 bytes) | 1040 | Ok, 0.003 s | `differentiate(x)`; its `simplify` | 0.50 s, 610 MB; 0.55 s, 612 MB | `differentiate` 15 MB; `simplify` 0.11 s, 16 MB |
| D2 | `prodchain300.txt` | 1000 | Ok | the same | 0.13 s, 180 MB; 0.19 s, 181 MB | |
| D3 | `divchain1000.txt`: `sin(y0/y1/.../y999)/(1-x)` | 500 | Ok | the same | 0.40 s, 495 MB; 1.50 s, 831 MB | 18 MB |
| D4 | the same | 1040 | Ok | the same | 0.80 s, 1 023 MB; 4.32 s, 1 706 MB | |
| D5 | `powtower1000.txt`: `sin(y0^y1^...^y999)/(1-x)` | 500 | Ok | the same | 0.50 s, 648 MB; 3.66 s, 1 412 MB (5.1e8 tree nodes) | `differentiate` 1.55 s, 225 MB |
| D6 | the same | 1040 | Ok, 0.004 s | the same | 1.08 s, 1 340 MB; 9.12 s, 2 896 MB (1.05e9 tree nodes) | |
| D7 | `powtower3000.txt` (16 901 bytes) | 340 | Ok, 0.009 s | the same | 1.21 s, 1 319 MB; 8.69 s, 2 862 MB | |
| D8 | `sinsum1000.txt`: `sin(sin(y0)+...+sin(y999))/(1-x)` | 500 | Ok | the same | 0.24 s, 251 MB; 0.13 s, 252 MB | |
| D9 | REPL `replD_prodchain1000_diff.txt` (debug) | 1040 | `t` 0.35 s | `d = diff(t, x)` | 4.99 s, 664 MB | |
| D10 | REPL `replT_powtower1000_diff.txt` (debug) | 1040 | `t` 0.30 s | `d = diff(t, x)` | 11.6 s, 2 118 MB | REPL `diff` of the input: 0.15 s, 33 MB |
| S1 | round 5: `sin(K/3 x 2535)/(1-x)`, K = `(2^254+1)^4096` (5 095 bytes) | 10 | refused since round 5 (`FoldWork`) | `simplify(substitute(x, 2))`, `simplify(expand)`, `simplify(differentiate)` | before the cap: 24.1 s, 23.7 s, 27.1 s; order 190: 418.6 s | |
| S2 | round 5: `sin(K*3 x 6000)/(1-x)` and `x 11000` (12 and 22 KB) | 1 | refused since round 5 (`TreeText`) | `simplify(differentiate(x))` | before the cap: 3.65 s, 774 MB | |
| S3 | round 4: `sin(K/3 x 12000)/(1-x)` and three relatives (24 to 60 KB) | 1 | refused since round 4 (`FoldWork`) | `simplify` | before the cap: 35.2 s to 169.5 s | |
| S4 | round 4: REPL session, 6 000 lines `a_k = a_{k-1}*3`, `t = taylor(sin(a6000)/(1-x), x, 0, 1)`, `d = diff(t, x)` | 1 | `t` refused since round 5 | `diff` | before: 840 MB | |
| I1 | `taylor(1/(1-x), x, 0, N)` | 250, 500, 1000 | Ok | `integrate(x)` | 0.78 s, 5.6 s, 43.3 s | the same polynomial typed as text costs the same (N = 250: 3.57 s HEAD debug) |

Rows S1 to S4 are refused today by the caps of S1. They are acceptance inputs for phase 2 only.

## 4. Acceptance

Tests written first (they fail on the code at the start of S2), in `rustmath-symbolic/tests/s2_*.rs`
and, for the REPL rows, in `rustmath-jupyter` (each REPL session in a child process of the test
binary, as `test_taylor_diff_of_a_chain_result_memory` does). Each memory measurement resets the
peak first (`/proc/self/clear_refs`, as `s1_review5_memory.rs`) or runs in a child.

1. E1, E3, E5: `R.expand()` and `simplify(R.expand())` each within 20 s and 512 MB (release,
   `#[ignore]` like the S1 budget tests); E1 in both profiles. E1 against E3: the scaling bound of
   section 2. E6 in the REPL: every line `Ok`, peak under 512 MB.
2. D1, D4, D6, D7: `R.differentiate(x)` and `simplify` of it each within 20 s and 512 MB; D1 and D3
   in both profiles. D5 against D6: the scaling bound. D9 and D10 in the REPL: every line `Ok`,
   peak under 512 MB.
3. `substitute(x, 2)` and `==` (against a clone and against an independently built equal tree) of
   every result above within the bar; `==` on a result as deep as the S1 depth tests allow
   (`tests/s1_depth.rs`) without a stack overflow on a 2 MB debug thread.
4. I1: `integrate(x)` of N = 1000 within 20 s; N = 250 against N = 1000 within the scaling bound.
   Before coding, the design measures where the cubic cost of HEAD's `integrate` is (tier 1 on a
   polynomial of degree N, or tier 3's recursion over the sum running tier 1 again on every
   subtree) and states the fix.
5. No regression: every existing test of rustmath-symbolic and of its dependents (benchmarks,
   calculus, curves, functions, geometry, jupyter, manifolds, plot3d) passes unchanged, except the
   expected outcomes section 5 allows to change, each with the reason. The S1 tests
   (`tests/s1_*.rs`, the matrix and the fuzz data) pass unchanged.
6. The depth reach of every converted function is at least HEAD's on a 2 MB debug thread and an
   8 MB release thread (`tests/s1_depth.rs` method: about 96 % of HEAD's limit).
7. The doc comments of S1 that state the measured scaling of these consumers
   (`series_const.rs` `MAX_RESULT_TEXT`, `simplify.rs` `simplify`, `series.rs`
   `taylor_with_budget`, `repl.rs` `TAYLOR_WORK`) are rewritten with the new measurements.

Phase 2 (optional, a decision for the owner after phase 1): with DAG-aware consumers, the
per-occurrence counts of S1 (`TextBound::total`, `lit_text`, the per-occurrence text) may go back
to per-distinct-node counts, so that rows S1 to S4 become values. Acceptance: those rows `Ok`,
then every consumer within the bar.

## 5. Constraints and open questions for the design

- Share by address within one call. A shared subtree is the same `Arc`; a memo keyed by its
  address, with every node alive during the walk, maps it to its output `Arc`. `simplify.rs`
  already has this machinery (parents counted inside the walked DAG, an entry released after its
  last use, a payload cap past which it recomputes as HEAD does). The design decides whether to
  factor it into one DAG-map walker for `expand`, `differentiate` and `substitute`.
- Keep the memory at most HEAD's live set plus a cap (the S1 round 4 and 5 lessons: a memo keyed
  by `Arc` counts owners outside the walk; a memo that keeps every value aborts on large DAGs).
- `substitute` returns the input node itself where nothing changed (no copy of a subtree free of
  the symbol).
- `differentiate` of a subtree free of the variable is 0, decided by one DAG walk before the
  recursion. This changes HEAD's output for such subtrees (`0*z + y*0` becomes `0`), so the
  callers' tests must be checked (rustmath-manifolds' slot assertions, rustmath-calculus). The
  design states whether S2 takes this rule or keeps HEAD's output and only shares it.
- `expand` keeps HEAD's rules (including the S1 rule under a minus, `expand.rs:254`); it shares
  the expansion of a shared subtree, so `sin(K)` at every term is expanded once.
- `PartialEq`: `Arc::ptr_eq` first, then an explicit stack, and a memo of compared pairs for
  shared nodes. It must stay an equivalence relation and equal the derived one in value.
- Iterative walks where the recursion is on the path of a result (the S1 results are as deep as
  their inputs plus at most 18 levels, so HEAD's reach is enough for them; inputs of other callers
  may be deeper).
- Out of scope: the S1 engine, the parser, `Display` and `Drop` (already linear and iterative).
- Zero `unsafe`; no new dependencies beyond the workspace's.

## 6. Pointers

- S1 records: `fu2/s1impl/S1_impl.md` (fix-up rounds 3 to 5 and "Finish under the narrowed bar"),
  `fu2/s1impl/S1.review4-resources.md`, `S1.review5-resources.md`, `S1.review6-resources.md`
  (B1 and B2, minor 3 on `integrate`).
- The doc comments that state the current scaling: `rustmath-symbolic/src/series_const.rs`
  (`MAX_RESULT_TREE`, `MAX_RESULT_TEXT`, `MAX_RESULT_FOLD_WORK`), `src/simplify.rs` (`simplify`),
  `src/series.rs` (`taylor_with_budget`), `rustmath-jupyter/src/repl.rs` (`TAYLOR_WORK`).
