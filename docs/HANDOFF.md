# Session handoff (2026-11-20, updated after Tasks 0-2, the L0-L1 fixes, their follow-ups, the final follow-up fixes and all 13 units of the open follow-ups)

Read this first, then `CLAUDE.md` and `docs/README.md`. It records where the project stands and
what to do next. **Tasks 0, 1 and 2 are done** (status notes below), and so are the L0-L1 defects
the review found (2026-11-07), their follow-ups and the final follow-up fixes (both 2026-11-08). In the
open follow-ups round **all 13 units are done**. Twelve were committed by 2026-11-15 (the last is MF1, `f29c492`).
The thirteenth, S1 (the rustmath-symbolic series engine and literal parser), is approved under the narrowed bar that
the owner accepted on 2026-11-16. Its design was never approved: after nine design reviews and four red-team rounds
the owner froze it (2026-11-15) and had it implemented test-first. Merged into `main`: the work-in-progress commit
`4763f31`, through PR #584. This run's commits (the finish under the narrowed bar, requests C-1 and C-4 of
`docs/workstreams/m23_golay/REQUESTS.md`, and these records) are on `claude/gracious-gates-e6b6l7` for the S1 pull
request. Open: the follow-up unit S2 (DAG-aware consumers) and S1's minor handoffs. See "Open follow-ups round" below
and `docs/workstreams/handoff-followups/README.md`, which has the designs, reports and reviews. After that come the
dead code, the mixin wiring and the `docs/PLAN.md` build targets.

## Where things stand

- **Direction:** RustMath is a pure-Rust port of MAGMA. The handbook digests in
  `rustmath-lava/docs/Chapter_NN_*.md` are the spec. SageMath is a secondary reference only.
  IGP24 is over; its code lives at https://codeberg.org/johnjanik/inverse_galois (map:
  `docs/igp24/SOURCES.md`).
- **Git:** one line of history. `main` contains everything (PRs #569, #570). The 19 old branches
  were deleted; their tips are tags `archive/<branch>` (`docs/BRANCHES.md`). Work on a new branch
  off `main`; cloud sessions can push only their own branch (no tags, no other branches).
- **Build and tests:** the whole workspace compiles, and `cargo test --workspace` is green:
  14,791 passed, 0 failed, 328 ignored, 203 targets at `f29c492` (`docs/TEST_BASELINE.md`). First build
  compiles GMP/MPFR for `rug` (minutes).
- **Disk:** the session's disk allowance is about 39 GB. With incremental compilation on, `target/`
  grew past 20 GB and filled it mid-session. This session set `[build] incremental = false` in
  `~/.cargo/config.toml` (not in the repo; set it again in a new container) and a clean full build
  plus test then fit comfortably. If space runs out, `rm -rf target` and rebuild.
- **Environment:** the RustMath cloud environment's setup script installs GAP (`gap-core`,
  `gap-libs`, `gap-transgrp`) and builds the gitignored degree-24 atlas files in
  `rustmath-groups/data/`. Verified on its first real run (2026-11-06): GAP at `/usr/bin/gap`, both
  atlas files 25,000 rows, `galois_narrowing` 4/4.
- **Lost for good:** seven IGP24 files never committed anywhere (`docs/igp24/SOURCES.md`).
  Nothing in the port depends on them.

## Key files

| File | What |
|---|---|
| `docs/magma/intrinsics.csv` | 13,948 handbook intrinsics (13,341 math scope, ch 17-159), one row each, with status |
| `docs/magma/status_overrides.csv` | Reviewed statuses with evidence; the generator never overwrites these |
| `docs/magma/STATUS.md` | Per-chapter roll-up (generated) |
| `tools/magma_tracker/build.py` | Regenerates the three generated files above |
| `docs/magma/reading_list_core.md` | 330 papers cited by core chapters (179 priority A); user is collecting PDFs as `<key>.pdf` |
| `docs/TEST_BASELINE.md` | Workspace test results and failures by crate |
| `PROJECT_TRACKER.md` | Algorithm inventory by `file:line`, with facade/broken flags |
| `docs/PLAN.md` | Phased plan (see the 2026-11-05 update at the top) |

## Task 0: verify the environment (5 minutes). DONE 2026-11-06

Passed as written; the setup script worked. Re-run it at the start of a session if in doubt.

```bash
which gap && wc -l rustmath-groups/data/transitive_24.jsonl rustmath-groups/data/transitive24_cycletypes.jsonl
cargo test -p rustmath-groups --test galois_narrowing     # expect 4 passed
```

If GAP or either file is missing: run `apt-get install -y gap-core gap-libs gap-transgrp`, then
the two commands in `rustmath-groups/data/README.md`, and tell the user the setup script failed.

## Task 1: fix the 57 failing tests. DONE 2026-11-06

All 58 fixed (the 57 below, plus `perm23_classifier::transitive24_db_load_default_fails_gracefully`,
which assumed the atlas data was absent). There are 13 commits, `eb50f9b`..`e0a9c23`, one or two
per crate. Each fix was checked by an independent reviewer agent before it was committed. Each of the
21 library-test failures involved a code bug; three of those tests also had a wrong fixture or
expected value, corrected with proof (GAP or the definition). Most of the doctest failures were stale
examples, but giving them real assertions exposed further bugs and facades (Artin relators, the
Mathieu groups, ghost brackets, multimodular Bernoulli). `docs/TEST_BASELINE.md` has the per-crate
causes, and the `PROJECT_TRACKER.md` Change Log row (2026-11-06) and inventory rows have the
details. Things to know:
- `rustmath-algebras` doctests behind the `full` feature run only in a workspace build (through
  `rustmath-jupyter`). Test that crate alone with `--features full`.
- `rustmath-misc` doctests no longer write `output.txt` or leave `.tmp_dir_*` behind.
- Facades found but **left open** (rows in the tracker): the `GroupElement::order` default (a
  bounded search that decides "infinite"), the `Group::is_abelian` default `false`, the
  `Group::exponent` default `None`, `GenericParentWrapper::contains_element` (always true),
  `FreeModuleWithBasis::cardinality`, lieconformal `degree()` (None for every nonzero element),
  affine `compute_structure_data` (empty tables, so its bracket panics), the f.p.-group SNF, and the
  Artin/braid word problem. The Virasoro, N=2, Neveu-Schwarz and structure-coefficient
  lambda-brackets now panic honestly instead of returning empty maps; they are unimplemented.
- Behaviour and API changes with no callers outside their crate: `ClusterQuiver::mutation_type`
  now returns `Result`, `is_mutation_equivalent` returns `Option<bool>`, and `to_quiver` returns
  `Result`. `recover_forms*` return phase-normalised kernel vectors. `GroupLibGAP::contains` and
  `elements` panic (`unimplemented!`) when a group cannot be enumerated; use `try_contains`.
- Commit `3fab92c` lacks the blank line after its subject (cosmetic; history not rewritten).

The original task text follows.

Baseline (`docs/TEST_BASELINE.md`, with atlas data present): 57 failures.

| Target | Count | Notes |
|---|--:|---|
| `rustmath-groups` lib | 18 | `libgap_group`, `libgap_mixin` (pure-Rust GAP name-alikes), `semidirect_product`, `nilpotent_lie_group`, `generic::test_merge_elements` |
| `rustmath-groups` doctests | 8 | `groups_catalog`, `generic`, `semimonomial_transformation*` |
| `rustmath-lieconformal` doctests | 13 | stale examples |
| `rustmath-trees` / `-numbertheory` / `-modules` / `-algebras` doctests | 5 / 4 / 2 / 2 | stale examples |
| `rustmath-stats`, `rustmath-misc` doctests | 1 each | |
| `rustmath-quivers` lib | 2 | `cluster_quiver::test_cartan_companion`, `test_type_d_quiver` |
| `rustmath-curves` lib | 1 | `belyi::modular_forms_hp::recover_forms_centered_z_a_matches_uncentered`, 3.36e-10 divergence |

Priority: `rustmath-groups` first (it underlies Galois work). Rules:
- Fix the code when the test is right. Never weaken an assertion just to go green. For a stale
  doctest, update the example to the current API and make sure it still checks something real.
- If a test exposes a **facade** (fabricated return value), make the function return an honest
  `Err`/`unimplemented!()` and record it, rather than making the facade pass.
- Whether the groups failures predate the belyi/lava merge is unknown; `git log` on the files will tell.
- `rustmath-misc` tests write `rustmath-misc/output.txt` and a `.tmp_dir_*`; do not commit those.
- Run `cargo test -p <crate>` per crate; update `docs/TEST_BASELINE.md` at the end.

## Task 2: review the tracker for the 45 remaining core chapters. DONE 2026-11-06

All 4,930 rows reviewed and committed, one commit per chapter (`826b982` (ch 32 pilot) through
`f2ef395`), plus a correction pass (`bd98093`). **Read `docs/magma/REVIEW_2026-11-06.md`**: method,
results and the defect table. In short:
- 16 verified, 326 done, 1,190 partial, 17 stub, 108 facade, 72 broken, 3,172 missing, 29 n/a.
  Of the generator's `candidate` rows, 46% are real (partial or better); 17% of its `missing` rows
  are real implementations under other names.
- Each chunk had a Sonnet reviewer, an independent verifier (95% of rows checked; it disputed 15%)
  and an adjudicator for disputes (it sided with the verifier in 666 of 699). Provenance is the tag
  at the end of each note. 245 `missing` rows are reviewer-only.
- A dep-info check found never-compiled source files, among them `rustmath-rings/src/number_field/`
  (`morphisms.rs`, `order.rs`, `s_unit_solver.rs`, `tests.rs`) and the genus-2 curves stack (gated
  off). 16 rows that relied on such code or on vague test citations were corrected.
- 180 rows are facade or broken (81 in rustmath-rings), and every one is now cited by a
  `PROJECT_TRACKER.md` inventory row. Of the 184 the review rated, 63 are in 22 files that the
  2026-11-07 defect list marked as absent from the tracker (7 of those files were in fact already
  cited): 61 are covered by its inventory (14 new rows, 8 updated, card T-247), and the other two
  were reclassified because the code does not fabricate: ch042.IsPerfect to missing,
  ch042.IsPrincipal to partial. The remaining 121 of the 184, in files the tracker already cited,
  were cross-checked one by one against its rows on 2026-11-07: 42 were stated accurately (Covers
  reference added), 54 sat in rows that were corrected, 23 were added (11 new rows, the rest as row
  extensions), and two were reclassified: ch042.PrimitiveElement to partial, ch020.Norm to missing.
  That leaves 61 + 119 = 180. Tracker deltas 🟢 -7, ⏳ -11, 🛑 +3, 🔴 +27, ⬜ -1 (1327 catalogued,
  282 🛑, 97 🔴), new card T-248; see its Change Log.
- The real cost was about 33M subagent tokens, not the 4-5M estimated below: every agent starts
  with about 65k tokens of fixed context, and verification plus adjudication tripled the work.

The original task text follows.

The automatic statuses (`candidate` = a same-named Rust fn exists, `missing` = none) are unreliable.
The pilot on ch 19, 38, 46 (231 rows) found only 17-42% of candidates were real, and some `missing`
rows were implemented under other names. Reviewed statuses go in `status_overrides.csv`.

**Remaining core chapters (4,930 rows):** 17, 18, 20-37, 39-45, 47-52, 57-68.

Method used in the pilot (about 90k Sonnet tokens per chapter; this task is about 4-5M tokens, so
confirm with the user before fanning out):
- One Sonnet subagent per chapter. It reads that chapter's rows from `intrinsics.csv`, searches
  the home crate (column `home_crates`) by **meaning**, not only by name, and writes
  `id,status,rust_path,note` for every row to a scratch CSV.
- Status vocabulary: `missing`, `stub`, `partial`, `facade`, `broken`, `done` (handbook semantics
  **and** a test exercising it, cite the test), `n/a`. Reserve `verified` for tests against
  external reference values (LMFDB, PARI/GP, Sage, GAP, handbook examples).
- Merge each CSV into `status_overrides.csv` (append `[review 2026-11-xx]` to `note`), spot-check
  2-3 claims per chapter against the code, run `python3 tools/magma_tracker/build.py`, commit.
- Subagents should write files rather than rely on their final message; in the pilot their
  reports did not come back, but the CSVs were complete.

## After these tasks

**L0-L1 defects: DONE 2026-11-07** on branch `claude/gracious-gates-e6b6l7`, commits `b475298..8e25fe2`.
Each unit had a design, an independent design review, an implementation, an implementation review
with reference values re-derived from PARI/GP or GAP, and a commit:
- `8b29b5a` core/category: `Parent::is_finite` is three-valued (`Option<bool>`); the
  `RingElementMethods` defaults became overridable `Ring` hooks (`unit_status`, `nilpotent_status`,
  `unit_inverse`, `multiplicative_order_status`, `static_characteristic`, enum `MultiplicativeOrder`)
  with exact overrides for Z, Q, R, C, Z/m, GF(q) and Galois rings; `Rational::reciprocal` of a
  negative is canonical.
- `0d7c126` integers: `mod_pow` residues in [0, |m|) and negative exponents, Legendre and Jacobi
  through the new `kronecker_symbol`, `sqrt_mod_prime` for negative input,
  `is_quadratic_residue_prime -> Result<bool>`, `mod_inverse`, generalised `binomial_mod`.
- `b4ded72` finitefields: complete square roots mod p^k (p = 2 and p | a), `all_sqrts`, `is_square`,
  `num_sqrts`.
- `191030a` polynomials: `coeff` past the degree is 0, `Quotrem` always terminates, `gcd`/`lcm`
  normalized (Euclid, then sub-resultant PRS), `exact_quotient`, `is_divisible_by`, pseudo-division,
  `normalize`, `content`, `primitive_part`.
- `5dfcd8c` matrix: consistent `solve`/`solve_left`, `solve_left_with_kernel`, vector-space
  coordinates and quotients, `reduce_vector`, the F[x] Smith form with a structural check,
  `invariant_factors`, `minimal_polynomial`, a certified rational form.
- Follow-ups: `8cc9f75` (DownUpAlgebra and manifold finiteness, f64 tolerance band, exact Galois-ring
  orders), `a2416ef` (docs, num-bigint >= 0.4.5), `e1ece67` (gcd over Q through Z[x]), `8e25fe2`
  (rational-form convention docs, REPL `solve_linear` reports non-uniqueness).

`PROJECT_TRACKER.md` (Change Log 2026-11-07), `docs/magma/status_overrides.csv` (57 rows updated, 37 with a new status)
and `docs/magma/REVIEW_2026-11-06.md` (defect table marked, counts recomputed) record the result.
Designs, reports and reviews are in the session scratchpad only (`l01/`), not in the repository.

**Follow-ups: DONE 2026-11-08** on the same branch, commits `a46a6a8..702d0a5`, with the same
design, review, implementation and review cycle (reference values from PARI/GP or GAP):
- `89b1c4e` polynomials: Berlekamp over GF(p) rewritten as `rustmath_polynomials::berlekamp` (Q rows
  x^(kp) mod f, an RREF null space, Yun with p-th roots, public `factor`, `squarefree_decomposition`,
  `q_matrix`; `berlekamp_factor_gf` is a wrapper with multiplicities); `zx::subresultant_gcd(p, 0)` keeps
  the content; the free `content`/`primitive_part` follow the methods and `poly_gcd_prem` is gone;
  `is_square_free` is content-blind (PARI `issquarefree`), and the characteristic-p square-free routines
  refuse what they cannot do; `Quotrem` over Z and `Content` over Q are documented as unverified.
- `5394aa3` polynomials: the `is_irreducible` facade is replaced (decided over Z and Q, `Err`
  otherwise; GF(p) by `berlekamp::is_irreducible`).
- `2a46bd0`, `8903c69` integers and rationals: generalised `SageInteger::binomial` (n < 0), non-negative
  `Integer` gcd/lcm/xgcd, `nth_root`, `is_power` and `is_perfect_power` of negatives (no exponent cap, a
  power-residue sieve), machine-word `pow_mod` returning `Result`, `SageRational::multiplicative_order`
  as `MultiplicativeOrder` and its `nth_root` of negatives.
- `029f7ab` finitefields and jupyter: squares in GF(2), `square_root_mod_prime` refuses a composite p,
  REPL `sqrt_mod` through `IntegerMod::sqrt` (card T-249 done), `GaloisRingElement::zero()`/`one()` as
  unbound sentinels (card T-243 done).
- `a813c49`, `527981b` rings, category, manifolds: three-valued `RingParentMethods`,
  `CommutativeRingParentMethods` and `RingMorphism` defaults (`is_zero_ring` from the parent, `random_*`
  are `Err`); T^(0,0) and Ω^0 are infinite on every manifold; the manifold rationale wording.
- `06945db`, `702d0a5` matrix: Krylov presentation for invariant factors, minimal polynomial and rational
  form (n = 20 random dense over Q in about 0.1 s release; n = 12 did not finish in 95 s), a determinant
  fallback in the rational-form certificate, faster transforms for small r > 1, GF(p) Krylov tests, and
  rational-form docs that state MAGMA's and PARI's block order and transform conventions correctly.
- `14a05a2` interfaces: the two tests sharing the object registry are serialised (a flaky failure).

`PROJECT_TRACKER.md` (Change Log 2026-11-08), `docs/magma/status_overrides.csv` (36 rows re-rated, 7 with
a new status; 226 more with path-only refreshes) and `docs/magma/REVIEW_2026-11-06.md` record the result.
The stale `file:line` paths in `status_overrides.csv` are refreshed: a script (session scratchpad
`fu/path_refresh/`) mapped every citation to the same item at `702d0a5` from the revision each row was
written against (`git blame`), and rewrote 299 citations in 256 rows (226 path-only rows tagged `path
refreshed 702d0a5`, the rest inside re-rated rows); the 27 citations it could not place were checked by
hand (25 point into files unchanged since their row was written, one was fixed by hand, one sits in a
re-rated row), so none remains unresolved.

**Final follow-up fixes: DONE 2026-11-08** on the same branch, commits `f0682b7..9c24389`, with the same cycle (a
design, two design reviews, an implementation and adversarial implementation reviews; reference values from
PARI/GP 2.15 or GAP):
- `f0682b7` polynomials: `fp_factor::equal_degree_factor` is Las Vegas (the open follow-up of the previous round:
  it gave up after 200 tries and returned the block unsplit, and Zassenhaus and `is_irreducible` over Z and Q
  relied on it). A block that the 200 legacy tries cannot split must pass a Rabin-type certificate and then gets
  up to 256 uniform vzGG 14.8 tries; the cap panics (probability at most 2^-256). New `pub const MAX_P` =
  3037000499; the public entry panics on f = 0, on a p that is not a prime <= `MAX_P` and on every contract
  violation; a nonzero constant gives `[]`.
- `7ce8117` finitefields: the `ff_poly` Cantor-Zassenhaus stack (`cantor_zassenhaus.rs`) and the canonical
  `poly_factor` EDF are certified the same way (the canonical one looped forever on an invalid block);
  `factor_squarefree` checks squarefreeness, the lava `factor`, `factor_squarefree` and EDF check the prime
  characteristic, `random_element` is exactly uniform (it drew only 0 and 1 for p > 2^64, so a reducible
  quadratic over GF(2^64 + 37) came back irreducible) and `PrimeField::from_int` is O(1) (it was O(p)).
- `6adaba2` integers: the QS, ECM and Pollard-rho complete-factorization drivers no longer push a composite as a
  prime (blocker B-04 closed; `ecm_factor_complete(3127, 1000, 10000, 0)` gave `[(53,1), (59,1), (3127,1)]`); the
  rho fallback is complete (Brent rho plus trial-division windows) and its output sorted; the dead QS
  `tonelli_shanks` has no z > 100 cap. 22 PARI `factor` rows in `tests/ch18_factorization_drivers.rs`.
- `874258b` jupyter: the REPL `binomial` goes to `SageInteger::binomial` for all integers (card T-250 done;
  `binomial(2^32 + 1, 1)` gave 1), 74 narrowing call sites use the checked `int_arg` / `index_arg` with the limits
  of `mod repl_limits` (no call the old code computed correctly in under about 10 s or a few hundred MB is
  refused), `fibonacci` / `lucas` accept n < 0, and `exterior_derivative` / `wedge` check degree and component
  counts.
- `da28d6f`, `a119d1e`, `9c24389`: doc fixes from the reviews (citation escapes, the composite-modulus wording,
  the psi_12 caveat in the QS and ECM driver docs, cost sentences).

`PROJECT_TRACKER.md` (Change Log "Final follow-up fixes"), `docs/magma/status_overrides.csv` (6 rows re-rated, one
with a new status: ch023.EqualDegreeFactorization from done to verified; 19 rows with path-only refreshes, tagged
`path refreshed 9c24389`) and `docs/magma/REVIEW_2026-11-06.md` record the result. Designs, reports and reviews
are in the session scratchpad only (`fu/final/`).

**Open follow-ups round: all 13 units done** (paused 2026-11-13, resumed 2026-11-14; S1 finished 2026-11-20). The list below was split
into 13 units by crate; every unit has a design with independent adversarial reviews, and each unit is
implemented with adversarial code review before its commit. Designs, reviews, reports, the workflow script and
the resume steps are in `docs/workstreams/handoff-followups/`. Committed and pushed:
- `80130e7` integers (I3): MAGMA/PARI signs for the i32/i64 gcd, lcm and xgcd; `fast_arith` unsigned gcd/lcm and
  `Result` modular ops; `Integer::sqrt` exact via `BigUint::sqrt`; Lehmer gcd/xgcd (half-gcd not done);
  `Integer::exact_root` and near-linear `is_power` / `is_perfect_power`; `SageInteger::binomial` by prime
  splitting, r < 0 cross-checked with gp and GAP; `sigma` overflow checks; `Integer::to_i64` now returns
  `Option<i64>` (137 call sites in 24 crates migrated); `class_number_quadratic` placeholder removed.
- `7c4d057` integers (I1): BPSW default test and an Atkin-Morain ECPP prover with checkable certificates
  (`try_is_prime`, `prove_primality`; Err when a proof is unresolved), so psi_12 is no longer accepted; a
  segmented sieve behind `prime_range`, `nth_prime` and `prime_pi`; the duplicate `is_prime_u64` removed;
  callers switched.
- `0ceaeaa` integers (I2): the quadratic sieve rewritten as SIQS (card T-223); `trial_division_with_primes`
  classifies its cofactor (the 🛑 row); `factorization_with_limits` returns the unfactored composites (MAGMA's
  third return value); the old QS `legendre_symbol` is gone with the rewrite, and the ECM module doc is fixed.
- `166ac34` polynomials (P1): `fp_factor` checks p and square-free input on every entry, sorts its output in PARI
  order and drops the legacy phase; typed Zassenhaus errors; `pgalois` and `padic_factor` check p; maclane's
  duplicate Rabin test removed; polynomial Ring hooks; Yun and Cohen citations fixed.
- `4b7f034` finitefields, core, polynomials (F1): every GF(p) and GF(q) factorization entry proves its
  characteristic prime (`try_is_prime`), both distinct-degree factorizations refuse 0 and non-squarefree input,
  `Gfpn::try_new` validates the modulus (Rabin, memoised), `random_irreducible` draws uniformly, `from_int` is
  O(1) on `Gfpn`; orders, `FactoredOrder` and `IsPrimitive` from a proven factorization (`group_order.rs`, Cohen
  Alg. 1.4.3; capped Ring hooks); `IsSquare` and `SquareRoot` in every GF(q); the `Ring` hooks
  `HAS_UNBOUND_CONSTANTS`, `is_unbound_constant`, `bind_to_parent_of`, used by the univariate, multivariate and
  Laurent polynomial containers.
- `fd081fa` rings, p-adics, category (R1): exact unbound p-adic `zero()`/`one()` (they panicked); value equality
  and `is_one` for `PadicRational` and extension elements; field certificates, field status and the inverse in
  certified p-adic extensions; decided p-adic and series Ring hooks; real `RingParentMethods` implementors
  (`category_parents.rs`, `ring_parents.rs`, `NumberField`, `DownUpAlgebra`, `DiffScalarFieldAlgebra`);
  `is_nonzero` through `domain_one`; composite characteristic is not a domain in both traits; six p-adic
  elliptic-curve tests un-ignored.
- `d3a133b` matrix (M1a): invariant factors, minimal polynomial and rational form by vector peeling
  (`frobenius.rs`, O((r+1) n^3) field operations), the Smith paths as certified fallbacks.
- `dc367a2` matrix, polynomials, finitefields (M1b): `FactorizationField` (Q, GF(p)), `primary_invariant_factors`,
  `primary_rational_form`, `jordan_form`, `is_similar` (`primary_forms.rs`) and the multi-RHS `IsConsistent` /
  `Solution` (`linear_solve.rs`).
- `2e000c9` combinatorics (C1): exact Integer counting (product-tree factorials, band-row Stirling and Eulerian,
  Bell, Motzkin, Schröder, derangements by binary splitting, multinomial without the u32 sum), the certified
  Hardy-Ramanujan-Rademacher `number_of_partitions`, `number_of_standard_tableaux`, falling and rising factorials
  over Integer and any ring, `Permutation::order_integer`, and a checked machine-word sweep of the crate.
- `bd03278` jupyter, rationals, finitefields (J1a): one depth-aware scanner and precedence chains in the REPL
  parser (S9), exact Q arithmetic, a nesting bound, correctly rounded `Rational::to_f64` and exact-first numerics
  (S11), chart dimension check (S14), checked plot options (N5), `pochhammer` is rising (S10), limits for graphs,
  `prime_pi` and `primes` (S15), `sqrt_mod` by one factorization, `num_square_roots_mod_prime_power`.
- `36c2170` jupyter (J1b): Arc-backed REPL values (F2, REPL part), growth guards on result size (S12), a display
  budget with GMP decimal conversion, interrupts, a time limit and a kernel that answers while it computes
  (S13), the C1 arms (any integer for falling/rising factorials, `partitions` up to 2·10^10), re-measured limits.
- `f29c492` manifolds (MF1; two comment lines in `rustmath-jupyter/src/repl.rs`): the exact exterior derivative and
  wedge product (Lee 2013 ch. 14, Spivak ch. 4) instead of the facades, written to every permutation with its sign and
  zero-tested slot by slot, so d∘d of a polynomial form is a literal 0; a budgeted exact zero test (`expr_zero.rs`:
  Zero, NonZero or Unresolved) with a derivative guard that refuses components whose derivative rustmath-symbolic
  fabricates; checked component counts (`component_count`, fallible allocation; `to_flat`, `total_rank` and
  `num_components` return `Result`); contraction and tensor product rewritten; the interior product, Cartan's Lie
  derivative, `Add`, `Neg`, `scale`, `vanishes` and a three-valued `is_closed`; constructors that check finiteness and
  antisymmetry; d checks the chart; the Lie derivative, symplectic, contact and topology routines delegate to the exact
  ones; three pullback and pushforward facades return `Err(NotImplemented)`. 49 new unit tests and 2 doc tests with
  PARI/GP 2.15 and Sage values.
- S1 symbolic (`rustmath-symbolic`, and the REPL's `taylor` arm in `rustmath-jupyter/src/repl.rs`): the work-in-progress
  commit `4763f31` (merged into `main` through PR #584) and this run's commits on `claude/gracious-gates-e6b6l7`. A
  truncated Laurent series engine over exact constants (MAGMA ch. 49; `series_engine.rs`, `series_const.rs`,
  `series_ball.rs`) behind `taylor`, `taylor_with_budget`, `series_expansion`, `laurent`, `asymptotic`,
  `series_coefficients` and the limit path, with a work budget, typed refusals and structural caps on the result; the
  owner's two rules at branch points and, from request C-1, a refusal when the argument leaves a branch cut; a parser
  with literals of any size that reads `a/b` as a quotient and runs on a heap stack; `expand` under a minus; `simplify`
  as a memoized work list with HEAD's results; a work-list `Display` and an iterative `Drop` for `Expr`. Reference
  values from PARI/GP 2.15.4 at 300 digits.

S1 in detail. What is merged: the work-in-progress commit `4763f31`, through PR #584 (`main` `28c4b69`). This run's
commits on `claude/gracious-gates-e6b6l7` add the finish under the narrowed bar, requests C-1 and C-4 and these records;
they reach `main` through the S1 pull request. The design (`docs/workstreams/handoff-followups/design/S1-symbolic.md`)
was never approved. Nine design reviews and four red-team rounds each found new holes over the fixed bar, so the owner
froze it (2026-11-15, `impl/S1-symbolic.owner_decision.md` in that folder) and asked for the regression tests first
(design review 9's two blockers, the red team's three holes, a singular-point matrix with gp values) and for two
behaviour rules: an exact-constant argument of a branch function is evaluated, and a non-constant argument at a branch
point is `NotLaurent`, never `DivisionByZero`. Six rounds of implementation review followed, each with a semantics and a
resources reviewer. The semantics reviewer approved every round but round 5 (a variable named pi or I could name a constant),
and in round 6 found no wrong value on 13,200 rows against a gp contour oracle. The resources reviewer kept finding consumers of a result (`simplify` in rounds 4 and 5; `expand`,
`differentiate` and the REPL's `diff` in round 6) whose cost is about the number of terms times their cost on the input's
kernel source. The owner then narrowed the bar (2026-11-16, `impl/S1-symbolic.owner_decision2.md`). A blocker is a wrong
or guessed result, an undecided answer reported as decided, a regression against HEAD-before-S1 (`8a0b310`) on an input
it handled, or a budget hole in S1's own entry points or in the printing, dropping and re-parsing of their results
(100 KB of input, 20 s, 512 MB, an 8 MB release or a 2 MB debug thread). The cost of a consumer applied to a result is
out of the bar: the doc comments give the measured scaling and claim no bound, and the follow-up unit S2 takes it. The
finish under that bar had two review rounds, each with a semantics and a resources reviewer: the first
(`impl/S1-symbolic.fin1-*.md`) was not approved (six distinct blockers, fixed test-first), the second
(`impl/S1-symbolic.fin2-*.md`) approved, and so did a check of its minor fixes (`impl/S1-symbolic.finminorscheck2.md`).
Requests C-1 and C-4 of `docs/workstreams/m23_golay/REQUESTS.md` were then approved by a semantics reviewer and a
regressions reviewer (`impl/S1-symbolic.c1-1-*.md`); their minor items were fixed, and no reviewer re-checked those
fixes. It is approved under the narrowed bar. The S1 items of the open list are fixed (the last bullet of the list) or
moved to the S2 and S1-handoff bullets.

C-1, the branch-cut rule. At each application of log, a non-integer power, arcsin, arccos, arctanh, arccosh, arctan or
arcsinh, when the constant term of the argument lies on an open cut and the engine cannot decide that every higher
coefficient is a real multiple of the cut's direction, the result is `NotLaurent` ("the argument leaves a branch cut at
the point"); an undecided real zero test is `UndecidedZero`. The variable is real. Free parameters are generic complex
indeterminates, the frozen design's convention, pending the owner. The five inputs of the request and their mirror
images refuse; 37 real controls on the cuts keep their values; the two-sided oracle of
`rustmath-symbolic/tests/data/s1_gp.py` passes the cut, matrix and fuzz data; one fuzz row changed with gp evidence
(`U:y12.1`). C-4. Every crate that depends on rustmath-symbolic passes: rustmath-calculus 184, rustmath-curves 345
(23 ignored), rustmath-functions 73, rustmath-geometry 523 (1 ignored), rustmath-manifolds 341 (11 ignored),
rustmath-plot3d 90, rustmath-benchmarks (no tests) and rustmath-jupyter 124 (3 ignored): 1,680 passed, 0 failed,
38 ignored. `cargo check --workspace --tests` has no error. rustmath-py is outside the workspace and was not built; by
inspection it uses only functions whose signatures S1 kept. rustmath-symbolic itself: 909 passed in debug and 793 in
release, 0 failed, and 23 other ignored tests pass (S1's 21 release measurements and child helpers, and HEAD's two
factor tests, `factor.rs:389` and `:402`); the ignored `test_airy_numerical_helper` fails, as at HEAD.

Conservative refusals, recorded (a later relaxation is welcome): a constant factor 0 does not hide a refused
subexpression, so 39 rows of the singular-point matrix of the form `x + 0*f(...)` refuse (14 `DivisionByZero`, 25
`NotLaurent`) where HEAD-before-S1 returned x, and two fuzz rows where HEAD returned 0; `sqrt(x^4)` and `abs(x)^2` at 0
are `NotLaurent`, where the function is x^2 (HEAD gave terms in 0/0 and sign(0)); abs and sign of a series whose
coefficients are real but not rational are `Unsupported` (`abs(x - sqrt(2))` at 0, where HEAD's value was right).

The trackers follow all thirteen units: `PROJECT_TRACKER.md` (Change Log "Open follow-ups round, part 1", "part 2",
"part 3" and "part 4 (S1)"), `docs/magma/status_overrides.csv` (part 2: 102 rows of the reviewed chapters re-rated, 57
of them with a new status, seven pilot rows, 18 rows added for chapters 83, 144 and 145, and 384 path-only refreshes
tagged `path refreshed 36c2170`; part 3: no row cites the code that `f29c492` changed, so none was re-rated; part 4: ten
chapter 49 rows from missing to partial, tagged `fixed S1`, and two path refreshes, tagged `path refreshed S1`) and
`docs/magma/REVIEW_2026-11-06.md`. The list below no longer has the items they fixed; its last four bullets name each
fixed item with its commit.

**Open follow-ups** (each has a tracker row or note unless marked; lines at `f29c492`, except in the files S1 changed,
`rustmath-symbolic` and `rustmath-jupyter/src/repl.rs`, where they are at the S1 tree of this run):
- MAGMA conventions still unverified against MAGMA itself (the handbook pages answer HTTP 401, no MAGMA
  binary): `Quotrem` over Z, `Content` over Q, the content in `IsIrreducible` over Z, `Binomial(n, r)` for
  r < 0, the shape of `QMatrix`, the generalised Jordan layout for deg p >= 2 and the block order of the primary
  forms over Q. Cross-checks in the code since `80130e7` and `166ac34`: Sage follows the
  conventions used here for `Content` over Q and for the content in `IsIrreducible`
  (`rustmath-polynomials/tests/ch23_is_irreducible.rs::sage_content_examples`); Sage's FLINT `quo_rem` over Z
  returns another remainder of the same minimal degree (`tests/ch23_quotrem_gcd.rs`
  `quotrem_over_z_minimal_remainder_is_not_unique`); PARI/GP, GAP and Sage agree with `SageInteger::binomial`
  for r < 0; the `q_matrix` doc gives the CompanionMatrix evidence for the shape (`berlekamp.rs:318-330`).
- Unbound constants in generic containers outside rustmath-polynomials (F1 design §7.6 and report; a
  correctness defect for mixed matrices and sparse matrices): `Matrix<R>` and `Vector<R>`
  (`rustmath-matrix/src/matrix.rs`, `vector.rs`: an entry that met only unbound constants stays an integer in Z,
  so `is_zero`, rank and determinant of a mixed matrix can see it), `SparseMatrix<F>` (`sparse.rs`: an unbound
  value that is 0 in the field is stored as nonzero), `PowerSeries` and `LaurentSeries`
  (`rustmath-powerseries/src/series.rs`, `laurent.rs`) and the rustmath-rings series types `PowerSeriesPari`,
  `PowerSeriesPoly` and `LaurentSeriesElement` (`power_series_pari.rs`, `power_series_poly.rs`,
  `laurent_series_ring.rs`). Bind them with
  `Ring::bind_to_parent_of` as `UnivariatePolynomial::new` does; M1a and M1b bind only their own inputs. The
  multi-RHS solvers return the unbound `F::zero()` for free variables over `PrimeField` (equal under `==`).
- Finite fields (F1 design §8, report and review; not tracked as rows unless marked):
  - `FiniteField::new(p, n)` without a Conway polynomial can take O(p) Rabin tests
    (`finite_field.rs:145`; `FiniteField::new(2^80+13, 3)` does not return, and `FiniteField::new(2, 256)` did
    not return within minutes in debug); a fix changes default moduli.
  - The canonical `poly_factor::roots(0)` and `factor(0)` return `[]` (`poly_factor.rs:998`, `:946`).
  - Trial-division sites that can call `group_order::factor_proven(.., Effort::Complete)`: `IntegerMod::sqrt`,
    `all_sqrts`, `num_sqrts`, `is_nilpotent` (`integer_mod.rs:348`, `:401`, `:443`, `:279`),
    `Zmod::factored_modulus` (`zmod.rs:123`), `ring_hooks::is_nilpotent_mod` (`ring_hooks.rs:91`).
  - `GaloisRing::with_modulus` (`galois_ring.rs:105`) runs the primality and Rabin tests twice.
  - Review notes (docs): the univariate `==` doc overstates for `LaurentPolynomial` coefficients
    (`rustmath-polynomials/src/univariate.rs:69-77`); `PrimeField::is_generator` cites MAGMA `IsPrimitive`
    without "for a prime modulus" (`prime_field.rs:159`; 2 mod 9 is primitive in MAGMA, false here); the capped
    order hooks' `# Cost` omits the ω(q - 1) + 1 exponentiations that follow the factorization (36.6 s for the
    generator of GF(2^200) in debug); `multiplicative_order` and `factored_multiplicative_order` should say
    "unbound nonzero element" in `# Panics` (`finite_field.rs:649`, `:663`). A failed capped factorization is
    recomputed on every hook call (only successes are cached).
- Rings, p-adics, category (R1 report and reviews):
  - `PadicExtensionElement::certified_valuation` (`extension.rs:902`) builds exact representatives, so the order
    hook and an undecided `inverse` are quadratic in the valuation (3.1 s at v = 10^5 in debug); V can be read
    from the stored valuations. `PadicRational` `Mul` and `pow` add the `i32` valuations unchecked
    (`padic_rational.rs:725`; `five.pow(3_000_000_000)` panics in debug and wraps in release).
  - `PadicRational::add` (`padic_rational.rs:660`) drops known digits and costs time linear in the valuation
    difference (1, 10, 104 ms at 10^3, 10^4, 10^5); `PadicInteger::lift` (`padic_integer.rs:243`) raises the
    precision without new digits; `PadicExtension::generator()` returns 0 in degree 1 (`extension.rs:610`);
    `PadicExtension::new` (`extension.rs:315`) accepts any f under any label, so `ramification_index` and
    `residue_degree` echo the label (OM's `ramification_data` could check it).
  - The extension inverse when only OM certifies the field (needs an integral basis), and the certificate and
    inverse for a proper prime-power p: `PadicRational::from_rational` rejects 1/2 in Q_4.
  - `impl Field for PadicRational` holds for every p > 1, but Q_p is a field only for a prime-power p
    (Q_6 ≅ Q_2 × Q_3); `MatrixAlgebra::<PadicRational>::full(1)` trusts it.
  - The parent objects for Z, Q and R[x] (`IntegerRingParent`, `RationalFieldParent`, `PolynomialRingParent`,
    `rustmath-rings/src/category_parents.rs:35`, `:39`, `:44`) carry only the ring predicates. They have no
    coercion or element constructors, R[x] has no generator, comparison, Category, Rank or ChangeRing, and
    `Integer`, `Rational` and `UnivariatePolynomial` values do not link back to them (Parent(x) has no API).
    `PolynomialRingParent::base()` (`:55`, BaseRing) is untested. Rows: ch018.IntegerRing, ch020.Rationals,
    ch023.PolynomialAlgebra and BaseRing (partial), and the ch023 structure rows (missing).
  - `SeriesRing` carries no base ring, so no series parent has ring predicates; IsIdempotent on series is the
    generic x*x == x and no series test calls it (ch049.IsNilpotent); `SimpleAlgebra` has no product;
    `nf_elem::NumberField` has no ring predicates (no `PartialEq` on `NfElem`; copy
    `rustmath-numberfields/src/category.rs`); `QuotientRingGeneric::is_integral_domain` and `is_field` are always
    false (`rustmath-rings/src/quotient_ring.rs:203`; ch017.IsDomain and ch017.IsField are partial for it).
  - Gaps in the ring predicates (status_overrides rows of chapters 17-25, 34, 37 and 48): the `RingParentMethods`
    of rustmath-rings (`category_methods.rs`) has no `is_pid`, `is_ufd` or `is_euclidean`, so Z, Q, Z_p, Q_p and
    R[x] do not answer IsPID, IsUFD or IsEuclideanDomain (the rustmath-category defaults decide only fields and
    non-domains); no parent answers IsUnitary, IsOrdered, IsDivisionRing, IsEuclideanRing or
    IsPrincipalIdealRing; the function-field order `is_pid` is still `class_number() == 1` with the class number
    hard-coded 1 (`function_field/order.rs:317`, `order_polymod.rs:207`; ch017.IsPID); no test asserts
    `is_euclidean` or `is_ufd` on a Galois ring, or the `GaloisRing` parent `is_commutative`; the chapter 37 digest
    says `IsEuclideanDomain(F)` errors for algebraic fields, while `NumberField` answers true (the chapter 34
    reading).
  - The `algebra_with_parent` and `down_up_algebra` tests of rustmath-algebras run only with feature `full`; a plain
    `cargo test -p rustmath-algebras` skips them. `rustmath-ellipticcurves/src/generic.rs:79-82` has a stale doc on
    the `PrimeField` zero/one.
  - The capped-relative, unramified and Eisenstein p-adic element types have no unit hook; real and complex types
    keep `static_is_commutative` = None.
- Matrices (M1a and M1b reports and reviews):
  - No bit-complexity bound over Q for the invariant factors and the forms (the peeling bounds field operations
    only); Storjohann's O(n^3) Frobenius form and [Ste97] are not implemented. Scalar-like input (r ≈ s ≈ n) is
    slower than the Smith path (2·I over GF(7), n = 100: 0.71 s against 0.05 s; documented). Some sentinel-only
    inputs over Z now answer where the Smith paths erred (documented). The old Smith path is much faster on this
    tree than at `c79a582` (cause not isolated).
  - `FactorizationField` exists for Q and GF(p) only: add `FiniteFieldElement` (GF(p^n), through
    `poly_factor::factor`) and the rings `GFp<P>`; `FactorableConstantField`
    (`rustmath-rings/src/function_field/typed/factor.rs`) could become an alias. A MAGMA-convention `RationalForm`
    (T·A·T^-1 = C^T) is `rational_canonical_form(&A^T)` transposed. Multi-RHS `IsConsistent` over Euclidean
    rings (Z) is missing. `is_similar` is unspecified for an odd composite `PrimeField` modulus.
  - Review notes: `peeled_forms_conjdiag20_s7_was_81_s` no longer detects a regression to the Smith path (the
    old calls take 12.7 s here, under its 20 s bound; add the caveat or lower the bound to about 5 s);
    `companion.rs:443` still says "(fast path)" for what is now the first fallback; the `release_timing_dense_q`
    label "Krylov path" means the public path; the `invariant_factors` docs use r for two things;
    `matrix_forms_gfp.rs:345` names two stages; the primary-forms docs do not say that the entries must lie in
    one field (two `PrimeField` moduli panic in the arithmetic); the `h26e9_magma_handbook_example` comment wording.
  - `cargo clippy -p rustmath-matrix --tests` fails on two old `erasing_op` errors in `integer_forms.rs` tests
    (`:645`, `:670`).
- Combinatorics (C1 design §1.7-1.8, report and review):
  - The machine-word sweep checked ten stated patterns across the crate; unfixed hits are bounded by memory,
    enumeration or 2^32 steps of work, guarded, or in flagged modules; arithmetic outside the patterns is not
    checked. Known to remain: the size invariant of partition-like types (`Partition::new(vec![1 << 63, 1 << 63]).sum()`
    is 0; about ten types need a checked constructor) and signed machine-word values (`Poset::mobius`, posets.rs:168,
    is 0 for a 194-element layered poset whose value is -2^64, tracker row ⏳;
    `PartitionTuple::degree_with_multicharge`, `SignedComposition::from_composition`, `AffinePermutation::apply`,
    `SixVertexConfiguration::energy`, `regular_sequences::sequence_sum` / `sequence_product` with `unwrap_or(0)`).
  - Wrong values found by the sweep, not fixed (each has a tracker row or a note in one, except the two outside
    the crate): `count_plane_partitions_in_box` (`plane_partition.rs:358`;
    (2,2,2) gives 15, gp 20); `polya_enumeration` uses w^L instead of L·w (`group_action.rs:256`); the one-part
    `hall_polynomial` case and `centralizer_group_cardinality` (`hall_polynomial.rs:117-133`,
    `similarity_class_type.rs:499`); `integer_lists` caps of 100 and `IntegerListIterator::next` always None
    (`integer_lists.rs:283-300`, `:615`, tracker 🛑 row); `integer_vectors_with_l1_norm_bounded` returns
    duplicates (`integer_vectors.rs:453`; (2, 1) gives 12 entries, 5 distinct); `signed_compositions`
    (`composition.rs:249`) differs from its doc and from Sage; `BlockDesign::new` checks only t <= 2 and v <= 10
    (`designs.rs:38`); `ColoredPermutation::order` returns 1000001 after 10^6 steps (`wreath_product.rs:159`);
    `partition_to_dyck_path` never returns for (2) and gives 2 N / 1 E steps for (1, 1) (`path_tableaux.rs:425`);
    `parking_functions.rs` is not compiled. Outside the crate (not tracked as rows): `rustmath-interfaces/src/gap_parser.rs:108`
    `Permutation::order` lcm overflow, and `rustmath-algebras/src/temperley_lieb_algebra.rs:317` `catalan_number`
    wraps from n = 33.
  - Review notes: `count_superpartitions` allocates n + 1 words before its first check
    (`superpartitions.rs:453`; n = 2^40 aborts instead of the documented panic); `falling_factorial_in` /
    `rising_factorial_in` hold all k factors at once (`factorials.rs:166`); the "n! is about n / 4 leaves" comment
    holds only below 2^16 (`factorials.rs:13`).
  - `cargo clippy -p rustmath-combinatorics` stops on five old deny lints (`binary_words.rs:336`,
    `debruijn.rs:101`, `path_tableaux.rs:454`, `super_tableaux.rs:297`, `word.rs:252`) and, with
    `--all-targets`, on `erasing_op` in a `composition.rs:775` test. C1's product tree (`factorials.rs:28`)
    duplicates I3's private BigUint tree in rustmath-integers.
- REPL (J1a and J1b reports and reviews; `rustmath-jupyter/src/repl.rs`; not tracked as rows unless marked):
  - Review notes of J1b, marked required and not applied: `print(...)` joins strings past STRING_BYTES (`:958`;
    four nested prints reach 1.9 GB); `eval_depth` is not reset after a panic that `eval_guarded` catches
    (`:665`, `:808-817`; reset it in `eval` and name it in the `eval_guarded` doc, `kernel.rs:122`; latent); the
    timing sentences of `BINOMIAL_WORK` (`:11150-11156`) and `checked_binomial` (`:12183-12194`) state pre-`80130e7`
    times (C(524288, 262144) now takes 0.03 s); `gcd`, `lcm` and `xgcd` (`:1021`, `:1054`, `:1035`) run the quadratic
    gcd without the `check_rational_gcd` estimate (6.9 s at 2^21 bits). Nits: the TimeoutError text shows the
    current limit, `charpoly` prints through the library Display, the `EVAL_DEPTH` doc, and the time models
    refuse trivial calls such as `multinomial(4294967295, [4294967295])` (which is 1) and `stirling1(n, n - 1)`
    (which is C(n, 2)).
  - Review notes of J1a: the phase-3 cost cap of `sqrt_mod_factor` (`:12622`, 20 s modelled) is above the worst
    case it bounds and restarts per piece; repeated factors are stripped one division at a time
    (`sqrt_mod(1, (2^10000)^30)` takes 59 s in debug; the finitefields `prime_power_sqrts` lift is bit by bit);
    probable primes above 4096 bits are refused where the earlier version gave a caveat (`sqrt_mod(4, 2^4423 - 1)`;
    an `is_probable_prime` after phase 3 would fix it); `Rational::to_f64` allocates on every call
    (`rustmath-rationals/src/rational.rs:163`); `N` of NaN says "outside the range of a double"; a plot range or `find_root` bound with a real operand is
    computed in doubles, so `10^20 + pi - 10^20` is 0 there (as in MAGMA at 53 bits; documented, J1a design §9).
  - Nested calls `abs(abs(...))` overflow a debug build's 8 MB stack at about 40 levels, below `EVAL_DEPTH` = 200
    (`:11301`; the kernel's 64 MiB thread holds 200).
  - No Float arithmetic in `apply_binary_op` (`2 * 1e-3` is a TypeError), and no Rational arm on symbolic values
    (`x / 2`, `x^(1/2)`); string literals are not unescaped; `integrate_num(x, x, 0, 1)` with a bare symbol is a
    TypeError; `Plot3DOptions` is unused and the 3D plots have fixed grids; graph commands other than the
    constructors (`edges`, adjacency output, `show_graph`) have no limit of their own (the constructor limit bounds
    their output); `median`, `min`, `max` in
    rustmath-stats panic on NaN (`statistics.rs:60`, `:129`, `:136`) and `mean([1e308, 1e308])` is inf; the
    num/den float division of S11 recurs in `rustmath-schemes/src/elliptic_curves/heegner.rs`,
    `rustmath-rings/src/qqbar.rs`, `rustmath-numberfields/src/kummer.rs`, `rustmath-combinatorics/src/derangements.rs`.
  - J1b design §7: `Matrix::pow` squares once more than needed (`rustmath-matrix/src/matrix.rs:410-436`); negative
    matrix exponents are refused (MAGMA allows them for invertible A); library calls cannot be interrupted (a
    cancellation hook in rustmath-core would extend S13); `Display for Integer` in rustmath-integers is quadratic
    (9.3 s at 2^25 bits; the REPL converts through rug); Clifford and Jordan displays are outside the budget, and
    `print` and `vars` apply it per value; a plot samples a DAG as a tree (`e = e + e`); list literals copy their
    Integers; `det`, `rank`, `matrix_inverse`, `charpoly`, `rref`, `factor` (`:1089`, trial division), `sigma`'s
    factoring (`:2790`) and `nth_prime` are unbounded library calls.
  - J1b's new items: every rational operation reduces with a quadratic Lehmer gcd and rustmath-rationals has no
    constructor from coprime parts, so the REPL refuses r^e of large fractions by its gcd estimate (27.7 s at
    2^22-bit parts); `POW_BITS` could be 2^26 (5.3 s measured); the binomial family and `PRIMES_COUNT` are
    conservative (`catalan(316227)` takes 0.03 s); `stirling_band`'s a-priori bounds are loose at the band edges
    for stirling2 with small k; `kernel_stops_on_shutdown_during_library_call` relies on `factor` being unbounded
    trial division. `TAYLOR_ORDER` (`:11195`) stays 20000 as an argument check; S1's `TAYLOR_WORK` (`:11261`) bounds
    the call (its calibration is an S1 handoff below); for `FORM_COMPONENTS` and `DIFF_FORM_DEGREE` see the manifolds
    items below. `is_prime`, `next_prime` and `sqrt_mod` prove primes up to `PROOF_BITS` = 1024 bits (`:12527`).
- Integer factorization and primality (not tracked as rows unless marked):
  - `factorint::factor_trial_division(n, None)` (`rustmath-integers/src/factorint.rs:199`) stops at 10^6 above
    2^128 and appends a composite cofactor, against its doc (tracker 🛑 row since 2026-11-14); `factorization::trial_division_stage`
    can serve it.
  - `prime::factor` and `prime::prime_divisors` (`prime.rs:634`, `:688`) are unbounded trial division (3.4 s in
    release on 100000007 * 1000000007; hours for a 12-digit second factor), used on 41 lines in 24 files of 8
    crates (the REPL `factor` and `sigma` at `rustmath-jupyter/src/repl.rs:1089` and `:2790` among them); rebuild
    them on `factorization_with_limits(n, &FactorizationLimits::default())` or document the cost (tracker row).
  - `ECMLimit`: a new ECM (projective Montgomery or Edwards curves, Suyama's parametrisation, a seeded sigma, a
    full stage 2) is needed before an `ecm_limit` in `FactorizationLimits` pays; without it a large composite
    with a medium factor costs the sieve's time for the whole number (hours to days from about 90 digits).
  - ECPP coverage: no unresolved prime up to 2048 bits in the design's experiments, 2 of 32 at 3072 and 4096
    bits; a 1024-bit prime takes 3.4 s in release. A faster prover above 2048 bits is open (I1 design §7).
  - `random_prime(a, b)` enumerates the whole range (`prime.rs:570`) and MAGMA `RandomPrime(n)` is missing;
    `prime_pi` above 2^64 (Lagarias-Miller-Odlyzko) is missing (tracker rows).
  - The Pratt certificate verifier in rustmath-constants (tracker: unsound) is separate from ECPP.
  - The private `is_prime_u64` copies in rustmath-modular, rustmath-ellipticcurves and rustmath-galois can call
    `rustmath_integers::prime::is_prime_u64`; rustmath-rings (`category_methods.rs:36-131`) and rustmath-category
    (`ring_category.rs:202-224`) keep private primality and prime-power helpers too.
  - Half-gcd is not implemented (Lehmer since `80130e7`; estimated crossover 10^5 to 10^6 bits), and
    `Integer::nth_root(n)` (MAGMA `Iroot`) is slow for a large n on a large input (num-bigint's Newton start; a
    start from the leading bits would fix it).
- Unit S2, DAG-aware consumers (spec `docs/workstreams/handoff-followups/design/S2-dag-consumers.md`; not designed, not
  implemented). A series result of S1 is a DAG whose kernels every term shares, but `expand`, `differentiate`,
  `substitute`, `PartialEq`, `integrate` and the REPL's `diff`, `expand`, `subs` and `integrate` walk it as a tree, so
  their cost on a result is about the number of terms times their cost on the input's kernel source: `expand` 4.0 s and
  2.5 GB and then `simplify` 21.8 s and 5.6 GB for a 110-byte input at order 1000; the REPL's `diff` 11.6 s and 2.1 GB
  for a 4.9 KB input; `integrate` 43.3 s at 1,000 terms (cubic, as at HEAD on the same polynomial typed as text). Goal:
  cost linear in the DAG, outputs that share what the result shares, values unchanged; the acceptance inputs E1 to E6,
  D1 to D10, S1 to S4 and I1 of the spec; iterative walks (`substitute` and `PartialEq` still recurse); an optional
  phase 2 may relax S1's per-occurrence caps. It changes rustmath-symbolic code outside S1, so it needs its own design
  and reviews.
- S1 handoffs (minor; `docs/workstreams/handoff-followups/impl/S1-symbolic.md`, the "Handoffs" sections of each round;
  tracker rows of rustmath-symbolic):
  - Engine scope: Puiseux and log terms, gamma and zeta, fast series arithmetic, products of radicals, the result-tree
    refusals of towers and a per-cell resource policy (design §7 items 1 to 10; rustmath-calculus `taylor.rs` keeps its
    own implementation). abs and sign of a series with real but not rational coefficients need the owner's decision on
    design §2.4 (11 first-step fuzz rows would become values; gp references in `tests/s1_review4.rs`). The thread-local
    work state could become the design's `&mut Work` (a refactor under the same tests). `TAYLOR_WORK` (1e8) is the
    design's provisional value: calibrate it with the design's §5.3 script, and port the design's §4.1 test list as
    `tests/series_engine.rs`.
  - Constant field: the numeric zero test has no complex coverage (trig and hyperbolic kernels of complex constants,
    inverse functions off their real domains, complex radicals are `UndecidedZero`); sqrt(-b) = i sqrt(b) for a
    rational b > 0 would decide `arctan(sqrt(-1)*x^0)` and reduce `((-4)^(1/2))^2`; fold exp, sin, cos, sinh and cosh
    at q i pi (`exp(I*pi)` stays a kernel) and log(-c) = log(c) + i pi, so that `I*(log(-2) - log(2))` and
    `exp(I*pi)` are decided real for the cut rule; Gaussian inverses for constants with parameters or more than 16
    terms; cache each kernel's enclosure in `KData` (the class of a `log(2 + sin(...))` chain is quadratic, charged);
    a float exponent with an integral value is undecided (`x^2.0` at 0, the design's rule of no float folding).
  - Branch cuts (C-1): a finer rule for an argument that leaves the cut to the same side for both signs of x (the
    eight rows of choice 2 in `tests/data/s1_cuts.tsv` are its first tests); a refused branch subexpression that cancels
    or is multiplied by 0 refuses the input (`log(-1 + I*x) - log(-1 + I*x)`, `x + 0*sqrt(-1 + I*x)`); two owner
    decisions were taken under the convention that parameters are generic complex (`log(-y^2 + I*x)` at 0, `log(sin(x))`
    at a symbolic point), with the reviewer's alternative, real parameters, recorded in the report; `log(tan(x))` at a
    symbolic point stays "cannot decide".
  - Parser: exempt the parser's own printed forms from the depth limit (a printed result with an integer above
    i64::MAX, a rational with such a denominator or `^ -(`, then more than 1,024 units of nesting, does not re-parse);
    the docs' short form "128 levels of parentheses and calls (or 1 024 of `^`)" is exact only for unmixed nesting (the
    limit is 1,024 units, 8 per parenthesis or call and 1 per `^`), and three byte counts in the docs are one too high
    (`impl/S1-symbolic.finminorscheck2.md`, n1 and n2); `5 / 0`, `(1)/(0)` and `10\r/0` are refused, where HEAD read
    Div(n, 0); a float literal that underflows to 0 is refused, where HEAD read 0.0.
  - Limits (HEAD's code; tracker 🔴 row since part 4): `Expr::limit`, `limit_advanced` and the `combine_limits_*` laws
    return symbolic `Finite` values without a zero test of a denominator (154 `Finite` values at a pole and 8 wrong
    values on 181 inputs that the S1 parser now reads); direct substitution takes `0^10*exp(0/0)` as 0, so the left
    limit of `1 + x^10*exp((sin(x^2)-x^2)/x^9)` at 0 is `Finite(1)` (it is +infinity); `log(-1 - I*x^2)` at 0 gives
    `Finite(log(-1))` (-i pi for a real x); `Expr::limit`, `limit_enhanced` and `substitute` recurse per node (a tower
    `x^-x^-...` of 1,400 levels aborts a 2 MB debug thread in this code of HEAD's, reached because S1's parser now
    reads the text that HEAD's parser refused). `limit_with_error` gives `Finite` only for a rational limit (1,097 of
    HEAD's symbolic answers became `Unknown`).
  - Tests and tools: `kernel_protocol.rs` picks ports by binding and releasing them, so parallel kernels can collide
    (`AddrInUse`) and `kernel_aborts_queued_cells` then waits forever (run it with `--test-threads=1`); the ignored
    `test_airy_numerical_helper` fails (`airy.rs:369`, as at HEAD); `expand` of `L = -(L + x) + x` needs memory
    quadratic in the depth (1.48 GB at 4,000 levels, as at HEAD).
- Manifolds after MF1 (MF1 design §4.4 and §5, report and code review; `rustmath-manifolds/src/` unless marked; not
  tracked as rows unless marked):
  - REPL (`rustmath-jupyter/src/repl.rs`, design §4.4): `DIFF_FORM_DEGREE` (`:11275`, used at `:4835`) can go, since the
    library returns `DegreeOverflow`; drop its test lines (`:14845-14846`, and the degree block of
    `test_form_component_limits` from `:15456`). The arms' own overflow checks (`:4878-4879`, `:4917-4918`) are
    redundant. `check_form_components` (`:12488`) and `missing_components_error` (`:12508`) should count with
    `rustmath_manifolds::tensor_field::component_count` ("too many components to store"). Re-measure `FORM_COMPONENTS`
    in release with J1b's harness (64 bytes a slot, held once) and rewrite its doc (`:11276-11281`, which still says
    each d enlarges the expressions) and the `repl_limits` overview (`:11115-11116`, which also still says that
    `TAYLOR_ORDER` waits for S1). Every REPL chart is named
    `user_chart` (`:4817`) and is not registered, so REPL charts need unique names or registration before forms get
    symbolic components. Optional: expose `is_closed`, `closedness`, `interior_product`, `lie_derivative` and the
    components (a form displays only its degree, `:550-553`).
  - The `differentiate` facades of rustmath-symbolic (tracker 🛑 row "Derivative of a named function":
    `rustmath-symbolic/src/differentiate.rs:230-234` named functions, `:90` Mod, `:184-188` gamma, `:191` factorial,
    `:204` zeta, `:194-198` erf with 500/443 for 2/sqrt(π), `:218-221` a `root_sum` whose polynomial contains the
    variable). d and the 1-form Lie derivative refuse them (`expr_zero::unsupported_derivative`, `expr_zero.rs:726`);
    the other callers in the crate are unguarded: `vector_field.rs:169`, `:198`, `:202`, `scalar_field.rs:149`,
    `maps.rs:126`, `transition.rs:143`, `utilities.rs:134`, `symplectic.rs:398`, `:405` (`from_hamiltonian`),
    `symmetries.rs:121-139`, `finsler.rs:160-161`, `:309-311`. Wiring the hooks of `function.rs` (`:23`,
    `FunctionRegistry` `:207`) into `differentiate` lifts the refusal; `unsupported_derivative` must then admit exactly
    the kinds that gained a derivative.
  - Integration (tracker 🔴 row since part 3): `integrate_form_on_chart` and `VolumeForm::evaluate_at` read
    `components[0]` (`integration.rs:310`, `:254`), which is 0 in every n-form with n >= 2; the quadrature for n >= 2
    covers one coordinate (`:644`); `VolumeForm::from_metric` allocates factorial(n) slots (`:201`). Stokes' `verify`
    (`:361`) compares with 0.
  - Pullback and pushforward (tracker 🛑 row): the Jacobian-minor pullback of p-forms, the Jacobian products for
    tensors (Lee 2013 ch. 12 and 14) and the inverse map for the pushforward (`maps.rs:327`, `:362`, `:428` return
    `Err(NotImplemented)`); `apply_to_one_form` (`:278`) does not compose ω with f; `apply_to_tangent_vector` pushes
    zeros and keeps the source base point (`:188-195`).
  - Symplectic (tracker 🛑 row): `matrix_at` (`symplectic.rs:261`) ignores the point, the chart and the form, so
    `is_non_degenerate` (`:244`) and `evaluate` (`:281`) use the standard form; `SymplecticForm::new` (`:169`) checks
    only the degree. The pattern of `is_contact` (`contact.rs:175`, one independent component of ω^n) gives an honest
    test.
  - `LieDerivative::apply_to_tensor` (`lie_derivative.rs:209`) returns `Err(NotImplemented)`: the correction terms of
    the Lie derivative of a general tensor are missing (formula in its doc and error, Lee 2013 ch. 12).
  - Charts are identified by name only (`tensor_field.rs`, `vector_field.rs:81`, `scalar_field.rs:31`,
    `riemannian.rs:79` key components by `chart.name()`); d checks a registered chart and trusts an unregistered one.
    Storing the coordinate names with the components would let every reader check the chart.
  - Unchecked counts outside MF1's files: `spin.rs:76`, `:78`, `:341`, `:343`, `almost_complex.rs:160`,
    `finsler.rs:153`, `:300`, `symmetries.rs:110`, `symplectic.rs:266` (`matrix_at`); the module rank sums
    `modules.rs:605`, `traits.rs:124`.
  - Dense storage: a p-form holds n^p slots of 64 bytes, not C(n, p); ω^4 on R^8 needs 8^8 slots, and from R^10 on
    `SymplecticForm::volume_form` ends in `ComponentAllocationFailed`. `DiffForm::new` means both "zero" and "no
    components in this chart". `tracker_08_09_tests.rs` is still not compiled (it reads the private field
    `PullBack::map`).
  - One public zero test: move `expr_zero` to rustmath-symbolic and merge it with S1's constant field
    (`series_const.rs`, whose zero test is three-valued too).
    rustmath-symbolic's `simplify` folds `0 * a` to 0 for any a (`oo`, NaN and `1/(y - y)` included) and keeps
    `Rational(1)` as a `Rational`; `Expr::from(f64)` maps NaN and the infinities to `Integer(0)`
    (`rustmath-symbolic/src/expression.rs:485-490`), so such a float becomes a literal 0 before the finiteness checks of
    forms see it.
  - `cargo clippy -p rustmath-manifolds` stops in rustmath-combinatorics on its five old deny lints (see Combinatorics;
    `--no-deps` passes), and with `--tests` also on `approx_constant` in `examples.rs:520`, `:522`.
  - Code review notes of MF1 (minor, not applied): two negation items of `test_zero_test_budget`
    (`expr_zero.rs:956-972`) are Unresolved by the nesting limit before the negation charge is reached, because the
    factors of P are balanced sums (build them left-deep with 100 terms each, or use 150 levels); `Symbol("π")`, which
    `rustmath-symbolic/src/solve.rs:911`, `:919`, `:938`, `:952`, `:977`, `:985` build for π, is a free indeterminate
    in the zero test (`reserved::NAMED_CONSTANTS`, `expr_zero.rs:94`, has only `pi` and `e`), so π - e is NonZero where
    pi - e is Unresolved (add "π", a doc sentence and a test row); the module doc (`diff_form.rs:49-50`) says every
    result component is simplified and zero-tested, but `Add` copies a summand when the other is a literal 0
    (`:945-948`) and `Neg` only simplifies.
- From the P1 design, section I (not tracked as rows): facade-like `Err` handling of Zassenhaus results at
  `rustmath-groups/src/galois_resolvent.rs:159-169`, `rustmath-curves/src/belyi/audit.rs:390-398`,
  `rustmath-igp24/src/main.rs:117` (`unwrap_or(false)`), `rustmath-functionfields/src/factor.rs:70` and
  `genus.rs:240-244`; `local_galois_candidates` (`rustmath-groups/src/pgalois.rs:217`) assumes a transitive
  local group at ramified primes; `LaurentPolynomial::is_one` is false for the zero polynomial of the zero
  ring (`rustmath-polynomials/src/laurent.rs:370`); `rustmath-numberfields/src/panayi.rs:326` duplicates
  `is_padic_square` without the prime check; `LaurentPolynomial`'s `Mul` adds the i32 exponents unchecked
  (`laurent.rs:287`; an overflow panics in debug; the hooks use a checked product).
- From the I3 report (not tracked as rows): call sites that `80130e7` turned into explicit panics guard
  pre-existing facades: `rustmath-powerseries/src/algebraic.rs` (`i128_of` callers turn `None` into an empty
  root list), `rustmath-algebraic/src/radicals.rs:42-43` (returns alpha itself as its n-th root; tracker row),
  `rustmath-numbertheory/src/quadratic_forms.rs` (counts 0 solutions for m <= 0) and
  `rustmath-quadraticforms/src/local_densities.rs:231` (counts only the first 1000 residues).
  `rustmath-symbolic/src/integrate.rs` `polynomial_degree` returns `Some(-1)` for x^(-1).
  `SageInteger::binomial` near its 2^32-bit cap needs about 10 GB (a segmented sieve would remove the K-byte
  sieve). `rustmath-py` is outside the workspace and was edited without a compile check in `80130e7` and
  `7c4d057`.
- Fixed by part 1 of the round and removed from this list: `80130e7`: `IsPower(n, k)` without a root
  (`Integer::exact_root`), the bits^2 `is_perfect_power`, the sign-arbitrary i32/i64 gcd/lcm/xgcd and the
  `div_rem(MIN, -1)` panic, the `ArithInt`/`ArithLLong` gcd and lcm overflow on MIN and `add_mod`/`mul_mod`
  for m <= 0, `Integer::sqrt` starting Newton at x = self, a faster balanced gcd (Lehmer; half-gcd above stays
  open), the slow `SageInteger::binomial` (S1), the unchecked `sigma` exponent (S5), the panicking inherent
  `Integer::to_i64` and the `series.rs` expansion point (S7, first two parts), and `Binomial(n, r)` for r < 0
  cross-checked with PARI/GP, GAP and Sage; `7c4d057`: no proving `is_prime` above psi_12 (BPSW and ECPP now;
  psi_12 is split by every driver), `square_root_mod_prime` testing primality twice, the repeated
  Miller-Rabin bases and the per-candidate `nth_prime` / `prime_pi` (S6), the duplicate
  `fast_arith::is_prime_u64`; `0ceaeaa`: the broken QS sieve stage and its two pinned tests (SIQS, card T-223),
  `trial_division_with_primes` reporting its cofactor as a prime, `factor_pollard_rho` without a capped variant
  (`factor_pollard_rho_limited`, `factorization_with_limits`), the QS `legendre_symbol`, the ECM module doc;
  `166ac34`: the unchecked squarefree input of `fp_factor::distinct_degree_factor`, the unchecked p in
  `fp_factor::factor` and `pgalois::unramified_local_galois_group`, the d + 1 rungs of `block_contract`, the
  redundant Rabin test in maclane, the literal in the typed factor, phase 1 and the unsorted `factor` output,
  the Zassenhaus error text, the Cohen citations in `zx.rs` and `univariate.rs`, and the Ring hooks for
  polynomials.
- Fixed by part 2 of the round and removed from this list: `4b7f034`: the unchecked squarefree input of both
  rustmath-finitefields distinct-degree factorizations and the unchecked lava `squarefree_factorization`,
  `check_prime_characteristic` calling `Integer::is_prime` (now `try_is_prime`), an undetected reducible `Gfpn`
  modulus (`Gfpn::try_new`), the non-uniform `random_irreducible` for p > 2^63, the O(min(n, p)) `Gfpn::from_int`,
  the O(m) order loops of `IntegerMod` / `PrimeField` and the trial-division factoring of q - 1 for GF(q) orders,
  `IsSquare` / `Sqrt` with the root on `PrimeField` and GF(p^n), `static_is_commutative` for the finite-field
  types, and the Galois-ring containers built from `R::one()` (degree 7 instead of 6); `fd081fa`: the panicking
  p-adic `zero()`/`one()`, `RingMorphismMethods::is_nonzero` reading the static `one()`, no real implementor of
  the three-valued `RingParentMethods`, the missing composite-characteristic rule in the category trait, and the
  Ring hooks of series and p-adics; `d3a133b`: no complexity bound for the rational form and invariant factors
  over a field (now O((r+1) n^3) field operations), the rational-form "Fallback" paragraph and the
  `polynomial_matrix` header naming only the `PrimeField` sentinel; `dc367a2`: the multi-RHS `IsConsistent`,
  `PrimaryRationalForm`, `JordanForm`, `IsSimilar`; `2e000c9`: the S2 u32 overflows in rustmath-combinatorics
  (catalan, rising factorial, multinomial, Schröder, `partition_count`), S3 an Integer p(n) for n > 416
  (`number_of_partitions`), S4 the faster callees (Fibonacci doubling, the Bell triangle, rolling Stirling rows,
  the derangement and Motzkin recurrences), the u32 `binomial` through `SageInteger::binomial`; `bd03278`: N5
  (plot options), S9 (the `-` before `*` split and commas inside string literals), S10 (`pochhammer` was the
  falling factorial), S11 (`to_f64` of large rationals), S14 (`chart` names against dim(M)), S15 (graph
  constructors, `prime_pi`, `primes`), the REPL `sqrt_mod` trial division (and the dropped 4 and 73 mod psi_12
  rows); `36c2170`: F2 in the REPL (values cloned on every assignment and lookup), S12 (growth guards for
  `matrix_power`, `**` and products), S13 (no timeout or interrupt), the REPL `falling_factorial` /
  `rising_factorial` and `partitions` arms limited to u32 and 416, and the 7-minute `test_table_limits`.
- Fixed by part 3 of the round and removed from this list: `f29c492`: F2 in the library (repeated d and wedge built
  ever larger expression trees, 574 MB for d(d(dx_0)) at 10^6 components, and the O(n^(p+2)) inner loop of d), the d
  and wedge facades (a flat-index sum and the raw tensor product, tracker 🔴 rows), N4 / S8 (the unchecked
  `n.pow(p + 1)` and `n.pow(p + q)` in `diff_form.rs` and `tensor_field.rs`), the library d without a chart check, the
  contraction that summed index values, the tensor product layout, `Add` and `Neg` returning an operand, the
  structural `is_closed`, `is_contact` and symplectic volume form built on the facades, `ContactForm::standard` with
  symbols that are not coordinates, and the three pullback and pushforward relabellings (now `Err(NotImplemented)`).
- Fixed by part 4 of the round (S1) and removed from this list: `4763f31` and this run's S1 commits: `Expr::taylor`
  differentiating without simplifying (memory exponential in the order, a stack overflow at order 20000; now the
  series engine under `TAYLOR_WORK`); the i64-only literal parser and its `rational` rule (`x^2/4` read as x^(1/2),
  `2/3^2` as 4/9, `1/x` refused; tracker 🔴 row since part 2); `expand` under a minus (`expand(-(x - 1))` gave x - 1;
  tracker 🔴 row since part 2); the `laurent` facade that dropped the principal part; a symbolic chain `w + w + ... + w`
  of 3000 terms aborting the debug driver in `Expr` Display and Drop; `-(-x)` displayed as `--x`; and the MF1 note on
  `impl Drop for Expr` (it compiles in rustmath-manifolds, and MF1's slot assertions pass).

**Next steps** (unchanged): wire up or delete the dead code listed in
`docs/magma/REVIEW_2026-11-06.md` (the `rustmath-rings/src/number_field/` orphan files, the gated
genus-2 curves stack, the commented-out calculus modules), and wire the `libgap_mixin` algorithms to
`GroupLibGAP`/`PermutationGroup`. Then the candidates from `docs/PLAN.md`: complete GF(p^n); general
Schreier-Sims (the keystone for groups and Galois chapters); and removing the duplicates left by the
merge: two GF(p^n) stacks in `rustmath-finitefields`, two Montes/OM implementations
(`rustmath-padics::montes` vs `rustmath-rings::padics`), and the placeholder
`rustmath-galois::stauduhar` vs the real descent. Implement from the cited papers; test against
LMFDB, PARI or GAP (GAP is installed).

## Working rules learned this session

- Commit and push often; the container can restart mid-task (it did several times). Detach long
  jobs with `nohup` so a restart does not kill them.
- Full `cargo test --workspace` takes about an hour; run it detached and log to the scratchpad.
- Crossref rate-limits after about 300 quick queries; `reading_list.py` caches and backs off.
- The user prefers no em-dashes and no stock phrases in prose.
