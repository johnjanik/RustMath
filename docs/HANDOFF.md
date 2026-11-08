# Session handoff (2026-11-08, updated after Tasks 0-2, the L0-L1 fixes and their follow-ups)

Read this first, then `CLAUDE.md` and `docs/README.md`. It records where the project stands and
what to do next. **Tasks 0, 1 and 2 are done** (status notes below), and so are the L0-L1 defects
the review found (2026-11-07) and their follow-ups (2026-11-08). The next step is "After these
tasks" at the end: the open follow-ups, then the dead code, the mixin wiring and the `docs/PLAN.md`
build targets.

## Where things stand

- **Direction:** RustMath is a pure-Rust port of MAGMA. The handbook digests in
  `rustmath-lava/docs/Chapter_NN_*.md` are the spec. SageMath is a secondary reference only.
  IGP24 is over; its code lives at https://codeberg.org/johnjanik/inverse_galois (map:
  `docs/igp24/SOURCES.md`).
- **Git:** one line of history. `main` contains everything (PRs #569, #570). The 19 old branches
  were deleted; their tips are tags `archive/<branch>` (`docs/BRANCHES.md`). Work on a new branch
  off `main`; cloud sessions can push only their own branch (no tags, no other branches).
- **Build and tests:** the whole workspace compiles, and `cargo test --workspace` is green:
  14,093 passed, 0 failed, 330 ignored, 189 targets at `702d0a5` (`docs/TEST_BASELINE.md`). First build
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

**Open follow-ups** (each has a tracker row or note unless marked):
- `fp_factor::equal_degree_factor` gives up after 200 random tries and returns the block unsplit (a
  bounded search that decides; about 2^-200 per block, never observed). Zassenhaus and so
  `is_irreducible` over Z and Q rely on it; Berlekamp's count r would certify the block (tracker ⏳ row).
- MAGMA conventions still unverified (the handbook pages answer HTTP 401, no MAGMA binary): `Quotrem`
  over Z, `Content` over Q, the content in `IsIrreducible` over Z (Sage's `(2*x).is_irreducible()` over
  ZZ and QQ would be a cross-check), `Binomial(n, r)` for r < 0, the shape of `QMatrix`.
- REPL `binomial` truncates n and k to u32 and refuses negative n (new card T-250, tracker 🔴).
- Rational form and invariant factors: no complexity bound (the fixed start basis has a worst case with
  r = n, and one 2x2 relation matrix over Q took 81 s in debug); [Ste97] or Storjohann would bound it.
  Doc nits from the last review: the rational-form "Fallback" paragraph names only the sentinel trigger,
  the `polynomial_matrix` header names only `PrimeField` sentinels.
- REPL `sqrt_mod` for a composite n factors by trial division (up to three times), so a large prime
  cofactor never finishes (stated in the help line); a proven factorization would fix it.
- p-adic `zero()`/`one()` still panic (`PadicInteger`, `PadicRational`, `PadicExtensionElement`), and
  `RingMorphismMethods::is_nonzero` reads the static `Self::Domain::one()`; no real ring type implements
  the three-valued `RingParentMethods`; the category trait lacks the composite-characteristic rule.
- Galois-ring containers built from `R::one()` or `UnivariatePolynomial::var()` keep unbound coefficients
  (degree 7 instead of 6 in the documented GR(8,2) example); pinned by a test that must change if
  containers ever coerce.
- Not tracked as rows: `IsPower(n, k)` returns no root; `is_perfect_power` still costs about bits^2 on huge
  non-powers (2^1000000+1: 12 s in debug); i32/i64 keep the sign-arbitrary default gcd/lcm/xgcd;
  `ArithInt`/`ArithLLong` gcd and lcm overflow on MIN, `add_mod`/`mul_mod` assume m > 0; `Integer::sqrt`
  starts Newton at x = self; a faster balanced gcd (Lehmer, half-gcd); MAGMA `IsSquare`/`Sqrt` returning
  the root on `PrimeField` and GF(p^n); `square_root_mod_prime` runs `is_prime` twice and
  `is_prime_miller_rabin` repeats its 12 bases 20 times; `fast_arith::is_prime_u64` duplicates
  `FastArith64`; Cohen GTM 138 algorithm numbers in `zx.rs` and `univariate.rs` are unchecked; the
  Zassenhaus error text blames every failure on the subset bound; Ring hooks for polynomials, series and
  p-adics (honest `None`); O(m) order loops for `IntegerMod`/`PrimeField` and trial-division factoring of
  q-1 for GF(q) orders; the multi-RHS `IsConsistent`, `PrimaryRationalForm`, `JordanForm`, `IsSimilar`.

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
