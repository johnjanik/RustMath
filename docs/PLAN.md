# RustMath Implementation Plan — full Sage + MAGMA parity

_The single phased plan. Grounded in [`SURVEY.md`](./SURVEY.md) (code-current, 2026-07-09). Supersedes and reconciles `MASTER_PORT_PLAN.md` (MAGMA/foundation-first waves), `SAGEMATH_RUST_ROADMAP.md` (Sage/PyO3 phases), `MVP_ANALYSIS_REPORT.md`, `SYMPY_ALTERNATIVE_ROADMAP.md`, and the status sprawl — all now in [`archive/`](./archive/)._

## Framing

RustMath is **one backend** that must satisfy **two front-ends**: SageMath (Python API shape, `sage.*` module semantics) and MAGMA (intrinsic names, handbook chapter semantics). Every crate therefore carries **two contracts** — a Sage module mapping and a MAGMA chapter mapping (both in the SURVEY matrix). The plan is **foundation-first**: fix the shared trait/numeric core and eliminate silent facades before fanning out to breadth.

**Guiding principle from the survey:** the danger is not what's obviously missing — it's the **category-3 facades** (real-looking APIs returning plausible constants) and the **lib-builds-but-tests-don't** crates. Honesty first, then depth, then breadth.

### Execution discipline (every crate task)
1. One owner per crate; new files preferred; only that crate's `lib.rs`/`Cargo.toml` touched on shared surface.
2. Consume `rustmath-core` traits + reuse base numeric crates — **no new private `num-*` wrappers**.
3. Zero `unsafe`. Keep the crate **build-green and test-green** (`cargo test -p <crate>`).
4. Cite the MAGMA chapter and Sage module in each new file header. Port the handbook/Sage doctests as tests.
5. Certifying output stays "UNRESOLVED vs decided"; a bounded search that finds nothing is never a decision. **Facades return `Err`/`unimplemented`, never a plausible constant.**
6. Isolated worktree → one PR per crate.

---

## Phase 0 — Green & honest (immediate; ~1–2 weeks)

Make the whole workspace build+test green and convert every silent facade into an explicit non-result. This is prerequisite to trusting anything downstream.

- **Build fixes.** `interfaces/src/test_long.rs` (3× `iterations as u32`); gate `databases` network/OpenSSL behind an optional feature so it builds without `libssl-dev`; fix test-mode build breaks in `modules` (388 errs — `BigInt: !Ring` dev-dep), `category` (16 errs), `schemes` (100 errs — `Rational` Zero/One, duplicate `tests` module, dimension shadowing). Target: `cargo test --workspace` compiles.
- **Live correctness bugs.** `finitefields::ExtensionField` `Div` returns `self` + never impls `Field` → implement inverse (ext-Euclid in `F_p[x]/(m)`) and the `Field` impl. `algebraic::AlgebraicReal` `sign()`/`cmp()` return `Equal` for irrationals → interval-refine or return a `Result` (no silent wrong order). `topology` 7 failing tests.
- **Facade audit → explicit `unimplemented`.** rings `function_field/`+`valuation/`, numerical LP/MIP/SDP backends, modular `HeckeAlgebra::eigenvalues`/`DirichletGroup`, schemes Chabauty/Heegner, `curves::find_singularities` (returns `[]` after building the ideal), manifolds cohomology, ellipticcurves `Sha`/descent. Each returns an honest error and is logged in the SURVEY as such.
- **Hygiene.** `cargo fix` + `clippy` pass (thousands of warnings); wire the 44 uncounted `jupyter` tests + 11 `automata` tests into the harness.

**Exit:** workspace builds + tests compile; no API returns a plausible-but-fake value; SURVEY "build/test health" section goes all-green.

## Phase 1 — Foundation: the shared core (Wave 0)

The trait vocabulary + arbitrary-precision numerics everything else consumes. (Largely built on the unmerged `magma-port/wave0-foundation` branch — land it.)

- **`rustmath-core` additive traits:** `ordering` (OrderedRing/Field), `valuation` (DiscreteValuation/Place), `analytic` (RealField/ComplexField/Ball), `morphism` (object-safe Hom/End layer — Ring is not dyn-safe), `nonassoc` (Lie/nonassociative), `coercion` (pushout over `parent.rs`). Expose as `pub mod` + selective path-qualified re-exports only — no root re-export (the trait names `RealField`/`ComplexField` deliberately collide with the existing structs `rustmath_reals::RealField`/`rustmath_complex::ComplexField`; policy documented in `rustmath-core/src/lib.rs`).
- **Arbitrary-precision reals/complex:** land the pure-Rust `BigFloat`/`BigComplex`/`Ball` — the native `Integer`-backed implementation (zero new deps, offline-buildable; the earlier astro-float/dashu option is superseded) — implementing `RealField`/`ComplexField`; keep `rug`/MPFR behind the same trait as the fast path. Resolve the "pure-Rust vs C-dep" story explicitly (reals currently ships both).
- **Canonicalize the duplicate types** (survey finding #2), one decision each, done centrally:
  - `Z/mZ`: one `Integers(m)` Parent (retire integers::ModularInteger / finitefields::IntegerMod / rings::quotient_ring overlap).
  - **p-adics:** unify `rustmath-padics` (skeleton) into `rings::padics` (the real one) — single home before extensions.
  - `Algebra` trait (core vs algebras), `CharacterTable` (groups vs combinatorics), elliptic curve (schemes vs ellipticcurves vs crypto).

**Exit:** core traits merged + re-exported; one arbitrary-precision real/complex behind a trait; the five duplicate-type collisions each have a single canonical home.

## Phase 2 — Foundation completion & core adoption

- **`num-*` → core normalization pass** (survey finding #1) for `ellipticcurves`, `modular`, `crypto::elliptic_curve`, `graphs` spectral, `geometry::toric` — replace `num-bigint`/`num-complex<f64>` with `Integer`/`Matrix`/`ComplexField`. Prerequisite for Phase 4.
- **Finite fields:** post-Phase-0 fix, add Cantor–Zassenhaus + irreducible-poly search + full Conway DB + field embeddings; **Galois rings GR(p^a,d)** (needs non-domain PIR marker) → unblocks MAGMA ch48 + coding over rings.
- **Power series:** un-comment `exp`/`log`/`integral`; Laurent/Puiseux with a Precision enum; lazy series.
- **Matrix/analysis:** exact eigen/charpoly off the f64 path; LLL/Gram–Schmidt over `RealField`; arbitrary-precision root-finding (Aberth/Durand–Kerner) on `BigComplex` — reused by number fields, Galois, elliptic curves.
- **Wire `category` into core** coercion (it's currently isolated) or formally scope it as documentation-only. — **DONE (wired):** `rustmath_category::core_bridge` lets the category `CoercionMap` graph drive `rustmath_core::coercion::{Coercible, Pushout}` (canonical pushout `Z, Q -> Q` verified against real `Integer`/`Rational` in `rustmath-category/tests/core_coercion_bridge.rs`), with type erasure through core's object-safe `morphism` layer. One-directional: category → core; core gained docs only, no dependency and no code change.

## Phase 3 — Discrete-algebra depth (parity for the strong tier)

The layer where RustMath is already good; make it Sage/MAGMA-complete.

- **General permutation groups (MAGMA 57–58, the keystone):** Schreier–Sims / BSGS stabilizer chain → membership, order, base/SGS, general orbits/blocks. Either native or a **real GAP FFI bridge** (today `libgap_*` are pure-Rust name-alikes). Unblocks exact character tables (cyclotomic, ch91), automorphism groups (ch67), soluble/polycyclic (ch63/72).
- **General Galois groups (MAGMA ch38 — the earlier high-priority ask):** on the BSGS backbone + the existing exact resolvents (`polynomials::resolvent`) + Frobenius sieve, build a general-degree `GaloisGroup(f)`: transitive-group lattice, p-adic/complex root labelling with a fixed ordering, Stauduhar descent with relative invariants, `GaloisProof` via absolute resolvents, subfield/tower reconstruction. New `rustmath-galois` crate over groups/polynomials/numberfields. (Degree-24 atlas + IGP24 path remain the specialized fast track.)
- **Fill the discrete gaps:** homology torsion via matrix SNF; modules over Dedekind domains + `Hom`/`End`; monoid Knuth–Bendix (shared FSA engine in `automata`, consumed by groups/monoids ch74/75/78); quiver path-algebras as `Ring` + representations; symmetric-functions Hall–Littlewood/Macdonald + Ring/Algebra trait tower; combinatorics skew-tableau (JeuDeTaquin) + matrix-RSK + plactic monoid.

## Phase 4 — Analytic & number-theoretic upper layers (the deep gaps)

Where the facades and skeletons concentrate (arithmetic geometry, function fields, class field theory, modular forms, CAS closure).

- **Number fields (34–39,44):** general S-unit/class-field theory; ray class **fields** (only numbers today); canonical `polredabs`; subfields (Klüners / van Hoeij); Artin representations. Retire the crate-root "toy" `NumberField` onto the real round2/classgroup machinery.
- **Function fields (rings, 41–43):** replace the string-typed `function_field/`+`valuation/` facades with typed generics; MacLane inductive valuations; places/divisors; **Riemann–Roch (Hess algorithm)**; class field theory for global function fields.
- **p-adics (47,51):** real unramified/Eisenstein extensions; **Montes/OM** local factorization + Newton-polygon templates + ramification certificates.
- **Arithmetic geometry (112–139):** `schemes` blow-ups + real affine ideals over Gröbner; `ellipticcurves` Tate's algorithm, 2-descent/Selmer, heights, L-functions, modular-symbol Hecke eigenvalues; hyperelliptic Jacobians (full Cantor) → new `rustmath-hyperelliptic`; `modular` Manin symbols + Hecke/Atkin–Lehner + Dirichlet characters + modular symbols. This is the largest single block.
- **CAS closure (Sage/SymPy side):** Risch integration; arbitrary-precision numeric backend for `symbolic` (off f64); restore `calculus` laplace/minpoly/pochhammer; full symbolic solve; multivariate limits.
- **Optimization (159):** real simplex + interior-point in a new `rustmath-optimization` crate over `OrderedField` (numerical LP is a stub and off-limits historically).

## Phase 5 — Breadth fill & new crates

Coverage the front-ends expect but no route yet needs:
- Coding: generic `LinearCode<F: Field>` over GF(q); weight enumerators/bounds; codes over rings (Z4), additive/quantum/AG/LDPC (152–157).
- Graphs: first-class Networks/MaxFlow (151); graph products; exact spectral over `Matrix<Integer>`; wire AutomorphismGroup to `PermutationGroup`.
- Lie/reps: reflection groups (99), groups of Lie type (103, new crate), reps of Lie algebras (104).
- Geometry: exact rational Polyhedron (H/V-rep, vertex enumeration); toric Cox rings; polar spaces + sesquilinear forms (29); finite planes/incidence (141/142); resolution graphs (115).
- New greenfield crates: `nearfields` (22), `hyperelliptic` (125), `genusone` (124), `hgm` (126), `sheaves` (113 free resolutions/Ext), `groupsoflietype` (103).
- Number theory: Dirichlet characters + L-functions; arithmetic dynamics (heights, preperiodic points); stats distributions (Gamma/Beta/t/F/χ²) + nonparametric/multivariate.

## Phase 6 — Front-end parity & interop

- **Sage side:** extend `rustmath-py` PyO3 bindings from the current narrow slice to all stable crates, matching `sage.*` constructor/method names; a `test_sagemath_compat.py` suite.
- **MAGMA side:** `lava` DSL intrinsic dispatch (`GaloisGroup(f)`, `NumberField(f)`, …) → backend calls; port handbook worked-examples as golden tests; the `parse` command (planned v0.2).
- **Cross-checks & certificates:** OSCAR/Hecke/GAP oracles for independent verification; certificate + replay layer for credible results.

---

## Sequencing & dependencies

```
Phase 0 (green+honest) ── gate ──▶ Phase 1 (core traits + bignum + de-dup)
                                        │
                        Phase 2 (core adoption, finite fields, series, arb root-find)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
       Phase 3 (groups/Galois/          Phase 4 (number fields,    │
       discrete depth)                  function fields, p-adics,  │
              │                         arith-geometry, CAS, LP)   │
              └─────────────┬───────────────────────┘             │
                       Phase 5 (breadth fill, new crates)          │
                                        │                          │
                       Phase 6 (PyO3 + lava DSL + cross-checks) ◀──┘
```

Phases 3 and 4 can run in parallel once Phase 2 lands (disjoint crates). Within each phase, one worker per crate, one PR per crate.

## Priority tiers (if resources are limited)

- **P0 (do first):** Phase 0 entirely — a green, honest workspace is worth more than any new feature.
- **P1 (highest leverage):** Phase 1 foundation + the two keystones in Phase 3 (BSGS groups, general `GaloisGroup`) — they unblock the most downstream chapters and were the original high-priority ask.
- **P2:** Phase 4 arithmetic-geometry + number-field/function-field depth (the largest coverage gap vs both front-ends).
- **P3:** Phase 5 breadth + Phase 6 interop.
