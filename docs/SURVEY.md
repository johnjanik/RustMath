# RustMath Survey — the single source of truth

_Code-grounded survey of all 66 crates as a shared computational backend for **both** SageMath and MAGMA. Generated 2026-07-09 from the current tree (branch `belyi/u0-unblock-and-true-root-detector`, which includes all merged work up to PR #567). Supersedes the 70 scattered root-level status/plan/analysis docs now in `docs/archive/`._

## How to read this

RustMath is the **compute engine**; SageMath and MAGMA are front-ends. This survey is **backend-centric**: each crate is placed next to the Sage module(s) and MAGMA chapter(s) it must satisfy, with one honest coverage verdict.

**Verdict legend**
- **Complete** — full parity for its scope; no known stubs.
- **Substantial** — the core is real, tested, usable; named advanced features missing.
- **Partial** — real pieces coexist with significant stubs/placeholders.
- **Stub / Skeleton** — types exist, algorithms mostly absent or return placeholders.
- **Non-math-infra** — trait vocabulary / IO / viz / bindings (not a math domain).

Coverage is judged against **current code**, not the (stale) legacy docs. Sizes are `src/**` LOC and `#[test]` counts.

## Build health (2026-07-09)

- **64 / 66 crates build clean** (`cargo build --workspace`).
- **`rustmath-databases`** — fails *in this environment only*: native `libssl-dev`/OpenSSL not installed (`openssl-sys` can't find `openssl.pc`). Not a code defect; needs `apt install libssl-dev` or an `OPENSSL_DIR`. Consider making the DB/network features optional so the crate builds without system OpenSSL.
- **`rustmath-interfaces`** — 3 real but trivial `error[E0308]` in `src/test_long.rs` (`elapsed / iterations` where `iterations: usize` but `Duration: Div<u32>`). One-line fixes (`iterations as u32`). Everything else in the crate compiles.
- Warnings are plentiful (unused vars, non-snake-case) but non-blocking; a `cargo fix` + clippy pass is a cheap hygiene win.

## The two legacy completion figures were inconsistent

Prior docs claimed **35%** (SYMPY_ALTERNATIVE), **68% / 364-of-539** (SAGEMATH_RUST_ROADMAP), and "~35%" (CLAUDE.md). None post-dated recent merges. This survey replaces them with per-crate verdicts; see the roll-up at the end.

## Scale

~**600K** src LOC and ~**11.9K** declared `#[test]` markers across 66 crates (test *markers*, not confirmed-passing — see build health). Ten largest by LOC: combinatorics (68K), rings (62K), algebras (37K), graphs (32K), symbolic (27K), liealgebras (21K), groups (21K), geometry (18K), curves (18K), manifolds (17K). Per-crate LOC/tests are shown inline in the Appendix (e.g. `[LOC, tests]`).
## Master coverage matrix (all 66 crates × Sage × MAGMA)

Verdict key: ✅ Complete · 🟩 Substantial · 🟨 Partial · 🟥 Stub/Skeleton · ⬜ Infra/front-end.
"MAGMA" = handbook chapter numbers (see `reference/magma_coverage.md`). Full per-crate detail in the Appendix below.

### Foundation & exact arithmetic
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| core | ⬜ | structure, categories | 17 | flat trait tower; no coercion; Wave-0 ordering/valuation/analytic traits unmerged |
| integers | ✅ | rings.integer, arith | 18 | no GMP/FLINT backend (perf ceiling) |
| rationals | 🟩 | rings.rational, continued_fraction | 20 | no CF-of-arbitrary-real; no localization |
| reals | 🟩 | rings.real_mpfr/real_double | 25 | intervals on f64 not MPFR; rug is C-dep vs "pure-Rust" plan; no arb RealBall |
| complex | 🟩 | rings.complex_mpfr/complex_arb | 25 | no in-crate QQbar; ball error-prop incomplete; no root-finding |
| numbertheory | 🟨 | arith.misc, quadratic_forms | 18 | thin; brute-force theta; no Dirichlet chars/L-funcs |
| constants | 🟨 | symbolic.constants, databases | — | digit tables ≤1000; tiny hardcoded DBs |
| finitefields | 🟨 | rings.finite_rings | 19,21,48 | **ExtensionField Div returns self (broken); never impls Field**; 31-entry Conway table |
| padics | 🟥 | rings.padics | 47,51 | skeleton; no extensions; **collides with rings::padics** |

### Rings, series, forms
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| rings | 🟨 | rings.* (quotient/fraction/localization/padics/valuation/function_field/series) | 17,36,40-43,45,47-50 | function_field/ (31% LOC) & valuation/ are **string facades**; number_field only quadratic; padic ext fake norm/trace/Galois; Riemann-Roch placeholder |
| powerseries | 🟨 | rings.power_series_ring | 49,50 | exp/log/integral commented out; no Laurent/Puiseux/lazy here |
| quadraticforms | 🟨 | quadratic_forms.* | 32,33,86 | exact binary/Hilbert/conic solid; mass formula/densities hardcoded; no class group/composition |

### Polynomials & number fields
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| polynomials | 🟩 | rings.polynomial.* | 23,24,46,105,106 | F4 is a stub; no van Hoeij; no multivariate factorization; **resolvents only, no Galois-group engine** |
| numberfields | 🟩 | rings.number_field.* | 34,35,37,38,39 | crate-root NumberField is a toy layer; polred≠polredabs; **no global GaloisGroup/Stauduhar engine**; no ray class *field* |
| algebraic | 🟨 | rings.qqbar | — | **sign()/cmp() return Equal for irrationals (correctness landmine)**; minpoly/conjugates stubbed |

### Linear algebra, modules, homological
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| matrix | 🟩 | matrix.* | 26,27,30,83 | eigen/Jordan only power-iteration; no BKZ; no iterative solvers |
| modules | 🟨 | modules.*, tensor.modules | 53,54,55,30,31 | **test suite fails to build (388 errs, BigInt≠Ring)**; ~30 one-line stub files; Submodule.rank=gen count |
| homology | 🟨 | homology.* | 56,140 | torsion hardcoded `[]` (SNF exists in matrix but unused); no persistent homology |
| category | 🟨 | categories.* | — | dyn-compat FIXED; but tests don't build; ~1 external user; not wired into core tower |

### Groups & discrete algebra
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| groups | 🟩 | groups.* | 57-73,90,91 | **no Schreier-Sims/BSGS** (degree-23/24 atlases only); libgap_* are pure-Rust fakes; f64 character tables |
| monoids | 🟨 | monoids.* | 77,78 | AutomaticSemigroup::mul is a stub; no Knuth-Bendix; Hecke/Trace skeletal |
| automata | 🟨 | combinat.words.automatic | 75 | solid DFA/NFA but **zero integration with groups/monoids** (no automatic-groups) |
| quivers | 🟨 | quivers.* | (90) | no path-algebra Ring; no quiver representations/AR theory |

### Algebras & Lie theory
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| algebras | 🟩 | algebras.* | 79,80,81,84,86,88 | fusion F-matrix/Hall/cluster simplified; quaternion orders p-adic stubbed; exotics behind `full` feature |
| liealgebras | 🟩 | algebras.lie_algebras, combinat.root_system | 94,96,98,100,101,104 | WeylGroup::longest_element=identity; classical gl/sl/so/sp brackets return zero; no reflection groups (99) |
| lieconformal | 🟩 | algebras.lie_conformal_algebras | — | several "simplified"; no vertex-algebra envelope; no LCA cohomology |
| crystals | 🟨 | combinat.crystals.* | — | KR crystals + rigged configs placeholder |
| quantumgroups | 🟨 | algebras.quantum_groups | 102 | q-scalars only; U_q(g) objects live in algebras; no R-matrices |

### Combinatorics
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| combinatorics | 🟩 | combinat.* (88 modules) | 92,144,145,147,148 | skew Rectify/JeuDeTaquin missing; matrix-RSK incomplete; no plactic monoid; brute-force canonical forms (no nauty) |
| symmetricfunctions | 🟩 | combinat.sf.*, ncsf_qsym | 146 | no Hall-Littlewood/Macdonald; no Ring/Algebra trait tower; incomplete transition matrices |
| sets | 🟩 | sets.* | — | limited lazy/infinite sets; ConditionSet no symbolic predicate |
| trees | 🟩 | combinat.binary_tree (partial) | — | CS data-structure lib; combinatorial trees duplicated in combinatorics |

### Symbolic / CAS & analysis
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| symbolic | 🟩 | symbolic.* | (111) | string parser + heuristic integration + limits EXIST; no Risch; no arbitrary-precision numerics |
| calculus | 🟩 | calculus.* | (111) | laplace/minpoly/pochhammer/product modules commented out; overlaps symbolic |
| functions | 🟩 | functions.{trig,log,hyperbolic,other} | — | thin wrapper; SymbolicFunction trait dead; no complex/arbitrary precision |
| special-functions | 🟨 | functions.{gamma,special,error,transcendental} | — | f64-only, no complex args; no Bessel I/K, incomplete gamma; duplicated in symbolic |
| logic | 🟩 | logic.* | — | no formula string parser; DPLL only (no CDCL); propositional only (no FOL) |
| numerical | 🟨 | numerical.optimize/mip | 159 | **entire LP/MIP/SDP layer returns zeros (stub)**; no arbitrary precision |
| stats | 🟨 | stats.*, probability.* | — | missing Gamma/Beta/t/F/χ² distributions; no ANOVA/nonparametric; no multivariate |

### Geometry & topology
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| geometry | 🟩 | geometry.* | 118,143 | core Polyhedron f64-only (no exact H/V-rep, no vertex enum); toric bypasses Integer/Matrix; no 141/142 |
| topology | 🟨 | topology.*, knots.* | — | **NO homology computation** (no boundary/SNF/Betti); knot invariants heuristic; 7/137 tests failing |
| manifolds | 🟩 | manifolds.* | — | builds clean (188-err doc claim stale); Phase-5 placeholder-heavy; Geodesic::at ignores t; cohomology hardcoded |
| dynamics | 🟨 | dynamics.complex_dynamics | — | **no arithmetic_dynamics** (heights/(pre)periodic points); f64 only |

### Arithmetic geometry (weakest tier)
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| schemes | 🟨 | schemes.* | 112,113,120-122 | **fails to compile in test mode (100 errs)**; no blow-ups; affine schemes build no ring; Chabauty/Heegner placeholders; hardcoded Cremona tables |
| affineschemes | 🟥 | schemes.generic.spec, affine | 112 | PhantomData wrappers; ideal primality/radical/dimension all stubs; no polynomial ring |
| curves | 🟨 | schemes.curves, hyperelliptic | 114,125 | **find_singularities builds ideal then returns [] (silently wrong)**; Riemann-Roch only deg≥2g-1; belyi/ is separate active subsystem |
| ellipticcurves | 🟨 | schemes.elliptic_curves.* | 120,122,127 | descent/L-func/BSD toy (Sha=1.0 hardcoded); conductor ad-hoc not Tate; ignores core (num-* + own f64 complex) |
| modular | 🟥 | modular.* | 128,132-139 | real pockets (arithgroup/cusps/Hecke q-exp/dims) atop ~50 near-empty files; eigenvalues=0; DirichletGroup trivial-only; 15 empty exotic modules |

### Applied (coding, crypto, graphs)
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| graphs | 🟩 | graphs.* | 149,150 | adjacency/Laplacian/eigen use f64 not exact Matrix; AutomorphismGroup not wired to groups; no Networks (151); no products |
| coding | 🟨 | coding.* | 152 | prime-field only (no GF(q>p)); no weight enumerator/bounds; no ring/AG/LDPC/quantum codes |
| crypto | 🟩 | crypto.* | 158 | no LFSR/Berlekamp-Massey (158); no standard curves/RSA padding; ECC hardcodes num-bigint; duplicates curves EC |

### Infrastructure, IO, viz, front-end
| Crate | V | Sage | MAGMA | Headline gap |
|---|:--:|---|---|---|
| matrix→(see above) | | | | |
| plot | 🟩 | plot.* | — | contour fill; multi-graphics render; show() no backend |
| plot3d | 🟩 | plot.plot3d.* | — | list_plot3d incomplete; no interactive rotation (known bug) |
| plot-core | ✅ | plot.primitive | — | thin foundation, complete |
| jupyter | 🟩 | repl.* | — | bespoke calculator lang not Sage/Python syntax; 44 tests uncounted |
| typesetting | ✅ | misc.latex, typeset.* | — | minor layout gaps |
| colors | ✅ | plot.colors | — | no CMYK/colorblind palettes |
| databases | 🟨 | databases.* | — | **build fails here (needs libssl-dev)**; LMFDB hardcoded stub |
| interfaces | 🟩 | interfaces.gap | — | **3 trivial compile errors in test_long.rs**; GAP real, PARI/Singular/FLINT absent |
| features | ✅ | — | — | complete detection layer (gated crates don't exist yet) |
| misc | 🟨 | misc.* | — | mrange/temporary_file solid; ~9 six-line stub modules |
| benchmarks | ⬜ | — | — | bins only; no real SymPy comparison |
| igp24 | ⬜ | — | — | thin JSON CLI over deg-24 Galois engine |
| lava | ⬜ | — | front-end | Magma-syntax format/highlight/test only; parse planned; no compute backend wired |
| py | 🟨 | (PyO3) | — | bindings for narrow slice; most crates unexposed; 0 tests |

---

## Cross-cutting structural findings (recur across many crates)

1. **`num-*` vs `rustmath-core` non-adoption.** ellipticcurves, modular, crypto's ECC, graphs' spectral, toric geometry hardcode `num-bigint`/`num-complex<f64>` instead of the `Integer`/`Matrix`/`ComplexField` tower. This is the #1 integration debt — a normalization pass is prerequisite for several Wave-3 crates.
2. **Duplicate/colliding types.** `Z/mZ` (integers::ModularInteger vs finitefields::IntegerMod vs rings::quotient_ring); p-adics (rustmath-padics vs rings::padics — the real one); CharacterTable (groups vs combinatorics); Algebra trait (core vs algebras); EC (schemes vs ellipticcurves vs crypto). Each needs one canonical home.
3. **Facade modules** — real-looking APIs that return constants: rings `function_field/`+`valuation/`, numerical LP/MIP/SDP backends, modular `HeckeAlgebra::eigenvalues`, schemes Chabauty/Heegner, curves `find_singularities`, manifolds cohomology. These are the most dangerous (silently wrong, not obviously missing). **Every one should log/return an explicit "unimplemented" rather than a plausible constant.**
4. **Lib-builds-but-tests-don't.** modules (388 errs), category (16 errs), schemes (100 errs, test mode) — their `#[test]` counts are declared, not passing. topology has 7 runtime failures. So workspace test-green is *not* established; only 64/66 *libs* build.
5. **Degree-specialized vs general.** The Galois/group stack (transitive23/24, pgalois, igp24) is purpose-built for the degree-24 Belyi/IGP24 research track, not general Sage/GAP/MAGMA machinery. A general `GaloisGroup(f)` needs a general BSGS backbone (absent) or the GAP bridge.
6. **f64 where exact is expected.** algebraic sign/cmp, geometry Polyhedron, graphs spectrum, topology, all special functions. Blocks exactness claims.
7. **ExtensionField division is broken** (finitefields) — a live correctness bug, not just a gap; blocks GF(q) across coding, and anything needing GF(p^n) field arithmetic.

## Roll-up by verdict (66 crates)

- **✅ Complete:** ~6 — integers; infra: plot-core, typesetting, colors, features (+ core as trait vocabulary).
- **🟩 Substantial:** ~24 — rationals, reals, complex, polynomials, numberfields, matrix, groups, algebras, liealgebras, lieconformal, combinatorics, symmetricfunctions, sets, trees, symbolic, calculus, functions, logic, geometry, manifolds, graphs, crypto, plot, plot3d, jupyter, interfaces.
- **🟨 Partial:** ~28 — numbertheory, constants, rings, finitefields, quadraticforms, powerseries, algebraic, numerical, modules, homology, category, monoids, automata, quivers, crystals, quantumgroups, special-functions, stats, topology, dynamics, schemes, curves, ellipticcurves, coding, databases, misc, py.
- **🟥 Stub/Skeleton:** ~3 — padics, affineschemes, modular.
- **⬜ Tooling/front-end:** benchmarks, igp24, lava.

**Honest headline:** RustMath is **broad and, in the foundation + discrete-algebra + CAS layers, genuinely deep** (integers, matrix, polynomials, numberfields, groups, algebras, liealgebras, combinatorics, symbolic all real and usable). The **analytic/number-theoretic upper layers** (arithmetic geometry, modular forms, function fields, class field theory, general Galois, arbitrary-precision analysis) are where the facades and skeletons concentrate. The single biggest *quality* risk is category-3 facades (plausible constants); the biggest *coverage* gaps are arithmetic geometry (ch 112–139) and general group theory (Schreier–Sims, ch 57–72).

---

# Appendix — per-crate detail (all 12 domain clusters)
## Cluster A — Exact-arithmetic foundation

### rustmath-core [849, 7]
- Verdict: Non-math-infra — foundational traits (Ring/Field/Group/Module), not a domain.
- Sage: sage.structure, sage.categories.rings/fields. MAGMA: 17.
- Implemented: Magma/Semigroup/Monoid/Group/Ring/Field/EuclideanDomain/Module traits + default gcd/lcm/xgcd; error types; Parent/UniqueRepresentation scaffolding.
- Gaps: no Category framework beyond flat hierarchy; no coercion system; only i32/i64 concrete; no ordered-ring/valuation traits (NOTE: Wave-0 branch adds these, unmerged).

### rustmath-integers [6944, 150]
- Verdict: Complete — arbitrary-precision integers on num-bigint, rich number theory, no stubs.
- Sage: sage.rings.integer, sage.arith.misc, integer_mod. MAGMA: 18.
- Implemented: gcd/xgcd/sqrt/nth_root/divisors/sigma/phi/moebius/Jacobi-Legendre/valuation, Miller-Rabin, Pollard rho/p-1, ECM, quadratic sieve, CRT, ModularInteger.
- Gaps: no GMP/FLINT backend (perf ceiling); no SIQS; no p-adic valuation-ring integration.

### rustmath-rationals [2345, 46]
- Verdict: Substantial — Rational, continued fractions (periodic/quadratic-irrational), Bernoulli/harmonic.
- Sage: sage.rings.rational, continued_fraction. MAGMA: 20.
- Gaps: no CF of arbitrary real; no localization type; no Farey/mediant.

### rustmath-reals [1756, 33]
- Verdict: Substantial — f64 Real + MPFR-backed RealMPFR (rug/GMP) with transcendentals + intervals.
- Sage: sage.rings.real_mpfr, real_double, real_interval_field. MAGMA: 25.
- Gaps: intervals on f64 not MPFR (not arb-grade); no RealBall(arb); rug/MPFR is C-dep (contradicts "pure-Rust" plan — NOTE Wave-0 BigFloat is the pure-Rust track, unmerged); no arbitrary-precision root-finding.

### rustmath-complex [4296, 107]
- Verdict: Substantial — f64 Complex, MPC ComplexMPFR, ComplexBall, complex interval field.
- Sage: sage.rings.complex_mpfr, complex_arb, complex_interval_field. MAGMA: 25.
- Gaps: no QQbar in-crate; ComplexBall lacks rigorous arb error prop; no complex root-finding exposed.

### rustmath-numbertheory [1427, 27]
- Verdict: Partial — thin: re-exports integer primes + Bernoulli mod-p + basic quadratic forms (brute force).
- Sage: sage.arith.misc, quadratic_forms, bernoulli_mod_p. MAGMA: 18 (primes), forms.
- Gaps: no reduction theory/class number/genus; theta/representation are bounded brute search; no Dirichlet characters/L-functions.

### rustmath-constants [1523, 29]
- Verdict: Partial — static digit tables (≤1000 digits) + small sequence DBs; not computational.
- Sage: sage.symbolic.constants, databases.oeis. MAGMA: none.
- Gaps: capped at 1000 digits (no MPFR compute); tiny hardcoded DBs not real OEIS/Cunningham; not runtime-extensible.

### rustmath-padics [681, 7]
- Verdict: Skeleton — minimal fixed-precision Z_p/Q_p + Hensel lifting only.
- Sage: sage.rings.padics.*. MAGMA: 47, 51.
- Gaps: no extensions (unramified/Eisenstein); no p-adic power series/L-functions; no parent/factory; no p-adic poly factorization; abs() f64. NOTE: real p-adic machinery lives in rustmath-rings/src/padics (collision — must unify).

**Cluster A total:** 8 crates — 4 substantial+, 3 partial, 1 skeleton.

## Cluster F — Algebras & Lie theory

### rustmath-algebras [36995, 751]
- Verdict: Substantial — huge breadth (70+ algebra types); core solid, exotic simplified/placeholder.
- Sage: sage.algebras.{clifford,free_algebra,group_algebra,quatalg,hecke_algebras,fusion_rings,cluster_algebra,quantum_groups,steenrod,jordan,yangian}. MAGMA: 79,80,81,84,86,88.
- Implemented: quaternion, Clifford/exterior, free, group, finite-dim, octonion, Iwahori-Hecke, sym-group, Weyl, tensor, Steenrod algebras w/ real arithmetic.
- Gaps: F-matrix/fusion pentagon-hexagon solving placeholder; Hall polynomials/splitting-algebra/cluster-sign simplified; quaternion order/ideal p-adic stubbed; exotic algebras behind non-default `full` feature.

### rustmath-liealgebras [21484, 368]
- Verdict: Substantial — deep abstract machinery; Weyl longest-element and matrix-realization brackets stubbed.
- Sage: sage.algebras.lie_algebras.*, sage.combinat.root_system.*. MAGMA: 94,96,98,100,101,104.
- Implemented: structure-coeff algebras w/ real bracket, Cartan types/matrices/root systems/Dynkin, PBW, Chevalley, affine/exceptional, Verma, BGG.
- Gaps: WeylGroup::longest_element returns identity (placeholder); classical gl/sl/so/sp matrix realizations have zero-returning bracket/Killing; BGG differentials + center-of-UEA simplified; no reflection groups (MAGMA 99) beyond Weyl.

### rustmath-lieconformal [6918, 149]
- Verdict: Substantial — wide catalog of named LCAs with real λ-bracket.
- Sage: sage.algebras.lie_conformal_algebras. MAGMA: none.
- Gaps: several "simplified" (Virasoro OPE, affine LCA, dual maps empty); no vertex-algebra envelope; no LCA cohomology/rep theory.

### rustmath-crystals [5485, 76]
- Verdict: Partial — generic Crystal/tableau/tensor core solid; KR crystals + rigged configs placeholder.
- Sage: sage.combinat.crystals.*, rigged_configurations. MAGMA: none.
- Gaps: KR combinatorial R-matrix placeholder; rigged-config bijection "simplified type A"; Nakajima monomials simplified; virtual-crystal machinery thin.

### rustmath-quantumgroups [1741, 41]
- Verdict: Partial — thin: q-analog scalars only; actual U_q(g) objects live in rustmath-algebras.
- Sage: sage.algebras.quantum_groups. MAGMA: 102 (partial, q-numbers only).
- Gaps: no U_q(g) generators/relations/PBW in-crate; no R-matrices; no crystal-limit q→0; consolidation w/ algebras needed.

**Cluster F total:** 5 crates — 3 substantial+, 2 partial, 0 stub.

## Cluster H — Combinatorics, symmetric functions, sets, trees, stats

### rustmath-combinatorics [68261, 1449]
- Verdict: Substantial — huge (88 modules); Murnaghan-Nakayama buggy tests, skew-tableau ops incomplete.
- Sage: sage.combinat.{partition,tableau,permutation,posets,designs,species,words,symmetric_group_representations,subset,set_partition,integer_vector,dyck_word,growth}. MAGMA: 92,144,145,147,148.
- Implemented: partitions/tableaux/permutations/posets/designs/species/words/ASMs/RSK/Coxeter/Tamari/plane-partitions, hook-length, MN, Latin squares, gen functions.
- Gaps: full skew Rectify/JeuDeTaquin missing; matrix/double-word RSK + inverse incomplete; no plactic monoid; BlockDesign lacks general IncidenceStructure; automorphism/canonical-form brute-force (no nauty).

### rustmath-symmetricfunctions [3752, 82]
- Verdict: Substantial — 5 classical bases + NCSF/QSym/FQSym Hopf algebras; not core-trait integrated.
- Sage: sage.combinat.sf.*, ncsf_qsym.*, fqsym. MAGMA: 146.
- Gaps: no Hall-Littlewood/Macdonald in-crate; not all transition matrices; no change-of-basis coercion; SymFun lacks Ring/Algebra trait tower; no Frobenius hom.

### rustmath-sets [4474, 115]
- Verdict: Substantial — trait-based Set hierarchy + union-find + ranges/families/real intervals.
- Sage: sage.sets.{set,finite_enumerated_set,disjoint_set,cartesian_product,condition_set,family,integer_range,primes,real_set,finite_set_maps}. MAGMA: none.
- Gaps: no disjoint-union combinator; limited lazy/infinite sets; ConditionSet no symbolic predicate; no iso testing.

### rustmath-trees [3523, 78]
- Verdict: Substantial (non-math-infra) — generic CS data-structure lib, not combinatorial-tree theory.
- Sage: partial overlap sage.combinat.binary_tree/ordered_tree (real ones in combinatorics). MAGMA: none.
- Gaps: no B/red-black/splay trees; combinatorial tree enumeration duplicated in combinatorics; no persistent variants.

### rustmath-stats [2388, 63]
- Verdict: Partial — basic descriptive stats + 5 distributions/tests; narrow.
- Sage: sage.stats.basic_stats, probability.probability_distribution. MAGMA: none.
- Gaps: missing Gamma/Beta/t/F/Chi-sq distributions; no ANOVA/non-parametric; no multivariate (covariance/PCA); no MCMC; no discrete Gaussian sampler.

**Cluster H total:** 5 crates — 4 substantial+, 1 partial, 0 stub.
## Cluster B — Rings, finite fields, forms, series

### rustmath-rings [61581, 1589]
- Verdict: Partial — real MPFR/series/ring-extension core, but function_field (31% of LOC) & valuation are string-typed facades.
- Sage: sage.rings.{ring,quotient_ring,fraction_field,localization,padics,valuation,number_field,function_field,laurent_series_ring,power_series_ring,puiseux_series_ring,lazy_series,qqbar,universal_cyclotomic_field,real_arb,asymptotic,invariants,semirings}. MAGMA: 17,36,40,41,42,45,47,48,49,50 (+34-37 number fields).
- Implemented: MPFR/ARB reals, capped-relative p-adic arithmetic, quotient/fraction/ring-extension rings, tropical semirings, asymptotic rings, classical invariant theory, exact quadratic number fields, Laurent/lazy series.
- Gaps: function_field/ subtree (places, valuations, ideals, Jacobian/Cantor) is string-typed formatting NOT real algebra; valuation/ MacLane (Gauss/inductive/limit) returns trivial constant regardless of input; number_field/ only exact for quadratic (general degree/S-unit/class/unit unimplemented); p-adic extensions FAKE norm/trace/Galois + unramified-via-Conway for deg>2; Riemann-Roch (divisor/differential) dimension/basis/canonical are placeholders.

### rustmath-finitefields [1818, 32]
- Verdict: Partial — PrimeField solid exact Field; ExtensionField division broken placeholder.
- Sage: sage.rings.finite_rings.{finite_field_prime_modn,integer_mod,integer_mod_ring,conway_polynomials}. MAGMA: 19,21,48.
- Implemented: PrimeField GF(p) full Field (inverse via xgcd, discrete_log, Legendre, mult order); IntegerMod w/ Lucas; ExtensionField add/sub/mul/Frobenius/norm/trace correct via poly-mod-irreducible.
- Gaps: Div for ExtensionField returns self (NO inverse) and ExtensionField never impls Field trait; Conway table ~31-entry hardcoded HashMap vs Sage full DB; no field iso/embedding; no irreducibility testing/generation; no optimized GF(2^n).

### rustmath-quadraticforms [2928, 48]
- Verdict: Partial — exact binary-form/Hilbert-symbol/conic core; genus mass & local densities hardcoded approximations.
- Sage: sage.quadratic_forms.{binary_qf,genus,quadratic_form}, arith.hilbert_symbol. MAGMA: 32,33.
- Implemented: exact binary QF reduction/discriminant/primitivity, exact theta series (verified vs r(n)), exact Hilbert symbols + conic/quaternion Hasse-Minkowski w/ ternary diagonalization (from dessin_engine, solid).
- Gaps: Smith-Minkowski-Siegel mass + local densities f64/hardcoded not exact; class_number_estimate heuristic not Dirichlet; no class group/form composition; Satake/L-coeffs float approx; no quaternary/general n-ary lattice theory.

### rustmath-powerseries [871, 13]
- Verdict: Partial — solid truncated arithmetic + Newton inverse; exp/log/integral commented-out dead code.
- Sage: sage.rings.power_series_ring/power_series_poly, combinat recognizable_series. MAGMA: 49,50.
- Implemented: truncated univariate PowerSeries<R:Ring> (shift/truncate/compose/derivative/±*neg), Newton inverse over Field, WeightedAutomaton→recognizable series via DP.
- Gaps: exp()/log()/integral() only inside /* */ comments (not compiled — unavailable); no multivariate here (in rings); no Laurent/Puiseux here; no lazy series; compose not precision-truncated.

**Cluster B total:** 4 crates — 0 substantial+, 4 partial, 0 stub.
## Cluster C — Polynomials, number fields, algebraic, numerical

### rustmath-polynomials [12127, 227]
- Verdict: Substantial — Zassenhaus/Cantor-Zassenhaus factorization, Buchberger Groebner, resolvents, p-adic tools; F4 stub.
- Sage: sage.rings.polynomial.*. MAGMA: 23,24,46,105,106.
- Implemented: exact ℤ[x] factorization (CZ→Hensel→recombination), CRT resultant/discriminant, Newton polygons, p-adic factorization (Ore), Sturm, Buchberger, exact Lagrange/pair-sum/k-subset resolvents, bivariate resultants.
- Gaps: F4 stub (delegates to Buchberger); no van Hoeij/LLL recombination (exponential worst case); no multivariate factorization; stale doc on factor_over_integers.
- Galois: builds exact resolvents + orbit signatures ONLY; group-matching delegated to rustmath-groups (deg 4-5 + deg-24 dataset). No general Stauduhar/subfield-tower.

### rustmath-numberfields [7425, 105]
- Verdict: Substantial — real Round-2 maximal order, class group/units (validated vs PARI), Dedekind-Kummer/Montes ideals; crate-root NumberField is a "toy".
- Sage: sage.rings.number_field.*. MAGMA: 34,35,37,38(partial),39(partial).
- Implemented: Round-2 max order + field disc; Dedekind-Kummer prime ideal factorization + Montes fallback; class group (LLL principality + index calculus); unit rank/regulator/roots-of-unity + ray class numbers; S-units/S-class group; trace form/different/codifferent; higher ramification filtration + Hasse-Arf; local decomposition, unramified extensions, Panayi root-finding/local-Galois; polred (not polredabs); algebraic-number recognition.
- Gaps: crate-root NumberField/Element is a separate toy layer (inverse/class_number/signature/galois_closure NotImplemented or heuristic) not wired to real machinery; polred not canonical polredabs; wild ramification needs single-wild-prime assumption; no ray class FIELD construction (only number); no global Galois engine.
- Galois: NO global GaloisGroup/Stauduhar. Local only: panayi::is_eisenstein_galois (any deg but totally-ramified Eisenstein only); ramification filtrations GIVEN group known; groups::pgalois gives exact unramified local Galois but only deg-4/5 candidate set at ramified primes. No general ramified namer, no global-to-local lift.

### rustmath-algebraic [1779, 32]
- Verdict: Partial — rational/quadratic construction+arithmetic work; comparison/sign/minpoly/conjugates are TODO stubs.
- Sage: sage.rings.qqbar. MAGMA: none.
- Implemented: AlgebraicNumber/AlgebraicReal descriptor-tree, sqrt/nth_root w/ isolating intervals, arithmetic ops, complex embeddings.
- Gaps: sign()/cmp() silently return Equal for irrationals (CORRECTNESS LANDMINE — no interval refinement); minimal_polynomial() returns placeholder x; galois_conjugates() returns [alpha]; no real QQbar exact-comparison guarantee.

### rustmath-numerical [3536, 64]
- Verdict: Partial — root-finding/optimization/integration/FFT/homotopy real; entire LP/MIP/SDP layer is a stub.
- Sage: sage.numerical.optimize, sage.numerical.mip (stub). MAGMA: 159 (stub).
- Implemented: bisection/Newton/secant/Brent, Gauss-Newton true-root detector, gradient descent/Nelder-Mead/golden-section, Simpson/Romberg/Gauss-Legendre quadrature, radix-2 FFT, parameter-homotopy to external HomotopyContinuation.jl + LLL exactify.
- Gaps: simplex() returns all-zeros; GenericBackend/GLPK/CVXOPT/SDP all return {0.0} regardless of input — NO real LP/MIP/SDP; no arbitrary-precision; FFT no mixed-radix/Bluestein.

**Cluster C total:** 4 crates — 2 substantial+, 2 partial, 0 stub.

## Cluster K — Graphs, coding, crypto

### rustmath-graphs [31988, 605]
- Verdict: Substantial — broad real algorithms; many outputs untyped/f64.
- Sage: sage.graphs.{graph,digraph,generic_graph,generators,connectivity,planarity,spectrum,automorphism_group,strongly_regular_db,tutte_polynomial}. MAGMA: 149,150(partial).
- Implemented: Graph/DiGraph/MultiGraph/WeightedGraph, traversals, connectivity/blocks/bridges, planarity, spanning trees, Tutte polynomial, SRG database, Cayley graphs, generators, backends.
- Gaps: no first-class Network/MaxFlow-MinCut type (151); Adjacency/Laplacian/eigenvalues use Vec<Vec<f64>> not exact Matrix<Integer>; AutomorphismGroup ad-hoc not wired to groups::PermutationGroup; ChromaticPolynomial not over Integer; no SPQR/triconnectivity; no graph products (Complement/LineGraph/Cartesian).

### rustmath-coding [2338, 35]
- Verdict: Partial — real Hamming/Golay/BCH/Reed-Solomon; locked to prime fields via u64.
- Sage: sage.coding.{linear_code,hamming_code,golay_code,bch_code,reed_solomon,grs}. MAGMA: 152(partial).
- Implemented: LinearCode w/ generator/parity matrices, syndrome decoding, Hamming/Golay/BCH/RS encode-decode.
- Gaps: no generic LinearCode<F:Field> over finitefields/matrix (GF(q>p) unsupported); no weight enumerator/MacWilliams; no bounds (Singleton/Hamming/GV/Griesmer); no codes-over-rings(155)/additive/quantum(156/157); no AG(153)/LDPC(154).

### rustmath-crypto [6812, 148]
- Verdict: Substantial — wide real primitives (DES, Ed25519, SHA-256/SHA3/BLAKE2 w/ NIST vectors); no LFSR/PRNG chapter.
- Sage: sage.crypto.{classical,block_cipher.des,miniaes,present,public_key.rsa,stream,lfsr}. MAGMA: 158(none).
- Implemented: classical ciphers, DES/S-DES/Mini-AES/PRESENT, RSA/DH/ElGamal/ECC+ECDSA/Ed25519, SHA256/SHA3/BLAKE2b, RC4/ChaCha20, GCM, PBKDF2/Argon2.
- Gaps: no LFSR/Berlekamp-Massey (158); no named standard curves or RSA padding (OAEP/PKCS1); ECC hardcodes num-bigint not Integer; no BBS/shrinking-generator PRNGs; elliptic_curve.rs duplicates curves (collision).

**Cluster K total:** 3 crates — 2 substantial+, 1 partial, 0 stub.
## Cluster D — Linear algebra, modules, homology, category

### rustmath-matrix [9992, 139]
- Verdict: Substantial — dense/sparse, LU/PLU/QR/Cholesky/SVD, HNF/SNF, LLL, char-poly all real; lib+tests compile clean.
- Sage: sage.matrix.{matrix2,constructor,special,matrix_integer_dense,berlekamp_massey}. MAGMA: 26,27,30,83.
- Implemented: LU/PLU/QR/Cholesky/Hessenberg/SVD, HNF/SNF, exact LLL (unimodular-checked), char/min poly, companion/rational canonical form, sparse CSR, Strassen.
- Gaps: eigenvalues/Jordan only via power iteration (no full QR algorithm/complex spectrum); no GMP/FLINT fast path; no BKZ; no iterative solvers (CG/GMRES); no decompositions over general non-Euclidean rings.

### rustmath-modules [9136, 196]
- Verdict: Partial — deep with_basis/tensor, but ~30 files are 4-6 line placeholder stubs; TEST SUITE FAILS TO BUILD.
- Sage: sage.modules.{free_module,free_module_element,submodule,quotient_module,fg_pid.fgp_module,with_basis}, sage.tensor.modules. MAGMA: 53,54,55,30,31.
- Implemented: FreeModule/Element/morphisms over any Ring; with_basis (morphism 554 LOC, homsets 445 LOC), tensor/ (comp 452 LOC) substantive.
- Gaps: `cargo test -p rustmath-modules --lib` fails 388 errors (num_bigint::BigInt !impl Ring) — with_basis/tensor coverage unverified; Submodule.rank() = generator count (no independence check); QuotientModule.are_equivalent() = a==b; ~30 files (ore_module*, fp_graded, vector_mod2/modn/numpy/symbolic_dense, diamond_cutting) one-line stubs; FreeQuadraticModule/TorsionQuadraticModule lack genus/discriminant-form/isometry.

### rustmath-homology [1271, 24]
- Verdict: Partial — real chain/cochain complexes + Betti homology, but torsion hardcoded empty despite matrix SNF.
- Sage: sage.homology.{chain_complex,chain_complex_homspace,homology_group}. MAGMA: 56,140.
- Implemented: ChainComplex/CochainComplex over Matrix<Integer>, boundary validation, rank/kernel Betti numbers, Euler char, simplicial_chain_complex helper.
- Gaps: homology()/cohomology() always return torsion:[] (comment admits needs SNF, though matrix has it); no persistent homology; no simplicial/CW/cubical builders beyond one helper; no cohomology ring/cup products.

### rustmath-category [6208, 178]
- Verdict: Partial — dyn-compat issue (root_cause_analysis) FIXED (lib compiles), but nearly unused elsewhere and own tests don't build.
- Sage: sage.categories.{category,functor,morphism,map,rings,modules,groups}. MAGMA: none.
- Implemented: Category/Morphism/Functor/NaturalTransformation traits (Category now dyn-compatible, Box<dyn Category> works), axiom system, coercion scaffolding; `cargo check` clean.
- Gaps: `cargo test --lib` fails 16 errors (missing num_bigint dev-dep, missing methods on test structs); only 1 external usage (modules imports Morphism once) — NOT wired into core Ring/Field/Module hierarchy; categories/axioms are string-based bookkeeping, no real coercion-graph resolution; no concrete functor category.

**Cluster D total:** 4 crates — 1 substantial+, 3 partial, 0 stub.

## Cluster I — Geometry, topology, manifolds, dynamics

### rustmath-geometry [18155, 363]
- Verdict: Substantial — broad (toric, polytopes, hyperbolic, Voronoi) but core Polyhedron f64-only, not exact.
- Sage: sage.geometry.{polyhedron,cone,fan,toric_variety,lattice_polytope,triangulation,hyperbolic_space,hyperplane_arrangement,voronoi_diagram,polyhedral_complex,newton_polygon}. MAGMA: 118(partial),143(partial).
- Implemented: cones/fans/toric divisors/Chow groups/blow-ups (1883 LOC), lattice polytopes over Integer (1355 LOC), Voronoi/Delaunay/hyperplane arrangements, hyperbolic space (4 models), ribbon graphs.
- Gaps: no exact H-rep/V-rep rational Polyhedron (current is f64, no vertex enumeration/duality); toric.rs uses Vec<i64>/f64 bypassing core/Integer/Matrix HNF/SNF; no 115 resolution graphs; no 141 finite planes / 142 incidence geometry; no Cox ring.

### rustmath-topology [5607, 137]
- Verdict: Partial — solid simplicial/knot combinatorics but NO homology computation and 7 failing tests.
- Sage: sage.topology.{simplicial_complex,cubical_complex,delta_complex,simplicial_set,filtered_simplicial_complex}, sage.knots.{knot,link}. MAGMA: none.
- Implemented: simplicial/cubical/delta/cell complexes w/ f-vectors + Euler char; simplicial sets (cone/suspension/product); knot/link/braid, Jones/HOMFLY/Kauffman, Reidemeister.
- Gaps: NO homology computation (no boundary matrices/SNF/Betti — "persistent homology" only tracks filtration values, no diagrams); knot invariants heuristic/bounds not exact; 7/137 tests FAILING (cell_complex, simplicial_complex, link); no fundamental group/covering spaces.

### rustmath-manifolds [17303, 275]
- Verdict: Substantial — deep Lie/symplectic/spin API; BUILDS CLEAN (lib, 130 warnings — the "188 errors on main" in MANIFOLDS_STATUS.md is STALE/incorrect; verified builds 2026-07-09).
- Sage: sage.manifolds.{manifold,chart,differentiable.*,metric,tensorfield,diff_form,symplectic_form,examples.*}. MAGMA: none.
- Implemented: charts/transitions, vector/tensor/diff-form fields w/ exterior calculus, tangent/cotangent, Lie groups/algebras, symplectic manifolds, fiber bundles, catalog (Minkowski/Schwarzschild/Kerr) — TODO-free through Phase 4.
- Gaps: Phase-5 modules placeholder-heavy (dirac 14 TODOs, spin 7, contact/finsler/subriemannian 5 each; Clifford mult/Atiyah-Singer/Nijenhuis unimplemented); Geodesic::at(t) ignores t (returns initial point, no integration); ParallelTransport::transport identity stub; de Rham cohomology/Betti/Chern hardcoded per known-space not computed.

### rustmath-dynamics [1241, 40]
- Verdict: Partial — real generic real-valued dynamics/fractals but NO arithmetic dynamics.
- Sage: sage.dynamics.complex_dynamics, generic_ds (NOT arithmetic_dynamics). MAGMA: none.
- Implemented: discrete map iteration/fixed points, RK4/Euler, Mandelbrot/Julia/Newton/Burning-Ship, Lyapunov, bifurcation, Poincaré sections.
- Gaps: no arithmetic_dynamics (projective/affine over number fields, canonical heights, (pre)periodic points — the number-theory-relevant part); no cellular automata; no interval exchange; f64 only.

**Cluster I total:** 4 crates — 2 substantial+, 2 partial, 0 stub. (manifolds build corrected to green.)
## Cluster E — Groups, monoids, automata, quivers

### rustmath-groups [20898, 497]
- Verdict: Substantial — broad group-type coverage but permutation-group core is thin; libgap_* are pure-Rust name-alikes, not GAP bindings.
- Sage: sage.groups.{perm_gps,matrix_gps,abelian_gps,free_group,finitely_presented,braid,artin,cactus_group,raag,class_function,conjugacy_classes,perm_gps.permgroup_named}. MAGMA: 57,58(partial),59-62(GLn/SLn only),69,70-71(no coset enum),73,74(rules only),90-91(f64).
- Implemented: Abelian (invariant factors), free/FP/Artin/braid/cactus/RAAG, Schreier-lemma orbit/stabilizer/block predicates, degree-23/24 Galois atlases, basic character/conjugacy arithmetic.
- Gaps: NO Schreier-Sims/BSGS stabilizer chain or general membership/order (only orbit+Schreier-gen primitives); libgap_*.rs are pure-Rust stand-ins (no real GAP FFI); no polycyclic/soluble-quotient, coset enumeration, SLP/black-box; RewritingSystem has no Knuth-Bendix; CharacterTable f64 (no exact cyclotomic) + duplicates combinatorics; matrix groups GLn/SLn only.
- General vs specialized: NOT general BSGS — perm_predicates.rs gives generator-only orbit/transitivity/primitivity/block/Schreier-stabilizer (any degree, no full chain); transitive23/24/pgalois are degree-23/24-specialized Galois classifiers (LMFDB deg-24 dataset) for Belyi track, not a general transitive-group DB.

### rustmath-monoids [2478, 103]
- Verdict: Partial — free/free-abelian/string monoids solid; automatic-semigroup/trace/Hecke skeletal.
- Sage: sage.monoids.{free_monoid,free_abelian_monoid,string_monoid,indexed_free_monoid,hecke_monoid}. MAGMA: 77(free only),78(AutomaticSemigroup::mul is a stub).
- Implemented: FreeMonoid/FreeAbelianMonoid (782 LOC), IndexedFreeMonoid, StringMonoid/string_ops (669 LOC).
- Gaps: AutomaticSemigroup::mul returns empty word (no automaton mult); TraceMonoid no normal form; HeckeMonoid bare data holder (37 LOC); no FP presentations/Knuth-Bendix; no growth/word-problem.

### rustmath-automata [1231, 0 (actually 11 tests)]
- Verdict: Partial — solid generic DFA/NFA/Moore/Mealy; ZERO integration with groups/monoids (no automatic-groups despite purpose).
- Sage: sage.combinat.words.automatic (loose). MAGMA: 75 (no group/FSA integration).
- Implemented: DFA/NFA (subset construction, minimization, complement, product), Moore/Mealy w/ interconversion, well-tested.
- Gaps: not wired to groups/monoids word-problem/rewriting; no automatic-group FSA (word-difference/multiplier automata); no regex↔automaton; no pushdown/CFG; single-file.

### rustmath-quivers [2057, 50]
- Verdict: Partial — quiver graph/path/cluster-mutation combinatorics solid; no path-algebra ring or representation theory.
- Sage: sage.quivers.{path_semigroup,paths}, combinat.cluster_algebra_quiver.quiver. MAGMA: none (closest 90).
- Implemented: Quiver (digraph, acyclicity, path enum), QuiverPath, PathSemigroup (idempotents/arrows/basis), ClusterQuiver (mutation, type A/D/E, mutation-type detection, Cartan companion).
- Gaps: no QuiverAlgebra as a Ring (no coeff ring/multiplication); no QuiverRep/representations (module category, proj/inj covers, AR translate, Ext); no relations/bound quivers; cluster algebra lacks seed/coefficient tracking.

**Cluster E total:** 4 crates — 1 substantial+, 3 partial, 0 stub.
## Cluster G — Symbolic, calculus, functions, logic

### rustmath-symbolic  [26973 LOC, 600 tests]
- **Verdict:** Substantial — full CAS core (parser, diff, integrate, limits, series, solve, patterns) all native Rust
- **Sage:** sage.symbolic.expression, sage.symbolic.assumptions, sage.symbolic.integration.*, sage.calculus.functional, sage.symbolic.relation, sage.symbolic.function
- **MAGMA:** none (ch111 Differential Rings has minor diffeq overlap)
- **Implemented:** Native nom-based string parser; recursive differentiation; heuristic symbolic integration (table+by-parts+trig-sub+partial fractions); L'Hôpital limits; Taylor/Laurent/Fourier series; assumptions system; pattern-matching rule engine; ODE solvers (RK4/Euler + closed-form 1st/2nd order); polynomial solve up to quartic; units; extensive specialfunctions submodule (Bessel/Airy/hypergeometric/orthogonal polys/Wigner symbols) as symbolic Expr forms.
- **Gaps:** No Risch algorithm; external CAS bridges (maxima_wrapper, integrate_external) are stubs returning None; cubic/general-degree symbolic solve incomplete; unit conversion factor calc incomplete; no arbitrary-precision numeric backend (f64 in numerical.rs).
- **Key CAS gaps:** string parsing EXISTS (parser.rs, nom, real); symbolic integration EXISTS (heuristic, not Risch); limits EXIST (sub + L'Hôpital); no full multivariate/asymptotic limit theory.

### rustmath-calculus  [8607 LOC, 169 tests]
- **Verdict:** Substantial — diff/integ/limits/series/ODE/transforms real; several modules disabled
- **Sage:** sage.calculus.calculus, sage.calculus.desolvers, sage.calculus.interpolation, sage.calculus.riemann, sage.calculus.transforms.fft
- **MAGMA:** none (ch111 tangential via ODE)
- **Implemented:** Symbolic diff/integ (table-based, thinner than symbolic's), limits, Taylor/Maclaurin, RK4/Euler/adaptive ODE + systems, cubic spline interpolation, DFT/FFT/DWT, Riemann mapping numerics.
- **Gaps:** laplace.rs, maxima_compat.rs, minpoly.rs, pochhammer.rs, product.rs commented out ("TODO: Fix for new Expr structure"); overlaps/duplicates symbolic rather than reusing; no PDE (lives in symbolic::pde).

### rustmath-functions  [1527 LOC, 72 tests]
- **Verdict:** Substantial — thin but complete symbolic+numeric wrapper over elementary functions
- **Sage:** sage.functions.trig, sage.functions.hyperbolic, sage.functions.log, sage.functions.other
- **MAGMA:** none
- **Implemented:** Dual-mode (Expr symbolic + f64) wrappers for trig/inverse/hyperbolic/exp/log/power/sign/floor/ceil; delegates to symbolic's Expr methods.
- **Gaps:** SymbolicFunction trait never implemented (dead); no complex support; no arbitrary precision; largely redundant with Expr methods.

### rustmath-special-functions  [944 LOC, 30 tests]
- **Verdict:** Partial — f64-only numeric special functions; symbolic/high-prec versions elsewhere
- **Sage:** sage.functions.gamma, sage.functions.special, sage.functions.error, sage.functions.transcendental
- **MAGMA:** none
- **Implemented:** f64 gamma/ln_gamma/digamma, beta, Riemann+Hurwitz zeta, Bessel J/Y (int order), erf/erfc.
- **Gaps:** f64-only, no complex args; no Bessel I/K, incomplete gamma/beta, no hypergeometric here (duplicated in symbolic::specialfunctions 5086 LOC); no elliptic integrals; shallower than symbolic's submodule.

### rustmath-logic  [2257 LOC, 46 tests]
- **Verdict:** Substantial — real DPLL SAT, CNF/DNF, propositional engine, resolution/nat-deduction proofs
- **Sage:** sage.logic.propcalc, sage.logic.boolformula, sage.logic.logicparser, sage.logic.logictable
- **MAGMA:** none
- **Implemented:** Formula AST with evaluate/simplify/truth_table/tautology/sat; CNF/DNF w/ De Morgan; DPLL w/ unit propagation; resolution-refutation + natural-deduction proofs.
- **Gaps:** No string parser for formulas; no CDCL/clause learning; propositional only (no FOL/quantifiers); no Karnaugh/circuit minimization.

**Cluster G total:** 5 crates — 4 substantial+, 1 partial, 0 stub/skeleton.
## Cluster J — Schemes, curves, elliptic curves, modular forms

### rustmath-schemes [11297, 227]
- Verdict: Partial — real projective embeddings + EC group law, but affine/Heegner/Chabauty stubbed AND fails to compile in test mode.
- Sage: sage.schemes.generic.spec, sage.schemes.projective.proj, sage.schemes.elliptic_curves.*. MAGMA: 112,113(O(d) only),120-122.
- Implemented: Veronese/Segre monomial maps, generic-Field Weierstrass group law, naive/canonical heights + bounded point search, division-polynomial recurrence, Vélu coefficients.
- Gaps: does NOT compile as tests (100 errors: Rational lost Zero/One; corrupted elliptic_curves/mod.rs duplicate tests; projective dimension shadowing) — "227 tests" is a static count; NO blow-ups; affine schemes never build a polynomial ring (AffinePoint::parent unimplemented!); Chabauty-Coleman returns constants; Heegner/Gross-Zagier/Kolyvagin placeholders; rational.rs hardcoded Cremona/rank tables not Tate's algorithm; three unconsolidated EC representations.

### rustmath-affineschemes [2410, 57]
- Verdict: Skeleton — nearly every type wraps PhantomData<R>; ideal membership/primality/radical hardcoded stubs.
- Sage: sage.schemes.generic.spec, affine.affine_space. MAGMA: 112(affine fragment).
- Implemented: Zariski open/closed set algebra, Spec(R) scaffolding, Display, trivial smoke tests.
- Gaps: is_prime/radical/is_maximal/primary_decomposition/ideal contains all placeholders; AffineScheme never constructs a polynomial ring (affine_space no-op); Krull dimension manually set; no Gröbner integration despite polynomials available; fiber products/morphisms PhantomData.

### rustmath-curves [19441, 270]
- Verdict: Partial — real Weierstrass/hyperelliptic invariants, but genus/Riemann-Roch/singularity detection formula-only approximations.
- Sage: sage.schemes.curves, hyperelliptic_curves, rings.function_field. MAGMA: 114,125.
- Implemented: correct Weierstrass b/c-invariants, discriminant, j-invariant; hyperelliptic squarefree/genus/discriminant; Brill-Noether/Clifford formulas; genus2 Mumford divisor + Cantor skeleton.
- Gaps: find_singularities builds a real Gröbner ideal then unconditionally returns [] (genus/parameterization silently wrong for singular curves); Riemann-Roch L(D) exact only for deg(D)≥2g-1 else non-exact lower bound (no Hess algorithm); Cantor composition not full ext-Euclidean; differentials.rs placeholders; belyi/ subsystem (~30 files, actively edited) is a separate genus-0 Belyi/dessins pipeline.

### rustmath-ellipticcurves [2313, 40]
- Verdict: Partial — correct short-Weierstrass group law, but descent/L-functions/modularity/BSD toy.
- Sage: sage.schemes.elliptic_curves.{ell_rational_field,ell_point,descent_two_isogeny,lseries_ell,heegner}. MAGMA: 120,122,127.
- Implemented: correct point add/double/scalar-mul/j-invariant over BigRational; naive point counting mod p; naive rational-point search.
- Gaps: 2-descent/Selmer "simplified" (rank_bound = log2 of element count, not real); conductor ad-hoc bad-prime product not Tate; Hecke eigenvalues from Ramanujan-bound heuristic not modular symbols; Sha hardcoded 1.0; ignores rustmath-core (hardcodes num-bigint/num-rational + own f64 ComplexNum).

### rustmath-modular [5170, 73]
- Verdict: Skeleton — wide API scaffold; real pockets (arithgroup, cusps, Hecke q-expansion, dims) atop ~50 near-empty files.
- Sage: sage.modular.{arithgroup,modform,modsym,hecke,dirichlet,abvar}. MAGMA: 128,132,133,134,136,137-139.
- Implemented: real SL2Z/Gamma0/Gamma1 action + index formulas; correct cusp reduction/equivalence; correct Hecke T_n q-expansion formula; Cohen-Oesterle dimension (simplified); eta-product q-expansion.
- Gaps: no Manin-symbol relations or basis reduction (modsym = data containers); HeckeAlgebra::eigenvalues returns zeros; Atkin-Lehner/diamond no-ops; DirichletGroup only trivial character; abvar.rs (~941 lines) almost entirely placeholder; 15 exotic modules (btquotients, buzzard, drinfeld_modform, local_comp, overconvergent, pollack_stevens, quasimodform, quatalg/brandt, ssmod, hypergeometric_motive) empty shells.

**Cluster J total:** 5 crates — 0 substantial+, 3 partial, 2 stub/skeleton.
## Cluster L — Infrastructure, IO, viz, front-ends

### rustmath-plot [9242, 213]
- Verdict: Substantial — full 2D graphics stack (primitives, plots, SVG/raster, animation), few TODO gaps.
- Sage: sage.plot.{graphics,plot,primitive,point,line,circle,contour_plot,histogram}. MAGMA: front-end.
- Gaps: hyperbolic polygon area; contour fill between levels; multi-graphics combined render; some markers; show() no display backend.

### rustmath-plot3d [4999, 78]
- Verdict: Substantial — surfaces, parametric, marching-cubes implicit, OBJ/STL export.
- Sage: sage.plot.plot3d.*. MAGMA: front-end.
- Gaps: list_plot3d incomplete; no interactive rotation (known bug); no glTF; minimal textures.

### rustmath-plot-core [1504, 29]
- Verdict: Complete — thin focused foundation (traits, options, bbox, types).
- Sage: sage.plot.primitive/graphics (shared types). MAGMA: front-end.

### rustmath-jupyter [12159, 0 (44 tests exist)]
- Verdict: Substantial — real async ZeroMQ Jupyter kernel + 10.8k-line REPL dispatcher wired to 17+ crates.
- Sage: sage.repl.{interpreter,display}. MAGMA: front-end.
- Gaps: REPL is a bespoke calculator language NOT Sage/Python syntax; 3D-plot rotation bug; no real completion/introspection.

### rustmath-typesetting [2720, 92]
- Verdict: Complete — multi-format math renderer (LaTeX/ASCII/Unicode/HTML), precedence-aware MathDisplay.
- Sage: sage.misc.latex, sage.typeset.*. MAGMA: front-end.

### rustmath-colors [1485, 44]
- Verdict: Complete — RGB/HSL/HSV + named colors + colormaps (viridis/plasma/jet).
- Sage: sage.plot.colors. MAGMA: front-end.
- Gaps: no colorblind-safe/CMYK; limited named set.

### rustmath-databases [3153, 54] — FAILS BUILD HERE (openssl-sys, env-only)
- Verdict: Partial — OEIS/Cunningham/Cremona real; LMFDB hardcoded stub.
- Sage: sage.databases.{oeis,cremona,cunningham_tables,lmfdb}. MAGMA: front-end.
- Implemented: OEIS HTTP client + analyzer, Cunningham tables, Cremona curve DB.
- Gaps: LMFDBClient returns builtin_* hardcoded only (no real API); needs libssl-dev; no download/sync tooling. FIX: gate network/DB features optional.

### rustmath-interfaces [4752, 63] — 3 trivial test_long.rs compile errors
- Verdict: Substantial — GAP process bridge real and deep; PARI/Singular/FLINT absent.
- Sage: sage.interfaces.gap. MAGMA: front-end.
- Implemented: GapProcess/GapInterface spawn+pipe, command/result translation, gap_parser, GapPermutationGroup high-level API, workspace save/load.
- Gaps: PARI/Singular/FLINT/GMP/MPFR "planned" only; test_long.rs 3 type errors block build; most tests #[ignore]d (need real GAP).

### rustmath-features [860, 22]
- Verdict: Complete — feature-detection/fallback framework, no_std-capable.
- Sage: none (infra). MAGMA: front-end.
- Note: the gmp/mpfr/flint/pari features it gates don't yet exist as crates.

### rustmath-misc [1638, 31]
- Verdict: Partial — mrange/temporary_file solid; ~9 near-empty 6-line stub modules.
- Sage: sage.misc.{mrange,temporary_file,table,verbose,sageinspect,...}. MAGMA: front-end.
- Gaps: sageinspect, sh, sphinxify, stopgap, superseded, trace, unknown, viewer, weak_dict are 6-line placeholders.

### rustmath-benchmarks [916, 0]
- Verdict: Non-math-infra — CLI bench harness (bins only).
- Gaps: no real SymPy comparison; no criterion; no regression tracking.

### rustmath-igp24 [288, 0]
- Verdict: Non-math-infra — thin JSON CLI over the degree-24 Galois-ID engine.
- Gaps: no own tests; single-purpose.

### rustmath-lava [front-end, 0]
- Verdict: Non-math-infra — vendored Magma-syntax tooling (format/highlight/test via tree-sitter-magma + topiary).
- MAGMA: front-end (L7 in lava_mvp.md; the L0-L6 compute backend is the unbuilt integration).
- Gaps: `parse` planned v0.2; no Magma semantic eval; not wired to any math crate.

### rustmath-py [1574, 0]
- Verdict: Partial — PyO3 bindings for narrow slice (Integer/Rational/Matrix/Symbolic/Plot); most of RustMath unexposed.
- Sage: PyO3 interop. MAGMA: front-end.
- Gaps: no bindings for polynomials/finitefields/numberfields/groups/graphs/combinatorics; 0 tests; no maturin config verified.

**Cluster L total:** 14 crates — 5 complete/substantial infra, 3 partial (databases, misc, py), 3 non-math tooling (benchmarks, igp24, lava), rest thin-complete.
