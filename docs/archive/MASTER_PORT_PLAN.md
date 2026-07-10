# MASTER PORT PLAN — MAGMA handbook → RustMath (single source of truth)

**Decision (2026-07-01):** Port the full MAGMA computational-algebra handbook
(`rustmath-lava/docs`, chapters **17–159**) into RustMath as the native back-end.
RustMath is the compute engine; MAGMA (via `lava`) and Sage are front-ends.
**Foundation-first**: harden the shared trait/numeric foundation before fanning
out to chapters. Derived from the `magma-port-survey` coverage of all 143
mathematical chapters (27 domain reports).

Locked decisions (user-confirmed):
1. **Scope** = math back-end, ch **17–159**. Language ch 1–16 stay with the `lava` front-end.
2. **Cadence** = foundation-first: **Wave 0** (core traits + arbitrary-precision reals/complex) lands and is reviewed **before** Wave 1 fan-out.
3. **Arbitrary-precision reals/complex** = trait-abstract (`RealField`/`ComplexField`/`Ball` in `rustmath-core`) + **pure-Rust** default backend (astro-float / dashu-float), swappable to `rug`/MPFR later behind the same trait. Zero-`unsafe`, portable.
4. **Integration** = **worktree per crate → one PR per crate**. Disjoint new files ⇒ clean merges.

---

## 0. Worker discipline (every port worker, no exceptions)
1. **Own exactly one crate.** Touch only files under it. Only edits to shared surface: that crate's own `src/lib.rs` (one `pub mod` line per new module) and its `Cargo.toml`. Never edit another crate. *(Wave-0 foundation is the sole exception: `rustmath-core`+`reals`+`complex` are co-designed by one owner.)*
2. **New files only.** Add beside stubs; do not rewrite existing module logic. Trait *impls* for existing types go in new `*_core_impls.rs` files.
3. **Implement/consume `rustmath-core` traits.** No private bignum/rational/complex wrappers — reuse `rustmath-integers`/`rustmath-rationals`/`rustmath-complex`/`rustmath-reals`/`rustmath-finitefields`/`rustmath-matrix`. **Do not** pull `num-bigint`/`num-complex` directly into new code.
4. **Zero `unsafe`.**
5. **Keep the crate green** (`cargo test -p <crate>`). Port the MAGMA-doc examples as tests. Cite the source MAGMA chapter in each new file's header.
6. **Status honesty.** Certifying/semialgorithmic output stays "UNRESOLVED vs decided"; a bounded search that finds nothing is never a decision.
7. **Isolated worktree, one PR per crate.** Rebase on merged Wave-0 before starting Wave 1+.

## Collision firewall — OFF-LIMITS crates (active IGP24 + dessin workers)
`rustmath-numberfields`, `rustmath-quadraticforms`, `rustmath-groups`,
`rustmath-polynomials`, `rustmath-numerical`, `rustmath-curves`,
`rustmath-igp24`, `rustmath-lava`. **No port worker touches these** until those
efforts merge. Chapters homed there are **Deferred** (§5). Wave-0 changes to
`rustmath-core` are **purely additive** so they never break the active workers on `main`.

---

## 1. Coverage snapshot (chapters 17–159)
From the survey, per-chapter coverage: **24 substantial · 44 partial · 35 stub · 40 none**.
The recurring integration gap is *not* missing math so much as **trait non-adoption**: existing
types use ad-hoc/`num-*` types instead of the `rustmath-core` tower. Wave 0 fixes the
vocabulary; each crate's Wave then adopts it.

Full per-chapter matrix + per-crate build backlogs: regenerate from the survey
(`magma-port-survey` workflow result) or see `docs/port/` once persisted.

---

## 2. WAVE 0 — Foundation (the gate; merges before fan-out)

Single owner for the co-designed triad `rustmath-core` + `rustmath-reals` + `rustmath-complex`.
**Purely additive** to `core` (new modules only; never touch `traits.rs` existing defs).

### W0.core — new trait vocabulary in `rustmath-core` (new files, `pub mod` each)
- `ordering.rs` — `OrderedRing: Ring + PartialOrd { fn sign()->i32; fn abs()->Self; }`, `OrderedField: Field + OrderedRing`. (Needed by LP simplex ch159, exact polytopes ch143, real comparisons.)
- `valuation.rs` — `DiscreteValuation<R> { fn valuation(&self,&R)->i64; fn uniformizer()->R; }` and a sign-aware `Place` value type incl. infinite/degree places. (Valuation rings ch45, p-adics 47/51, function fields 41–43.)
- `analytic.rs` — `RealField: OrderedField`, `ComplexField: Field { type Real: RealField; re/im/abs/conj/arg }`, `Ball { center/radius/contains }`; precision carried via the `Parent` (see below), not a global. (Real/complex ch25, Newton polygons 46, L-functions 127, modular forms 132, exact eigen.)
- `morphism.rs` — object-safe `Morphism { type Domain; type Codomain; fn apply(&self,&Domain)->Codomain }` + boxed erased layer. **`Ring` is not dyn-safe** (PartialEq + by-value ops), so Hom/End, chain maps, scheme & ring morphisms need this erased layer.
- `nonassoc.rs` — `NonAssociativeAlgebra<F: Field>: Module<F> { fn mul }` and `LieAlgebra<F: Field>: Module<F> { fn bracket }`. (Lie brackets are not `Ring::mul`; ch100–104.)
- `coercion.rs` — pushout/common-overstructure helper on top of `parent.rs` (`R ! a`, automatic coercion of MAGMA §17.3).
- (Evaluate, don't duplicate) marker traits `GcdDomain`/`Ufd`/`Pid`/`LocalRing`/non-domain-`PIR` — **note:** `rustmath-rings` already has `NoetherianRing`/`DedekindDomain`/`PrincipalIdealDomain`; decide central home before adding. Non-domain PIR marker is genuinely missing (Galois rings ch48).
- (Optional) `form.rs` — `BilinearForm`/`SesquilinearForm`/`HermitianForm` traits (polar spaces ch29).

### W0.reals — `rustmath-reals`: pure-Rust arbitrary-precision real
New files (e.g. `bigfloat.rs`): `BigFloat` on astro-float (or dashu-float), implementing
`Ring`/`CommutativeRing`/`Field`/`OrderedField`/`RealField` + `NumericConversion`, with a
precision-carrying `Parent` (`RealField(prec)`). Keep the existing f64 `Real`. Elementary
functions (sqrt/exp/ln/sin/…) + π/e at precision.

### W0.complex — `rustmath-complex`: arbitrary-precision complex + ball
New files (`bigcomplex.rs`, `ball.rs`): `BigComplex` over `W0.reals::BigFloat` implementing
`ComplexField` + `Ring`/`Field`; `Ball`/`Arb`-style interval type for certified numerics.
Also add `Ring`/`Field` impls to the existing MPFR/MPC `RealMPFR`/`ComplexMPFR` so they are
first-class (survey: they currently implement neither).

**Wave-0 exit test:** `cargo test -p rustmath-core -p rustmath-reals -p rustmath-complex` green;
one foundation PR; reviewed before Wave 1 branches off it.

---

## 3. WAVE 1 — Leaf algebraic crates (fan out after W0 merges)
Each = one worker / one crate / worktree → PR. Minimal foundation deps.

| Crate | Chapters | Headline backlog |
|---|---|---|
| `rustmath-finitefields` | 19, 21, 48 | Unify the two `Z/mZ` types behind one `Integers(m)` Parent; Cantor–Zassenhaus + irreducible-poly search + BSGS discrete log; Conway embeddings via Parent; **Galois rings GR(p^a,d)** (needs non-domain PIR marker). |
| `rustmath-matrix` | 26, 27, 28, 30, 83 | `Vector` → impl `Module<R>`/`VectorSpace<F>`; sparse SNF/HNF over `EuclideanDomain`; exact eigen/charpoly path (off f64); LLL/Gram-Schmidt over `RealField`; matrix algebras (ch83). |
| `rustmath-powerseries` | 49, 50, 52 | **`PowerSeries` impl core `Ring`** (top gap); Laurent/Puiseux (Rational exponents) w/ Precision enum; lazy (closure-backed); algebraic series. |
| `rustmath-combinatorics` | 92, 144, 145, 147, 148 | Sym-group reps (ch92); enumerative (ch144); tableaux/plactic monoid → `Monoid`; designs (147); Hadamard (148). Unify `CharacterTable` duplication w/ groups (coordinate). |
| `rustmath-symmetricfunctions` | 146 | `SymFun`/QSym/NCSF/FQSym → impl `Ring`/`CommutativeRing`/`Algebra<F>`; more bases + transition matrices over `Rational`. |
| `rustmath-graphs` | 149, 150, 151 | Adjacency/Laplacian → `Matrix<Integer>`; chromatic/Tutte → `Polynomial`; networks over `Integer` capacities; `AutomorphismGroup` → PermutationGroup (defer group side). |
| `rustmath-monoids` + `rustmath-automata` | 74, 75, 77, 78 | Adopt core `Monoid`/`Semigroup`/`Group`; build the **Knuth–Bendix completion + FSA reduction engine once in `rustmath-automata`** (shared by 74/75/78); `Order()` returns `Option`/BigInt (infinite-safe). |
| `rustmath-crypto` | 158 | LFSR / Berlekamp–Massey / BBS over `rustmath-finitefields` + `rustmath-polynomials` (read-only dep). |

## 4. WAVE 2 — Structures (after Wave 1)
| Crate | Chapters | Headline backlog |
|---|---|---|
| `rustmath-rings` | 17, 36, 40, 41, 42, 43, 45 | Ideal arithmetic/predicates; cyclotomic fields; **algebraic closure** (lazy, arena-backed); **refactor the ~18k-LOC `function_field/` String scaffolding into typed generics** (41/42/43); valuation rings (DVR = `EuclideanDomain`, norm=valuation). Skew/Ore polys + Witt vectors for 43. |
| `rustmath-padics` | 47, 51 | **Unify with `rings::padics`** (two impls collide); extension elements → `Module<Qp>`; wire `ResidueClassField` to `rustmath-finitefields`; weak-equality convention vs `Ring::eq`. |
| `rustmath-modules` | 53, 54, 55 | Reconcile local `Module` trait → core `Module<R>`; `Hom`/`End` (End is a `Ring`, via `morphism.rs`); Dedekind modules (defer fractional-ideal dep on numberfields). |
| `rustmath-homology` | 56, 68, 140 | `ChainComplex` generic over `EuclideanDomain`/`Field` coeffs; simplicial homology; group cohomology H^1/H^2 + extensions (defer fp/pc-group dep on groups). |
| `rustmath-algebras` | 79, 80, 81, 82, 84, 85, 87, 88, 89 | Reconcile the crate's own `Algebra` tower with core; associative algebras impl `Ring`+`Algebra<F>` w/ Parent-carrying elements; Wedderburn/ideals; Clifford (88); *-algebras (87); basic algebras (85); modules over an algebra (89). |
| `rustmath-liealgebras` (+ `rustmath-crystals`) | 94–101, 104 | Coxeter/root systems/root data; `WeylGroupElement`→`Group`; weight/root lattices→`Module<Integer>`; non-crystallographic H3/H4/I2(m) need cyclotomic/`RealField`; reps decomposition. |
| `rustmath-quantumgroups` | 102 | Q(q) fraction-field + Z[q,q^-1] Laurent coeffs; quantized UEA → `Ring`/`Algebra`; canonical bases. |
| `rustmath-geometry` | 29, 115, 118, 141, 142, 143 | Exact polytopes over `Rational`/`Integer` (off f64); toric divisors → `Module<Integer>`, Hilbert series over Z[t]; finite planes/incidence over `Field`; polar spaces + `SesquilinearForm`; resolution graphs on `rustmath-graphs`. |
| `rustmath-coding` | 152–157 | Refactor `Vec<Vec<u64>>` → `LinearCode<F: Field>` w/ `Matrix<F>` generator/parity; codes over rings (Z4, ch155); additive/quantum over GF(q^2); AG codes (153) blocked on function fields. |

## 5. WAVE 3 — Higher / analytic (after Wave 2 + arbitrary precision)
| Crate | Chapters | Notes |
|---|---|---|
| `rustmath-schemes` + `rustmath-affineschemes` + **new `rustmath-sheaves`** | 112, 113, 116, 117 | Real coordinate rings/ideals over Gröbner; graded modules + **free resolutions/syzygies/Ext** (new abstraction) for coherent sheaves; surfaces; Hilbert series over Z[t]. |
| `rustmath-ellipticcurves` | 120–123, 127 | **Normalize off `num-*` onto RustMath numerics first**; point group → `AbelianGroup`; L-functions/heights need `RealField`/`ComplexField` + special functions; consolidate w/ `rustmath-crypto::elliptic_curve`. |
| `rustmath-modular` | 128–139 | **num-* → RustMath normalization pass is prerequisite** (crate uses `num_bigint`/`num_complex<f64>` throughout, zero core impls); spaces → `VectorSpace<F>`, Hecke ops → `Matrix`; Fuchsian/Shimura need `RealField`; GL2(Qp) needs p-adics. |
| **new** `rustmath-hyperelliptic` (125), `rustmath-genusone` (124), `rustmath-hgm` (126), `rustmath-groupsoflietype` (103) | 103, 124, 125, 126 | Greenfield; build on curves/schemes/liealgebras foundations. |
| `rustmath-nearfields` (**new**) | 22 | Nearfields violate left distributivity ⇒ bespoke `Nearfield` trait (units = `Group`, addition = `AbelianGroup`), not `Ring`. Small. |
| **new** `rustmath-optimization` | 159 | LP homes in `rustmath-numerical`, which is OFF-LIMITS → build LP in a new crate over `OrderedField` to avoid the collision. |
| reps-characters | 89, 93 | Buildable parts: A-modules (`GModule<F> = {dim, Vec<Matrix<F>>}`), Meataxe over finite fields; ch90/91 (K[G]-modules, characters) are homed in `rustmath-groups` → **Deferred**. φ-modules (93) need Laurent series over finite fields. |

## 5b. Deferred (off-limits until IGP24/dessin land)
`rustmath-polynomials` (23, 24, 46, 105–111 Gröbner/commutative algebra/invariant theory/differential rings),
`rustmath-numberfields` (34–39, 44), `rustmath-quadraticforms` (32, 33, 86, 29-part, 119),
`rustmath-groups` (57–73 general group theory, 66 DBs, 90, 91), `rustmath-curves` (114),
`rustmath-numerical` (159 → rerouted). Pick up after those branches merge to `main`.

---

## 6. Cross-cutting coordination tracks (central design, NOT parallel fan-out)
These are duplicate-type unifications the survey surfaced; each needs one decision, done
centrally (not by independent crate workers), and some touch off-limits crates (partial defer):
- **`Z/mZ`**: `integers::ModularInteger` vs `finitefields::IntegerMod` vs `rings::quotient_ring` → one canonical `Integers(m)` Parent.
- **p-adics**: `rustmath-padics` vs `rings::padics` → unify before extensions.
- **Character/CharacterTable/Representation**: `groups::representation` vs `combinatorics::symmetric_group_representations` (groups side deferred).
- **Algebra trait**: `core::Algebra` vs `algebras::traits::Algebra` → reconcile.
- **num-\* → RustMath numerics** normalization pass for `modular` and `ellipticcurves` (prerequisite refactor).

---

## 7. Status
- Foundation decisions locked; coverage map complete (survey, 27 domains).
- **Next:** Wave 0 foundation (core traits → reals → complex), one PR, reviewed before Wave 1.
- Detailed per-crate backlogs: `magma-port-survey` workflow result (`.build_backlog` per domain).
