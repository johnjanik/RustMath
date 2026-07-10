# lava MVP — a Rust computational replacement for MAGMA

**Goal.** Replace the MAGMA calls our IGP24 inverse-Galois pipeline depends on
(degree-24 Galois-group verification, number-field invariants, polynomial
factorization, resolvents, local/ramification data) with a self-contained Rust
stack, reusing the existing **RustMath** crates wherever possible and adding only
the genuinely missing pieces.

**Scope.** This is an *MVP*, not all of MAGMA. MAGMA's handbook is 160+ chapters
(Lie theory, coding theory, modular forms, full algebraic geometry, graphs, …);
we target only the computational-number-theory / Galois / group-theory core that
the competition actually exercises. Out-of-scope chapters are listed in §7.

**Starting point.** `lava` today is *Magma-language* tooling — `lava-core` +
`lava-cli` wrap `tree-sitter-magma` and `topiary` to format/highlight Magma source.
That is the **front-end** (§6, Module L7). The MVP adds the **compute back-end**
underneath it, drawing on RustMath.

---

## 1. Strategy: reuse RustMath first

RustMath (`/Users/john/Documents/rm-galois/`, ~70 crates on `num-bigint`) already
implements most of the MVP's lower layers as working, tested code — including the
crown-jewel **degree-24 transitive-group atlas** and a production **`rustmath-igp24`
CLI** that already does native degree-24 Galois ID without MAGMA. The job is
therefore mostly **integration + a few hard new modules**, not greenfield.

Reuse verdicts used below:
- **REUSE** — exists and is production-usable; wire it in as-is.
- **EXTEND** — partially present; finish/strengthen it.
- **BUILD** — absent or stub; new implementation (or external bridge) required.

---

## 2. Architecture — dependency layers

```
L0  Exact kernel        ℤ ℚ ℤ/n 𝔽_p 𝔽_q          (foundation)
        │
L1  Polynomials + linear algebra + lattices
        │
L2  Number fields │ local fields/p-adics │ finite-field extensions │ function fields
        │
L3  Groups: permutation groups + degree-24 transitive atlas
        │
L4  Galois engine: Frobenius sieve → resolvents → Stauduhar descent → label ID
        │
L5  Construction engines: signature seeds, CRT-LLL, Newton templates, Mestre, covers
        │
L6  Databases + certificates + replay (target-pair cube, ghost ledger, provenance)
        │
L7  Front-end: Magma-syntax parser/DSL + CLI + orchestration   ← lava today lives here
```
Each layer depends only on those above it. The MVP can be built and trusted
bottom-up; L0–L3 are almost entirely REUSE.

---

## 3. Module list (MVP) — module → MAGMA chapter → RustMath → status

### L0 — Exact arithmetic kernel
| Module | MAGMA ch. | Key algorithms | RustMath crate | Status |
|---|---|---|---|---|
| Integers ℤ, modular ℤ/n | 17, 18, 19 | Euclid/Lehmer gcd, CRT, Miller–Rabin, Pollard ρ/p−1, ECM, MPQS | `rustmath-integers` | **REUSE** |
| Rationals ℚ, continued fractions | 18 | exact field arith, CF convergents, rational reconstruction | `rustmath-rationals` | **REUSE** |
| Finite fields 𝔽_p, 𝔽_{p^n} | 21, 48 | Berlekamp; Conway lookup; discrete log (BSGS) | `rustmath-finitefields` | **EXTEND** (𝔽_{p^n} reduction simplified; add Cantor–Zassenhaus + irreducible-poly search) |
| Number-theory helpers | — | Bernoulli, quadratic forms, Jacobi/Legendre, sqrt mod p | `rustmath-integers`, `rustmath-numbertheory` | **REUSE** |

### L1 — Polynomials, linear algebra, lattices
| Module | MAGMA ch. | Key algorithms | RustMath crate | Status |
|---|---|---|---|---|
| Univariate poly + factorization | 23 | Zassenhaus, Berlekamp, Hensel lift, squarefree, subresultant `resultant`/`discriminant` (modular-CRT fast path) | `rustmath-polynomials` | **REUSE**; **EXTEND** with **van Hoeij** (large-degree ℚ-factoring) + Cantor–Zassenhaus |
| Real-root isolation, Sturm | 23, 25 | Sturm sequences, Descartes, Aberth/Durand–Kerner, `real_roots.rs` | `rustmath-polynomials`, `rustmath-numerical` | **REUSE**/**EXTEND** |
| Multivariate poly + Gröbner | 24, 105, 106 | Buchberger, **F4/F5**, FGLM, elimination | `rustmath-polynomials` (`groebner.rs` skeleton) | **BUILD** (needed only for resolvent/Mestre elimination) |
| Real & complex fields | 25 | interval/ball arithmetic, root isolation | `rustmath-numerical`, `rustmath-complex` (rug) | **REUSE** |
| Dense/sparse linear algebra | 26, 27, 28 | Bareiss/fraction-free, LU/QR/SVD, char-poly, Wiedemann | `rustmath-matrix` | **REUSE** |
| Lattices: LLL / HNF / SNF | 30, 31 | **LLL**, Hermite & Smith normal form, (add **BKZ**, Fincke–Pohst enumeration) | `rustmath-matrix` (`lll.rs`, `integer_forms.rs`) | **REUSE**; **EXTEND** (BKZ/enumeration) |

### L2 — Number fields, local fields, function fields
| Module | MAGMA ch. | Key algorithms | RustMath crate | Status |
|---|---|---|---|---|
| Number fields & orders (𝒪_K, Δ_K, signature, integral basis, polred/polredabs) | 34, 37 | Pohst–Zassenhaus **Round 2**, Durand–Kerner→T2-lattice→LLL polred, Faddeev–LeVerrier | `rustmath-numberfields` | **REUSE** (core); **EXTEND** |
| Ideal arithmetic & prime decomposition | 34, 37 | HNF ideals, **Montes**/Round-2 p-maximalization, different/conductor | `rustmath-numberfields` (`different.rs`, `round2.rs`) | **EXTEND** |
| Class groups, units, regulator | 34, 39 | Buchmann subexponential, relation collection | `rustmath-numberfields` (partial) | **BUILD** (secondary; only for class-field routes) |
| Galois theory of number fields (subfields, automorphisms, splitting field) | 38 | Klüners–Pohst subfields, Trager splitting field, fixed fields | `rustmath-numberfields` + `rustmath-groups` | **EXTEND** (subfields/closure partial for deg≥4) |
| Local fields, p-adics, Newton polygons, ramification | 45, 46, 47, 51 | Hensel, **Newton-polygon/Montes/OM**, Krasner, Panayi, single-factor lifting, (e,f) invariants | `rustmath-padics` (thin), `rustmath-polynomials` (`newton.rs`, `padic_factor.rs`) | **EXTEND**/**BUILD** (Montes/OM is the big local-template gap) |
| Galois rings | 48 | Galois-ring arithmetic over ℤ/p^k | `rustmath-finitefields` | **EXTEND** |
| Function fields ℚ(t)[x] / K(C) | 41, 42 | Trager, Newton–Puiseux, specialization, branch locus | — | **BUILD** (needed for regular-cover scanning) |
| Class field theory | 39, 43 | ray class groups, Artin map, conductor–discriminant | — | **BUILD** (secondary) |

### L3 — Groups
| Module | MAGMA ch. | Key algorithms | RustMath crate | Status |
|---|---|---|---|---|
| Permutation groups | 57, 58 | Schreier–Sims, BSGS, orbits/stabilizers, conjugacy, blocks, primitive tests | `rustmath-groups` (degree-24 specialized); `rustmath-interfaces` GAP bridge | **REUSE** (deg-24); **BUILD**/bridge (general Schreier–Sims) |
| **Degree-24 transitive-group atlas** | 58, 66 | all 25 000 groups + generators + cycle-type support, block systems, involution profiles | `rustmath-groups` (`transitive24.rs` + `data/transitive_24.jsonl`, `..._cycletypes.jsonl`) | **REUSE** ★ crown jewel |
| Automorphism & character theory | 67, 91 | Dixon–Schneider character table, permutation characters, involution fixed-counts | — / GAP bridge | **BUILD**/bridge (involution profiler for signature admissibility) |
| Matrix / FP / soluble groups | 59–63, 70–72 | MeatAxe, Todd–Coxeter, Knuth–Bendix | `rustmath-interfaces` (GAP) | **BUILD**/bridge (secondary; branch-cycle/rigidity work) |

### L4 — Galois group engine
| Module | MAGMA ch. | Key algorithms | RustMath crate | Status |
|---|---|---|---|---|
| Frobenius shadow sieve | 38, 58 | Dedekind/Chebotarev, mod-p factor → cycle type, candidate filtering, blind-cell ID | `rustmath-groups` (`CycleTypeSupport`), `rustmath-polynomials` (`padic_factor::cycle_type`) | **REUSE** ★ |
| Root labeling (ℂ + p-adic) | 38 | Aberth/Durand–Kerner, Hensel root lift, symmetric reconstruction | `rustmath-numerical`, `rustmath-polynomials` | **REUSE**/**EXTEND** |
| Resolvent construction | 38 | Lagrange/Stauduhar relative invariants, block & k-subset orbit signatures | `rustmath-polynomials` (`resolvent.rs`), `rustmath-groups` (`ksubset_orbits.rs`) | **REUSE** |
| **Stauduhar descent (generic)** | 38 | Stauduhar + **Fieker–Klüners/Geißler–Klüners**, Tschirnhausen, subgroup-tree backtracking | partial (deg-24 narrowing only) | **BUILD** ★ hardest module |
| Galois label identification G↦24Tt | 38, 58 | staged fingerprint (\|G\|, blocks, μ_G, stabilizers), DB lookup | `rustmath-groups`, `rustmath-igp24` | **REUSE** |
| Exact oracle (cross-check) | — | Fieker–Klüners via **OSCAR/Hecke** | external (`oscar_bridge`) | **REUSE** (bridge) |

### L5 — Construction engines (currently Sage; PORT to Rust)
| Module | MAGMA ch. | Key algorithms | source today | Status |
|---|---|---|---|---|
| Signature seed generator | — | Sturm-certified real-root seeds, Hermite–Biehler | `frobenius/generate.sage` | **BUILD/PORT** |
| CRT-LLL coefficient lift | — | CRT + Garner + rational reconstruction + LLL near a seed | Sage | **BUILD/PORT** |
| Newton-polygon template constructor | 46 | slope prescription, residual polys, Hensel certification | Sage/`newton.rs` | **BUILD** |
| Mestre–Vila A₂₄ family | — | square-disc pencils, `P′H−PH′=R²` solver, even-degree descent | `frobenius/mestre*.sage`, `mestre.rs` | **BUILD/PORT** |
| Regular-cover specialization scanner | 38, 42 | Hilbert irreducibility specialization, bad-value avoidance | Sage | **BUILD** |
| Real-fiber analyzer | 25 | Sturm over parameter intervals, signature map t↦r | Sage | **BUILD** |
| Relative-fiber / directed constructor | — | `Res_y(g, X²−γ(y))`, block-top-group matching | `frobenius/directed_construct*.sage` | **BUILD/PORT** |

### L6 — Databases, certificates, replay
| Module | Key content | source today | Status |
|---|---|---|---|
| Target-pair cube | one row per admissible (24Tt, r) + status/method | `igp24-coverage/data/*.csv` | **REUSE**/formalize |
| Frobenius-blind atlas | μ_G partition, separator ledger | `frobenius/blind_classes.json` | **REUSE** |
| Ghost ledger | failed candidate → actual group, separator learned | (ad hoc) | **BUILD** |
| Certificate schema + replay | irreducibility/signature/disc-squareclass/Frobenius/Galois-label certs; deterministic recheck | — | **BUILD** ★ (credibility) |

### L7 — Front-end
| Module | Key content | RustMath/lava | Status |
|---|---|---|---|
| Magma-syntax parser / formatter | tree-sitter-magma + topiary | `lava-core`, `lava-cli` | **REUSE** (already built) |
| DSL / intrinsic dispatch | map Magma intrinsics (`GaloisGroup(f)`, `NumberField(f)`, …) → back-end calls | — | **BUILD** (thin) |
| Orchestration / batch / checkpointing | parallel search, content-addressed cache | `frobenius/*.py`, `run.sh` | **BUILD/PORT** |

---

## 4. External Rust crate dependencies

**Already used across RustMath (inherit these):**
- `num-bigint`, `num-integer`, `num-rational`, `num-traits` — the bignum/rational backend.
- `ndarray`, `rayon` — dense linear algebra + parallelism (`rustmath-matrix`).
- `rand` — primality/factoring randomization.
- `serde`, `serde_json` — group atlas + database serialization.
- `once_cell` / `lazy_static`, `thiserror`.
- `rug` (GMP/MPFR/MPC) — **only** in `rustmath-complex` (high-precision/ball complex).

**lava front-end (already present):** `tree-sitter`, `tree-sitter-magma`,
`topiary-core`, `clap`, `tokio`, `anyhow`, `tracing`.

**To add for the MVP:**
- A **Gröbner** capability — implement F4/F5 in `rustmath-polynomials`, or bind a crate.
- Optional `rustmath-interfaces` **GAP bridge** (feature-gated) for general
  permutation-group algorithms (Schreier–Sims, character tables) until native.
- External **OSCAR/Hecke** as an exact Galois oracle for cross-checking (already wired
  as `oscar_bridge.py`; keep as a verification dependency, not a runtime one).

**Bundled data (ship with the binary):**
- `transitive_24.jsonl` (~25 MB, 25 000 groups + generators)
- `transitive24_cycletypes.jsonl` (~32 MB, per-group cycle-type support)
- Conway polynomial table; the IGP24 coverage CSVs / blind-class atlas.

**Key decision — bignum backend.** RustMath is on **`num-bigint`** (pure Rust,
portable, already integrated). The wishlist's `IGPExactKernel` recommends wrapping
**GMP/FLINT** (via `rug`) for speed. Recommendation for the MVP: **stay on
`num-bigint`** (it already works end-to-end and `rustmath-igp24` is production-ready);
treat a GMP swap as a later, cross-cutting performance optimization behind the same
arithmetic trait — not an MVP blocker.

---

## 5. What must be built (the real work)

Everything else is REUSE/EXTEND. The genuinely new modules, in rough hardness order
(matching the wishlist's tiers):

1. **Generic Stauduhar descent** (L4) — the hardest. We have sound degree-24
   *narrowing* + resolvents, but not a general subgroup-descent driver. For the MVP
   the degree-24 atlas path + OSCAR oracle may suffice; a native Stauduhar is the
   stretch goal.
2. **p-adic local-factorization (Montes/OM)** (L2) — completes Newton-polygon
   templates and ramification certificates; current p-adics are thin.
3. **Construction engines** (L5) — port the Sage engines (signature seeds, CRT-LLL,
   Mestre, cover/real-fiber, directed constructor) to Rust.
4. **Gröbner (F4/F5)** (L1) — needed for resolvent/Mestre elimination.
5. **Certificate + replay** (L6) — independent rechecking for credible submissions.
6. **GF(p^n) completion, function fields, class groups** — fill-ins as routes demand.
7. **General Schreier–Sims / character theory** (L3) — or keep the GAP bridge for MVP.
8. **DSL intrinsic dispatch + orchestration** (L7) — glue lava's existing Magma
   parser to the back-end.

---

## 6. Phased build plan (reuse-first)

- **Phase 0 — Integrate (mostly REUSE).** Wire `rustmath-integers/rationals/
  polynomials/matrix/finitefields/numberfields/groups` + `rustmath-igp24` into a
  single `lava` workspace; expose them behind a stable API. This alone replaces most
  of our MAGMA verification calls (factorization, discriminants, signatures,
  Frobenius, degree-24 Galois ID).
- **Phase 1 — Diagnostics & databases (L6).** Formalize target-pair cube, blind
  atlas, ghost ledger, coverage maps.
- **Phase 2 — Local layer (L2).** Montes/OM p-adic factorization + Newton-polygon
  templates + ramification certs.
- **Phase 3 — Galois engine (L4).** Strengthen resolvents; prototype generic
  Stauduhar; keep OSCAR oracle as cross-check; certificate generation.
- **Phase 4 — Construction (L5).** Port Sage engines to Rust (signature/CRT/Mestre/
  cover/real-fiber/directed).
- **Phase 5 — Front-end (L7).** DSL dispatch over lava's Magma parser; orchestration.

---

## 7. Deliberately out of MVP scope

These MAGMA areas are not needed for IGP24 and are excluded (revisit only if a
specific route demands them): multivariate/commutative-algebra beyond elimination
(most of 105–113), algebraic geometry / schemes / curves at scale (114–127),
modular curves & forms / Hilbert & Bianchi forms (128–139) — *except* as external
sources of projective covers, Lie theory / Coxeter / root systems / Kac–Moody
(94–104), representation theory beyond involution characters (89–92), coding theory
& designs (150–157), graphs & combinatorics (142–149), quantum groups, braid/
automatic/semigroup machinery (73–77). RustMath has partial crates for several of
these (elliptic curves, modular forms, quadratic forms) that can be promoted later
if the projective/modular construction routes are pursued.

---

## 8. One-line summary

> **~70–80% of the MVP already exists in RustMath** (exact arithmetic, polynomial
> factorization, LLL/HNF/SNF, number-field invariants + polredabs, and the complete
> degree-24 transitive-group atlas with a working native Galois-ID CLI). lava's job
> is to **integrate those crates behind its existing Magma front-end** and **build
> the five hard new pieces**: generic Stauduhar descent, Montes/OM local
> factorization, the construction engines (port from Sage), Gröbner (F4), and a
> certificate/replay layer.
