# M23 BELYI → DESCENT CONIC — RustMath module specification (approaches 1–4)

**Author-context (2026-07-02):** Derived from the M₂₃ campaign
(`/home/john/inverse_galois/M23/m23_hidden_law.tex` + `dessin_build/`). Governs
what must exist in RustMath — **ported from the MAGMA handbook where a source
exists, written natively where MAGMA has none** — to carry the
M₂₄→M₂₃ point-stabiliser portal to a decision.

Companion SSOTs: `MASTER_PORT_PLAN.md` (MAGMA ch 17–159 → crates) and
`DESSIN_REFACTOR_PLAN.md` (dessin_engine → existing crates). This spec is the
**math-goal view**: it slices those plans by the *one computation we need* and
records, per approach, the exact module list.

---

## REVISION 2 (2026-07-02) — review-note corrections + build-state discovery

Two things changed the plan:

**A. What already exists in RustMath (verified by inspection).** The DESSIN
refactor already landed the *entire decision half and verification pipeline*:
- `rustmath-quadraticforms`: `hilbert::hilbert_symbol`, `quaternion::{QuaternionAlgebra,BrauerClass}`,
  `conic::{DiagonalConicQ, ConicBrauerReport, Verdict<T>, MathStatus, VerdictKind}`,
  `ternary::{TernaryForm, diagonalize}` — **incl. the `(-1,-1)→x²+y²+z²`, ram {2,∞} gate.**
- `rustmath-curves/src/belyi/`: `pipeline` (`decide_from_solved_cover`, `assemble_2_12_5_homotopy`,
  `g6_mueller_gate`, `uniform_law`), `pinned` (`pinned_system_2_12_5`, `psi`, `p_star`),
  `encode` (`GenusZeroBelyiAnsatz`), `monodromy` (`Permutation`, genus), `audit`
  (`frobenius_types`, `classify_m24`, `audit_m23_residual`, `M23Witness`), `bad_locus`,
  `descent` (`mobius_from_three_pairs`, `QuaternionClassQ::to_conic`), `verify`, `bridge`.

So **M1–M4 (P0) are DONE**; the remaining gap is precisely the campaign's wall:
**produce a solved cover** (`ExactBelyiCover`) to feed `decide_from_solved_cover`.
The port work refocuses on the **construction half + the typed gate**.

**B. Review-note corrections (adopted; supersede §0/§3/§6 where they conflict).**
1. **Claim boundary.** A conic ℚ-point is *necessary, not sufficient*. Portal
   realization also requires: exact Belyi identity, ramification `2⁸1⁸·12²·5⁴1⁴`,
   monodromy = M₂₄, rational point **outside the bad locus `Z_C`**, and residual
   degree-23 Galois group = M₂₃. Encode this as a **type pipeline**
   `NumericalCoverCandidate → ExactCover → VerifiedM24Cover → DescentConic → PortalVerdict`.
2. **`HasRationalPoint=false` ⇒ `VerdictKind::LocallyEmpty`** (a *local* obstruction
   for *this portal/component*), never a global M₂₃/ℚ negative.
3. **Field of moduli ≠ field of definition.** Recognize over a coefficient field L;
   compute the Galois descent cocycle `[g_σ]∈H¹(Gal(L/ℚ),PGL₂)`; its image in
   `Br(ℚ)[2]` is the conic. For `L=ℚ(√δ)` with `ĝ_σ·σ(ĝ_σ)=βI`, class `=(δ,β)`.
4. **Stage the conic port.** P0 = diagonal ternary → quaternion → Hilbert → verdict
   (done). Simon `MinimalModel` / Holzer / parametrization are P1/P2 (**not** blockers).
5. **Approach 3 renamed:** *KMSV/Sijsling–Voight* — **3A** source-port (if licence
   permits) or **3B** clean-room from the papers. No item assumes package source.

**Revised sequencing (supersedes §6):**
- **P0 (done):** hilbert · quaternion · diagonal conic · `Verdict` types · quadratic
  cocycle→quaternion. Gate: `(-1,-1)↦x²+y²+z²`, Ram={2,∞}. *(verify green.)*
- **P1 (next):** exact factorized Belyi-identity gate · permutation/genus/ramification
  audit *(exists — wire it)* · **4E flag triangulation** *(the fix to the campaign's
  circle-packing blocker; note §9)* · **bounded-frame LM + true-root detector**
  *(the cheap construction gamble; note §11–12 + the detector that unmasked the
  spurious LM minimum)*.
- **P2:** spherical circle packing on the 4E flags · robust root/evaluation Newton ·
  robust monodromy tracker (fixes near-coincident-root failures).
- **P3:** KMSV reimplementation (3B) / package-port (3A); homotopy only if P2 stalls.

All new construction modules land in `rustmath-curves/src/belyi/` (the reserved
dessin home) unless a piece is genuinely generic (then `rustmath-numerical`),
keeping everything inside the firewall and beside the existing dessin code.

---

## 0. The goal, in one line

Realise M₂₃/ℚ via the regular M₂₄/ℚ(t) cover whose degree-24 M₂₃-fixed field is a
genus-0 curve X; decide whether X ≅ ℙ¹_ℚ. Concretely: **construct the three
`[2,12,5]` M₂₄ Belyi maps** `φ = A²B/(λR⁵S)` (passport `2⁸1⁸ · 12² · 5⁴1⁴`),
**recognise each over its field of moduli**, **read the descent conic
(δ,β) ∈ Br(ℚ)[2]**, and **decide ℚ-points by Hilbert symbols**. Any conic with a
ℚ-point ⇒ M₂₃/ℚ.

The chain has a **construction half** (get the cover numerically — the hard,
mostly non-MAGMA part) and a **decision half** (recognise + conic + Hilbert — the
portable MAGMA part). All four approaches share the decision half; they differ
only in the construction half.

```
        CONSTRUCTION HALF (approach-specific)          DECISION HALF (shared)
 dessin ─▶ conformal/algebraic start ─▶ Newton ─▶ cover ─▶ recognise ─▶ conic ─▶ Hilbert ─▶ verdict
 (§4.1 circle pack | §4.2 bounded LSQ | §4.3 Belyi pkg | §4.4 homotopy)     (§3.1)  (§3.1)  (§3.2)
```

---

## 1. The port boundary (what MAGMA gives us, what it does not)

| Capability | In MAGMA handbook? | Consequence for RustMath |
|---|---|---|
| Numerical Belyi map / dessins d'enfants | **No** (only tangential mentions: Ch 112 Schemes; monodromy appears in Ch 126/133/135 in other senses) | The construction half is **original code**, not a chapter port. Approach 3 ports MAGMA's *separate* Belyi **package** (Sijsling–Voight / KMSV), which is not in the ch 17–159 handbook. |
| Circle packing (Thurston/Collins–Stephenson) | **No** | Original. No source to port. |
| Homotopy continuation / polyhedral (BKK) start systems | **No** (Gröbner ch 105/106 is symbolic; mixed volume touches ch 143) | Original numerical solver; only the mixed-volume/Newton-polytope combinatorics have a MAGMA cognate (ch 143/46). |
| Conics: rational-point decision over ℚ | **Yes — Ch 119**, fully (Simon [Sim05], Hasse–Minkowski, Holzer [CR03]) | **Port directly.** This is the verdict engine. Highest value. |
| Hilbert symbols / quaternion algebras / Brauer | **Yes — Ch 86** (+ quadratic forms Ch 32/33) | Port. |
| Permutation-group ID (is ⟨σ₀,σ₁⟩ = M₂₄?) | **Yes — Ch 58** (+ almost-simple ID Ch 65, characters Ch 91) | Port (much already targeted by `rustmath-groups`/`igp24`). |
| Algebraic-number recognition + Galois-group cross-check | **Yes — Ch 34/37/38** (`GaloisGroup`), arb-prec ℂ Ch 25 | Port. |
| Arbitrary-precision ℝ/ℂ, matrices, polynomials | **Yes — Ch 25 / 26 / 23–24** | Wave-0 + leaf crates (already planned). |

**Net:** the *decision half* is a clean, high-value MAGMA port (§3). The
*construction half* is mostly native numerical engineering (§4) sitting on ported
foundations (arb-prec ℂ, dense linear algebra, polynomials).

---

## 2. Crate homes (all inside the DESSIN/IGP24 firewall — §"OFF-LIMITS" of MASTER_PORT_PLAN)

Every module below lands in a crate already reserved by the active dessin/IGP24
workers, so this spec is **not** a port-worker fan-out; it is the work plan for
those reserved crates.

| Crate | Role in this computation |
|---|---|
| `rustmath-complex`, `rustmath-reals` | arb-prec ℂ/ℝ, `Ball` (all numerics) — Wave-0 foundation |
| `rustmath-numerical` | Newton/least-squares, analytic continuation, homotopy, circle packing, mixed volume |
| `rustmath-polynomials` | dense univariate ℂ[x], resultants, `A²B−λR⁵S` assembly |
| `rustmath-graphs` | ribbon graph / dessin combinatorics, Tutte & spectral embeddings |
| `rustmath-groups` | monodromy permutation group, transitivity, M₂₄ identification |
| `rustmath-numberfields` | `algdep`/LLL recognition, field of moduli, `GaloisGroup` cross-check |
| `rustmath-curves` | **the conic**: `Conic`, `HasRationalPoint`, Legendre reduction (Ch 119) |
| `rustmath-quadraticforms` | Hilbert symbols, quaternion Brauer class, local solubility |
| `rustmath-igp24` | orchestration: passport → construct → recognise → conic → verdict |

---

## 3. DECISION HALF — shared modules (needed by ALL of 1–4). Port these first.

These are the payoff and the cheapest, most certain wins: they are genuine MAGMA
ports with published algorithms, and they turn *any* constructed cover into a
verdict. **Recommended to build regardless of which construction approach wins.**

### 3.1 Descent conic + rational-point decision — **Ch 119** → `rustmath-curves`
The heart of the verdict. Port Simon's algorithm and the local-global machinery.

| MAGMA intrinsic (Ch 119) | RustMath target | Algorithm to port |
|---|---|---|
| `Conic([a,b,c])`, `Conic(M)`, `IsConic(S)` | `Conic::from_diagonal`, `Conic::from_matrix` | direct construction |
| `ReducedLegendreModel(C)`, `ReducedLegendrePolynomial(C)` | `conic::reduced_legendre` | diagonalise, square-free & pairwise-coprime reduction |
| `MinimalModel(C)` | `conic::minimal_model` | Simon minimisation, prime-by-prime discriminant reduction **[Sim05]** |
| `HasRationalPoint(C)` → bool, point | `conic::has_rational_point` | **[Sim05]** over ℚ; Hasse–Minkowski, only primes \| disc, Hensel |
| `RationalPoint(C)`, `Random(C:Bound,Reduce)` | `conic::rational_point` | parametrisation eval |
| `IsReduced(p)`, `Reduction(p)` | `conic::holzer_reduce` | Holzer bound, Mordell/Cremona reduction **[CR03]** |
| `Parametrization(C[,p])`, `Conic(C)` (genus-0 curve→conic) | `conic::parametrize` | anti-canonical / 2-uple embedding **[Sim05]** |

Notes: over ℚ the algorithm is complete (Simon). We need exactly ℚ (and possibly
a quadratic field if recognition lands there → the number-field Lagrange variant,
also in Ch 119). `BadPrimes` / `IsLocallySoluble` give the ramified places of the
Brauer class = the conic's obstruction. **This module alone converts the known
Granboulan conic `x²+y²+z²` and each new `[2,12,5]` conic into the M₂₃/ℚ verdict.**

### 3.2 Hilbert symbols / Brauer class — **Ch 86 (+ 32/33)** → `rustmath-quadraticforms`
The conic (δ,β) ↦ ℚ-solubility is a product of Hilbert symbols; already partly
present as `dessin_engine`'s quaternion/`quadraticforms` code (harvest per
DESSIN_REFACTOR_PLAN).

| MAGMA (Ch 86/32) | RustMath target | Purpose |
|---|---|---|
| `HilbertSymbol(a,b,p)` | `hilbert::symbol` | local invariant at each place |
| `QuaternionAlgebra<Q|a,b>`, `IsSplit`, `RamifiedPrimes` | `quaternion::brauer_class` | (δ,β) as a Brauer(ℚ)[2] element; its ramified set |
| `IsIsotropic`, `IsLocallySolvable` (ternary form) | `qform::is_isotropic` | conic ↔ ternary quadratic form bridge |

### 3.3 Monodromy group = M₂₄ verification — **Ch 58 (+ 65, 91)** → `rustmath-groups`
Every construction must be *certified* to have monodromy M₂₄ (not another
transitive passport group). Given ⟨σ₀,σ₁⟩ from analytic continuation:

| MAGMA (Ch 58/65/91) | RustMath target | Purpose |
|---|---|---|
| `sub<Sym(24)|σ0,σ1>`, `IsTransitive`, `IsPrimitive`, `Order` | `perm::group_from_gens`, `is_transitive` | basic monodromy sanity |
| `CompositionFactors`, `IsIsomorphic(G, M24)` / name (Ch 65) | `group::identify_simple` | confirm the group is **M₂₄** |
| `ClassMap`, structure-constant count (Ch 91) | `chartable::structure_constant` | reproduce the "3 covers" count (2A,12B,5A) |
| `IsConjugate` of triples in S₂₄ | `perm::simultaneous_conjugacy` | match a numeric cover to a target passport triple / detect the Galois orbit |

### 3.4 Recognition + field of moduli + Galois cross-check — **Ch 34/37/38 (+ 25)** → `rustmath-numberfields`
Turn the high-precision numeric cover into exact algebraic data and get an
independent check.

| MAGMA (Ch 34/38/25) | RustMath target | Purpose |
|---|---|---|
| `PowerRelation`/`MinimalPolynomial`(ℂ approx) ~ `algdep` (LLL) | `recognize::algdep` | each coefficient → minimal polynomial → field of moduli |
| `NumberField`, `Compositum`, `OptimizedRepresentation` (Ch 34/37) | `numberfield::*` | assemble the field the cover is defined over (ℚ? quadratic? cubic Galois orbit?) |
| `GaloisGroup(f)` (Ch 38) | `galois::galois_group` | **independent** verification that the resolvent field realises M₂₃ — the publication gate |
| arb-prec ℂ, precision-carrying parent (Ch 25) | `rustmath-complex` | ≥150-digit coefficients feed LLL |

---

## 4. CONSTRUCTION HALF — per-approach module lists

Legend: **[PORT]** = MAGMA chapter port · **[NATIVE]** = original (no MAGMA source)
· **[PKG]** = port of MAGMA's out-of-handbook Belyi package.

### 4.1 Approach 1 — Circle packing (conformal embedding) — *recommended construction*
Rationale: the campaign proved harmonic starts (Tutte/spectral) give non-root
minima; a *conformal* start is required, and circle packing is the KMSV-standard.

| Module | Kind | Crate | Notes |
|---|---|---|---|
| Ribbon graph / dessin from (σ₀,σ₁,σ_∞) | [NATIVE] | `rustmath-graphs` | done in `dessin_build/ribbon.py`; port + the **4E half-edge / flag triangulation** (the piece that blocked us — faces need the two-sides-per-edge model, not the naive 2E flags) |
| Spherical circle packing (Thurston/Collins–Stephenson) + Möbius normalisation | [NATIVE] | `rustmath-numerical` | interior-angle-sum→2π radius solve on the sphere; the genuinely hard new component |
| arb-prec ℂ, `Ball` | [PORT Ch 25] | `rustmath-complex` | positions + certified refinement |
| Dense ℂ linear solve (Jacobian) | [PORT Ch 26] | `rustmath-matrix`/`numerical` | Newton step |
| Newton on the **root/evaluation** system (well-conditioned form validated in `solve_roots.py`) | [NATIVE] | `rustmath-numerical` | unknowns = ramification points; identity by weighted evaluation (avoids Wilkinson) |
| Univariate ℂ[x] assembly of `A²B−λR⁵S` | [PORT Ch 23] | `rustmath-polynomials` | — |
| → then §3 (recognise/conic/Hilbert/monodromy) | — | — | — |

### 4.2 Approach 2 — Bounded-frame multi-restart least-squares — *cheap gamble*
Rationale: put both order-12 points finite (no ∞ spread ⇒ no Wilkinson);
many perturbed restarts; keep only true roots (now detectable: residual ~1e-14 vs
spurious ~1e-9). Lowest new-code cost; works only if the true basin is near a
harmonic start.

| Module | Kind | Crate | Notes |
|---|---|---|---|
| Bounded-frame system `A²B−λR⁵S = c(x−w₀)¹²(x−w₁)¹²` (evaluation form) | [NATIVE] | `rustmath-numerical` | reformulation of the validated root system |
| Trust-region / Levenberg–Marquardt least-squares | [NATIVE] | `rustmath-numerical` | (scipy-`least_squares` analogue; no MAGMA source) |
| Deterministic multi-start generator (Tutte + spectral + perturbations) | [NATIVE] | `rustmath-graphs`/`numerical` | Tutte/spectral already in `tutte.py`/`spectral_embed.py` |
| arb-prec ℂ, matrices, polynomials | [PORT Ch 25/26/23] | complex/matrix/polynomials | — |
| **True-root detector** (Newton non-divergence test) | [NATIVE] | `rustmath-numerical` | the diagnostic that unmasked the spurious LM min |
| → then §3 | — | — | — |

### 4.3 Approach 3 — Port MAGMA's Belyi package — *most direct if source available*
Rationale: MAGMA (Sijsling–Voight, building on KMSV) has production Belyi-map
computation for exactly this passport style. This is a **[PKG]** port (out of
the ch 17–159 handbook), so it is a separate acquisition/port task.

| Module | Kind | Crate | Notes |
|---|---|---|---|
| `BelyiMap` / numerical dessin solver (KMSV) | [PKG] | `rustmath-numerical`/`igp24` | port the package's circle-packing + Newton pipeline (subsumes §4.1) |
| Puiseux / Newton polygon for degenerate starts | [PORT Ch 46] | `rustmath-powerseries`/`numerical` | Granboulan-style degenerate-cover start |
| Function-field / curve plumbing (cover as map of curves) | [PORT Ch 114/42] | `rustmath-curves` | if porting the algebraic (not just numeric) path |
| Gröbner fallback for small algebraic pieces | [PORT Ch 105/106] | `rustmath-polynomials` | exact solve where degree permits |
| → then §3 | — | — | — |

Dependency risk: requires obtaining the Magma Belyi package source; not in the
local handbook. Treat as "acquire, then port," parallel to §4.1 as insurance.

### 4.4 Approach 4 — Homotopy continuation + monodromy filter — *most general, heaviest*
Rationale: enumerate the whole passport by continuation, filter to M₂₄. Needs a
*reliable* tracer (ours failed on near-coincident roots) and the passport may hold
thousands of covers (A₂₄/S₂₄ dominate).

| Module | Kind | Crate | Notes |
|---|---|---|---|
| Polyhedral / total-degree homotopy start systems | [NATIVE] | `rustmath-numerical` | HomotopyContinuation.jl/PHCpack analogue |
| Mixed volume / BKK (Newton polytopes) | [PORT-ish Ch 143] | `rustmath-numerical` (+`geometry`) | convex-polytope mixed volume; ch 143 is the cognate |
| Path tracking with singular endgame (extended precision) | [NATIVE] | `rustmath-numerical` | the earlier Julia campaign's pain point |
| **Robust numerical monodromy** (analytic continuation of all 24 sheets, handles near-coincident roots) | [NATIVE] | `rustmath-numerical` | fixes `dessin_build/monodromy.py`'s tracking failures |
| Passport filter: cycle types + ⟨σ₀,σ₁⟩ = M₂₄ | [PORT Ch 58/65] | `rustmath-groups` | = §3.3, applied at scale |
| → then §3 | — | — | — |

---

## 5. Consolidated module manifest

| # | Module | Kind | MAGMA ch | Crate | Approaches | Priority |
|---|---|---|---|---|---|---|
| M1 | Conic `HasRationalPoint`/Simon/Legendre/Holzer | PORT | 119 | curves | all | **P0** |
| M2 | Hilbert symbol / quaternion Brauer class | PORT | 86,32,33 | quadraticforms | all | **P0** |
| M3 | Monodromy group ID = M₂₄ | PORT | 58,65,91 | groups | all | **P0** |
| M4 | algdep/LLL recognition + field of moduli | PORT | 34,37,25 | numberfields | all | **P0** |
| M5 | `GaloisGroup` cross-check (publication gate) | PORT | 38 | numberfields | all | P1 |
| M6 | arb-prec ℂ/ℝ + `Ball` | PORT | 25 | complex,reals | all | **P0** (Wave-0) |
| M7 | dense ℂ linear algebra (Jacobian) | PORT | 26 | matrix,numerical | 1,2,4 | P1 |
| M8 | univariate ℂ[x] assembly / resultants | PORT | 23 | polynomials | all | P1 |
| M9 | ribbon graph + 4E flag triangulation | NATIVE | — | graphs | 1,(2) | P1 |
| M10 | spherical circle packing + Möbius norm | NATIVE | — | numerical | 1,3 | P2 (hard) |
| M11 | root/evaluation Newton (well-conditioned) | NATIVE | — | numerical | 1,2 | P1 |
| M12 | trust-region/LM least-squares + true-root detector | NATIVE | — | numerical | 2 | P1 |
| M13 | Belyi package (KMSV) port | PKG | — | numerical,igp24 | 3 | P2 (acquire) |
| M14 | Puiseux/Newton-polygon degenerate start | PORT | 46 | powerseries,numerical | 3 | P2 |
| M15 | homotopy continuation + endgame | NATIVE | — | numerical | 4 | P3 |
| M16 | mixed volume / BKK | PORT-ish | 143 | numerical,geometry | 4 | P3 |
| M17 | robust numerical monodromy tracer | NATIVE | — | numerical | 4,(all verify) | P2 |
| M18 | Tutte + spectral embeddings | NATIVE | — | graphs | 1,2 | done (harvest) |

---

## 6. Recommendation & sequencing

1. **Build the decision half first (M1–M6, all P0).** It is pure MAGMA port with
   published algorithms, it is shared by every approach, and it *immediately*
   reproduces the known Granboulan verdict (`x²+y²+z²` anisotropic) as a
   regression test — validating M1/M2 before any new cover exists.
2. **Construction: pursue §4.1 (circle packing, M9–M11) and §4.3 (acquire+port
   the Belyi package, M13) in parallel** — §4.1 is the principled native build,
   §4.3 is insurance if the package source is obtainable. Keep §4.2 (M12) as a
   fast cheap probe run *before* committing to M10 (it may just work in the
   well-conditioned bounded frame).
3. **§4.4 (homotopy, M15–M17) only if 1/3 stall** — heaviest, and M17 (robust
   monodromy) is independently useful for *verifying* covers from any approach.
4. **Gate every result on M3 (monodromy = M₂₄) and M5 (`GaloisGroup`)** before any
   M₂₃/ℚ claim.

The cheapest path to a *verdict-capable* system is M1+M2+M6 (conic + Hilbert +
arb-prec ℂ): with those, one clean cover from any construction method closes the
problem.

---

## 7. Discipline (inherited from MASTER_PORT_PLAN §0 and DESSIN_REFACTOR_PLAN)

- All crates here are **firewall crates** (active dessin/IGP24 workers). This spec
  is their work plan, not a port-worker fan-out; no outside worker touches them.
- **Status honesty:** a bounded search that finds no ℚ-point is **UNRESOLVED**, not
  "M₂₃ ∤ ℚ". Only `HasRationalPoint = false` over ℚ (Hasse–Minkowski complete over
  ℚ) is a *decision* — and it decides only *this portal*, not the IGP for M₂₃.
- **Cite sources** in every new file: MAGMA chapter for [PORT], the KMSV/Granboulan
  paper for [NATIVE] numerics, `dessin_build/*.py` for harvested prototypes.
- **Reference implementations to harvest:** `/home/john/inverse_galois/M23/dessin_build/`
  (`ribbon.py`, `solve_roots.py`, `tutte.py`, `spectral_embed.py`, `refine_roots.py`,
  `monodromy.py`) and `dessin_engine` (conic/descent/quaternion, per DESSIN_REFACTOR_PLAN).
