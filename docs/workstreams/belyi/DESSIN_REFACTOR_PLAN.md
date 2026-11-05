# Dessin-engine → RustMath refactor plan (single source of truth)

**Decision (1 July 2026):** *Refactor first, then solve.* Port the proven,
tested `dessin_engine` implementations into **existing** RustMath crates (no new
crate), filling the stubs the surveys found. M23/Belyi logic folds into existing
crates. Once the primitives live in RustMath, build the λ-pinned encoder + the
parameter-homotopy solve there and run it.

Source crate to harvest from: `/home/john/inverse_galois/M23/dessin_engine`
(standalone, 101 tests green, deps only num-bigint/num-integer/num-traits/serde/
thiserror). It is the *reference implementation*; RustMath is the *destination*.

---

## Rules (every agent, no exceptions)

1. **Own exactly one crate.** Touch only files under that crate's directory. The
   only editable shared file is that crate's own `src/lib.rs` (one `pub mod` line
   per new module) and its own `Cargo.toml` (to add dependencies). **Never edit
   another crate.**
2. **New files only.** Do not modify existing module logic. If an existing module
   is a stub/placeholder, leave it and add a new module beside it (e.g. add
   `conic.rs`, don't rewrite `quadratic_form.rs`).
3. **Use RustMath foundation types, not dessin_engine's.** Where dessin_engine
   uses its own `Rat`/`BigInt` wrappers, target RustMath's existing
   `rustmath-integers` / `rustmath-rationals` (add as Cargo deps). Match the
   destination crate's existing conventions. Port the *logic and the tests*, not
   the private helper types.
4. **Keep the crate green.** `cargo test -p <your-crate>` must pass when you
   finish (build only your crate, not the whole workspace). Port dessin_engine's
   unit tests for the code you move, adapted to the new types.
5. **Cite sources.** In each new file's header, note the dessin_engine source
   path it was ported from.
6. **Status discipline stays.** Any certifying/semialgorithmic output keeps its
   "UNRESOLVED vs decided" honesty; a bounded search that finds nothing is never
   a decision. (Relevant to conic/recognize/verify.)
7. **Do not run the solve, do not touch `/home/john/inverse_galois`.** This phase
   is porting only.

---

## Wave 1 — leaf crates (fully parallel, independent)

### Agent Q — `rustmath-quadraticforms`  (greenfield: it has NO conic machinery)
Port the conic/descent reader. New files:
- `src/hilbert.rs` ← dessin_engine `src/hilbert.rs` (Hilbert symbols at odd p / 2 /
  ∞; `Place`; candidate ramified places).
- `src/ternary.rs` ← dessin_engine `src/quadratic_form.rs` (`TernaryForm`, exact
  3×3 congruence diagonalization).
- `src/conic.rs` ← dessin_engine `src/conic.rs` (`DiagonalConicQ`,
  `from_ternary_form`, `brauer_report`, Hasse `verdict`).
- `src/quaternion.rs` — quaternion algebra `(a,b)`, ramified places, Br(ℚ)[2]
  class (extract from conic's `brauer_report` if not already separable).
- Do **not** touch `quadratic_form.rs` (binary forms), `theta_series.rs`, etc.
- **Acceptance test:** the Hamilton quaternion `(-1,-1)` is anisotropic, ramified
  exactly at `{2,∞}` (this is the Müller M23 conic — the canonical gate).

### Agent N — `rustmath-numberfields`  (has rigorous module suite; toy OO layer)
Port element arithmetic + algebraic recognition. New files (do **not** edit the
toy OO `lib.rs`):
- `src/quadratic.rs` ← dessin_engine `src/quad_field.rs` (`QuadField`/`QuadElem`
  for ℚ(√δ): norm, inverse, conjugate σ(√δ)=−√δ) — the sugar constructor Survey 2
  recommended.
- `src/nf_elem.rs` ← dessin_engine `src/number_field.rs` (`PolyQ`, `NfElem`
  arithmetic incl. inverse via xgcd) — a *working* element layer beside the
  crate's stubbed OO `inv`.
- `src/recognize.rs` ← dessin_engine `src/recognize.rs` **plus**
  `exactification::recognize_complex_algebraic` (rational reconstruction + LLL
  minimal-polynomial recovery with the shortness gate). May use
  `rustmath-matrix::lll` or port dessin_engine's exact-rational LLL locally.
- **Acceptance test:** recognize a known quadratic irrational from a
  high-precision float; reconstruct a rational from a residue (CRT) example.

### Agent G — `rustmath-groups`  (strong; two gaps)
- Add permutation-group predicates absent from `src/permutation_group.rs`:
  `is_transitive`, `orbits`, `blocks`, `stabilizer`, `is_primitive` (adding
  methods to this crate's own file is allowed since you own the crate; prefer a
  new `src/perm_predicates.rs` with free functions if cleaner).
- `src/transitive23.rs` — degree-23 classifier via Burnside's prime-degree
  dichotomy (the 7 transitive groups C23, D23, F23=23:11, AGL(1,23), M23, A23,
  S23 have pairwise-distinct cycle-type sets; use the discriminant-square test to
  split ⊆A23 from the rest, then cycle-type-set matching; M23 pinned by its
  ~12-type fingerprint). This is the "galois23" the notebook wanted, for the
  1+23 residual.
- Fix the `Db::load_default` path discrepancy (points at missing
  `data/transitive_24.jsonl`; the working path is `CycleTypeSupport::load_default`
  → `data/transitive24_cycletypes.jsonl`). Fix within this crate only.
- **Acceptance test:** classify synthetic degree-23 permutation samples of each
  of the 7 groups correctly; M23 vs A23 vs S23 separation.

### Agent P — `rustmath-polynomials`  (multivariate + Buchberger present)
- `src/poly_system.rs` ← dessin_engine `src/mpoly.rs` (`MPolySystem`: eval &
  Jacobian mod m, exact rational eval, `is_exact_solution`) built on the existing
  `multivariate.rs`. This is the shared system type the numerical & Belyi layers
  consume — **highest priority in this crate.**
- *(Stretch, only if the above is solid and green)* fill real stubs with new
  files: `src/elimination.rs`, `src/saturation.rs`, `src/staircase.rs` (vdim from
  a Gröbner basis). Do **not** touch the placeholder bodies in `ideal.rs`/
  `quotient.rs`; add free functions instead.
- Do **not** port the F4/FGLM/multimodular solver stack yet — Gröbner is demoted;
  that is optional future work.
- **Acceptance test:** build a small multivariate system, verify exact-solution
  check and mod-m Jacobian against a hand example.

---

## Wave 2 — `rustmath-numerical`  (after N's `recognize` interface exists)

### Agent U — `rustmath-numerical`  (greenfield for solving)
- `src/homotopy.rs` ← dessin_engine `src/homotopy_adapter.rs` **generalized to a
  PARAMETER homotopy**: emit a HomotopyContinuation.jl script with a `@var`
  parameter block, a start system, and a *free start solution* (random seed
  `z0`, `p0 = Ψ(z0)`), tracking `p0 → p*`. Keep the working candidate importer
  (`parse_result_json` → numerical solutions).
- `src/exactify.rs` ← dessin_engine `src/exactification.rs::exactify`
  orchestration (numerical candidate → `recognize` → exact back-substitution),
  calling `rustmath-numberfields::recognize`.
- *(Optional)* `src/newton_system.rs` — multivariate Newton (residual + Jacobian
  via `rustmath-matrix`); the numerics may stay in Julia, so this is a backstop.
- **Interface contract with N:** `recognize(re: f64, im: f64, max_deg) ->
  Option<Vec<BigInt>>` (minimal polynomial coeffs). Agree this signature; U calls
  it, N provides it.

---

## Wave 3 — `rustmath-curves`  (the Belyi integrator; depends on all above)

### Agent C — `rustmath-curves`
Fold the Belyi/cover machinery here (semantic home: covers of curves). New files:
- `src/belyi/mod.rs` + encoders ← dessin_engine `belyi_system.rs`,
  `belyi_encode.rs`, `belyi_linearize.rs`, `portal.rs`, **and the new
  `belyi_pinned.rs`** (the λ-pinned 25-var system `A²B − λR⁵S = c·x¹²`, order-5
  point pinned at x=1 — the genuine solve gap).
- `src/belyi/monodromy.rs` ← dessin_engine `permutation.rs` (`BelyiTriple`,
  `genus_from_branch_cycles` via Riemann–Hurwitz). *(Genus lives with covers, not
  in the placeholder `riemann_roch.rs`.)*
- `src/belyi/verify.rs` ← `cover_verify.rs`; `src/belyi/bad_locus.rs` ←
  `bad_locus.rs`; `src/belyi/bridge.rs` ← `conic_bridge.rs`;
  `src/belyi/descent.rs` ← `descent.rs` (uses `rustmath-quadraticforms` conic/
  quaternion + `rustmath-numberfields` quadratic/nf).
- `src/belyi/pipeline.rs` ← `pipeline.rs` + `portal::uniform_law` (M23 decision
  logic folds in here per "everything into existing crates").
- Cargo deps to add (this crate's Cargo.toml): quadraticforms, numberfields,
  numerical, polynomials, groups.
- **Acceptance gate (the Müller reproduction):** end-to-end, recover the Müller
  conic `(-1,-1)` ramified `{2,∞}`, `LocallyEmpty`, from a cover — the G6 gate.

---

## Integration (after waves land)

- Each agent worked in an isolated worktree; reconcile by merging each crate's new
  files back to `main`. Source merges are disjoint (different crates); regenerate
  `Cargo.lock` if it conflicts.
- Then: build the pinned system, run the parameter homotopy in Julia, exactify,
  verify (genus/monodromy via `rustmath-groups` + `rustmath-igp24`), read the
  descent conic. That is the *solve*, now on RustMath rails.
- `dessin_engine` is retired once `rustmath-curves::belyi` reproduces its 101
  tests' coverage and the Müller gate.

---

## Dependency DAG (for scheduling)

```
Wave 1 (parallel):  quadraticforms(Q)   numberfields(N)   groups(G)   polynomials(P)
Wave 2:                                  numerical(U) ── needs N.recognize, P.poly_system
Wave 3:             curves/belyi(C) ── needs Q, N, U, P, G
```
Sole cross-agent coordination points: the `recognize` signature (N↔U) and the
`MPolySystem` type (P↔U,C). Fix those two contracts up front; everything else is
independent.
