# Stauduhar improvement plan (degree-24 unique-t)

Grounds `Stauduhar_bottleneck.md` in the actual `rustmath-galois` code. The note's
ranked levers, mapped to what exists and what to build.

## Current state (commit ab7a052)

- `descent.rs` — classical Stauduhar for small n (3,4,5): complex-root labeling,
  start S_n/A_n, enumerate **all** maximal subgroups, build the relative resolvent
  over **all** `[G:H]` cosets, test for a simple rational root. **Works, validated.**
- `deg24.rs` — degree-24 narrowing: Frobenius cycle-type candidate class, then
  builds+factors the **degree-2024 (k=3) absolute subset-sum resolvent over ℤ**.
  **This is the >300 s bottleneck** — exactly the anti-pattern the note names.
- `short_coset.rs` (NEW) — `short_cosets(G,H,σ)`, `descent_impossible`,
  `conjugation_perm`. The #1 lever's group-theory core. Validated.

## The architectural fix (note §"most important rule")

> Do not compute a degree-24 resolvent unless the candidate list is already tiny.

Replace `deg24.rs`'s absolute resolvent with the OSCAR/Magma pipeline:
`Frobenius → subfields/blocks → wreath fingerprint → short cosets → Tschirnhaus-
selected relative invariant → proof resolvent only at the end`.

## STATUS (2026-06-24): P1, P1.5, P2, P3 DONE and merged to integrate-lava-galois

- **P1** (5339327) GaloisCtx with explicit Frobenius — 8 tests. ✅
- **P1.5** (a1300d5) GFPN extension-field factoring fix (discovered prerequisite) — 52 tests. ✅
- **P2** (da9bdbf) common-ring Z_{p^M} embedding + separable relative invariants — 40 tests,
  0.04s (was 617s once GFPN fixed). ✅
- **P3** (a6a59c1) short-coset degree-24 descent — **24T2672 narrows in ~21s, SOUND, no
  degree-2024 resolvent** (was >300s). ✅ short_cosets primitive (ab7a052) + conjugation
  Frobenius already in.

**Remaining to reach unique-t (tightening, not architecture):** P3's Accept path lacks a
**separability gate**, so it over-accepts (sound but loose narrowing). Add: establish the
relative invariant's separability (mod-p collision test / precomputed) so an integer value at
a short coset is a *simple* root ⇒ genuine descent. Then P4 (Tschirnhaus preselection) and P5
(block-2 (B,V,[a]) shortcut) sharpen toward unique-t. The >300s→21s architecture win is done;
unique-t is incremental from here.

## Build order (each piece is testable on its own)

### P1 — p-adic `GaloisCtx` (the keystone primitive)
Mirror OSCAR: roots of `f` in an unramified extension `Q_{p^m}` to controllable
precision, **labeled so Frobenius σ is an explicit permutation** (within each
irreducible factor of `f mod p` of degree d, σ is a d-cycle). Exposes:
`roots(ctx, prec)`, `frobenius(ctx) -> Perm`, `isinteger(ctx, value, bound)`.
Reuse `rustmath_polynomials::root_label::padic_roots` + `zp_hensel`; extend to the
unramified-extension case (roots of the irreducible mod-p factors lifted into
`GF(p^d)` then Hensel-lifted). **This is what makes short cosets usable at deg 24.**

### P1.5 — **GFPN extension-field factoring fix** (newly-found prerequisite for P2)
P2's common-ring embedding finds roots of `g_i` in `GF(p^M)` by factoring over that
extension — which hits the **known-broken** `rustmath-finitefields` GF(pⁿ) equal-degree
split (Wave-1 `test_factor_over_gfpn` is `#[ignore]`'d: it draws the splitting element
from the prime subfield `GF(p)` not the full `GF(q)`). Symptom in P2: `root_in_gfpm`
returns `None` and the suite runs 617 s (pathological retry). **Fix this first**, then P2.

### P2 — separable **relative invariants** (not absolute subset-sum resolvents)
For `Stab_G(I)=H`, build `I` from group/block structure (note §Stage 3):
block sums `S_i=α_{i,0}+α_{i,1}`, block discriminants `D_i=(α_{i,0}-α_{i,1})²`,
products/sums over B-orbits of subsets `T⊆{1..12}`. Separability of `I` for `(G,H)`
established **once** (orbit of I under G has distinct values) or by a mod-p test —
NOT by evaluating all cosets of f. This is what makes short-coset evaluation sound.

### P3 — short-coset descent (consume `short_coset.rs` + P1 + P2)
Replace `deg24.rs`'s resolvent step: for each `(G,H)` on the atlas descent graph,
`σ = ctx.frobenius()`; `cosets = short_cosets(G,H,σ)`; if empty → reject H
(evaluation-free); else evaluate `I` p-adically on those few cosets and test
`isinteger`. A rational/integer value descends. Objective: pick the observed
Frobenius σ minimizing `|（G/H)_σ|` (note §Stage 1).

### P4 — Tschirnhaus preselection (deterministic, modular)
Before any p-adic evaluation, pick the transform `T_c(x)=x+c2 x²+c3 x³` and the
invariant whose orbit values **don't collide mod cheap primes** (note §Stage 4).
Mirrors OSCAR `upper_bound(ctx, I, tsch)`.

### P5 — imprimitive block shortcut (the IGP24 specialization, note §Stage 2)
For block-2 degree-24, compute `B, V=G∩F2^12, [a]∈H¹(B,P/V)` directly and match
the fingerprint — this is the native port of `inverse_galois/.../ghost_classify.sage`.
Often pins t (e.g. 24T2672 is the unique (B=12T34, dimV=4, cyclic)) with **no**
resolvent. Needs B = Galois group of the degree-12 block resolvent (recursive ID).

### P6 — proof mode (note §Stage 5)
Only after the candidate list is size 1: emit a certificate (simple rational root
of the relative resolvent / block-subfield / Frobenius lower bound). Geißler-Klüners
split: fast short-coset narrowing first, proof only for the survivor.

## OSCAR primitive mirror (target API for the ctx)
`GaloisCtx · roots(ctx,prec) · frobenius(ctx) · upper_bound(ctx,I,tsch) ·
isinteger(ctx,bound,value) · resolvent(ctx,G,H) · fixed_field(ctx,H)`

## Quick interim mitigation (optional, low value)
`deg24.rs` could default to k≤2 (degree-276 resolvent) and make k=3 opt-in/budgeted
— keeps most cases under the timeout but narrows less. The note says fix the
architecture, not tune k; treat this only as a stopgap.

## Ranking (do P1→P3 first; that is the actual OSCAR-retirement path)
1 short cosets (core in place) · 2 GaloisCtx (P1) · 3 relative invariants (P2) ·
4 block shortcut (P5, IGP24-specific) · 5 Tschirnhaus (P4) · 6 proof (P6).
