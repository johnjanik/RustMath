# CFT construction stack — phased fleet plan

Goal (from `docs/cft_construction_engine.md`): a native class-field-theory engine that **constructs a
degree-24 field of a specified solvable `24Tt` at a specified signature `r` with small discriminant**,
to claim the competition's open cells. Build order (Entry 44): `rayclass → abext → artin → solvtower →
embed`. Verifier = `rustmath-igp24 --galois-short` (already shipped).

## Coordination rules (lessons from Waves 1–3)

- **Worktrees branch from `integrate-lava-galois`** (NOT main) — I create them manually and point each
  agent at its path.
- **Agents cannot run `cargo test`** (sandbox) — they write tests + `cargo build`; **I validate** in the
  main checkout, cross-check against **PARI (`gp`/Sage `bnrinit`)**, back out failures, merge only green.
- **Each agent owns disjoint files** — the only shared edits are each crate's `lib.rs` mod-line and the
  workspace `Cargo.toml`, which I merge. New crates/files = zero conflict.
- **GRH-conditional** class/unit/ray-class work is acceptable for now — must be flagged in output.

## File ownership (all NEW files → no cross-agent conflict)

| module | crate / file | dep |
|---|---|---|
| rayclass | `rustmath-numberfields/src/rayclass.rs` | classgroup, units, ideals (existing) |
| disc_score | `rustmath-numberfields/src/disc_score.rs` | round2 disc, ideal_norm (existing) |
| solvtower | `rustmath-groups/src/solvtower.rs` | transitive24 gens, group_closure (existing) |
| abext+artin | `rustmath-numberfields/src/{abext.rs, artin.rs}` | **rayclass**, s_unit_gens |
| embed (cohomology) | `rustmath-groups/src/cohomology.rs` | solvtower (only for non-split targets) |
| construct (driver) | `rustmath-numberfields/src/construct.rs` + `rustmath-igp24` CLI | all of the above |

Numberfields `lib.rs` collects ≤5 `pub mod` lines; groups `lib.rs` ≤2; I merge them.

## Phases (maximise independent parallelism)

### Phase 0 — independent foundations (3 agents, fully parallel; disjoint files, different crates)
- **P0-RAYCLASS** (critical path): ray class group `Cl_m(K)` for a modulus `m = m_0·m_∞` (finite part ×
  **real infinite places**), as an `AdditiveAbelianGroup` (reuse `additive_abelian_group`) + the map
  ideal↦class and discrete log. This single object carries **both** levers (conductor→disc, m_∞→signature).
- **P0-SOLVTOWER**: pure group theory — given a transitive-group's gens, compute a **chief series** with
  **abelian quotients** (derived/elementary-abelian layers) for solvable `24Tt`; expose the layer data
  the driver loops over. Different crate (`rustmath-groups`).
- **P0-DISCSCORE**: port the scorer from `frobenius/disc_score.sage`: `score = 2·log|D_K| + log N(cond)`;
  rank candidate moduli. Small, self-contained.

### Phase 1 — abelian-extension realization (1 agent, after RAYCLASS merges)
- **P1-ABEXT**: `abext.rs` (`AbelianExtension`/`RayClassField`: realize the field cut out by a finite-index
  norm subgroup of the ray class group — Kummer via existing `s_unit_gens` when `μ_n⊂K`, else explicit)
  + `artin.rs` (`ArtinMap`, `Conductor`, `NormGroup`: reciprocity + **exact conductor ⇒ discriminant**).
  Coupled, so one agent. Validation gate: a *known* small abelian extension with predicted disc + signature.

### Phase 2 — integration (after Phase 0 + Phase 1)
- **P2-CONSTRUCT**: `construct.rs` driver `construct_field(t,r) -> Option<Polynomial>`: solvtower decomposes
  `G`, the abelian-layer loop builds each layer via rayclass/abext/artin choosing the **conductor that
  minimizes disc_score** and the **m_∞ giving signature r**, composes the tower, `polredabs`. Plus the CLI
  `rustmath-igp24 --construct 24Tt,r`, verified by `--galois-short`.
- **P2-EMBED** (only if a chosen target is a non-split central extension): `cohomology.rs` (`H^2`,
  `TwoCocycle`, `Extension`). Many targets are split — defer until a target needs it.

## Validation gates (I run; agents cannot)

1. **rayclass** → cross-check group invariants **and the map** against PARI `bnrinit` (`gp`/Sage) on
   several small fields, **including a modulus with real infinite places** (signature lever).
2. **abext+artin** → build a cyclic cubic of prescribed conductor and a quadratic with a chosen real
   place ramified; verify **disc + signature** match the conductor–discriminant formula and PARI
   `bnrclassfield`.
3. **solvtower** → chief factors of known solvable `24Tt` cross-checked against GAP (`libgap` bridge) /
   known structure.
4. **End-to-end (the milestone)** → one small *solvable open* `24Tt` (from `open_cells.json`),
   constructed at chosen `r`, `polredabs`-reduced, confirmed by `--galois-short` to be exactly `t` with
   real-root count exactly `r`. **Scale only after this closes.**

## STATUS (2026-06-25)

- **Phase 0 ✅ merged** (f35ef1e): rayclass (PARI-`bnrinit`-validated, incl. signature lever),
  solvtower (chief series), disc_score. 
- **Phase 1 ✅ merged** (0af9fea): artin (Artin map / NormGroup / Conductor→disc) + abext.
  Construction gates pass: cyclic cubic cond 7→disc 49, cyclic quartic cond 5→disc 125, Q(√5)→disc 5,
  real-place-ramified signature lever. Two tracked defects: (a) Artin map discrete-log is **wrong for
  ideals with primes outside the Minkowski factor base** (`#[ignore]`'d FIXME); (b) abext realizes
  **only K=ℚ** (Gaussian periods, prime conductor) + partial Kummer — general relative abelian
  extensions over K≠ℚ are `ConstructionMethod::Abstract`.

## Phase-2 scoping (the milestone is harder than a driver)

Crossing `open_cells.json` (116,426 cells) with the atlas: **19,911 / 20,405 open `24Tt` are solvable**
(confirms the thesis) but **0 are abelian** (abelian degree-24 are in LMFDB ⇒ excluded from open). The
smallest-order solvable open targets are **|G| = 96 = 2⁵·3, all non-abelian** (e.g. 24T87 r=8, 24T127
r=4, 24T140 r=6). So every milestone candidate is a **multi-layer non-abelian tower**, whose layers
after the first live over **K ≠ ℚ**. Therefore Phase 2 needs, before the driver:

- **P2a — relative abelian extension over K≠ℚ** (the real ray-class-field construction; abext is ℚ-only)
  **+ Artin-map completeness** (large-prime principalization — needed for relative conductors/norm groups).
  This is the CFT crux that the ℚ-specialized Phase-1 abext does not yet provide.
- **P2b — tower driver** `construct_field(t,r)`: solvtower layers → relative abext per layer (conductor
  min disc, m_∞ for signature) → compose → polredabs → verify `--galois-short`. **Milestone: one of the
  |G|=96 targets** (e.g. 24T127 r=4), group + signature confirmed.

## P2a attempt #1 — FAILED validation (NOT merged; preserved on branch `cft-relabext`)

De-risk first succeeded: `gaussian_period_poly(73,24)` → degree-24 C₂₄ field → `--galois-short`
`unique_t=1, confident` — the construct→verify pipeline works end-to-end at degree 24 (committed as a
smoke test). Then the relative-abext agent's deliverable **failed**:
- **relabext (all 3 gates fail):** `hcf_qsqrtm23` panics (relabext.rs:1196), `real_quadratic_rcf_qsqrt3`
  **hangs** (a non-terminating coefficient/compositum search — ran 91 min before being killed),
  `kummer_over_qi` panics in `bivariate.rs:79` (the compositum resultant). The relative class-field
  construction does not work.
- **artin map fix:** correct in isolation (5/5, `artin_total_qi_m5` un-ignored — the whole-ideal
  principalization fixes the wrong-class bug) BUT it makes the map **slow** (`is_principal` on growing
  ideals), which **regresses abext into a hang** (40 s+). Net: not mergeable.

Lesson: relative CFT (ray-class-field over K≠ℚ + the compositum/primitive-element) is the genuinely hard
piece and needs an **incremental, bounded, per-case** build — NOT one big agent. Re-do path:
1. **Bounded primitives first:** a hang-proof `compositum`/primitive-element (the `bivariate.rs:79`
   crash) and a fast principal-ideal test; every search MUST be budgeted (no unbounded loops).
2. **One relative case at a time, PARI-validated:** start with the **HCF of Q(√−23)** alone (unramified
   cyclic cubic → S₃ sextic, disc −23³), cross-checked with `gp` `bnrclassfield`/`rnfdisc` and the
   absolute group via Sage, before any generalization.
3. Keep the artin map fix but **make it fast** (cache principalizations / bound the search) so abext
   doesn't regress — or gate the slow path so abext uses only the factor-base-smooth fast path.
**P2b (tower driver) is BLOCKED** until a relative-abext case is validated.

## P2a redo path — VALIDATED against PARI (`docs/algorithm_notes/abext_notes.md`, 2026-06-25)

The note diagnoses every P2a#1 hang/crash as an **unbounded element/coefficient search** and replaces
each with **finite linear algebra over an explicit finite space**. I confirmed its Q(√−23) worked example
end-to-end in `gp` (`/tmp/chk23.gp`): `rnfequation(K, x³−x−1) = y⁶+67y⁴−2y³+1588y²+140y+13249` (matches the
note), `h_K=[3]`, rel disc `d_{F/K}=1`, `polredabs → x⁶−3x⁵+5x⁴−5x³+5x²−3x+1`, `polgalois` order 6 = S₃,
`|disc|=23³`. Build order (note's "final implementation priorities"), each bounded + validatable:

1. **Alg 1A — multiplication-matrix `rnfequation`** ✅ DONE + validated
   (`rustmath-numberfields/src/rnfeq.rs`, `absolute_defining_polynomial`). Basis `{θⁱαʲ}`; builds
   mult-by-θ and mult-by-α matrices once, `M_s = T + sA`, `F_s = charpoly(M_s)`; primitivity tested by
   **square-freeness** of `F_s` (≡ distinct conjugates; with `h` irred / `g` irred ⟹ `F_s` irreducible
   of degree `dn`); bounded loop `s = 0..C(dn,2)`. Keystone — unblocks absolute polys for BOTH Kummer
   towers and descent, and **replaces the `bivariate.rs:79` resultant crash**. **Oracle PASSED:**
   `x³−x−1` over Q(√−23) returns exactly `y⁶+67y⁴−2y³+1588y²+140y+13249` at `s=1` (exact 6-coeff match
   vs PARI `rnfequation`); 5/5 tests green. Square-free test depends on the `univariate.rs` fix
   (committed separately as the post-merge baseline).
2. **Alg 2 — finite `K(S,n)` Kummer enumeration** ✅ core DONE + validated
   (`rustmath-numberfields/src/kummer.rs`, `n=2`). `k_s_2_space` = the finite `𝔽₂`-generating set
   (torsion `−1` + fundamental units + `S`-prime generators = the `O_{K,S}^×/squares` part);
   `kummer_quadratic_candidates` enumerates the bounded `2^{#gens}−1` combinations, computing **exact
   infinite-place ramification** (real embeddings where `γ<0` — the IGP **signature lever**) and the
   **tame finite conductor** (`v_𝔭(γ)` odd, `𝔭∤2`). This structurally eliminates the unbounded
   coefficient search that hung `real_quadratic_rcf_qsqrt3`. **Validated:** narrow HCF of Q(√3) — the
   finite space contains the torsion `−1`, which is totally negative ⇒ both real places ramified,
   giving `L=K(√−1)`, relative poly `x²+1` (matches gp `bnrclassfield`); the old prime-only
   `kummer_generator` misses this unit/torsion generator entirely. 3/3 tests green. **Scope/TODO:** the
   wild `2`-adic conductor exponent (local Kummer-symbol membership), the `Cl_{K,S}[n]` torsion part of
   `K(S,n)`, general `n>2`, and the H-annihilator γ-selection are the remaining refinements for the
   tower driver (P2b).
3. **§3B — imaginary-quadratic HCF** ✅ Step 1 DONE + validated
   (`rustmath-numberfields/src/hcf.rs`). `reduced_forms(d)` enumerates the `h(d)` reduced
   primitive pos-def forms; `hilbert_class_field_from_hcp(d, H_d)` assembles the **absolute** HCF as
   the compositum `ℚ(√d, j)` by **reusing Alg 1A** (`g = T²−d`, `H_d` as the monic relative poly over
   `K`). **Validated on D=−23:** class number 3, absolute degree 6, field discriminant exactly
   `(−23)³ = −12167` via the model-independent invariant `poldisc = d^h·index²` (index ≈ 3.2×10¹⁸ — the
   raw compositum model is highly non-maximal). Replaces the `hcf_qsqrtm23` panic. 4/4 tests green.
   **Note:** the in-repo `polredabs`/`field_discriminant` are too weak to *reduce/identify* a model
   this large — recorded as a downstream reduction gap (not a §3B blocker). ✅ **Step 2 DONE +
   validated:** `hilbert_class_polynomial(d)` computes `H_d` from high-precision `j(τ) = E₄³/Δ`
   (`rug`/MPC, `ComplexMPFR`) over the reduced-form CM points, rounding the symmetric functions to
   integers; `hilbert_class_field(d)` chains it into the Alg 1A assembly. **Validated:** `H_{−23}`
   matches PARI `polclass(-23)` exactly; `H_{−3}/H_{−4}/H_{−163}` (incl. the `j=−640320³` precision
   stress) correct; full pipeline field disc `= d^h` for `−23,−31,−47`. **Foundation fixes required
   en route** (separate commit): `ComplexMPFR::with_val_reals` rounded through `f64` (capped ~53 bits);
   `RealMPFR::round/floor/ceil` converted through `f64` with a `0` placeholder above `i64::MAX` — both
   now exact bignum; added `Integer::from_decimal_str`, `RealMPFR::as_float`.

   **Strategic payoff:** an imaginary quadratic with `h=12` has a **degree-24** HCF with generalized-
   dihedral group `Cl_K ⋊ C₂` — a direct candidate source for open `24Tt` cells (pending a good
   reduction step for small discriminant).
4. **Alg 4 — fast ray Artin log** ✅ DONE + validated
   (`artin.rs` `artin_gen_vector_fast`/`reduce_ideal` + `classgroup::short_ideal_elements`).
   Per call: one **LLL ideal reduction** `𝔟 = (η)·a⁻¹` (`η ∈ a` near-minimal-norm via `short_ideal_
   elements`, coprime to `m₀`) → factor the small `𝔟` over the factor base →
   `[a]_m = R(η) − Σ_j v_{𝔭_j}(𝔟)·[𝔭_j]_m`. **No per-call principality search** (fixes the P2a
   regression) and **total on primes above the Minkowski bound** (fixes the documented Phase-1 defect).
   Falls back to the old bounded search if `𝔟` isn't factor-base-smooth. **Validated:** the previously
   `#[ignore]`'d `artin_total_qi_m5` now passes (total on `(11+5i)`, prime above 73); new
   `artin_nonprincipal_prime_above_minkowski` maps `𝔭₇` of Q(√−5) (norm 7 > MK bound ≈ 5, outside FB)
   to an order-2 class — cross-checked with gp `bnfisprincipal = [1]~`. artin 6/6, classgroup/abext/
   rayclass all green (no regression).

**Anti-hang invariants (must hold in every primitive):** all radical candidates come from a finite
`K(S,n)` basis; all ray logs use precomputed factor-base logs, never a per-call principality search;
every primitive-element shift loops a bounded `s ≤ C(dn,2)`. Validate each primitive against `gp`
(`rnfequation`/`bnrclassfield`/`rnfdisc`/`polgalois`) on the Q(√−23) case before generalizing.

## Execution

I launch each phase's agents in worktrees off `integrate-lava-galois`, validate against PARI + tests,
back out anything that fails, and fast-forward `integrate-lava-galois` per merged module. Phase 0's three
agents start now in parallel.
