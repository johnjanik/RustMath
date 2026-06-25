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

## Execution

I launch each phase's agents in worktrees off `integrate-lava-galois`, validate against PARI + tests,
back out anything that fails, and fast-forward `integrate-lava-galois` per merged module. Phase 0's three
agents start now in parallel.
