# Handoff: build the CFT construction stack (the next wishlist)

**To: the RustMath-building session (you already shipped Stauduhar `--galois-short`).**
**From: the IGP24/frobenius strategy session, 2026-06-25.**

## Mission

Build the class-field-theory construction stack in `rustmath-numberfields` so we can
**construct a degree-24 number field realizing a *specified* solvable Galois group `24Tt`
with a *specified* real-root signature `r` and a *small* discriminant.** This is the engine
that lets us claim the competition's **116,426 open cells** (each worth 1.0 points; none has
a published field anywhere — they must be built from scratch).

## Why this, why now (full derivation: living notes Entries 38–44)

`/Users/john/Documents/inverse_galois/inverse_galois_notes.tex`, Entries 38–44. In one paragraph:
the leaderboard intelligence proved our problem is **value, not volume** — we're rank 3 but last in
points/discovery (0.282), because we mass-produce contested pairs (~0.05 each) and never reach the
open frontier. Three laptop shortcuts are **all dead, tested**: forward sampling (re-treads easy
groups), the Galois-ID pre-screen (only 21% identified, 0/5 of those open), and LMFDB transfer (LMFDB
degree-24 = the 622 baseline = excluded from open by construction). The open cells have **no published
field** — pure construction. Degree 24 = 2³·3, so the vast majority of open `24Tt` are **solvable**,
and CFT builds solvable extensions as abelian-layer towers with **both knobs we need under control**:
conductor → discriminant, infinite-place modulus → signature `r`. This is *constructed to
specification*, not sampled.

## What already exists — DO NOT rebuild

- `rustmath-numberfields`: `classgroup`, `units`, `s_units`/`s_unit_gens`, `ideals`, `different`,
  `ramification`, `local_field`, `round2`, `panayi`, `polred`. These are the CFT prerequisites.
- `rustmath-galois`: `descent24`/`short_coset`/`galois_ctx`/`relative_invariant` — the Stauduhar ID
  (`rustmath-igp24 --galois-short`). Use it as the construction **verifier**.
- Spec = Magma compatibility. The handbook chapters are at `rustmath-lava/docs/`:
  Ch.39 Class Field Theory, Ch.63 Finite Soluble Groups, Ch.68 Cohomology and Extensions.
- Branch: `integrate-lava-galois`.

## Build order (the wishlist — Entry 44)

| # | module | Magma ch. / intrinsics | enables |
|---|---|---|---|
| 1 | **`rayclass`** | 39: `RayClassGroup`, `RayResidueRing` | conductor **and** signature control — the keystone |
| 2 | **`abext`** | 39: `AbelianExtension`, `RayClassField` | realize each abelian layer from a norm subgroup |
| 3 | **`artin`** | 39: `ArtinMap`, `Conductor`, `NormGroup` | reciprocity + exact conductor/disc |
| 4 | **`solvtower`** | 63: `ChiefSeries`, `AbelianQuotient` | decompose target `G` into abelian layers (absent in `rustmath-groups`) |
| 5 | **`embed`** | 68: `CohomologyModule`, `H^2`, `TwoCocycle`, `Extension` | non-split 2-kernel embedding problems (last; many targets don't need it) |

**Rationale for the order:** `rayclass` sits directly on existing `classgroup`/`units`/`ideals`, and a
ray class group whose modulus includes the **infinite places** already delivers both levers in one
object. `abext`+`artin` turn a chosen finite-index norm subgroup into the actual abelian field (Kummer
when `μ_n ⊂ K` — feed it the existing `s_units` — else an explicit class-field construction).
`solvtower` is pure group theory and drives the layer loop. `embed` (the `H^2` obstruction) is only for
non-split central extensions and comes last.

## Validation gates (prove each before moving on)

1. `rayclass`: reproduce `RayClassGroup` structure (group invariants + the map) against PARI
   `bnrinit`/Magma on a handful of small fields, **including a modulus with real infinite places**.
2. `abext`+`artin`: construct a *known* small abelian extension (e.g. a cyclic cubic of prescribed
   conductor; a quadratic with a chosen real place ramified) and verify its **discriminant and
   signature** match the conductor-discriminant prediction.
3. End-to-end: pick **one** small *solvable* `24Tt` that is **open** (cross-ref
   `/Users/john/Documents/inverse_galois/frobenius/open_cells.json`), construct it at a chosen `r`,
   `polredabs`, and confirm with `--galois-short` that the group is exactly `t` and the real-root count
   is exactly `r`. **Closing the loop on one cell is the milestone** — scale only after.

## Integration target

A callable `construct_field(t: u32, r: u32) -> Option<Polynomial>` and a CLI
`rustmath-igp24 --construct 24Tt,r` that emits a degree-24, `polredabs`-reduced, `--galois-short`-verified
polynomial of the specified group and signature, minimizing disc via the conductor choice (rank
candidate moduli by `2·log|D_K| + log N(conductor)` — the scorer logic from
`frobenius/disc_score.sage`). Targets and their published `D_min` (for the disc lever) are in
`open_cells.json` and `frobenius/disc_attack_list.csv`.

## Notes / constraints

- GRH-conditional class/unit computation is fine for now — flag it in output (mirror Magma's
  rigorous/GRH/heuristic distinction).
- This is ROG-class compute (heavy class/unit/ray-class work); correctness first, performance second.
- Report back: which `24Tt` the engine can construct, at which `r`, with what disc vs the published
  `D_min`. That list feeds the submitter directly.

The strategic conclusion this engine answers (Entry 42/43): *the durable lever is not a better sampler
— it is the class-field-theory construction stack.* Start with `rayclass`.
