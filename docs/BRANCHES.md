# Branch inventory and consolidation (2026-11-05)

All 20 remote branches were audited against `main` (`57fc436`, PR #567). Method: commit
containment (`git merge-base --is-ancestor`), patch equivalence (`git cherry`), and, for the
`magma-port/wave*` branches, a file-by-file comparison against the `belyi` branch.

**Result:** every commit of value is now in `claude/sleepy-carson-6kl13v`, which is
`belyi/u0-unblock-and-true-root-detector` + a merge of `integrate-lava-galois` + `main`.
No branch has been deleted. Deleting the superseded ones is a separate decision (list below).

## Summary

| Branch | Tip | Date | vs main | Status | Action |
|---|---|---|---|---|---|
| `belyi/u0-unblock-and-true-root-detector` | `70918e0` | 2026-07-18 | +134 | **Most advanced line.** Re-landed Wave 0/1, docs consolidation, PROJECT_TRACKER, Belyi/dessin stack, elliptic/modular depth, Risch, Schreier-Sims, exact polytopes | Integration base |
| `integrate-lava-galois` | `77112d5` | 2026-06-29 | +35 | Unique work not in belyi: generic + degree-24 Stauduhar descent (`rustmath-galois`), CFT stack (rayclass, artin, abext, solvtower), K(S,2) Kummer enumeration, Hilbert class polynomial, `rustmath-functionfields`, Montes in `rustmath-padics`, GF(p^n) Cantor-Zassenhaus fix | **Merged** (conflicts resolved, see below) |
| `magma-port/wave0-foundation` | `71614d5` | 2026-07-01 | +1 | Superseded: belyi `d3ad8a5` re-landed it; all 17 files present in belyi, 11 grown further | Safe to delete |
| `magma-port/wave1-combinatorics` | `64772cd` | 2026-07-01 | +2 | Superseded by belyi `8a86800` (39 files, all present) | Safe to delete |
| `magma-port/wave1-crypto` | `b4c1a25` | 2026-07-01 | +2 | Superseded by belyi `29488a8` | Safe to delete |
| `magma-port/wave1-finitefields` | `1d7fb56` | 2026-07-01 | +2 | Superseded by belyi `f1d9b30`. Its `residue_ring.rs` was intentionally replaced by `zmod.rs` (canonical `Integers(m)`) | Safe to delete |
| `magma-port/wave1-graphs` | `dd6c058` | 2026-07-01 | +2 | Superseded by belyi `5b80c4a` | Safe to delete |
| `magma-port/wave1-matrix` | `826daa0` | 2026-07-01 | +2 | Superseded by belyi `0eba9d1` (one file differs by a single line) | Safe to delete |
| `magma-port/wave1-monoids-automata` | `b1aef88` | 2026-07-01 | +2 | Superseded by belyi `aa84d80` (1 commit patch-identical) | Safe to delete |
| `magma-port/wave1-powerseries` | `b7fbaaa` | 2026-07-01 | +2 | Superseded by belyi `56bcc27` | Safe to delete |
| `magma-port/wave1-symmetricfunctions` | `e2a52cf` | 2026-07-01 | +2 | Superseded by belyi `3fee773` | Safe to delete |
| `igp24-cli-galois` | `fc69a9b` | 2026-06-21 | 0 | Fully merged into main | Safe to delete |
| `p0-mestre-tools` | `5517cff` | 2026-06-19 | 0 | Fully merged | Safe to delete |
| `p1-prime-ideals` | `2e72236` | 2026-06-19 | 0 | Fully merged | Safe to delete |
| `padic-factorization-port` | `90c6b35` | 2026-06-18 | 0 | Fully merged | Safe to delete |
| `phase1-factorization-over-z` | `81273e6` | 2026-06-18 | 0 | Fully merged | Safe to delete |
| `phase2-local-analysis` | `54bc719` | 2026-06-18 | 0 | Fully merged | Safe to delete |
| `phase3-discriminant-lll` | `5683358` | 2026-06-18 | 0 | Fully merged | Safe to delete |
| `phase4-galois-narrowing` | `3966f2c` | 2026-06-18 | 0 | Fully merged | Safe to delete |
| `main` | `57fc436` | 2026-06-23 | 0 | Default branch; 134 + 35 commits behind the integration branch | Fast-forwardable via PR from the integration branch |

"Superseded" means each file the wave branch adds/modifies exists in belyi, and in every case
but one belyi's copy is the same or longer (later edits on top). The tips stay reachable as long
as the branches exist; tag them before deleting if you want a permanent pointer.

## Merge of `integrate-lava-galois` into belyi: conflict resolutions

Both lines developed in parallel from `f22ac17` and several times solved the same problem
independently.

| File(s) | Conflict | Resolution |
|---|---|---|
| `rustmath-galois/*` | Both branches created a `rustmath-galois` crate (belyi: exact low-degree engine + sieve + Cayley sextic; lava: generic/deg-24 Stauduhar descent). Only `lib.rs`/`Cargo.toml` overlapped | Union. Belyi's `galois_group` stays the crate-level API (honesty contract); lava's driver is re-exported as `stauduhar_galois_group` (`DescentConfig`, `DescentResult`). All lava modules kept; deps unioned |
| `rustmath-finitefields/src/lib.rs` | Two GF(p^n) stacks; re-exports `FiniteFieldElement`, `find_irreducible`, `is_irreducible`, `random_irreducible` collided | Belyi's canonical exports kept at the root; lava's reachable by module path (`ff_poly::`, `irreducible::`), which is how all lava callers already import them |
| `rustmath-padics/src/lib.rs` | Belyi turned the crate into a re-export shim over `rustmath_rings::padics`; lava added `montes.rs` + `newton_polygon.rs` | Shim kept, lava's two modules added (`rustmath-integers` moved to a normal dependency) |
| `rustmath-curves/*` (7 files) | Lava's only change was a compile-fix commit; belyi fixed the same 25 errors and then rewrote the crate | Belyi side |
| `rustmath-reals/src/mpfr.rs`, `rustmath-complex/src/mpc.rs`, `rustmath-polynomials/src/univariate.rs` | Same bug fixed on both sides (float->integer exactness, f64 round-trip, square-free test) | Belyi side (a superset of lava's additions) |
| `Cargo.toml` | `rustmath-galois` listed twice | Deduplicated |

**Known duplication to resolve later (tracked, not urgent):** two GF(p^n) factorization stacks in
`rustmath-finitefields`; two Montes/OM implementations (`rustmath-padics::montes` and
`rustmath-rings::padics` MacLane/OM); belyi's `rustmath-galois::stauduhar` placeholder vs lava's
real descent.

## Not visible from here

Branches that exist only on the Mac (worktree `/Users/john/Documents/rm-galois`, any unpushed local
branches) and the off-repo IGP24 material in `/Users/john/Documents/inverse_galois/` are not in
this audit. Push them (or attach the folder via Remote Control) to include them.
