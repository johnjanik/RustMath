# Inverse-Galois competition — build handoff

**Read this first.** It tells a fresh session exactly what to build next to advance our
standing in the SAIR IGP24 competition, what already exists (reuse, don't rebuild), and how to
validate. Everything here was established in prior sessions; this is the contract. The full
MAGMA-replacement module→crate map is **Appendix A** below (infrastructure reference).

---

## 0. TL;DR — the next build

**Build the S-unit squareclass Kummer constructor.** Prior work proved directed construction
*works in principle* (we can confine output to a target's top-group neighbourhood) but naive
split-prime support forcing is **too leaky** — it lands the wrong F₂[B]-module (measured: 9/9
wrong-V). The S-unit squareclass search is the fix: it controls the module **and** minimizes the
discriminant (which also closes our scoring gap to the leader). Prototype in Sage/PARI, then port
natively into `rustmath-numberfields` — the machinery is already on this branch (§5).

---

## 1. Mission & scoring

Realize transitive groups of degree 24 as Galois groups over ℚ. A scoreable label is a pair
**`(24Tt, r)`** — group `24Tt` (t=1..25000) at real-signature `r` (real-root count, even).
**Decoded scoring (exact, from 49k scraped placements):**

```
points(24Tt, r) = 2^(1-k) · ln(D_min)/ln(D_you)
```
- `k` = #teams holding the pair; `D_min` = smallest field-disc among holders; `D_you` = ours.
- **A sole-claimed (k=1) open pair = exactly 1.0, disc-irrelevant** (you are the min). This is the prize.
- Contested pairs decay exponentially in k AND penalize you unless you hold the smallest disc.

**Standing:** rank **3** — "Algebraically Delicious" (`IGP24-T00009`,
`teamv2_bffbe96483e641ee98ea408277d1efff`), ~9.6k pairs at **0.28/pair** vs KLPB's 0.70. Our discs
run **60–110 digits vs KLPB's 28–32** — a ramification deficit the S-unit engine fixes.
Closes **2026-08-15**. The play: sole-claim **open** pairs (1.0, disc-proof); disc-improvement and
co-claiming are dead levers until our discs shrink.

---

## 2. Why we're stuck, and the path out

- The broad relative-fiber samplers (`inverse_galois/frobenius/generate.sage`) are **tapped out**
  — 0 new pairs in ~940 evals vs 123,703 open targets; they re-mine ~6,500 easy groups. **Don't
  reinvest in sampling.**
- **Path — directed construction.** To hit `24Tt` via its size-2 block system: the kernel is an
  **F₂[B]-module** `V = G ∩ F₂¹²`, top group `B = 12Tj`; full invariant **`(B, V, [a])`**,
  `[a] ∈ H¹(B, P/V)`. For `f = Res_y(g, X²−γ)` the realized module is `V_γ = ⟨B·[γ]⟩`. **The
  constructor must choose `V_γ`, not just vary γ.**
- **Diagnosis (measured):** misses are **wrong-V** — we land the *full* wreath kernel (dim 12)
  while targets are *proper* cyclic submodules (e.g. dim 4). Naive support forcing → 9/9 wrong-V.
  Per the directive, wrong-V dominating ⇒ **S-unit squareclass control**. The cocycle `[a]` layer
  is **deferred** until V is controlled and wrong-cocycle actually appears.

Full reasoning: `inverse_galois/strategy_update_20260623.md` (the directive) and
`inverse_galois/inverse_galois_notes.tex` Entries 31–33.

---

## 3. Build queue (priority order)

### Build A — S-unit squareclass Kummer constructor  ★ the next build
Given target `(B, V, r)` and base `K=ℚ[y]/g` with `Gal(g)=B`, produce γ with `V_γ = V`,
signature `r`, and **minimal** relative discriminant (directive §6):
- `S = {𝔭|2} ∪ {𝔭|p, p ≤ P_max} ∪ {target split primes}`.
- Search `O_{K,S}^×/(O_{K,S}^×)²` — finite F₂-space `γ = ∏ uⱼ^{eⱼ} ∏_{𝔭∈S} π_𝔭^{e_𝔭}`.
- Keep `V_γ = V ∧ r(γ)=r`; score `2·log|D_K| + log N(𝔡_{K(√γ)/K})` (smaller = better;
  `D_L = D_K²·N(𝔡_{L/K})`). Signature via unit signature map `O_K^×/sq → {±1}^12`; 2-adic
  conditioning (`γ ≡ 1 mod 4`) to avoid wild blowup.
- Prototype Sage/PARI (`K.S_units`), measure hit-rate + ghost classes, **then** port to Rust (§5).

### Build B — fix support-vector forcing (cheap interim)
Build-3 wrong-V may be partly a **support→B-orbit alignment** bug (support mapped to base roots by
index, ignoring the B-orbit). Align the forced subset to the target generator's B-orbit, re-measure.

### Build C — cocycle `[a]` layer (only if needed)
After V is controlled, re-classify. If **wrong-cocycle** then dominates, build twisted-wreath /
local-squareclass control (directive §12). The H¹ cohomologous test is written + unit-tested (§4).

### Build D — discriminant-quality levers (ongoing)
Base-field pool per top group sorted by `|D_K|` (squared lever — pick the smallest base); units for
signs; 2-adic conditioning. Keep field-disc vs poly-height scores separate (polredabs = presentation
only).

### Build E — native MAGMA-replacement (longer arc)
Port the identification + construction stack into the monorepo so we stop calling MAGMA — see
**Appendix A** (the L0–L7 module→crate map). Not required to score this week; the S-unit constructor
(Build A) is the first native piece and belongs in `rustmath-numberfields`.

---

## 4. Validation discipline (do not skip)

**Measure before grinding. The milestone is one batch of *diagnosed* candidates, not more candidates.**
- Every constructed poly → identify → classify the miss by **`(B,V,[a])`**:
  `wrong_B | wrong_V | wrong_cocycle | exact`. Prototype `inverse_galois/frobenius/ghost_classify.sage`
  (H¹ cohomologous test over F₂ implemented + unit-tested; B/V triage validated on known ghost pairs).
- Reachability already audited (`fingerprints.jsonl`): **93% of cluster groups reachable by a single
  relative quadratic** (cyclic V); **~1,600 low-weight (≤3)** — start there. Decision forks (directive §13):
  wrong-V → tighten squareclass control (S-unit) ← *where we are*; right-V/exact → scale; wrong-B → base selection broke.

---

## 5. What already exists — REUSE, don't rebuild

### Native Rust (this monorepo, branch `integrate-lava-galois`) — the S-unit build's foundation
The merge of `igp24-cli-galois` + `p1-prime-ideals` put **both** halves the S-unit constructor needs
in one tree:
- `rustmath-numberfields/src/` — **`s_units.rs`, `units.rs`, `ramification.rs`, `local_field.rs`,
  `polred.rs`, `round2.rs`** (maximal order, S-units, ray-class infra, relative ramification).
  **Build A goes here, reusing this.**
- `rustmath-groups/` — full degree-24 transitive-group atlas + block systems + Frobenius narrowing.
- `rustmath-igp24/` — native JSON Galois-ID CLI (factor / p-adic / disc-polred / narrow).
- `rustmath-matrix/` (LLL/HNF/SNF), `rustmath-polynomials/` (factorization, resultants), `rustmath-finitefields/`.
- `rustmath-lava/` — Magma-language facade. Build: `cargo build -p rustmath-numberfields` (toolchain 1.92; workspace = 65 crates, builds clean).

### Sage/PARI prototypes (`/Users/john/Documents/inverse_galois/frobenius/`)
- `fingerprint_audit.sage` → `fingerprints.jsonl` (per (t,12×2): B, dim V, cyclic, min_gen_weight, gen_supports).
- `ghost_classify.sage` — H¹ cohomologous test + (B,V,[a]) classifier.
- `build3_support_vector.sage` — split-prime support forcing (leaky; reuse its construction scaffolding).
- `directed_engine.sage` — interim neighbour-harvester. `generate.sage` — tapped-out broad samplers (has `resolve_gated`, `bank_append`, `load_admissible`).

### Data (`/Users/john/Documents/inverse_galois/`)
- `frobenius/all_placements/*.jsonl` — **all 51 teams, 95,578 placements** (full global k-map: `t,r,points,kTeams,scoringDiscAbs,minScoringDiscAbs`). Scrapers `scrape_placements.py`/`scrape_all.py` (public API, no auth: `GET server-9527.sair.foundation/api/igp24/leaderboard/teams/{teamv2_id}/placements?limit≤100&cursor=`).
- `frobenius/fingerprints.jsonl` — reachability audit. `igp24-coverage/data/hit_list.csv` — 123,703 ranked open pairs.
- `frobenius/directed_targets.csv` — 23,634 KLPB-cluster open targets. `frobenius/lmfdb_nf12_seeds.json` — **301 degree-12 bases keyed by `12Tj`** (the directed-construction base library).

### Submission pipeline (running — leave alone)
`frobenius/submitter.py` (pair-aware, KLPB-cluster priority) drains `frobenius/bank_pending.jsonl` →
SAIR API ≤99/day. New constructions bank there tagged `{coeffs,t,r,src}` and submit automatically.

---

## 6. Concrete first steps
1. `cd /Users/john/Documents/RustMath`; confirm branch `integrate-lava-galois`; `cargo build -p rustmath-numberfields`.
2. Read `inverse_galois/strategy_update_20260623.md` §6 (S-unit), §10–11 (disc), §13 (forks).
3. **Build A** Sage/PARI prototype on low-weight cyclic targets (weight ≤2, ~1,224 groups) from `fingerprints.jsonl`, base from `lmfdb_nf12_seeds.json`. Enumerate bounded S-unit squareclasses; keep `V_γ=V ∧ r(γ)=r ∧ min 𝔡`; classify via `ghost_classify.sage`.
4. Report the **ghost-class hit-rate table** before scaling. If exact hits appear, bank (tag `src:"sunit"`) — submitter sends them.
5. Confirm method in Sage, then port the squareclass search into `rustmath-numberfields` (reuse `s_units.rs`/`units.rs`/`ramification.rs`).

**Guardrails:** don't restart the tapped-out broad engines; don't grind without classifying; keep field-disc vs poly-height scores separate; smaller `D_K` (squared) is how we beat KLPB later. Append a stamped entry to `inverse_galois/inverse_galois_notes.tex` (Entries 1–33) when a build lands.

## 7. Repo / environment
- Monorepo `/Users/john/Documents/RustMath` (main checkout); worktree `/Users/john/Documents/rm-galois` (`igp24-cli-galois`). Integration on **`integrate-lava-galois`** — **not yet on `main` or pushed**.
- `rustmath-lava` = vendored Magma facade (MIT, havarddj fork) — keep its `LICENSE`.
- Sage: `sage` on PATH; prototypes use PARI + GAP (`TransitiveGroup`, `AllBlocks`, `TransitiveIdentification`) via Sage's `gap`/`pari`.

---
---

# Appendix A — Full MAGMA-replacement module map (infrastructure reference)

> This is the longer-arc "Build E": replacing MAGMA in our pipeline with a native Rust stack.
> The S-unit constructor above is its first piece. Below is the complete module→MAGMA-chapter→
> RustMath-crate→status map (REUSE / EXTEND / BUILD).

**Goal.** Replace the MAGMA calls our IGP24 pipeline depends on (degree-24 Galois verification,
number-field invariants, factorization, resolvents, local/ramification data) with a self-contained
Rust stack, reusing RustMath crates and adding only the missing pieces. `lava` today is Magma-*language*
tooling (`lava-core`+`lava-cli` over tree-sitter-magma/topiary) — that's the front-end (L7); the MVP
adds the compute back-end underneath.

## A.2 Architecture — dependency layers
```
L0  Exact kernel        ℤ ℚ ℤ/n 𝔽_p 𝔽_q
L1  Polynomials + linear algebra + lattices
L2  Number fields │ local fields/p-adics │ finite-field extensions │ function fields
L3  Groups: permutation groups + degree-24 transitive atlas
L4  Galois engine: Frobenius sieve → resolvents → Stauduhar descent → label ID
L5  Construction engines: signature seeds, CRT-LLL, Newton templates, Mestre, covers
L6  Databases + certificates + replay
L7  Front-end: Magma-syntax parser/DSL + CLI + orchestration   ← lava lives here
```

## A.3 Module map (module → MAGMA ch. → RustMath crate → status)

### L0 — Exact kernel
| Module | ch. | crate | status |
|---|---|---|---|
| Integers ℤ, ℤ/n (gcd, CRT, Miller–Rabin, ρ/p−1, ECM) | 17–19 | `rustmath-integers` | REUSE |
| Rationals ℚ, continued fractions | 18 | `rustmath-rationals` | REUSE |
| Finite fields 𝔽_p, 𝔽_{p^n} (Berlekamp, Conway, BSGS) | 21,48 | `rustmath-finitefields` | EXTEND (𝔽_{p^n} reduction; add Cantor–Zassenhaus + irreducible search) |

### L1 — Polynomials, linear algebra, lattices
| Module | ch. | crate | status |
|---|---|---|---|
| Univariate poly + factorization (Zassenhaus/Berlekamp/Hensel, resultant/disc) | 23 | `rustmath-polynomials` | REUSE; EXTEND (van Hoeij, Cantor–Zassenhaus) |
| Real-root isolation / Sturm | 23,25 | `rustmath-polynomials`,`-numerical` | REUSE/EXTEND |
| Multivariate + Gröbner (Buchberger/F4/F5/FGLM) | 24,105,106 | `rustmath-polynomials` (skeleton) | BUILD (resolvent/Mestre elimination) |
| Real & complex fields | 25 | `rustmath-numerical`,`-complex` (rug) | REUSE |
| Dense/sparse linear algebra | 26–28 | `rustmath-matrix` | REUSE |
| Lattices LLL/HNF/SNF (+BKZ, Fincke–Pohst) | 30,31 | `rustmath-matrix` | REUSE; EXTEND |

### L2 — Number / local / function fields
| Module | ch. | crate | status |
|---|---|---|---|
| Number fields & orders (𝒪_K, Δ_K, sig, integral basis, polred/polredabs; Round 2) | 34,37 | `rustmath-numberfields` | REUSE core; EXTEND |
| Ideals & prime decomposition (HNF ideals, Montes, different/conductor) | 34,37 | `rustmath-numberfields` | EXTEND |
| Class groups / units / regulator / **S-units** (Buchmann) | 34,39 | `rustmath-numberfields` (`s_units.rs`,`units.rs` present) | EXTEND (this is Build A's base) |
| Galois theory of number fields (subfields, automorphisms, splitting field) | 38 | `-numberfields`+`-groups` | EXTEND |
| Local fields / p-adics / Newton polygons / ramification (Hensel, Montes/OM, Krasner, Panayi) | 45–47,51 | `rustmath-padics`(thin),`-polynomials` | EXTEND/BUILD (Montes/OM = big gap) |
| Function fields ℚ(t)[x] (Trager, Newton–Puiseux, specialization) | 41,42 | — | BUILD (regular-cover scanning) |
| Class field theory (ray class, Artin map) | 39,43 | — | BUILD (secondary) |

### L3 — Groups
| Module | ch. | crate | status |
|---|---|---|---|
| Permutation groups (Schreier–Sims, blocks, conjugacy) | 57,58 | `rustmath-groups`; `-interfaces` GAP | REUSE (deg-24); BUILD/bridge (general) |
| **Degree-24 transitive atlas** (25k groups + cycle types + blocks) | 58,66 | `rustmath-groups` (`transitive24.rs`) | REUSE ★ |
| Automorphism / character theory (Dixon–Schneider, involution profiler) | 67,91 | — / GAP | BUILD/bridge |

### L4 — Galois engine
| Module | ch. | crate | status |
|---|---|---|---|
| Frobenius shadow sieve (Dedekind/Chebotarev, blind-cell) | 38,58 | `rustmath-groups`,`-polynomials` | REUSE ★ |
| Root labeling (ℂ + p-adic) | 38 | `-numerical`,`-polynomials` | REUSE/EXTEND |
| Resolvent construction (Lagrange/Stauduhar, k-subset orbits) | 38 | `-polynomials`,`-groups` | REUSE |
| **Generic Stauduhar descent** (Fieker–Klüners) | 38 | partial | BUILD ★ hardest |
| Galois label ID G↦24Tt | 38,58 | `-groups`,`-igp24` | REUSE |
| Exact oracle (Fieker–Klüners via OSCAR/Hecke) | — | external `oscar_bridge` | REUSE (bridge) |

### L5 — Construction engines (Sage → port to Rust)
Signature seeds, CRT-LLL lift, Newton-polygon templates, Mestre–Vila A₂₄, regular-cover scanner,
real-fiber analyzer, relative-fiber/directed constructor — all **BUILD/PORT** from
`inverse_galois/frobenius/*.sage`.

### L6 — Databases / certificates / replay
Target-pair cube + Frobenius-blind atlas (REUSE/formalize from `igp24-coverage/data`); ghost ledger
+ certificate schema + deterministic replay (BUILD — credibility).

### L7 — Front-end
Magma parser/formatter (`lava-core`/`lava-cli`, REUSE); intrinsic dispatch `GaloisGroup(f)`/`NumberField(f)`
→ back-end (BUILD, thin); orchestration/batch (PORT from `frobenius/*.py`).

## A.4 External crates
Inherit: `num-bigint/-integer/-rational/-traits`, `ndarray`, `rayon`, `rand`, `serde`, `once_cell`,
`thiserror`; `rug` (GMP/MPFR/MPC) only in `rustmath-complex`. Front-end: `tree-sitter(-magma)`,
`topiary-core`, `clap`, `tokio`. Add: Gröbner (F4/F5), optional GAP bridge, OSCAR/Hecke oracle.
Bundled data: `transitive_24.jsonl` (~25 MB), `transitive24_cycletypes.jsonl` (~32 MB), Conway table.
**Bignum:** stay on `num-bigint` for the MVP; a GMP/`rug` swap is a later cross-cutting optimization.

## A.5 The hard new pieces (rough order)
1. Generic Stauduhar descent (L4). 2. Montes/OM p-adic factorization (L2). 3. Construction engines port (L5).
4. Gröbner F4/F5 (L1). 5. Certificate + replay (L6). 6. GF(p^n)/function fields/class groups. 7. General
Schreier–Sims/character theory (or GAP bridge). 8. DSL intrinsic dispatch (L7).

## A.6 Out of MVP scope
Commutative algebra beyond elimination, algebraic geometry/schemes/curves at scale, modular curves/forms
(except as projective-cover sources), Lie theory/Coxeter/Kac–Moody, representation theory beyond
involution characters, coding/designs, graphs/combinatorics, quantum/braid/automatic groups.

> **One line:** ~70–80% of the MVP already exists in RustMath (exact arithmetic, factorization,
> LLL/HNF/SNF, number-field invariants + polredabs, the degree-24 atlas + native Galois-ID CLI, and now
> S-unit/ray-class infra). The work is integration behind the Magma front-end + the few hard pieces above —
> with the **S-unit squareclass constructor (Build A) first**, because it's what restarts harvesting and
> closes the discriminant gap.
