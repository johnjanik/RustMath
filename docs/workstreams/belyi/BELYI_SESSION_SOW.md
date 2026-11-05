# Statement of Work — Belyi session

**Audience:** a Claude session working on the Belyi / dessin crates.
**Author:** the Phase-0 session (RustMath "green + honest" sprint).
**Date:** 2026-07-09. **Branch:** `belyi/u0-unblock-and-true-root-detector` @ `2bca72d`.

Everything below was verified against the working tree on the date above. Where a
pre-existing document contradicts this one, this one is newer — see §8.

---

## 0. TL;DR — read these five things

1. **A doctest allocated 89 GB and the kernel OOM-killed the terminal, crashing a
   session.** Root cause found and fixed. See §3. It was *not* your sweep.
2. **Every `cargo` command must run inside a memory-capped cgroup.** See §5. This
   is not optional. An earlier OOM (2026-07-05) already reaped a `rustmath_curves`
   process.
3. **Nothing is committed.** 27 crates carry uncommitted Phase-0 work. Do **not**
   run `git stash`, `git checkout -- .`, `git reset --hard`, or `git clean` at repo
   scope. You would destroy a day of work in crates you do not own. See §2.
4. **Your two `dump_2_12_5_matrix_ext_streamed` jobs are alive and progressing.**
   Do not kill them, and do not run anything that competes for all 24 cores. See §5.
5. **Phase-0 cannot affect the Belyi stack.** This was verified, not assumed. See §6.

---

## 1. Ownership and boundaries

### You own

| Crate / path | Note |
|---|---|
| `rustmath-curves/src/belyi/` | 32 modules, ~12,270 lines, ~130 `#[test]`, 6 `#[ignore]`d. |
| `rustmath-curves/src/bin/belyi-atlas.rs` | Auto-discovered bin (no `[[bin]]` stanza). |
| `rustmath-curves/python/oa/` | The Python/Sage fan + ladder pipeline. |
| `rustmath-igp24/` | Thin JSON CLI over the number-theory engine. |
| The `sweep_m24_348` campaign | `/home/john/sweep_m24_348`, 48 GB, 5,767 files. |

### You must not touch

Everything else. In particular, 27 crates hold **uncommitted** Phase-0 changes
(§2). Editing or reverting them will collide.

### Correction: `rustmath-lava` is not a Belyi crate

`rustmath-lava/crates/{lava-core,lava}` is a **vendored Magma-language tooling CLI**
(tree-sitter formatter, highlighter, test discovery; from `github.com/havarddj/lava`,
MIT). A grep for `belyi|dessin|triangle_group|modular_form` across it returns zero
hits. It is not in the Belyi dependency graph.

Do not confuse it with `rustmath-lava/lava_mvp.md`, which *is* a live planning
document (the "Rust replacement for MAGMA" MVP) and shares the directory.

### Correction: the `rustmath-active-workers` memory is stale

That memory (2026-07-01) lists `numberfields, quadraticforms, groups, polynomials,
numerical, curves, igp24, lava` as off-limits. As of today:

| Crate | Uncommitted changes | Owner |
|---|---|---|
| `numberfields` | **0** | — |
| `quadraticforms` | **0** | — |
| `polynomials` | **0** | — |
| `igp24` | **0** | — |
| `curves` | 5 | **you** |
| `lava` | 1 (README) | you |
| `numerical` | 7 | Phase-0 |
| `groups` | 10 | Phase-0 (test code only) |

The IGP24 crates show no in-tree activity. Whoever updates that memory should say so.

---

## 2. State of the working tree

**Nothing has been committed.** `HEAD` is `2bca72d`. The working tree carries:

- **Phase-0 sprint + remediation** across 27 crates (`rings` 47 files, `manifolds` 30,
  `combinatorics` 16, `algebras` 15, `modules` 12, `liealgebras` 12, `groups` 10,
  `modular` 8, `numerical` 7, …). Two of those crates (`groups`, `combinatorics`)
  were still being worked when this was written.
- **Your own uncommitted work:** `rustmath-curves/src/belyi/modular_forms_hp.rs`
  (+56/−4), `rustmath-lava/README.md`, and four untracked files under
  `rustmath-curves/python/oa/`.
- A `docs/` reorganisation (70 root `.md` files moved to `docs/archive/`).

### Your `modular_forms_hp.rs` diff is complete, not mid-edit

Three coherent, backward-compatible changes, all inside `mod tests`:

1. **Degree generalisation** (L748-753): `assert_eq!(s0.len(), 24)` → `assert_eq!(s0.len(), s1.len())`
   plus a degree log line, so non-degree-24 passports (M23 = 23) run the same harness.
2. **Env-configurable compactify** (L783-787): the hardcoded `compactify_with(&tg64, 0.996, 40)`
   now reads `M_RPRUNE` (default `0.996`) and `M_LMAX` (default `40`) — needed for
   larger triangle groups such as (3,4,8). Defaults preserve prior behaviour.
3. **New `#[ignore]`d test `print_reps_348`** (L822-889): dumps hp coset reps and
   `z_a/z_b/z_c`, `delta_a/b/c` matrices at full precision, for the Python
   multi-frame atlas glue.

It is safe to commit as-is. **Task:** commit it, and decide whether the four
untracked `python/oa/` files are tracked or ignored (see §7).

### Danger: repo-scope git commands

During the Phase-0 remediation, several agents ran repo-wide `git stash` /
`git stash pop` on a tree with concurrent writers. Each reported clean restoration,
but this is exactly how uncommitted work in 27 crates gets destroyed. **Scope every
git command to your own paths**, e.g. `git add rustmath-curves/`, never `git add -A`.

---

## 3. What produced the OOM

### The evidence

`/var/log/kern.log`, 2026-07-09 11:07:14:

```
kdeconnectd invoked oom-killer: gfp_mask=0x140cca, order=0, oom_score_adj=200
oom-kill: ... task=rust_out,pid=1041977,uid=1000
Out of memory: Killed process 1041977 (rust_out)
  total-vm:134221284kB  anon-rss:89095528kB  pgtables:188340kB
ptyxis-spawn-45c258e9-….scope: Failed with result 'oom-kill'
```

`rust_out` is the binary name **rustdoc gives a doctest executable**. One doctest
took 89 GB resident (134 GB virtual) on a 122 GiB box, tripped the *global* OOM
killer, and took the terminal scope — and the session — with it.

There is precedent. 2026-07-05: `Out of memory: Killed process 2872293 (claude)`,
followed by `oom_reaper: reaped process 3238274 (rustmath_curves)`. **A previous OOM
already destroyed a `rustmath_curves` run.** Today's spared your two jobs only by luck.

### It was not the sweep

Measured, not assumed. The two `dump_2_12_5_matrix_ext_streamed` processes:

| | |
|---|---|
| CPU time vs elapsed | 378,607 s / 49,521 s and 296,327 s / 39,143 s |
| ⇒ | ~7.6 cores each, pinned at ~760% CPU, 10 threads apiece |
| **Peak memory** | **683 MB** (`VmPeak`), 53 MB resident |

Compute-bound, memory-trivial. Innocent.

### The culprit

`GelfandTsetlinPattern::to_tableau`, `rustmath-combinatorics/src/gelfand_tsetlin.rs`.
**Three pre-existing bugs were stacked, each concealing the next.**

**Bug 1 — unsigned underflow into an unbounded loop.** `rows: Vec<Vec<i64>>`. In the
`level == 0` branch:

```rust
let start = if i == 0 { 0 } else { current_row[i - 1] };  // 3
let end   = current_row[i];                                // 2
let count = (end - start) as usize;   // (2-3) = -1  →  18446744073709551615
if count > 0 { for _ in 0..count { tableau_rows[i].push(1); } }
```

Gelfand–Tsetlin rows are weakly **decreasing** *by definition*, so `end - start` is
negative for essentially every non-trivial pattern. `as usize` wraps to ~1.8×10¹⁹ and
the loop pushes until the kernel intervenes. (The sibling `else` branch had an
`if tableau_rows[i].len() < n` cap; the `level == 0` branch had none.)

**Bug 2 — the interlacing check in `new()` was backwards.** It coded
`rows[i-1][j+1] >= rows[i][j]`; the true GT condition is `upper[j] >= lower[j] >= upper[j+1]`.
So `new()` *rejected valid patterns*. Verified counterexamples, both valid, both
returning `None`: `[[4,2,1,0],[3,2,0],[2,1],[1]]` and `[[2,0],[1]]`.

**Bug 3 — a broken test target hid both.** `cargo test -p rustmath-combinatorics`
aborted on 48 `#[cfg(test)]` compile errors *before doctests ever ran*.

### The chain

Bug 2 rejected the original doctest's input, so `.unwrap()` panicked — the doctest
failed fast and never reached `to_tableau`. An agent, fixing that failure, substituted
`[[3,2,1],[2,1],[1]]`, which the buggy `new()` happens to accept. `to_tableau` finally
executed, hit Bug 1, and ate 89 GB.

**The agent's edit was reasonable.** It simply removed the accidental guard that had
been hiding a live 89 GB defect in shipped library code. The lesson is Bug 3: *a
broken test target is not inert — it is a blindfold.*

### The underflow class, honestly scoped

A grep for `(a - b) as usize` matches **32 sites across 21 crates**. That is a
**syntactic count, not 32 bugs.** Every one that was checked is guarded:

- `integers/fast_arith.rs:370` — guarded by `if start >= end || end <= 2 { return }`, and `u64` anyway.
- `numbertheory/bernoulli.rs:277` — guarded by `assert!(p >= 5)`.
- `groups/free_group.rs:461` — inside the `else` of `if e > 0`, so `-e >= 0`.
- `curves/riemann_roch.rs:534` — guarded by `if degree < g { 0 } else { … }`. **Safe.**
  It also has no non-test callers; the live paths (`RiemannRochSpace::dimension`,
  `riemann_roch_dimension`) use `.max(0)`.

The `to_tableau` site was the only unguarded one in the workspace. A sweep of
`rustmath-combinatorics` found no others. Treat the pattern as a *code smell worth a
guard*, not as an emergency.

---

## 4. The combinatorics fix

`rustmath-combinatorics` does **not** appear in the Belyi dependency graph, so none
of this changes behaviour you depend on. It is recorded here because it is the reason
your machine went down, and because the bug class generalises.

### What changed (all in `src/gelfand_tsetlin.rs`, signature-stable, crate-local)

| Item | Fix |
|---|---|
| `to_tableau()` | Rewritten as the real GT ↔ SSYT bijection: with `λ⁽ᵏ⁾ = rows[n−k]`, row *i* holds exactly `λ⁽ᵏ⁾ᵢ − λ⁽ᵏ⁻¹⁾ᵢ` copies of *k*. `usize::try_from` replaces `as usize`; added a triangle guard `rows.len() == rows[0].len()`. |
| `new()` | Backwards interlacing corrected to `upper[j] >= lower[j] >= upper[j+1]`. |
| `is_valid()` | Same bug, mirrored — corrected identically, so `new()` and `is_valid()` agree. |
| `generate_next_row_recursive()` (private) | Enumerated `0..=min(prev[pos], prev[pos+1])`, a different and wrong pattern set. Now enumerates `prev[pos+1]..=prev[pos]`. |
| `from_tableau()` | Was a masked defect (its doctest had been weakened to `let _pattern = …` with an excuse comment). Reimplemented as the exact inverse of `to_tableau`, with a round-trip test. |

### Ripple, handled honestly

Because `new()` now accepts strictly more patterns, three unit tests had encoded the
*old buggy* behaviour. They were corrected, not weakened: `test_invalid_interlacing`
and `test_specific_interlacing_conditions` now use genuinely invalid patterns, and
`test_generate_patterns_counts` for top row `[1,0]` asserts the correct count **2**
(was 1) — matching the two SSYT of shape (1).

### Result

`cargo test -p rustmath-combinatorics`: **lib 1429 passed / 0 failed / 7 ignored**
(all 7 pre-existing, in untouched files); **doctests 185 passed / 0 failed / 11 ignored**.
Zero new `#[ignore]`s. `to_tableau`'s doctest now asserts the exact tableau
`[[1,2,3],[2,3],[3]]` and completes in 0.54 s.

---

## 5. Operating rules — mandatory

The box has **24 cores / 122 GiB RAM / 8 GiB swap**. Your sweep already pins ~15 cores.

### 5.1 Cap every cargo command

`systemd-run --user --scope` with cgroup-v2 `MemoryMax` is confirmed working. A
runaway then dies inside its own cgroup instead of tripping the global OOM killer.

```bash
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  timeout -s KILL 900 \
  cargo test -p rustmath-curves --lib -j4
```

- **Never** run a bare `cargo` command.
- **Always** `-j4`. The sweep owns the rest of the cores.
- Exit **137** or **15** means the cap or timeout fired. That is a **bug to
  investigate** — an unbounded loop, an underflow cast, an accidental enumeration of
  an infinite object. **Never respond by raising the cap.**

### 5.2 Never run the full workspace

`cargo test --workspace` builds test binaries for 66 crates, competes with the sweep
for every core, and (as of today) several crates' test targets are only just repaired.
Build and test **only your crate**.

### 5.3 Doctests are executable code

This whole incident was a doctest. `cargo test -p <crate>` runs them. When you touch a
`///` example, you are writing a program that will run. If a doctest is a *dump* or a
*probe*, fence it ```` ```ignore ```` or ```` ```no_run ````.

### 5.4 Do not disturb the sweep

Two live jobs, actively checkpointing (verified: `.progress` mtimes seconds old):

| PID | Started | `M_OUT` | Params |
|---|---|---|---|
| 594774 | Jul 9 03:31 | `sweep_m24_348/m348_B0c4.bin` | `M_N=13200 M_K=4 M_PREC=200 M_LIMBS=3` |
| 622785 | Jul 9 06:24 | `sweep_m24_348/m348_B0c21.bin` | idem |

They write distinct outputs (no checkpoint collision) and resume from `<out>.progress`.
`M_N=13200` — far above the `600` default — is why they run for 14 h.

---

## 6. Verified: Phase-0 cannot reach the Belyi stack

Checked explicitly, because Phase-0 converted a number of silent facades into
`unimplemented!()` panics — an API that used to return a plausible wrong number now
aborts. If Belyi called one, it would newly panic.

**It does not.** `belyi/` imports exactly:

`rustmath_polynomials` (11 uses) · `rustmath_quadraticforms` (9) · `rustmath_numerical` (7) ·
`rustmath_groups` (4) · `rustmath_rationals` (8) · `rustmath_integers` (4) ·
`rustmath_core` (4) · `rustmath_powerseries` (1) · `rustmath_numberfields` (1) · external `rug`.

It does **not** use `rustmath_rings`, `rustmath_schemes`, `rustmath_finitefields`,
`rustmath_symbolic`, or `rustmath_matrix` (hp linear algebra is hand-rolled on `rug`
in `mp_svd.rs`).

| Risk | Finding |
|---|---|
| `numerical` — Phase-0 made LP/MIP panic | Belyi uses `exactify`, `homotopy`, `root_finding`. Phase-0 touched only `linear_programming`, `backends/*`, `brent`, `gauss_legendre`, `lib.rs`. **No overlap.** |
| `rings` — Phase-0 made 8 `divisor` methods panic | The sole reference is inside a ```` ```ignore ```` fence in a `riemann_roch.rs` doc comment. The only real `use` there is `rustmath_core::Field`. **No dependency.** |
| `groups` — a Phase-0 agent is editing it | `transitive23.rs`, `transitive24.rs`, `perm_predicates.rs` — the three files Belyi imports — are **untouched**. |
| `polynomials`, `quadraticforms`, `numberfields` | **Zero** uncommitted changes. |

`cargo build -p rustmath-curves --tests` is **exit 0** today (lib and test target both).

---

## 7. What needs to be done

Ordered. Items 1–3 are housekeeping that unblocks everything else.

### 1. Commit your own work (scoped)

`modular_forms_hp.rs` (+56/−4) is complete (§2). Commit it and `rustmath-lava/README.md`.
Scope the command: `git add rustmath-curves/ rustmath-lava/README.md`. **Never `git add -A`** —
27 other crates are dirty.

### 2. Decide the four untracked `python/oa/` files

- `c343_ladder.py` — (3,4,3) precision ladder: square holomorphic Newton on
  `A³ − λR³S − c·W⁴ = 0`, analytic Jacobian, dps-doubling rungs, 25 complex unknowns.
- `pM_fan_stage1.py` — M12:2 specialization fan, stage 1: polish θ (Newton, FD Jacobian,
  dps 820); per `t₀` build the degree-24 fiber resolvent, CF-recognize coefficients as
  exact rationals → `pM_fan_t{tag}.json`.
- `pM_fan_stage2.sage` — stage 2 (Sage): build `F ∈ ℚ[y]`, irreducibility, `polredbest`,
  real-root count, `nfdisc` (900 s guard), gate coeff-digits ≤ 40 → `pM_fan_stage2.json`.
- `pM_fan_stage3_bank.py` — stage 3: bank passing polynomials to `bank_pending.jsonl`
  (dedup) + full metadata to `pM_fan.jsonl`.

Track them (they are pipeline source, and `ff5abc8` already tracks `python/oa`) or
`.gitignore` them. Do not leave them untracked — they are load-bearing.

### 3. Retire or rewrite `CURVES_BUILD_BLOCKERS_SPEC.md`

**It is stale and actively misleading.** It asserts (2026-07-02) that the `curves` lib
does not compile — 25 errors in `cantor.rs` / `divisor.rs` / `jacobian.rs` /
`hyperelliptic.rs` — and that `belyi` is therefore blocked.

Today: `cargo build -p rustmath-curves --tests` is **exit 0**. The lib *and* the test
target compile. Strategy A (fix the errors) evidently landed.

Two loose ends it leaves:

- `rustmath-curves/Cargo.toml:41` still declares a **`genus2 = []` feature**, but
  `lib.rs` declares `hyperelliptic`, `divisor`, `cantor`, `jacobian` as plain
  ungated `pub mod`. The feature is **vestigial**. Remove it, or gate something with it.
- `CURVES_note.md` (844 lines) is a review of that now-obsolete spec. Fold what is
  still true into a short status note; archive the rest to `docs/archive/`.

### 4. Close the `riemann_roch` placeholder question

`belyi/monodromy.rs:15` explicitly calls `riemann_roch.rs` a placeholder and computes
genus locally via Riemann–Hurwitz (`genus_from_branch_cycles`) instead. That is the
right call for Belyi, but the module remains a trap for other callers
(`differentials.rs:337`, `special_divisors.rs:93,98`).

Either implement it honestly or mark it `unimplemented!()` per the Phase-0 honesty
rule. Meanwhile, add the missing defensive `.max(0)` at `riemann_roch.rs:534` so
`expected_dimension` does not depend solely on its guard (its sibling at L517/520
already has it).

### 5. The acceptance gates that define "done"

From `DESSIN_REFACTOR_PLAN.md` §Wave-3 and §Integration:

- **The Müller / G6 gate:** end-to-end, recover the Müller conic `(-1,-1)` ramified
  exactly at `{2, ∞}`, verdict `LocallyEmpty`, from a cover. `pipeline::g6_mueller_gate`
  exists — **prove it passes end-to-end**, not just in isolation.
- **dessin_engine retirement:** the plan retires `/home/john/inverse_galois/M23/dessin_engine`
  once `rustmath-curves::belyi` reproduces its **101 tests' coverage** plus the Müller
  gate. `belyi/` currently has ~130 `#[test]`. **Audit coverage parity** — a larger test
  count is not the same as covering the same behaviour. Then retire it explicitly.

### 6. Reconcile the domain-radius finding with the sweep

`genus0_map.rs:300` (`explore_2_12_5_domain`, `#[ignore]`d) records ρ ≈ 0.9906 ⇒ **N ≈ 3000**.
The live sweep runs **N = 13200**, 4.4× that. Either the sweep is deliberately
over-resolved (say so, in the atlas metadata) or the estimate is stale. This is worth
knowing before another 14-hour job.

### 7. Guard the frozen numerics

`belyi-atlas.rs` states the numerics are **FROZEN**: every matrix entry must be
bit-identical to `dump_2_12_5_matrix_ext_streamed` for the same resolved parameters,
because both funnel through `run_atlas_dump` (`modular_forms_hp.rs:452`).

There is no test asserting that. **Add one** — a small-N (`M_N` ≈ 40) comparison of
`run_atlas_dump` against the harness — so the invariant is enforced rather than
documented. Without it, a refactor silently invalidates the 48 GB atlas.

---

## 8. Documents to distrust

| Document | Status |
|---|---|
| `CURVES_BUILD_BLOCKERS_SPEC.md` | **Stale.** Its 25 lib errors are fixed; `curves` builds. See §7.3. |
| `CURVES_note.md` | Review of the above. Mostly obsolete. |
| memory `rustmath-active-workers` | **Stale.** Four of its eight "off-limits" crates show zero activity. See §1. |
| `docs/SURVEY.md`, `docs/PLAN.md` | Current as of 2026-07-09, but written *before* the 15 broken test targets were found. Test-health numbers there are optimistic. |
| `CLAUDE.md` "Known Issues" | Says `rustmath-category` has dyn-compat errors. Fixed; 144 tests pass. |

Authoritative as of today: **this file**, and `docs/SURVEY.md` for scope (not health).

---

## 9. Definition of done for this SOW

- [ ] `modular_forms_hp.rs` + `README.md` committed, scoped to your paths.
- [ ] `python/oa/` files tracked or ignored, deliberately.
- [ ] `CURVES_BUILD_BLOCKERS_SPEC.md` retired/rewritten; `genus2` feature removed or used.
- [ ] `riemann_roch.rs:534` given its `.max(0)`; the module made honest.
- [ ] A bit-identity test pins `run_atlas_dump` to the frozen harness.
- [ ] The Müller / G6 gate demonstrated end-to-end.
- [ ] dessin_engine coverage parity audited; retirement decided.
- [ ] `cargo test -p rustmath-curves` green **under a memory cap**, `-j4`.

---

## 11. Campaign isolation — cross-session findings (2026-07-09, later)

Added after the Belyi session reviewed §6 and identified a coupling this document
had not closed. Their analysis was right about the mechanism. Verified from both
sides; two of their three hazards survive, one is closed.

### The coupling

`/home/john/sweep_m24_348/dump348.sh:10-14` does `cd /home/john/RustMath/rustmath-curves`,
then `cargo test --release dump_2_12_5_matrix_ext_streamed -- --ignored --nocapture`.
So **every coset launch recompiles `rustmath-curves` and the libs of its dependency
crates.** No `-j` limit, no `nice`, no memory cap.

### CLOSED — Phase-0 cannot change the numbers

The frozen dump path has **zero `rustmath_*` imports**:
`modular_forms_hp.rs`, `mp_svd.rs`, `coset_graph.rs`, `triangle_group.rs`,
`triangle_group_hp.rs` are pure `rug` + std. Only `hypergeometric.rs` reaches out —
to `rustmath_core`, `rustmath_powerseries`, `rustmath_rationals` — and **none of those
three crates carries any uncommitted change.**

⇒ No Phase-0 edit can perturb a matrix entry. The "changed numerics" hazard is real
for *future* edits to Belyi's own numeric files; it is not live for the current diff.

### REAL — broken build ⇒ silent coset loss

This is the hazard that matters. If the tree does not compile when a coset fires,
`cargo test` fails, no `.bin` is produced, `is_done` stays false, and the coset is
dropped **silently**. `rustmath-groups` is a `curves` dependency and was being edited
by a Phase-0 agent while the campaign was live.

Status: that agent is **stopped**. `cargo build -p rustmath-groups --release` → exit 0.
The exact coset build (`cargo test --release dump_… --no-run -j4`) → **exit 0**.
Safe as of 2026-07-09 18:20.

### REAL — uncapped rebuild ⇒ contention

`dump348.sh`'s cargo invocation has no `-j`, no `nice`, no cgroup cap. A triggered
rebuild grabs all 24 cores and competes with the two running jobs.

### Disclosure: the test binary was rebuilt

Verifying the coset build (`--no-run`) rewrote
`target/release/deps/rustmath_curves-59c1b7410a347124`. Cargo's filename hash derives
from package/feature/profile metadata, **not source content**, so a rebuild reuses the
same path.

- The two running jobs are **unaffected** — they hold the old inode
  (`readlink /proc/<pid>/exe` shows `… (deleted)`).
- The next coset would have triggered the identical rebuild anyway, since Phase-0 had
  already dirtied `groups` and `numerical`.
- sha256, running vs rebuilt: `73d91ce2…` vs `3e230a1e…` — **they differ** (different
  dep libs linked). The dump path's *source* is unchanged, so output should be
  identical; nothing proves it. Precisely the gap §7.7 exists to close.

**The Jul-8 frozen binary was recovered from `/proc/594774/exe` and preserved:**

```
/home/john/sweep_m24_348/bin/m348_dump_frozen_jul8
sha256 73d91ce29a0223a99fa508ae3cfdf2db…    4,436,240 bytes
```

Both running jobs share that exact inode. To undo: `rm -rf /home/john/sweep_m24_348/bin`.

### Recommended fix — pin to the *preserved copy*, not the `target/` path

The Belyi session proposed pinning `dump348.sh` to
`target/release/deps/rustmath_curves-59c1b7410a347124`. **That path is not stable** —
cargo just overwrote it, as demonstrated above. Pin to the preserved copy instead:

```bash
# dump348.sh, replacing the `cargo test --release …` line
    /home/john/sweep_m24_348/bin/m348_dump_frozen_jul8 \
        dump_2_12_5_matrix_ext_streamed --ignored --nocapture \
```

Same cwd, same `M_*` env. This removes cargo from the campaign's critical path
entirely: no rebuild, no core contention, no numeric drift, immune to anything the
RustMath session does. It makes the frozen invariant **enforced by construction**
rather than by convention.

If you keep `cargo test`, at minimum add `nice -n 15` and `-j4`.

### Checked, not a bug

`dump348.sh` passes `M_COSET`, which is missing from the module's documented env
surface. It **is** read (`modular_forms_hp.rs:796, 802, 808`, inside the center
dispatch). The two live jobs differ correctly: `M_BASE=0 M_CENTER=c2`, `M_COSET=4` vs `21`.

### dessin_engine retirement

§7.5 gates retiring `/home/john/inverse_galois/M23/dessin_engine` on a coverage-parity
audit. That is a **deletion in a different tree**. Coordinate before removing; do not
let a RustMath session do it.

---

## 10. Open questions for John

1. **`groups`** — is it still contested by IGP24/dessin? A Phase-0 agent repaired its
   159 test-target compile errors (test code only; `transitive23/24` and
   `perm_predicates` untouched). Keep, or revert and defer?
2. **The 27 uncommitted crates** — Phase-0 has not been committed and has no
   authorisation to commit. Belyi commits should land first, scoped, to avoid
   entangling the two.
3. **`.claude/worktrees/`** — 9 stale `magma-port/wave*` worktrees, last commits 8 days
   old, **7.3 GB**. They also pollute every `grep`. Safe to remove?
4. **N = 13200 vs N ≈ 3000** (§7.6) — deliberate over-resolution, or a stale estimate?
