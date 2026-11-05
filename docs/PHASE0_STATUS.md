# Phase 0 — status and backlog (paused 2026-07-09)

Development was **paused** here at John's instruction. Nothing is committed. This file
records what landed, what is half-done, and what remains, so it can be resumed cold.

Companion: `../BELYI_SESSION_SOW.md` (the Belyi-session handoff, incl. the OOM
post-mortem and the m348 campaign coupling). `SURVEY.md` / `PLAN.md` for scope.

---

## 0. Tree state

- **Nothing committed.** `HEAD` = `2bca72d`, branch `belyi/u0-unblock-and-true-root-detector`.
- **27 crates carry uncommitted Phase-0 work.** Plus the Belyi session's own
  `rustmath-curves/src/belyi/modular_forms_hp.rs` (+56/−4) and 4 untracked `python/oa/` files.
- `cargo build --workspace` → **exit 0**.
- The m348 coset build (`cargo test --release dump_… --no-run`) → **exit 0**.
  ⇒ The live campaign will not lose a coset to a broken tree.

> **Do not run repo-scope git commands** (`git stash`, `git checkout -- .`,
> `git reset --hard`, `git add -A`, `git clean`). Two efforts share this tree.

---

## 1. What landed

### Sprint wave — "green + honest" (13 crates)
Converted category-3 **facades** (real-looking APIs returning plausible constants) into
`unimplemented!()` with stable signatures. Crates: `interfaces`, `finitefields`,
`category`, `modules`, `schemes`, `algebraic`, `topology`, `databases`, `numerical`,
`rings`, `modular`, `manifolds`, `ellipticcurves`.

**7 real bugs found and fixed**, not just facades:
1. `numerical` `golden_section_search` — wrong interval endpoint updates (errors to 0.47).
2. `topology` — `klein_bottle()` was an **octahedron** (S², χ=2); `torus()` didn't close (χ=2); `dunce_hat()` gave χ=−1. Replaced with verified SageMath triangulations.
3. `modular` cusp count — missing `euler_phi` in the Σ formula.
4. `modular` dimension formula — crude; `dim S₁₂(SL₂ℤ)` came out 0, should be 1 (⟨Δ⟩). → Cohen–Oesterlé.
5. `schemes` `find_points_up_to_height` — origin double-counted (5 vs 4).
6. `finitefields` `ExtensionField` division returned `self` (no inverse). Implemented ext-Euclid in `F_p[x]/(m)`.
7. `algebraic` `sign()`/`cmp()` silently returned `Equal` for distinct irrationals. Replaced with exact-rational interval refinement.

### Adversarial review + remediation
74 verification agents; 61/62 confirmed. Caught **one regression the sprint introduced**:
`modular_forms_gamma0(2,11)` returned 3 (true value 2) — `dims::eisen` overcounted the
weight-2 Eisenstein space (`#cusps`, should be `#cusps − 1`), masked by a `> 0` assertion.
Fixed and doubly verified: `M₂(Γ₀(11))=2`, `M₂(SL₂ℤ)=0`, `S₁₂=1`, `M₁₂=2`, `S₂(Γ₀(11))=1`.
Also relocated `unimplemented!()` out of `J0/J1::new` (they panicked on construction) into
lazy accessors, and restored `sqrt_poly` to return `None` rather than panic.

### Test-target wave (13 of 15 crates)
15 crates had **pre-existing** broken `#[cfg(test)]` targets (~319 errors; libs all fine).
Root causes: `num_*` non-adoption (`num_bigint::BigInt`/`num_traits` never implement
`rustmath_core::Ring` — orphan rule), API drift, renamed test symbols, a duplicated test block.

Green: `affineschemes`, `algebras`, `category`, `complex`, `graphs`, `liealgebras`,
`misc`, `padics`, `plot`, `plot3d`, `quantumgroups`, `special-functions`,
`symmetricfunctions`, and `combinatorics`.

### combinatorics — the 89 GB bug
`GelfandTsetlinPattern::to_tableau` had `(end - start) as usize` over `i64` rows that are
weakly **decreasing by definition** → wrapped to ~1.8×10¹⁹ → unbounded push → **89 GB →
kernel OOM → session crash.** Two more bugs sat beside it: `new()`'s interlacing check was
backwards (rejecting valid patterns), and `from_tableau` was a masked defect.
Fixed; `is_valid()` and the private generator carried the same interlacing bug and were
fixed too. **1429 lib + 185 doctests pass, 0 failed, 0 new ignores.**
Full post-mortem in `../BELYI_SESSION_SOW.md` §3.

---

## 2. Half-done — resume here

### `rustmath-groups` — STOPPED MID-EDIT
The agent was stopped (it edits a `curves` dependency while the m348 campaign is live).

- **13 files modified, 8 added `fn`s, unverified.** Lib compiles (`--release` exit 0).
- Belyi's three imports — `transitive23.rs`, `transitive24.rs`, `perm_predicates.rs` —
  are **untouched**. Confirmed.
- Its 159 test-target compile errors are **not** cleared.
- It reported a **real bug** just before being stopped, unfixed:
  `merge_elements` computes `a^(m/gcd)·b^(n/gcd)` — for `a` of order 4 and `b` of order 3
  that is `a⁴·b³ = e`, order **1** instead of **12**. The lcm construction needs a coprime
  decomposition `m'·n' = lcm(m,n)`. Verify whether `merge_elements` has non-test callers.
- An **earlier, killed** agent modified lib code against instructions: added
  `AdditiveAbelianGroup::element()` and a `mul()` alias on `FinitelyPresentedGroupElement`.
  These are still in the diff. **Review: justify as additive/signature-stable, or revert.**

**Decision needed first:** is `groups` still contested by IGP24/dessin? See §5.

### `rustmath-category` integration test
`tests/integration_tests.rs` fails to compile (14 errors): ambiguous `standard` glob import
between `coercion::*` and `algebraic_morphisms::*`, and `ProductRing::new(0i32, 0.0f64)`
needs `f64: Ring`. **Confirmed pre-existing** (reproduced via `git stash`). Lib + doctests
are green (144 + 3). This alone will red-light a naive `cargo test --workspace`.

---

## 3. Not started

1. **Full-workspace test-green.** Blocked on `groups` + `category`'s integration binary.
   Note `cargo test --workspace` aborts at the first crate whose test target fails to
   compile — use `cargo build --workspace --tests --keep-going` to enumerate breakage.
2. **Commit plan.** Nothing is committed and nothing is authorised to be. Per-crate,
   scoped commits; the Belyi session's commits should land first to avoid entangling.
3. **`as usize` underflow guards.** 32 syntactic matches across 21 crates. **Not 32 bugs** —
   every one checked is guarded (`fast_arith.rs:370`, `bernoulli.rs:277`,
   `free_group.rs:461`, `riemann_roch.rs:534`). `to_tableau` was the only unguarded one.
   Treat as a lint-worthy smell, not an emergency. Candidate for a clippy lint or a
   `checked_sub` policy.
4. **`num_*` non-adoption policy.** The single most common root cause across the workspace
   (110× E0277). Worth a workspace lint forbidding `num_bigint`/`num_traits` in favour of
   `rustmath_integers::Integer` / `rustmath_rationals::Rational`.
5. **Deferred honest-facade debt.** `ellipticcurves::curve.rs:224 two_torsion_rank()` still
   returns a hardcoded `1` (dead code, zero callers — harmless today, a trap if rewired).
   `modular::dims::eisen` still uses `num_divisors(N)` as the cusp count, exact only for
   squarefree `N` at weight ≥ 4 (affects no tested value).
6. **Cross-crate bug, unfixed.** `rustmath-symbolic` `differentiate()` matches symbols by
   unique id, but `Symbol::new("x")` mints a fresh id each call ⇒ `∂x/∂x = 0`. Surfaced by
   the manifolds work. Needs a fix in `symbolic`.

---

## 4. Operating rules learned the hard way

- **Cap every cargo command.** cgroup v2 + `systemd-run --user --scope` is confirmed working:
  ```bash
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    nice -n 15 timeout -s KILL 900 cargo test -p <crate> -j4
  ```
  Exit 137/15 means the cap fired — that is **a bug to investigate**, never a reason to
  raise the cap.
- **`-j4`, always.** The m348 sweep pins ~15 of 24 cores.
- **Never `cargo test --workspace`** while the campaign runs.
- **Doctests are executable code.** The crash was a doctest. Fence dumps/probes
  ```` ```ignore ```` or ```` ```no_run ````.
- **A broken test target is not inert — it is a blindfold.** `combinatorics`' 48 compile
  errors hid an 89 GB defect in shipped library code for who knows how long.
- **Never weaken an assertion to make a test pass.** The `> 0` assertion is what let the
  modular dimension regression through; `let _pattern = …` is what hid `from_tableau`.

---

## 5. Open decisions (John)

1. **`groups`** — still contested by IGP24/dessin, or free? The `rustmath-active-workers`
   memory says off-limits, but `numberfields`, `quadraticforms`, `polynomials`, `igp24`
   show **zero** uncommitted changes, so that memory looks stale. Keep the agent's 13-file
   diff, or revert `groups` to `HEAD` and defer?
2. **Commit authorisation** for the 27 dirty crates.
3. **`.claude/worktrees/`** — 9 stale `magma-port/wave*` worktrees, last commits 8 days old,
   **7.3 GB**, and they pollute every `grep`/`git grep`. Safe to `git worktree remove`?
4. **`dump348.sh` pinning** — see `../BELYI_SESSION_SOW.md` §11. Pin to the preserved
   frozen binary (`/home/john/sweep_m24_348/bin/m348_dump_frozen_jul8`), not to the
   `target/` path, which cargo overwrites.
