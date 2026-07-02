# rustmath-curves — build status

## Current state (2026-07-02)

- **Default build** (`cargo build -p rustmath-curves`): **GREEN.** Compiles the
  genus-0 / Belyi / conic path and all lib-green curve modules.
- **`belyi` path**: usable now. Integration-tested by
  `tests/belyi_build_gate.rs` (`cargo test -p rustmath-curves --test belyi_build_gate`):
  portal `[2,12,5]` is degree-24 genus-0; Riemann–Hurwitz genus = 0; the
  Granboulan/Müller conic gate reads `LocallyEmpty`.
- **`--features genus2` (lib)**: **GREEN** (U1 done). The 25 lib errors in
  `divisor`/`hyperelliptic`/`cantor`/`jacobian` are fixed. Group law exercised by
  `tests/genus2_jacobian.rs`
  (`cargo test -p rustmath-curves --features genus2 --test genus2_jacobian`):
  builds y²=x⁵−x over ℚ, checks P+0=P and 2·P=double(P) via Cantor.
- **`--features genus2` (full `cargo test`)**: still **RED** (U2) — test-only code
  in `riemann_roch`/`singularities`/`weierstrass`/`parameterization`/
  `special_divisors`/`lfunction`.

## Milestones (see CURVES_BUILD_BLOCKERS_SPEC.md)

- **U0 — belyi-unblock — DONE.** `genus2` feature added (off by default);
  `divisor`, `hyperelliptic`, `cantor`, `jacobian` gated in `lib.rs` (modules +
  re-exports); default build green; `tests/belyi_build_gate.rs` green.
- **U1 — genus2-restore — DONE.** All 25 lib errors fixed (order divisor →
  hyperelliptic → cantor → jacobian):
  - RC-1: `degree()→Option` (`unwrap_or(0)`/`is_some_and`/`==Some(0)`); `div_rem`
    now `Result` (handled via `?`-style match / `.expect` on nonzero divisors).
  - RC-2: added `+ EuclideanDomain` bounds (hyperelliptic impl, jacobian impls).
  - RC-3: **dropped** the unused `NumericConversion` bound from the main
    `HyperellipticCurve` impl (used only by `discriminant`, now in its own impl
    block) — so `contains_point` needs only `EuclideanDomain`, and `jacobian`
    never has to require `NumericConversion`.
  - **Foundation fix (rustmath-polynomials):** `UnivariatePolynomial::is_square_free`
    tested `gcd(f,f').is_one()`, but `gcd` is not normalized to monic, so every
    square-free f with a non-1 unit gcd was a false negative (blocked all curve
    construction). Now tests `gcd.degree() == Some(0)` (nonzero constant).
- **U2 — full-tests-restore — TODO.** `cargo test -p rustmath-curves --features
  genus2` green, incl. test-only repairs in `riemann_roch`, `singularities`,
  `weierstrass`, `parameterization`, `special_divisors`, `lfunction`.

## Debt (feature-gated under `genus2`)

`divisor`, `hyperelliptic`, `cantor`, `jacobian`.
Repair order: `divisor` → `hyperelliptic` → `cantor` → `jacobian`.

## Rule (status honesty)

The absence of `genus2` from the default build is **temporary debt, not
completion** of the genus-≥2 stack. Do not treat a green default build as
"hyperelliptic/Jacobian done." U1/U2 remain open.
