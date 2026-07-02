# rustmath-curves — build status

## Current state (2026-07-02)

- **Default build** (`cargo build -p rustmath-curves`): **GREEN.** Compiles the
  genus-0 / Belyi / conic path and all lib-green curve modules.
- **`belyi` path**: usable now. Integration-tested by
  `tests/belyi_build_gate.rs` (`cargo test -p rustmath-curves --test belyi_build_gate`):
  portal `[2,12,5]` is degree-24 genus-0; Riemann–Hurwitz genus = 0; the
  Granboulan/Müller conic gate reads `LocallyEmpty`.
- **`--features genus2`**: **RED** (tracked debt) — the hyperelliptic/Jacobian
  stack has 25 lib errors. Enabling the feature will not compile until U1.

## Milestones (see CURVES_BUILD_BLOCKERS_SPEC.md)

- **U0 — belyi-unblock — DONE.** `genus2` feature added (off by default);
  `divisor`, `hyperelliptic`, `cantor`, `jacobian` gated in `lib.rs` (modules +
  re-exports); default build green; `tests/belyi_build_gate.rs` green.
- **U1 — genus2-restore — TODO.** Fix the 25 lib errors so
  `cargo build -p rustmath-curves --features genus2` is green. Repair order and
  patch classes P1–P5 in CURVES_BUILD_BLOCKERS_SPEC.md §4–5.
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
