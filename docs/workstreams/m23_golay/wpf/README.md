# WP-F records

The records of work package WP-F (the Frobenius table of the M23 curve X). The brief is
`../HANDOFF.md`, Section 3; its Status block says what was reproduced and what is open.

| File | What |
|---|---|
| `design/WPF.md` | The design, with the prototype results and the test plan |
| `design/WPF.review1.md` | The design review (decision D1 rejected: Decided only for the nodal certificate) |
| `impl/WPF.md` | The implementation report: changes, tests, regenerated tables, follow-ups F-1 to F-11 |
| `impl/WPF.review1.md` | The code review (approved, six minor items) |
| `impl/WPF_docs_check.md` | The independent check of the tracker and handoff edits (approved, six minor items, fixed; item 6 by a note on the `ch118.IsWeil` override) |
| `impl/WPF_minors.md` | The fixes of the code review's six minor items, with the new tests and their PARI/GP values, and the fix-up after the checks of those fixes |
| `gp/` | The PARI/GP scripts behind the reference values of `rustmath-curves/tests/biprojective_small_curves.rs` and of the overflow test in `rustmath-curves/src/weil.rs`, each with its output (`*.out.txt`, PARI/GP 2.15.4) |

The reports are copied unchanged from the session that wrote them. Line numbers in `impl/WPF.md`,
the reviews and the docs check refer to commit `a62da12`; the minor fixes moved some of them.
`impl/WPF_minors.md` lists the new lines of the main items (its last section gives those moved
again by the fix-up), and the Status block of `../HANDOFF.md` is current. Paths of the form
`scratchpad/m23/...` name that session's scratch directory, which is not kept.

The PARI/GP scripts and raw outputs of that directory are not copied, except the small-curve
scripts in `gp/`. Some of the others hold the coefficients of HJLPPZ's `polynomial_F.m`, which has
no licence and must not enter this repository; the scripts in `gp/` hold no HJLPPZ data. Run them
from `gp/` (`small_curves4.gp` reads `small_curves2.gp`) with `gp -q <name>.gp < /dev/null`. Their
origin in the scratch directory:

| Script | Origin | Test cases |
|---|---|---|
| `small_curves.gp`, `branch.gp` | `design/WPF/gp/` (the design) | curves A to I2, E1, E2 |
| `tangent.gp` | `design/WPF/review1/` (the design review) | curve T3 (`tacnode_tangent_to_the_fibre`) |
| `small_curves2.gp`, `small_curves4.gp` | `impl/gp/` (the implementation) | curves J, J', J2, K1, K2 |
| `verify.gp` | `impl/minors/` (the minor fixes) | the six curves taken from code review 1, Section 2.2, and E6 |
| `two_torsion_overflow.gp` | `impl/fixup/lcm_case.gp` (the fix-up) | `weil.rs::two_torsion_order_above_u64_panics` |
